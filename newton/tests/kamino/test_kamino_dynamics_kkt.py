# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DYNAMICS: KKT PROBLEM
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics import KKTProblem, ProblemFormulation
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig, ConstraintComplianceConfig
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_cts_jacobians
from newton.tests.kamino.utils.make import (
    make_containers,
    make_generalized_mass_matrices,
    make_inverse_generalized_mass_matrices,
    update_containers,
)
from newton.tests.utils.basics import make_basics_heterogeneous_builder

###
# Helpers
###


class TestSetup:
    def __init__(self, device: wp.DeviceLike, sparse: bool, preconditioning: bool):
        builder = make_basics_heterogeneous_builder()
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = wp.spatial_vectorf(0.5, 0.0, -0.2, 0.0, 0.3, 0.0)
        model = ModelKamino.from_newton(builder.finalize(device=device))
        self.model, self.data, self.state, self.limits, detector, self.jacobians = make_containers(
            model=model, max_world_contacts=12, sparse=sparse
        )
        self.contacts = detector.contacts
        update_containers(
            model=self.model,
            data=self.data,
            state=self.state,
            limits=self.limits,
            detector=detector,
            jacobians=self.jacobians,
        )
        config = KKTProblem.Config(
            dynamics=ConstrainedDynamicsConfig(preconditioning=preconditioning),
            compliance=ConstraintComplianceConfig(
                joints=1e-3, limits=2e-3, contacts_normal=3e-3, contacts_tangential=4e-3
            ),
        )
        self.problem = KKTProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
            config=config,
            sparse=sparse,
        )
        self.problem.build(
            model=self.model,
            data=self.data,
            jacobians=self.jacobians,
            limits=self.limits,
            contacts=self.contacts,
        )
        self.num_worlds = self.model.size.num_worlds
        self.nbd = self.model.info.num_body_dofs.numpy()
        self.dim = self.data.info.num_total_cts.numpy()
        self.maxdim = self.model.info.max_total_cts.numpy()
        self.vio = self.model.info.total_cts_offset.numpy()

    def world_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        return [x[self.vio[w] : self.vio[w] + self.dim[w]] for w in range(self.num_worlds)]

    def stacked_blocks(self, x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Splits stacked KKT vectors into their generalized, active and inactive constraint parts."""
        kvio = self.problem.data.kkt_vio.numpy()
        blocks = []
        for w in range(self.num_worlds):
            x_w = x[kvio[w] : kvio[w] + self.nbd[w] + self.maxdim[w]]
            blocks.append(
                (x_w[: self.nbd[w]], x_w[self.nbd[w] : self.nbd[w] + self.dim[w]], x_w[self.nbd[w] + self.dim[w] :])
            )
        return blocks

    def jacobians_np(self) -> list[np.ndarray]:
        J_np = extract_cts_jacobians(self.model, self.limits, self.contacts, self.jacobians, only_active_cts=True)
        return [J_np[w][: self.dim[w], : self.nbd[w]] for w in range(self.num_worlds)]


###
# Tests
###


class TestKKTProblem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.rng = np.random.default_rng(seed=13)

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_kkt_problem(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(self.default_device, sparse=sparse, preconditioning=True)
                problem = test.problem
                self.assertEqual(problem.formulation, ProblemFormulation.KKT)
                self.assertIsNone(problem.linear_solver)
                self.assertIsNotNone(problem.data.R)
                kkt_maxdim = test.nbd + test.maxdim
                np.testing.assert_array_equal(problem.data.kkt_maxdim.numpy(), kkt_maxdim)
                np.testing.assert_array_equal(problem.data.kkt_dim.numpy(), test.nbd + test.dim)
                np.testing.assert_array_equal(problem.data.kkt_vio.numpy(), np.cumsum(kkt_maxdim) - kkt_maxdim)
                self.assertEqual(problem.data.k.size, int(np.sum(kkt_maxdim)))
                self.assertEqual(problem.data.ux.size, int(np.sum(kkt_maxdim)))

    def test_02_kkt_matvec(self):
        for sparse in (False, True):
            for preconditioning in (False, True):
                with self.subTest(sparse=sparse, preconditioning=preconditioning):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioning=preconditioning)
                    problem = test.problem
                    n_c = test.model.size.sum_of_max_total_cts
                    eta_np = self.rng.uniform(0.1, 1.0, n_c).astype(np.float32)
                    problem.data.eta.assign(eta_np)
                    x_np = self.rng.standard_normal(problem.data.k.size).astype(np.float32)
                    y = wp.zeros_like(problem.data.k)
                    problem.kkt_matvec(x=wp.array(x_np, device=test.model.device), y=y)

                    J_np = test.jacobians_np()
                    M_np = make_generalized_mass_matrices(test.model, test.data)
                    P_w = test.world_blocks(problem.data.P.numpy())
                    R_w = test.world_blocks(problem.data.R.numpy())
                    eta_w = test.world_blocks(eta_np)
                    for w, ((x_u, x_c, _), (y_u, y_c, y_inactive)) in enumerate(
                        zip(test.stacked_blocks(x_np), test.stacked_blocks(y.numpy()), strict=True)
                    ):
                        R_eff = R_w[w] + eta_w[w] / P_w[w] ** 2
                        np.testing.assert_allclose(y_u, M_np[w] @ x_u + J_np[w].T @ x_c, rtol=1e-4, atol=1e-4)
                        np.testing.assert_allclose(y_c, J_np[w] @ x_u - R_eff * x_c, rtol=1e-4, atol=1e-4)
                        np.testing.assert_array_equal(y_inactive, 0.0)

    def test_03_build_rhs(self):
        test = TestSetup(self.default_device, sparse=False, preconditioning=True)
        problem = test.problem
        b_np = self.rng.standard_normal(test.model.size.sum_of_max_total_cts).astype(np.float32)
        problem.build_rhs(wp.array(b_np, device=test.model.device))

        M_np = make_generalized_mass_matrices(test.model, test.data)
        u_f_np = problem.data.u_f.numpy().reshape(-1)
        bio = test.model.info.bodies_offset.numpy()
        for w, (k_u, k_c, k_inactive) in enumerate(test.stacked_blocks(problem.data.k.numpy())):
            u_f_w = u_f_np[6 * bio[w] : 6 * bio[w + 1]]
            np.testing.assert_allclose(k_u, M_np[w] @ u_f_w, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(k_c, test.world_blocks(b_np)[w])
            np.testing.assert_array_equal(k_inactive, 0.0)

    def test_04_gemv(self):
        for sparse in (False, True):
            for preconditioned in (False, True):
                with self.subTest(sparse=sparse, preconditioned=preconditioned):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioning=True)
                    problem = test.problem
                    n_c = test.model.size.sum_of_max_total_cts
                    x_np = self.rng.standard_normal(n_c).astype(np.float32)
                    y0_np = self.rng.standard_normal(n_c).astype(np.float32)
                    y = wp.array(y0_np, dtype=wp.float32, device=test.model.device)
                    problem.gemv(
                        x=wp.array(x_np, device=test.model.device),
                        y=y,
                        alpha=2.0,
                        beta=0.5,
                        preconditioned=preconditioned,
                    )

                    J_np = test.jacobians_np()
                    invM_np = make_inverse_generalized_mass_matrices(test.model, test.data)
                    P_w = test.world_blocks(problem.data.P.numpy())
                    R_w = test.world_blocks(problem.data.R.numpy())
                    for w, (x_w, y0_w, y_w) in enumerate(
                        zip(
                            test.world_blocks(x_np), test.world_blocks(y0_np), test.world_blocks(y.numpy()), strict=True
                        )
                    ):
                        A = J_np[w] @ invM_np[w] @ J_np[w].T + np.diag(R_w[w])
                        P = np.diag(P_w[w]) if preconditioned else np.eye(len(P_w[w]))
                        np.testing.assert_allclose(y_w, 2.0 * P @ A @ P @ x_w + 0.5 * y0_w, rtol=1e-4, atol=1e-4)

    def test_05_linear_solve_is_not_implemented(self):
        test = TestSetup(self.default_device, sparse=False, preconditioning=True)
        problem = test.problem
        x = wp.zeros_like(problem.data.v_f)
        with self.assertRaises(NotImplementedError):
            problem.update_regularization()
        with self.assertRaises(NotImplementedError):
            problem.solve(v=problem.data.v_f, x=x)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=20000, suppress=True)

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
