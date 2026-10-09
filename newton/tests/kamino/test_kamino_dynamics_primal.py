# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DYNAMICS: PRIMAL PROBLEM
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics import ProblemFormulation
from newton._src.solvers.kamino._src.dynamics.augmented_mass import (
    AugmentedMassOperator,
    BlockSparseMatrixFreeAugmentedMassOperator,
)
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.dynamics.primal import PrimalProblem
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, LLTBlockedSolver
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig
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


def assert_close_normwise(actual: np.ndarray, desired: np.ndarray, tol: float):
    """Asserts that two vectors agree up to a tolerance relative to the magnitude of the reference."""
    scale = max(1.0, float(np.max(np.abs(desired)))) if desired.size > 0 else 1.0
    np.testing.assert_array_less(np.abs(actual - desired), tol * scale)


class TestSetup:
    def __init__(self, device: wp.DeviceLike, sparse: bool, preconditioning: bool, sparse_jacobian: bool | None = None):
        builder = make_basics_heterogeneous_builder()
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = wp.spatial_vectorf(0.5, 0.0, -0.2, 0.0, 0.3, 0.0)
        model = ModelKamino.from_newton(builder.finalize(device=device))
        self.model, self.data, self.state, self.limits, detector, self.jacobians = make_containers(
            model=model, max_world_contacts=12, sparse=sparse if sparse_jacobian is None else sparse_jacobian
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

        # Create the dual and primal problems of the same system
        config = DualProblem.Config(dynamics=ConstrainedDynamicsConfig(preconditioning=preconditioning))
        solver = ConjugateResidualSolver if sparse else LLTBlockedSolver
        kwargs = {
            "model": self.model,
            "data": self.data,
            "limits": self.limits,
            "contacts": self.contacts,
            "jacobians": self.jacobians,
            "config": config,
            "sparse": sparse,
        }
        self.dual = DualProblem(solver=solver, **kwargs)
        self.primal = PrimalProblem(solver=solver, **kwargs)
        for problem in (self.dual, self.primal):
            problem.build(
                model=self.model,
                data=self.data,
                jacobians=self.jacobians,
                limits=self.limits,
                contacts=self.contacts,
            )

    def world_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        vio = self.model.info.total_cts_offset.numpy()
        dim = self.data.info.num_total_cts.numpy()
        return [x[vio[w] : vio[w] + dim[w]] for w in range(self.model.size.num_worlds)]

    def body_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        bio = self.model.info.bodies_offset.numpy()
        return [x[6 * bio[w] : 6 * bio[w + 1]] for w in range(self.model.size.num_worlds)]

    def delassus_matrices(self) -> list[np.ndarray]:
        J_np = extract_cts_jacobians(self.model, self.limits, self.contacts, self.jacobians, only_active_cts=True)
        dim = self.data.info.num_total_cts.numpy()
        nbd = self.model.info.num_body_dofs.numpy()
        invM_np = make_inverse_generalized_mass_matrices(self.model, self.data)
        J_np = [J_np[w][: dim[w], : nbd[w]] for w in range(self.model.size.num_worlds)]
        return J_np, [J_np[w] @ invM_np[w] @ J_np[w].T for w in range(self.model.size.num_worlds)]


###
# Tests
###


class TestPrimalProblem(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.rng = np.random.default_rng(seed=11)

    def tearDown(self):
        self.default_device = None

    def test_01_allocate_primal_problem(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(self.default_device, sparse=sparse, preconditioning=True)
                problem = test.primal
                self.assertEqual(problem.formulation, ProblemFormulation.PRIMAL)
                self.assertEqual(problem.sparse, sparse)
                operator_type = BlockSparseMatrixFreeAugmentedMassOperator if sparse else AugmentedMassOperator
                self.assertIsInstance(problem.operator, operator_type)
                self.assertIsInstance(problem.linear_solver, ConjugateResidualSolver if sparse else LLTBlockedSolver)
                self.assertEqual(problem.data.W.size, test.model.size.sum_of_max_total_cts)
                self.assertEqual(problem.data.du.size, test.model.size.sum_of_num_body_dofs)
                # The constraint-space data is shared by all formulations
                for name in ("v_f", "P", "mu", "bound_lower", "bound_upper"):
                    np.testing.assert_allclose(
                        getattr(problem.data, name).numpy(), getattr(test.dual.data, name).numpy(), rtol=1e-6
                    )

    def test_02_solve_matches_dual_problem(self):
        for sparse in (False, True):
            for preconditioning in (False, True):
                with self.subTest(sparse=sparse, preconditioning=preconditioning):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioning=preconditioning)
                    n_c = test.model.size.sum_of_max_total_cts

                    # Apply the same proximal regularization to both problems
                    # NOTE: Some worlds are hyperstatic, i.e. their Delassus matrix is singular,
                    # so the regularization must keep the float32 systems well-conditioned
                    eta_np = self.rng.uniform(0.1, 1.0, n_c).astype(np.float32)
                    v_np = self.rng.standard_normal(n_c).astype(np.float32)
                    x = {}
                    for name, problem in (("dual", test.dual), ("primal", test.primal)):
                        problem.data.eta.assign(eta_np)
                        problem.update_regularization()
                        x_wp = wp.zeros(n_c, dtype=wp.float32, device=test.model.device)
                        problem.solve(v=wp.array(v_np, device=test.model.device), x=x_wp)
                        x[name] = test.world_blocks(x_wp.numpy())

                    # Compare against the NumPy reference: x = (P D P + diag(eta))^{-1} v
                    # NOTE: Without preconditioning, the float32 solves of hyperstatic worlds are less accurate
                    tol = 1e-3 if preconditioning else 2e-2
                    _, D_np = test.delassus_matrices()
                    P_w = test.world_blocks(test.primal.data.P.numpy())
                    eta_w = test.world_blocks(eta_np)
                    v_w = test.world_blocks(v_np)
                    for w in range(test.model.size.num_worlds):
                        P = np.diag(P_w[w].astype(np.float64))
                        A = P @ D_np[w] @ P + np.diag(eta_w[w])
                        x_ref = np.linalg.solve(A, v_w[w])
                        assert_close_normwise(x["primal"][w], x_ref, tol=tol)
                        assert_close_normwise(x["primal"][w], x["dual"][w], tol=tol)

    def test_03_solve_satisfies_momentum_balance(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(self.default_device, sparse=sparse, preconditioning=True)
                problem = test.primal
                n_c = test.model.size.sum_of_max_total_cts
                problem.data.eta.fill_(0.5)
                problem.update_regularization()
                x_wp = wp.zeros(n_c, dtype=wp.float32, device=test.model.device)
                v_np = self.rng.standard_normal(n_c).astype(np.float32)
                problem.solve(v=wp.array(v_np, device=test.model.device), x=x_wp)

                # The pair (u, lambda) with u = u_f + du and lambda = P x satisfies M du = J^T lambda
                J_np, _ = test.delassus_matrices()
                M_np = make_generalized_mass_matrices(test.model, test.data)
                du_w = test.body_blocks(problem.data.du.numpy())
                lam_w = test.world_blocks(x_wp.numpy() * problem.data.P.numpy())
                for w in range(test.model.size.num_worlds):
                    assert_close_normwise(M_np[w] @ du_w[w], J_np[w].T @ lam_w[w], tol=1e-3)

    def test_04_gemv_matches_dual_problem(self):
        for sparse in (False, True):
            for preconditioned in (False, True):
                with self.subTest(sparse=sparse, preconditioned=preconditioned):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioning=True)
                    n_c = test.model.size.sum_of_max_total_cts
                    x_np = self.rng.standard_normal(n_c).astype(np.float32)
                    y0_np = self.rng.standard_normal(n_c).astype(np.float32)
                    y = {}
                    for name, problem in (("dual", test.dual), ("primal", test.primal)):
                        y_wp = wp.array(y0_np, dtype=wp.float32, device=test.model.device)
                        problem.gemv(
                            x=wp.array(x_np, device=test.model.device),
                            y=y_wp,
                            alpha=2.0,
                            beta=0.5,
                            preconditioned=preconditioned,
                        )
                        y[name] = test.world_blocks(y_wp.numpy())

                    _, D_np = test.delassus_matrices()
                    P_w = test.world_blocks(test.primal.data.P.numpy())
                    x_w = test.world_blocks(x_np)
                    y0_w = test.world_blocks(y0_np)
                    for w in range(test.model.size.num_worlds):
                        P = np.diag(P_w[w]) if preconditioned else np.eye(len(P_w[w]))
                        y_ref = 2.0 * P @ D_np[w] @ P @ x_w[w] + 0.5 * y0_w[w]
                        np.testing.assert_allclose(y["primal"][w], y_ref, rtol=1e-4, atol=1e-4)
                        np.testing.assert_allclose(y["primal"][w], y["dual"][w], rtol=1e-4, atol=1e-4)

    def test_05_dense_problem_with_sparse_jacobians(self):
        test = TestSetup(self.default_device, sparse=False, preconditioning=True, sparse_jacobian=True)
        n_c = test.model.size.sum_of_max_total_cts
        eta_np = self.rng.uniform(0.1, 1.0, n_c).astype(np.float32)
        v_np = self.rng.standard_normal(n_c).astype(np.float32)
        x = {}
        for name, problem in (("dual", test.dual), ("primal", test.primal)):
            problem.data.eta.assign(eta_np)
            problem.update_regularization()
            x_wp = wp.zeros(n_c, dtype=wp.float32, device=test.model.device)
            problem.solve(v=wp.array(v_np, device=test.model.device), x=x_wp)
            x[name] = test.world_blocks(x_wp.numpy())

        _, D_np = test.delassus_matrices()
        P_w = test.world_blocks(test.primal.data.P.numpy())
        for w, (eta_w, v_w) in enumerate(zip(test.world_blocks(eta_np), test.world_blocks(v_np), strict=True)):
            P = np.diag(P_w[w].astype(np.float64))
            x_ref = np.linalg.solve(P @ D_np[w] @ P + np.diag(eta_w), v_w)
            assert_close_normwise(x["primal"][w], x_ref, tol=1e-3)
            assert_close_normwise(x["primal"][w], x["dual"][w], tol=1e-3)

    def test_06_compliance_enters_effective_stiffness(self):
        test = TestSetup(self.default_device, sparse=False, preconditioning=True)
        problem = test.primal
        n_c = test.model.size.sum_of_max_total_cts
        R_np = self.rng.uniform(1e-2, 1e-1, n_c).astype(np.float32)
        eta_np = self.rng.uniform(0.1, 1.0, n_c).astype(np.float32)
        problem.data.R = wp.array(R_np, dtype=wp.float32, device=test.model.device)
        problem.data.eta.assign(eta_np)
        problem.update_regularization()
        x_wp = wp.zeros(n_c, dtype=wp.float32, device=test.model.device)
        v_np = self.rng.standard_normal(n_c).astype(np.float32)
        problem.solve(v=wp.array(v_np, device=test.model.device), x=x_wp)

        # x = (P (D + R) P + diag(eta))^{-1} v
        _, D_np = test.delassus_matrices()
        P_w = test.world_blocks(problem.data.P.numpy())
        x_w = test.world_blocks(x_wp.numpy())
        for w, (R_w, eta_w, v_w) in enumerate(
            zip(test.world_blocks(R_np), test.world_blocks(eta_np), test.world_blocks(v_np), strict=True)
        ):
            P = np.diag(P_w[w].astype(np.float64))
            A = P @ (D_np[w] + np.diag(R_w)) @ P + np.diag(eta_w)
            assert_close_normwise(x_w[w], np.linalg.solve(A, v_w), tol=1e-3)


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
