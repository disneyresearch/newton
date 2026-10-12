# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DYNAMICS: AUGMENTED MASS MATRIX PRECONDITIONING
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
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

PRECONDITIONERS = ("none", "jacobi", "block_jacobi")


def assert_close_normwise(actual: np.ndarray, desired: np.ndarray, tol: float):
    """Asserts that two vectors agree up to a tolerance relative to the magnitude of the reference."""
    scale = max(1.0, float(np.max(np.abs(desired)))) if desired.size > 0 else 1.0
    np.testing.assert_array_less(np.abs(actual - desired), tol * scale)


class TestSetup:
    def __init__(self, device: wp.DeviceLike, sparse: bool, preconditioner: str | None):
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

        dynamics = ConstrainedDynamicsConfig(preconditioning=True, augmented_mass_preconditioner=preconditioner)
        self.primal = PrimalProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
            solver=ConjugateResidualSolver if sparse else LLTBlockedSolver,
            config=DualProblem.Config(dynamics=dynamics),
            sparse=sparse,
        )
        self.primal.build(
            model=self.model, data=self.data, jacobians=self.jacobians, limits=self.limits, contacts=self.contacts
        )

    def world_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        vio = self.model.info.total_cts_offset.numpy()
        dim = self.data.info.num_total_cts.numpy()
        return [x[vio[w] : vio[w] + dim[w]] for w in range(self.model.size.num_worlds)]

    def body_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        bio = self.model.info.bodies_offset.numpy()
        return [x[6 * bio[w] : 6 * bio[w + 1]] for w in range(self.model.size.num_worlds)]

    def augmented_mass_matrices(self) -> list[np.ndarray]:
        """Returns the reference matrices ``M + J^T W J`` of each world in float64."""
        J_np = extract_cts_jacobians(self.model, self.limits, self.contacts, self.jacobians, only_active_cts=True)
        M_np = make_generalized_mass_matrices(self.model, self.data)
        dim = self.data.info.num_total_cts.numpy()
        nbd = self.model.info.num_body_dofs.numpy()
        W_w = self.world_blocks(self.primal.data.W.numpy())
        matrices = []
        for w in range(self.model.size.num_worlds):
            J = J_np[w][: dim[w], : nbd[w]].astype(np.float64)
            matrices.append(M_np[w].astype(np.float64) + J.T @ np.diag(W_w[w].astype(np.float64)) @ J)
        return matrices

    def delassus_matrices(self) -> list[np.ndarray]:
        J_np = extract_cts_jacobians(self.model, self.limits, self.contacts, self.jacobians, only_active_cts=True)
        invM_np = make_inverse_generalized_mass_matrices(self.model, self.data)
        dim = self.data.info.num_total_cts.numpy()
        nbd = self.model.info.num_body_dofs.numpy()
        delassus = []
        for w in range(self.model.size.num_worlds):
            J = J_np[w][: dim[w], : nbd[w]].astype(np.float64)
            delassus.append(J @ invM_np[w] @ J.T)
        return delassus


###
# Tests
###


class TestAugmentedMassPreconditioning(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.rng = np.random.default_rng(seed=7)

    def tearDown(self):
        self.default_device = None

    def _regularize(self, test: TestSetup):
        eta_np = self.rng.uniform(0.1, 1.0, test.model.size.sum_of_max_total_cts).astype(np.float32)
        test.primal.data.eta.assign(eta_np)
        test.primal.update_regularization()
        return eta_np

    def test_01_default_preconditioner_selection(self):
        for sparse, expected in ((False, "none"), (True, "jacobi")):
            with self.subTest(sparse=sparse):
                test = TestSetup(self.default_device, sparse=sparse, preconditioner=None)
                self.assertEqual(test.primal.operator.preconditioner, expected)
        for preconditioner in PRECONDITIONERS:
            with self.subTest(preconditioner=preconditioner):
                test = TestSetup(self.default_device, sparse=False, preconditioner=preconditioner)
                self.assertEqual(test.primal.operator.preconditioner, preconditioner)

    def test_02_invalid_preconditioner(self):
        with self.assertRaises(ValueError):
            ConstrainedDynamicsConfig(augmented_mass_preconditioner="ilu")

    def test_03_scaled_matrix_structure(self):
        for preconditioner in ("jacobi", "block_jacobi"):
            with self.subTest(preconditioner=preconditioner):
                test = TestSetup(self.default_device, sparse=False, preconditioner=preconditioner)
                self._regularize(test)
                info = test.primal.operator.info
                mio = info.mio.numpy()
                dims = info.dim.numpy()
                mat = test.primal.operator.matrix.numpy()
                A_ref = test.augmented_mass_matrices()
                for w in range(test.model.size.num_worlds):
                    n = dims[w]
                    A_s = mat[mio[w] : mio[w] + n * n].reshape(n, n).astype(np.float64)
                    if preconditioner == "jacobi":
                        np.testing.assert_allclose(np.diag(A_s), 1.0, rtol=1e-5)
                        s = 1.0 / np.sqrt(np.diag(A_ref[w]))
                        np.testing.assert_allclose(A_s, s[:, None] * A_ref[w] * s[None, :], rtol=1e-3, atol=1e-5)
                    else:
                        for k in range(n // 6):
                            block = A_s[6 * k : 6 * k + 6, 6 * k : 6 * k + 6]
                            np.testing.assert_allclose(block, np.eye(6), atol=1e-4)
                    # The scaled matrix is congruent to the original one and thus remains symmetric
                    np.testing.assert_allclose(A_s, A_s.T, rtol=1e-4, atol=1e-5)

    def test_04_operator_solves_match_reference(self):
        for sparse in (False, True):
            for preconditioner in PRECONDITIONERS:
                with self.subTest(sparse=sparse, preconditioner=preconditioner):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioner=preconditioner)
                    self._regularize(test)
                    n_u = test.model.size.sum_of_num_body_dofs
                    b_np = self.rng.standard_normal(n_u).astype(np.float32)
                    x_wp = wp.zeros(n_u, dtype=wp.float32, device=test.model.device)
                    test.primal.operator.solve(b=wp.array(b_np, device=test.model.device), x=x_wp)
                    A_ref = test.augmented_mass_matrices()
                    for w, (x_w, b_w) in enumerate(
                        zip(test.body_blocks(x_wp.numpy()), test.body_blocks(b_np), strict=True)
                    ):
                        assert_close_normwise(x_w, np.linalg.solve(A_ref[w], b_w), tol=1e-3)

    def test_05_problem_solves_match_reference(self):
        for sparse in (False, True):
            for preconditioner in PRECONDITIONERS:
                with self.subTest(sparse=sparse, preconditioner=preconditioner):
                    test = TestSetup(self.default_device, sparse=sparse, preconditioner=preconditioner)
                    eta_np = self._regularize(test)
                    n_c = test.model.size.sum_of_max_total_cts
                    v_np = self.rng.standard_normal(n_c).astype(np.float32)
                    x_wp = wp.zeros(n_c, dtype=wp.float32, device=test.model.device)
                    test.primal.solve(v=wp.array(v_np, device=test.model.device), x=x_wp)

                    # x = (P D P + diag(eta))^{-1} v
                    D_np = test.delassus_matrices()
                    P_w = test.world_blocks(test.primal.data.P.numpy())
                    x_w = test.world_blocks(x_wp.numpy())
                    for w, (eta_w, v_w) in enumerate(
                        zip(test.world_blocks(eta_np), test.world_blocks(v_np), strict=True)
                    ):
                        P = np.diag(P_w[w].astype(np.float64))
                        x_ref = np.linalg.solve(P @ D_np[w] @ P + np.diag(eta_w), v_w)
                        assert_close_normwise(x_w[w], x_ref, tol=1e-3)


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
