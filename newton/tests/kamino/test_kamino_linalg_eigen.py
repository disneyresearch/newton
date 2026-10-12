# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: LINEAR ALGEBRA: EIGENVALUE ESTIMATION
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.dynamics.primal import PrimalProblem
from newton._src.solvers.kamino._src.dynamics.spectral import (
    make_augmented_mass_operator,
    make_mass_operator,
    make_problem_operator,
    make_problem_solve_fn,
)
from newton._src.solvers.kamino._src.linalg import (
    ConjugateResidualSolver,
    DenseLinearOperatorData,
    DenseSquareMultiLinearInfo,
    LLTBlockedSolver,
    PowerIteration,
)
from newton._src.solvers.kamino._src.linalg.conjugate import BatchedLinearOperator
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


def make_spd_matrices(rng: np.random.Generator, dims: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Creates random SPD matrices with well-separated extreme eigenvalues, and returns them with their spectra."""
    matrices, spectra = [], []
    for n in dims:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        eigs = np.sort(rng.uniform(1.0, 2.0, n))
        eigs[0], eigs[-1] = 0.25, 6.0
        matrices.append((Q * eigs) @ Q.T)
        spectra.append(eigs)
    return matrices, spectra


class DenseTestOperator:
    """A batch of dense SPD matrices wrapped as a batched linear operator."""

    def __init__(self, matrices: list[np.ndarray], device: wp.DeviceLike):
        self.info = DenseSquareMultiLinearInfo[wp.float32, wp.int32]()
        self.info.finalize(dimensions=[m.shape[0] for m in matrices], dtype=wp.float32, itype=wp.int32, device=device)
        flat = np.concatenate([m.astype(np.float32).ravel() for m in matrices])
        self.data = DenseLinearOperatorData[wp.float32, wp.int32](
            info=self.info, mat=wp.array(flat, dtype=wp.float32, device=device)
        )
        self.operator = BatchedLinearOperator.from_dense(self.data)


class ProblemTestSetup:
    def __init__(self, device: wp.DeviceLike, sparse: bool):
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
        kwargs = {
            "model": self.model,
            "data": self.data,
            "limits": self.limits,
            "contacts": self.contacts,
            "jacobians": self.jacobians,
            "solver": ConjugateResidualSolver if sparse else LLTBlockedSolver,
            "config": DualProblem.Config(dynamics=ConstrainedDynamicsConfig(preconditioning=True)),
            "sparse": sparse,
        }
        self.dual = DualProblem(**kwargs)
        self.primal = PrimalProblem(**kwargs)
        for problem in (self.dual, self.primal):
            problem.build(
                model=self.model, data=self.data, jacobians=self.jacobians, limits=self.limits, contacts=self.contacts
            )

    def world_blocks(self, x: np.ndarray) -> list[np.ndarray]:
        vio = self.model.info.total_cts_offset.numpy()
        dim = self.data.info.num_total_cts.numpy()
        return [x[vio[w] : vio[w] + dim[w]] for w in range(self.model.size.num_worlds)]

    def jacobians_and_delassus(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        J_np = extract_cts_jacobians(self.model, self.limits, self.contacts, self.jacobians, only_active_cts=True)
        invM_np = make_inverse_generalized_mass_matrices(self.model, self.data)
        dim = self.data.info.num_total_cts.numpy()
        nbd = self.model.info.num_body_dofs.numpy()
        J_w = [J_np[w][: dim[w], : nbd[w]].astype(np.float64) for w in range(self.model.size.num_worlds)]
        return J_w, [J @ invM_np[w] @ J.T for w, J in enumerate(J_w)]


###
# Tests
###


class TestPowerIteration(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.rng = np.random.default_rng(seed=3)

    def tearDown(self):
        self.default_device = None

    def test_01_invalid_arguments(self):
        with self.assertRaises(ValueError):
            PowerIteration(max_iterations=0)
        with self.assertRaises(ValueError):
            PowerIteration(tolerance=-1.0)
        with self.assertRaises(TypeError):
            PowerIteration(operator=np.eye(3))

    def test_02_extreme_eigenvalues_of_dense_matrices(self):
        dims = [5, 12, 1, 8]
        matrices, spectra = make_spd_matrices(self.rng, dims)
        test = DenseTestOperator(matrices, self.default_device)
        pi = PowerIteration(operator=test.operator, max_iterations=500, tolerance=1e-7)
        lambda_max = pi.compute_max().numpy()
        lambda_min = pi.compute_min(method="shifted").numpy()
        for w, eigs in enumerate(spectra):
            np.testing.assert_allclose(lambda_max[w], eigs[-1], rtol=1e-3)
            np.testing.assert_allclose(lambda_min[w], eigs[0], rtol=1e-2, atol=1e-3)

        # The iterates are normalized approximations of the extreme eigenvectors
        vio = test.info.vio.numpy()
        for w, (A, n) in enumerate(zip(matrices, dims, strict=True)):
            x = pi.eigenvector_max.numpy()[vio[w] : vio[w] + n].astype(np.float64)
            np.testing.assert_allclose(np.linalg.norm(x), 1.0, rtol=1e-4)
            np.testing.assert_allclose(A @ x, spectra[w][-1] * x, atol=1e-2)

    def test_03_inverse_iteration(self):
        dims = [6, 10]
        matrices, spectra = make_spd_matrices(self.rng, dims)
        shift = 0.5
        test = DenseTestOperator(matrices, self.default_device)
        shifted = DenseTestOperator([A + shift * np.eye(A.shape[0]) for A in matrices], self.default_device)
        solver = LLTBlockedSolver(operator=shifted.data, device=self.default_device)
        solver.compute(A=shifted.data.mat)

        def solve_fn(b, x):
            solver.solve(b=b, x=x)

        pi = PowerIteration(operator=test.operator, max_iterations=200, tolerance=1e-7)
        shifts = wp.full(len(dims), shift, dtype=wp.float32, device=self.default_device)
        lambda_min = pi.compute_min(method="inverse", shift=shifts, solve_fn=solve_fn).numpy()
        for w, eigs in enumerate(spectra):
            np.testing.assert_allclose(lambda_min[w], eigs[0], rtol=1e-3)
        with self.assertRaises(ValueError):
            pi.compute_min(method="inverse")
        with self.assertRaises(ValueError):
            pi.compute_min(method="lanczos")

    def test_04_world_mask_and_warm_start(self):
        dims = [7, 9, 4]
        matrices, spectra = make_spd_matrices(self.rng, dims)
        test = DenseTestOperator(matrices, self.default_device)
        pi = PowerIteration(operator=test.operator, max_iterations=500, tolerance=1e-6)

        # Masked worlds keep their previous estimates
        mask = wp.array([True, False, True], dtype=wp.bool, device=self.default_device)
        pi.lambda_max.fill_(-1.0)
        lambda_max = pi.compute_max(world_mask=mask).numpy()
        self.assertEqual(lambda_max[1], -1.0)
        self.assertEqual(pi.iterations.numpy()[1], 0)
        np.testing.assert_allclose(lambda_max[[0, 2]], [spectra[0][-1], spectra[2][-1]], rtol=1e-3)

        # Warm-starting from the converged iterates requires only a few iterations
        cold_iterations = pi.iterations.numpy()[[0, 2]]
        pi.compute_max(world_mask=mask)
        warm_iterations = pi.iterations.numpy()[[0, 2]]
        self.assertTrue(np.all(warm_iterations <= 3))
        self.assertTrue(np.all(warm_iterations < cold_iterations))

        # Resetting the iterates restores the cold start
        pi.reset()
        pi.compute_max(world_mask=mask)
        np.testing.assert_array_equal(pi.iterations.numpy()[[0, 2]], cold_iterations)

    def test_05_problem_operators(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = ProblemTestSetup(self.default_device, sparse=sparse)
                J_w, D_w = test.jacobians_and_delassus()
                for name, problem in (("dual", test.dual), ("primal", test.primal)):
                    P_w = test.world_blocks(problem.data.P.numpy())
                    R_w = test.world_blocks(problem.data.R.numpy()) if problem.data.R is not None else None
                    if R_w is not None:
                        D_w = [D + np.diag(R_w[w]) for w, D in enumerate(test.jacobians_and_delassus()[1])]
                    for preconditioned in (False, True):
                        pi = PowerIteration(
                            operator=make_problem_operator(problem, preconditioned=preconditioned),
                            max_iterations=1000,
                            tolerance=1e-7,
                        )
                        lambda_max = pi.compute_max().numpy()
                        for w, D in enumerate(D_w):
                            P = np.diag(P_w[w]) if preconditioned else np.eye(D.shape[0])
                            expected = np.linalg.eigvalsh(P @ D @ P)[-1] if D.size > 0 else 0.0
                            with self.subTest(problem=name, preconditioned=preconditioned, world=w):
                                np.testing.assert_allclose(lambda_max[w], expected, rtol=5e-3)

                # The mass and augmented mass matrices are positive definite
                eta_np = self.rng.uniform(0.1, 1.0, test.model.size.sum_of_max_total_cts).astype(np.float32)
                test.primal.data.eta.assign(eta_np)
                test.primal.update_regularization()
                M_w = make_generalized_mass_matrices(test.model, test.data)
                W_w = test.world_blocks(test.primal.data.W.numpy())
                operators = {
                    "mass": (make_mass_operator(test.model, test.data), list(M_w)),
                    "augmented_mass": (
                        make_augmented_mass_operator(test.primal),
                        [M_w[w] + J.T @ np.diag(W_w[w]) @ J for w, J in enumerate(J_w)],
                    ),
                }
                for name, (operator, references) in operators.items():
                    pi = PowerIteration(operator=operator, max_iterations=2000, tolerance=1e-8)
                    lambda_max = pi.compute_max().numpy()
                    lambda_min = pi.compute_min(method="shifted").numpy()
                    for w, A in enumerate(references):
                        eigs = np.linalg.eigvalsh(A)
                        with self.subTest(operator=name, world=w):
                            np.testing.assert_allclose(lambda_max[w], eigs[-1], rtol=5e-3)
                            np.testing.assert_allclose(lambda_min[w], eigs[0], rtol=5e-2, atol=1e-3 * eigs[-1])

    def test_06_inverse_iteration_with_problem_solve(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = ProblemTestSetup(self.default_device, sparse=sparse)
                _, D_w = test.jacobians_and_delassus()
                eta = 0.5
                for name, problem in (("dual", test.dual), ("primal", test.primal)):
                    problem.data.eta.fill_(eta)
                    problem.update_regularization()
                    pi = PowerIteration(operator=make_problem_operator(problem), max_iterations=200, tolerance=1e-7)
                    shifts = wp.full(test.model.size.num_worlds, eta, dtype=wp.float32, device=test.model.device)
                    lambda_min = pi.compute_min(
                        method="inverse", shift=shifts, solve_fn=make_problem_solve_fn(problem)
                    ).numpy()
                    P_w = test.world_blocks(problem.data.P.numpy())
                    R_w = test.world_blocks(problem.data.R.numpy()) if problem.data.R is not None else None
                    for w, D in enumerate(D_w):
                        if D.size == 0:
                            continue
                        P = np.diag(P_w[w])
                        R = np.diag(R_w[w]) if R_w is not None else 0.0
                        expected = np.linalg.eigvalsh(P @ (D + R) @ P)[0]
                        with self.subTest(problem=name, world=w):
                            np.testing.assert_allclose(lambda_min[w], expected, atol=5e-3)


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
