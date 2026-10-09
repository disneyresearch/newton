# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the primal Proximal-ADMM Solver."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.dynamics.primal import PrimalProblem
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver
from newton._src.solvers.kamino._src.solvers.padmm_primal import PADMMPrimalSolver
from newton._src.solvers.kamino.config import ConstraintComplianceConfig, PADMMPrimalSolverConfig, PADMMSolverConfig
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics

###
# Helpers
###


def assert_close_normwise(actual: np.ndarray, desired: np.ndarray, tol: float):
    """Asserts that two vectors agree up to a tolerance relative to the magnitude of the reference."""
    scale = max(1.0, float(np.max(np.abs(desired)))) if desired.size > 0 else 1.0
    np.testing.assert_array_less(np.abs(actual - desired), tol * scale)


class TestSetup:
    def __init__(
        self,
        builder_fn,
        device: wp.DeviceLike,
        sparse: bool,
        max_world_contacts: int = 8,
        config: DualProblem.Config | None = None,
        **kwargs,
    ):
        builder = builder_fn(**kwargs)
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = wp.spatial_vectorf(1.0, 0.0, -0.5, 0.0, 0.2, 0.0)
        model = ModelKamino.from_newton(builder.finalize(device=device))
        self.model, self.data, self.state, self.limits, detector, self.jacobians = make_containers(
            model=model, max_world_contacts=max_world_contacts, sparse=sparse
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
        solver = ConjugateResidualSolver if sparse else LLTBlockedSolver
        kwargs = {
            "model": self.model,
            "data": self.data,
            "limits": self.limits,
            "contacts": self.contacts,
            "jacobians": self.jacobians,
            "solver": solver,
            "config": config,
            "sparse": sparse,
        }
        self.dual = DualProblem(**kwargs)
        self.primal = PrimalProblem(**kwargs)
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


###
# Tests
###


class TestPADMMPrimalSolverConfig(unittest.TestCase):
    def test_01_conversion_to_dual_units(self):
        config = PADMMPrimalSolverConfig(
            primal_tolerance=1e-3,
            dual_tolerance=1e-5,
            compl_tolerance=1e-4,
            eta=1e-4,
            rho_0=2.0,
            rho_min=0.1,
            rho_max=100.0,
            max_iterations=17,
        )
        dual = config.to_dual_units()
        self.assertIs(type(dual), PADMMSolverConfig)
        self.assertEqual(dual.primal_tolerance, 1e-5)
        self.assertEqual(dual.dual_tolerance, 1e-3)
        self.assertEqual(dual.compl_tolerance, 1e-4)
        self.assertEqual(dual.eta, 1e-4)
        self.assertAlmostEqual(dual.rho_0, 0.5)
        self.assertAlmostEqual(dual.rho_min, 0.01)
        self.assertAlmostEqual(dual.rho_max, 10.0)
        self.assertEqual(dual.max_iterations, 17)

        # The default primal penalty bounds map to the default dual penalty bounds
        dual = PADMMPrimalSolverConfig().to_dual_units()
        self.assertAlmostEqual(dual.rho_min, PADMMSolverConfig().rho_min)
        self.assertEqual(dual.rho_max, math.inf)

    def test_02_validation(self):
        for kwargs in ({"rho_0": 0.0}, {"rho_min": -1.0}, {"rho_min": 2.0, "rho_max": 1.0}, {"rho_max": math.inf}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PADMMPrimalSolverConfig(**kwargs)

    def test_03_dual_solver_rejects_primal_config(self):
        with self.assertRaises(TypeError):
            PADMMSolver._check_config(model=None, config=PADMMPrimalSolverConfig())


class TestPADMMPrimalSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _check_iterates_match_dual_padmm(self, test: TestSetup, use_acceleration: bool):
        """
        Checks the iterate-wise equivalence of the primal PADMM on the primal problem with penalty
        `rho` and of the (dual) PADMM on the dual problem with penalty `1 / rho`.
        """
        rho = 2.0
        common = {"primal_tolerance": 0.0, "dual_tolerance": 0.0, "compl_tolerance": 0.0, "eta": 1e-3}
        common["max_iterations"] = 25
        solvers = {
            "dual": (
                PADMMSolver(
                    model=test.model,
                    config=PADMMSolverConfig(rho_0=1.0 / rho, **common),
                    use_acceleration=use_acceleration,
                    use_graph_conditionals=False,
                ),
                test.dual,
            ),
            "primal": (
                PADMMPrimalSolver(
                    model=test.model,
                    config=PADMMPrimalSolverConfig(rho_0=rho, **common),
                    use_acceleration=use_acceleration,
                    use_graph_conditionals=False,
                ),
                test.primal,
            ),
        }
        results = {}
        for name, (solver, problem) in solvers.items():
            solver.coldstart()
            solver.solve(problem=problem)
            results[name] = (
                test.world_blocks(solver.data.solution.lambdas.numpy()),
                test.world_blocks(solver.data.solution.v_plus.numpy()),
                solver.data.status.numpy(),
            )

        lambdas_d, v_plus_d, status_d = results["dual"]
        lambdas_p, v_plus_p, status_p = results["primal"]
        for w in range(test.model.size.num_worlds):
            assert_close_normwise(lambdas_p[w], lambdas_d[w], tol=1e-3)
            assert_close_normwise(v_plus_p[w], v_plus_d[w], tol=1e-3)
            # NOTE: With zero tolerances, a solve only converges early once its float32
            # residuals vanish exactly, which depends on the rounding of each formulation
            if not (status_p[w]["converged"] or status_d[w]["converged"]):
                self.assertEqual(status_p[w]["iterations"], status_d[w]["iterations"])
            # The residuals are reported in the conventions of the respective algorithm
            np.testing.assert_allclose(status_p[w]["r_p"], status_d[w]["r_d"], rtol=1e-2, atol=1e-7)
            np.testing.assert_allclose(status_p[w]["r_d"], status_d[w]["r_p"], rtol=1e-2, atol=1e-7)

    def test_01_iterates_match_dual_padmm(self):
        for builder_fn in (basics.build_box_on_plane, basics.build_box_pendulum):
            for sparse in (False, True):
                for use_acceleration in (False, True):
                    with self.subTest(builder=builder_fn.__name__, sparse=sparse, use_acceleration=use_acceleration):
                        test = TestSetup(builder_fn, self.default_device, sparse=sparse)
                        self.assertIsNone(test.primal.data.R)
                        self._check_iterates_match_dual_padmm(test, use_acceleration)

    def test_02_iterates_match_dual_padmm_with_compliance(self):
        config = DualProblem.Config(
            compliance=ConstraintComplianceConfig(
                joints=1e-4, limits=1e-4, contacts_normal=1e-3, contacts_tangential=1e-3
            )
        )
        for sparse in (False, True):
            for use_acceleration in (False, True):
                with self.subTest(sparse=sparse, use_acceleration=use_acceleration):
                    test = TestSetup(
                        basics.build_boxes_fourbar,
                        self.default_device,
                        sparse=sparse,
                        config=config,
                        z_offset=-0.01,
                        dynamic_joints=True,
                        implicit_pd=True,
                    )
                    self.assertIsNotNone(test.primal.data.R)
                    np.testing.assert_array_equal(test.primal.data.R.numpy(), test.dual.data.R.numpy())
                    self.assertTrue(np.any(test.primal.data.R.numpy() > 0.0))
                    self._check_iterates_match_dual_padmm(test, use_acceleration)

    def test_03_solve_converges(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(basics.build_box_on_plane, self.default_device, sparse=sparse)
                config = PADMMPrimalSolverConfig(max_iterations=500)
                solver = PADMMPrimalSolver(model=test.model, config=config, use_graph_conditionals=False)
                solver.coldstart()
                solver.solve(problem=test.primal)
                status = solver.data.status.numpy()[0]
                self.assertTrue(bool(status["converged"]))
                self.assertLessEqual(status["r_p"], config.primal_tolerance)
                self.assertLessEqual(status["r_d"], config.dual_tolerance)

                # The solution satisfies the constraint-space dynamics: v_plus = D lambda + v_f
                problem = test.primal
                v_plus = wp.zeros_like(solver.data.solution.v_plus)
                problem.gemv(x=solver.data.solution.lambdas, y=v_plus, preconditioned=False)
                v_f = problem.data.v_f.numpy() / problem.data.P.numpy()
                v_plus_ref = test.world_blocks(v_plus.numpy() + v_f)[0]
                assert_close_normwise(test.world_blocks(solver.data.solution.v_plus.numpy())[0], v_plus_ref, tol=1e-4)


class TestSolverKaminoPrimalPADMM(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _make_solver(self, builder: newton.ModelBuilder, dynamics_solver: str, sparse: bool, **config_kwargs):
        model = builder.finalize(device=self.default_device, skip_validation_joints=True)
        config = SolverKamino.Config(
            dynamics_solver=dynamics_solver,
            use_collision_detector=True,
            sparse_jacobian=sparse,
            sparse_dynamics=sparse,
            **config_kwargs,
        )
        return model, SolverKamino(model, config=config)

    def _simulate_fourbar(self, dynamics_solver: str, sparse: bool, num_steps: int, **kwargs) -> np.ndarray:
        builder_kwargs = {k: kwargs.pop(k) for k in ("dynamic_joints", "implicit_pd") if k in kwargs}
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        basics.build_boxes_fourbar(builder=builder, z_offset=0.05, ground=True, **builder_kwargs)
        model, solver = self._make_solver(builder, dynamics_solver, sparse, **kwargs)
        state_in, state_out = model.state(), model.state()
        for _ in range(num_steps):
            solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
            state_in, state_out = state_out, state_in
        return state_in.body_q.numpy()

    def test_01_box_slides_and_sticks_on_plane(self):
        """Checks the analytic Coulomb friction response of a box resting on a plane on four contacts."""
        mu, dt, g = 0.5, 1e-3, 9.81
        for sparse in (False, True):
            for v_x in (1.0, 1e-4):
                with self.subTest(sparse=sparse, v_x=v_x):
                    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                    SolverKamino.register_custom_attributes(builder)
                    basics.build_box_on_plane(builder=builder, friction=mu, use_custom_shape_cfg=True)
                    model, solver = self._make_solver(builder, "padmm_primal", sparse)
                    state_in, state_out = model.state(), model.state()
                    state_in.body_qd.assign(np.array([[v_x, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
                    solver.step(state_in, state_out, control=None, contacts=None, dt=dt)

                    # Slip if the initial velocity exceeds what friction can remove in one step, else stick
                    expected_v_x = max(v_x - mu * g * dt, 0.0)
                    qd = state_out.body_qd.numpy()[0]
                    np.testing.assert_allclose(qd[0], expected_v_x, atol=1e-4)
                    np.testing.assert_allclose(qd[1:], 0.0, atol=1e-4)

    def test_02_trajectories_match_padmm(self):
        """Checks that simulations with the primal and dual PADMM solvers coincide."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                body_q = {s: self._simulate_fourbar(s, sparse, num_steps=20) for s in ("padmm", "padmm_primal")}
                np.testing.assert_allclose(body_q["padmm_primal"], body_q["padmm"], atol=1e-4)

    def test_03_trajectories_match_padmm_with_compliance(self):
        """Checks that compliant simulations with implicit joint dynamics coincide for both PADMM solvers."""
        compliance = ConstraintComplianceConfig(
            joints=1e-4, limits=1e-4, contacts_normal=1e-3, contacts_tangential=1e-3
        )
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                body_q = {
                    s: self._simulate_fourbar(
                        s, sparse, num_steps=20, dynamic_joints=True, implicit_pd=True, compliance=compliance
                    )
                    for s in ("padmm", "padmm_primal")
                }
                np.testing.assert_allclose(body_q["padmm_primal"], body_q["padmm"], atol=1e-4)


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
