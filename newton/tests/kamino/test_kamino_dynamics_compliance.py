# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the constraint compliance of constrained forward dynamics problems."""

import unittest
from collections.abc import Callable

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.config import ConstrainedDynamicsProblemConfig
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig, ConstraintComplianceConfig
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_cts_jacobians, extract_delassus, extract_problem_vector
from newton.tests.kamino.utils.make import (
    make_constraint_compliance,
    make_containers,
    make_inverse_generalized_mass_matrices,
    set_fourbar_body_states,
    update_containers,
)
from newton.tests.utils import basics

###
# Helper functions
###


def build_effort_limited_pendulum() -> ModelBuilder:
    """Builds a pendulum whose joint has an armature and an effort-limited implicit PD actuator."""
    builder = ModelBuilder()
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        com=wp.vec3f(0.5, 0.0, 0.0),
        xform=wp.transformf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=newton.Axis.Y,
        parent_xform=wp.transformf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        effort_limit=5.0,
        target_ke=100.0,
        target_kd=10.0,
        armature=0.1,
        friction=0.01,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([joint])
    builder.add_ground_plane()
    builder.end_world()
    return builder


def make_compliance_config(scale: float = 1.0) -> ConstraintComplianceConfig:
    """Makes a compliance config with distinct non-zero values for each constraint group."""
    return ConstraintComplianceConfig(
        joints=1.0e-3 * scale,
        limits=2.0e-3 * scale,
        contacts_normal=3.0e-3 * scale,
        contacts_tangential=4.0e-3 * scale,
    )


def compute_expected_compliance(
    model: ModelKamino, data: DataKamino, configs: list[ConstrainedDynamicsProblemConfig]
) -> list[np.ndarray]:
    """Computes the expected compliance of the active constraints of each world from the constraint layout."""
    ndyn = model.info.num_joint_dynamic_cts.numpy()
    dyn_offset = model.info.joint_dynamic_cts_offset.numpy()
    kcgo = model.info.joint_kinematic_cts_group_offset.numpy()
    nkin = model.info.num_joint_kinematic_cts.numpy()
    ecgo = model.info.joint_effort_cts_group_offset.numpy()
    neff = model.info.num_joint_effort_cts.numpy()
    eff_offset = model.info.joint_effort_cts_offset.numpy()
    lcgo = data.info.limit_cts_group_offset.numpy()
    nl = data.info.num_limits.numpy()
    ccgo = data.info.contact_cts_group_offset.numpy()
    nc = data.info.num_contacts.numpy()
    dim = data.info.num_total_cts.numpy()
    inv_m_j = data.joints.inv_m_j.numpy()
    inv_m_a = data.joints.inv_m_a.numpy()

    R_np: list[np.ndarray] = []
    for w in range(model.size.num_worlds):
        c = configs[w].compliance
        R_w = np.zeros(shape=(dim[w],), dtype=np.float32)
        R_w[: ndyn[w]] = inv_m_j[dyn_offset[w] : dyn_offset[w] + ndyn[w]]
        R_w[kcgo[w] : kcgo[w] + nkin[w]] = c.joints
        R_w[ecgo[w] : ecgo[w] + neff[w]] = inv_m_a[eff_offset[w] : eff_offset[w] + neff[w]]
        R_w[lcgo[w] : lcgo[w] + nl[w]] = c.limits
        R_w[ccgo[w] : ccgo[w] + 3 * nc[w]] = np.tile(
            [c.contacts_tangential, c.contacts_tangential, c.contacts_normal], nc[w]
        )
        R_np.append(R_w)
    return R_np


def compute_rigid_delassus(setup: "TestSetup") -> list[np.ndarray]:
    """Computes the rigid Delassus matrix ``J M^-1 J^T`` of the active constraints of each world."""
    J_np = extract_cts_jacobians(setup.model, setup.limits, setup.contacts, setup.jacobians, only_active_cts=True)
    invM_np = make_inverse_generalized_mass_matrices(setup.model, setup.data)
    return [J_np[w] @ invM_np[w] @ J_np[w].T for w in range(setup.model.size.num_worlds)]


class TestSetup:
    def __init__(
        self,
        builder_fn: Callable[..., ModelBuilder],
        set_state_fn: Callable[[ModelKamino, DataKamino], None] | None = None,
        max_world_contacts: int = 12,
        device: wp.DeviceLike = None,
        **kwargs,
    ):
        # Construct the model description using model builders for different systems
        self.builder: ModelBuilder = builder_fn(**kwargs)

        # Create the model and containers from the builder
        # NOTE: Sparse Jacobians are used to build both dense and sparse problems
        model = ModelKamino.from_newton(self.builder.finalize(device=device))
        self.model, self.data, self.state, self.limits, self.detector, self.jacobians = make_containers(
            model=model, max_world_contacts=max_world_contacts, sparse=True
        )
        self.contacts: ContactsKamino = self.detector.contacts

        # Optionally perturb the system state to activate joint limits and contacts
        if set_state_fn is not None:
            set_state_fn(model=self.model, data=self.data)

        # Update the sim data containers
        update_containers(
            model=self.model,
            data=self.data,
            state=self.state,
            limits=self.limits,
            detector=self.detector,
            jacobians=self.jacobians,
        )

    def make_problem(
        self,
        sparse: bool = False,
        compliance: ConstraintComplianceConfig | list[ConstraintComplianceConfig] | None = None,
        preconditioning: bool = True,
    ) -> DualProblem:
        """Creates and builds a dual problem with the given compliance configuration(s)."""
        if not isinstance(compliance, list):
            compliance = [compliance] * self.model.size.num_worlds
        config = [
            DualProblem.Config(
                dynamics=ConstrainedDynamicsConfig(preconditioning=preconditioning),
                compliance=c if c is not None else ConstraintComplianceConfig(),
            )
            for c in compliance
        ]
        problem = DualProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
            solver=ConjugateResidualSolver if sparse else LLTBlockedSolver,
            config=config,
            sparse=sparse,
        )
        problem.build(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
        )
        return problem

    def make_random_vector(self, seed: int) -> wp.array:
        """Creates a random constraint-space vector that is non-zero only on the active constraints."""
        rng = np.random.default_rng(seed=seed)
        vio = self.model.info.total_cts_offset.numpy()
        dim = self.data.info.num_total_cts.numpy()
        x_np = np.zeros(shape=(self.model.size.sum_of_max_total_cts,), dtype=np.float32)
        for w in range(self.model.size.num_worlds):
            x_np[vio[w] : vio[w] + dim[w]] = rng.standard_normal(dim[w]).astype(np.float32)
        return wp.array(x_np, dtype=wp.float32, device=self.model.device)


def make_setup_fourbar(device: wp.DeviceLike) -> TestSetup:
    """Makes a four-bar with dynamic joints, joint friction, and active joint limits and contacts."""
    return TestSetup(
        builder_fn=basics.build_boxes_fourbar,
        set_state_fn=set_fourbar_body_states,
        device=device,
        dynamic_joints=True,
        implicit_pd=True,
    )


def make_setup_heterogeneous(device: wp.DeviceLike, dynamic_joints: bool = True) -> TestSetup:
    """Makes a heterogeneous multi-world model with active joint limits and contacts."""
    return TestSetup(
        builder_fn=basics.make_basics_heterogeneous_builder,
        set_state_fn=set_fourbar_body_states,
        device=device,
        dynamic_joints=dynamic_joints,
        implicit_pd=dynamic_joints,
    )


###
# Tests
###


class TestConstraintCompliance(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_00_compliance_config(self):
        """Validate compliance configs and convert them to the problem config struct."""
        self.assertFalse(DualProblem.Config().has_compliance)
        for name in ("joints", "limits", "contacts_normal", "contacts_tangential"):
            with self.subTest(name=name):
                self.assertTrue(DualProblem.Config(compliance=ConstraintComplianceConfig(**{name: 1.0})).has_compliance)
                with self.assertRaises(ValueError):
                    ConstraintComplianceConfig(**{name: -1.0})

        config_struct = DualProblem.Config(compliance=make_compliance_config()).to_struct()
        self.assertAlmostEqual(config_struct.compliance_joints, 1.0e-3)
        self.assertAlmostEqual(config_struct.compliance_limits, 2.0e-3)
        self.assertAlmostEqual(config_struct.compliance_contacts_normal, 3.0e-3)
        self.assertAlmostEqual(config_struct.compliance_contacts_tangential, 4.0e-3)

    def test_01_rigid_problem_has_no_compliance(self):
        """Omit the compliance of problems whose constraints are all rigid."""
        setup = TestSetup(builder_fn=basics.build_box_on_plane, device=self.default_device)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                problem = setup.make_problem(sparse=sparse)
                self.assertIsNone(problem.data.R)
                self.assertIsNone(problem.delassus.compliance)
                if sparse:
                    self.assertIsNone(problem.delassus._combined_regularization)

                # Any non-zero compliance requires the allocation
                problem = setup.make_problem(sparse=sparse, compliance=ConstraintComplianceConfig(contacts_normal=1e-3))
                self.assertIsNotNone(problem.data.R)
                self.assertIs(problem.delassus.compliance, problem.data.R)

    def test_02_build_compliance_per_constraint_group(self):
        """Assign the configured compliance to each constraint group of each world."""
        setups = {
            "fourbar": make_setup_fourbar(self.default_device),
            "heterogeneous": make_setup_heterogeneous(self.default_device),
            "effort": TestSetup(builder_fn=build_effort_limited_pendulum, device=self.default_device),
        }
        for name, setup in setups.items():
            with self.subTest(model=name):
                # Use different compliance configurations for each world
                num_worlds = setup.model.size.num_worlds
                compliance = [make_compliance_config(scale=float(w + 1)) for w in range(num_worlds)]
                problem = setup.make_problem(compliance=compliance)

                # Ensure that all constraint groups are present in the test model
                model_size = setup.model.size
                if name == "fourbar":
                    self.assertGreater(model_size.sum_of_num_dynamic_joint_cts, 0)
                    self.assertGreater(model_size.sum_of_num_kinematic_joint_cts, 0)
                    self.assertGreater(model_size.sum_of_num_friction_joint_cts, 0)
                    self.assertGreater(int(np.sum(setup.data.info.num_limits.numpy())), 0)
                    self.assertGreater(int(np.sum(setup.data.info.num_contacts.numpy())), 0)
                elif name == "effort":
                    self.assertGreater(model_size.sum_of_num_effort_joint_cts, 0)

                # Compare the compliance of active constraints and ensure inactive constraints are rigid
                R_expected = compute_expected_compliance(setup.model, setup.data, problem.config)
                R_active = extract_problem_vector(problem.delassus, problem.data.R.numpy(), only_active_dims=True)
                R_all = extract_problem_vector(problem.delassus, problem.data.R.numpy(), only_active_dims=False)
                for w in range(num_worlds):
                    np.testing.assert_allclose(R_active[w], R_expected[w], rtol=1e-6, atol=0.0)
                    np.testing.assert_array_equal(R_all[w][R_active[w].size :], 0.0)

                # The standalone builder must reproduce the compliance of the problem
                if num_worlds == 1:
                    R_standalone = make_constraint_compliance(setup.model, setup.data, problem.config[0])
                    np.testing.assert_array_equal(R_standalone.numpy(), problem.data.R.numpy())

    def test_03_dense_delassus_includes_compliance(self):
        """Add the compliance to the diagonal of the dense Delassus matrix."""
        setup = make_setup_heterogeneous(self.default_device, dynamic_joints=False)
        problem_rigid = setup.make_problem(preconditioning=False)
        D_rigid = extract_delassus(problem_rigid.delassus, only_active_dims=True)
        self.assertIsNone(problem_rigid.data.R)

        problem = setup.make_problem(compliance=make_compliance_config(), preconditioning=False)
        D = extract_delassus(problem.delassus, only_active_dims=True)
        R = extract_problem_vector(problem.delassus, problem.data.R.numpy(), only_active_dims=True)
        for w in range(setup.model.size.num_worlds):
            self.assertTrue(np.any(R[w] > 0.0))
            np.testing.assert_allclose(D[w], D_rigid[w] + np.diag(R[w]), rtol=1e-5, atol=1e-5)

    def test_04_gemv_applies_compliance(self):
        """Compute products with the compliant operator consistently across dense and sparse problems."""
        setup = make_setup_heterogeneous(self.default_device)
        D_rigid = compute_rigid_delassus(setup)
        x = setup.make_random_vector(seed=42)

        results = {}
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                problem = setup.make_problem(sparse=sparse, compliance=make_compliance_config())
                delassus = problem.delassus
                R = extract_problem_vector(delassus, problem.data.R.numpy(), only_active_dims=True)
                P = extract_problem_vector(delassus, problem.data.P.numpy(), only_active_dims=True)
                x_np = extract_problem_vector(delassus, x.numpy(), only_active_dims=True)

                # Compute the products with the compliant operator in physical and preconditioned coordinates
                y = wp.zeros_like(x)
                y_p = wp.zeros_like(x)
                problem.gemv(x=x, y=y, preconditioned=False)
                problem.gemv(x=x, y=y_p, preconditioned=True)
                y_np = extract_problem_vector(delassus, y.numpy(), only_active_dims=True)
                y_p_np = extract_problem_vector(delassus, y_p.numpy(), only_active_dims=True)
                for w in range(setup.model.size.num_worlds):
                    A_w = D_rigid[w] + np.diag(R[w])
                    np.testing.assert_allclose(y_np[w], A_w @ x_np[w], rtol=1e-4, atol=1e-3)
                    np.testing.assert_allclose(y_p_np[w], P[w] * (A_w @ (P[w] * x_np[w])), rtol=1e-4, atol=1e-3)
                results[sparse] = (problem.data.P.numpy(), problem.data.v_f.numpy(), y.numpy(), y_p.numpy())

        # Dense and sparse problems must agree, including their preconditioners
        for dense_values, sparse_values in zip(results[False], results[True], strict=True):
            np.testing.assert_allclose(dense_values, sparse_values, rtol=1e-4, atol=1e-3)

    def test_05_implicit_joint_dynamics_compliance(self):
        """Represent the implicit joint dynamics as the compliance of dynamic and effort-limited joint constraints."""
        setups = {
            "fourbar": make_setup_fourbar(self.default_device),
            "effort": TestSetup(builder_fn=build_effort_limited_pendulum, device=self.default_device),
        }
        for name, setup in setups.items():
            D_rigid = compute_rigid_delassus(setup)
            for sparse in (False, True):
                with self.subTest(model=name, sparse=sparse):
                    problem = setup.make_problem(sparse=sparse, preconditioning=False)
                    self.assertIsNotNone(problem.data.R)

                    # The compliance must only be non-zero for dynamic and effort-limited joint constraints
                    R_expected = compute_expected_compliance(setup.model, setup.data, problem.config)
                    R = extract_problem_vector(problem.delassus, problem.data.R.numpy(), only_active_dims=True)
                    np.testing.assert_allclose(R[0], R_expected[0], rtol=1e-6, atol=0.0)
                    num_compliant = setup.model.size.sum_of_num_dynamic_joint_cts
                    num_compliant += setup.model.size.sum_of_num_effort_joint_cts
                    self.assertEqual(int(np.count_nonzero(R[0])), num_compliant)

                    # The Delassus matrix must be the rigid Delassus matrix plus the compliance
                    D = extract_delassus(problem.delassus, only_active_dims=True)
                    np.testing.assert_allclose(D[0], D_rigid[0] + np.diag(R_expected[0]), rtol=1e-4, atol=1e-4)

    def test_06_padmm_solve_with_compliance(self):
        """Solve compliant problems with PADMM such that ``v_plus = (D + R) lambda + v_f``.

        The perturbed four-bar violates its joints and limits, which requires more iterations than the default.
        """
        setups = {
            "box_on_plane": TestSetup(builder_fn=basics.build_box_on_plane, device=self.default_device),
            "fourbar": make_setup_fourbar(self.default_device),
        }
        for name, setup in setups.items():
            D_rigid = compute_rigid_delassus(setup)
            for sparse in (False, True):
                with self.subTest(model=name, sparse=sparse):
                    problem = setup.make_problem(sparse=sparse, compliance=make_compliance_config(scale=10.0))
                    solver = PADMMSolver(
                        model=setup.model,
                        config=PADMMSolver.Config(max_iterations=2000),
                        warmstart=PADMMWarmStartMode.NONE,
                        use_acceleration=True,
                        use_graph_conditionals=False,
                    )
                    solver.reset()
                    solver.coldstart()
                    solver.solve(problem=problem)

                    # Check the solver convergence
                    status = solver.data.status.numpy()[0]
                    self.assertTrue(bool(status["converged"]))

                    # Check the solution against the compliant dual problem in physical coordinates
                    delassus = problem.delassus
                    R = extract_problem_vector(delassus, problem.data.R.numpy(), only_active_dims=True)[0]
                    P = extract_problem_vector(delassus, problem.data.P.numpy(), only_active_dims=True)[0]
                    v_f = extract_problem_vector(delassus, problem.data.v_f.numpy(), only_active_dims=True)[0] / P
                    lambdas = extract_problem_vector(
                        delassus, solver.data.solution.lambdas.numpy(), only_active_dims=True
                    )[0]
                    v_plus = extract_problem_vector(
                        delassus, solver.data.solution.v_plus.numpy(), only_active_dims=True
                    )[0]
                    v_plus_expected = (D_rigid[0] + np.diag(R)) @ lambdas + v_f
                    tolerance = 1.0e-5 * max(1.0, np.max(np.abs(v_f)))
                    self.assertLessEqual(np.max(np.abs(v_plus - v_plus_expected)), tolerance)

                    # Ensure that the compliance contributes measurably to the post-event velocity
                    self.assertGreater(np.max(np.abs(R * lambdas)), 2.0 * tolerance)

    def test_07_solver_forwards_compliance_config(self):
        """Forward the constraint compliance config of the solver to its forward dynamics problem."""
        builder = basics.build_boxes_fourbar()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        compliance = make_compliance_config()
        solver = SolverKaminoImpl(
            model=model,
            contacts=ContactsKamino(model=model),
            config=SolverKaminoImpl.Config(compliance=compliance),
        )
        self.assertIs(solver.config.compliance, compliance)
        self.assertIs(solver._problem_fd.config[0].compliance, compliance)
        self.assertIsNotNone(solver._problem_fd.data.R)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
