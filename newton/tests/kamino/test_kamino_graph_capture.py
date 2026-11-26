# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph capture of :meth:`SolverKamino.step` on never-stepped solvers and states."""

import gc
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino import config as kamino_config
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_test_output, setup_tests, test_context
from newton.tests.kamino.utils.solver_configs import (
    make_padmm_dense_config,
    make_padmm_sparse_config,
    make_scc_dense_config,
    make_scc_sparse_config,
)
from newton.tests.unittest_utils import NewtonTestCase

DT = 0.002
_TARGETS = (0.2, -0.2, 2.0, -2.0)


def _build_model() -> newton.Model:
    """Build one revolute joint per world with dynamic, friction, and effort-limit rows."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    for _ in _TARGETS:
        builder.begin_world()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-4), lock_inertia=True)
        joint = builder.add_joint_revolute(
            -1,
            body,
            axis=newton.Axis.Y,
            armature=0.011,
            friction=0.01,
            effort_limit=0.5,
            target_ke=10.0,
            target_kd=2.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_chain() -> newton.Model:
    """Build two worlds, each a 64-link revolute chain swinging under gravity.

    Each link adds five kinematic and one dynamic row, so a world has the 384
    bilateral rows that select SCC's persistent joint-block solver.
    """
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    for _ in range(2):
        builder.begin_world()
        parent, joints = -1, []
        for _ in range(64):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-2), lock_inertia=True)
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform((0.2 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                    armature=0.01,
                    effort_limit=0.5,
                    target_ke=10.0,
                    target_kd=1.0,
                    actuator_mode=newton.JointTargetMode.POSITION,
                )
            )
            parent = body
        builder.add_articulation(joints)
        builder.end_world()
    return builder.finalize()


def _build_large_batch() -> newton.Model:
    """Build 2048 worlds, each an 8-link effort-limited revolute chain swinging under gravity.

    Each world has 48 bilateral and 8 effort-limit rows, which selects SCC's
    nonpersistent joint-block solver with packed coupling and effort-limit condensation.
    """
    chain = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(chain)
    parent, joints = -1, []
    for _ in range(8):
        body = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-2), lock_inertia=True)
        joints.append(
            chain.add_joint_revolute(
                parent,
                body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform((0.2 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                armature=0.01,
                effort_limit=0.5,
                target_ke=10.0,
                target_kd=1.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
        )
        parent = body
    chain.add_articulation(joints)
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.replicate(chain, 2048)
    return builder.finalize()


def _make_lox_config(**kwargs) -> SolverKamino.Config:
    return SolverKamino.Config(dynamics_solver="lox", use_fk_solver=False, use_collision_detector=False, **kwargs)


def _make_scc_joint_block_config() -> SolverKamino.Config:
    return make_scc_sparse_config(scc=kamino_config.SCCSolverConfig(use_schur_complement=True))


def _make_simulation(model: newton.Model, config: SolverKamino.Config, targets=None):
    """Create a never-stepped solver with fresh ping-pong states."""
    solver = SolverKamino(model, config)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    if targets is not None:
        control.joint_target_q.assign(targets)
    return solver, state_0, state_1, control


def _step_twice(solver, state_0, state_1, control):
    solver.step(state_0, state_1, control, None, DT)
    solver.step(state_1, state_0, control, None, DT)


class TestKaminoGraphCapture(NewtonTestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        setup_test_output(self)
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_first_use_capture_and_replay(self):
        """Capture the first-ever steps of a solver and replay them against eager stepping."""
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        if not wp.is_conditional_graph_supported():
            self.skipTest("The captured solver loops require conditional CUDA graphs")
        model, chain, large_batch = _build_model(), _build_chain(), _build_large_batch()
        # Each row allocates or branches differently on its first captured step.
        rows = {
            "padmm_dense": (model, make_padmm_dense_config),
            "padmm_sparse": (model, make_padmm_sparse_config),
            "padmm_sparse_solver_info": (model, lambda: make_padmm_sparse_config(collect_solver_info=True)),
            "padmm_sparse_metrics": (model, lambda: make_padmm_sparse_config(compute_solution_metrics=True)),
            "scc_dense": (model, make_scc_dense_config),
            "scc_sparse": (model, make_scc_sparse_config),
            "lox": (model, _make_lox_config),
            # Only taken while capturing, so eager stepping is the reference for its
            # optimized solve; the Jacobian changes on every replayed step.
            "scc_sparse_joint_blocks": (chain, _make_scc_joint_block_config),
            "scc_sparse_joint_blocks_large_batch": (large_batch, _make_scc_joint_block_config),
        }
        for name, (row_model, config_factory) in rows.items():
            with self.subTest(config=name):
                targets = _TARGETS if row_model is model else None
                solver, state_0, state_1, control = _make_simulation(row_model, config_factory(), targets)
                for _ in range(2):
                    _step_twice(solver, state_0, state_1, control)
                expected_qd = state_0.joint_qd.numpy()
                self.assertGreater(np.abs(expected_qd).max(), 1.0e-3)

                solver, state_0, state_1, control = _make_simulation(row_model, config_factory(), targets)
                with wp.ScopedCapture() as capture:
                    _step_twice(solver, state_0, state_1, control)
                if name.startswith("scc_sparse_joint_blocks"):
                    path = solver._solver_kamino._solver_fd._sparse_path
                    self.assertIsNotNone(path.joint_block_solver)
                    large_batch_paths = name.endswith("large_batch")
                    self.assertEqual(path.joint_block_solver._persistent, not large_batch_paths)
                    self.assertEqual(path.packed_coupling is not None, large_batch_paths)
                    self.assertEqual(path.effort_limit_condensation is not None, large_batch_paths)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                np.testing.assert_allclose(state_0.joint_qd.numpy(), expected_qd, rtol=1.0e-4, atol=1.0e-6)

    def test_graph_replacement_on_fresh_states(self):
        """Replay a recaptured graph after the first, never-launched capture is destroyed."""
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = _build_model()

        solver, state_0, state_1, control = _make_simulation(model, make_padmm_dense_config(), _TARGETS)
        for _ in range(2):
            _step_twice(solver, state_0, state_1, control)
        expected_qd = state_0.joint_qd.numpy()
        expected_effort = state_0.joint_lambdas_tau.numpy()
        self.assertTrue(np.all(np.abs(expected_qd) > 1.0e-3))

        solver, state_0, state_1, control = _make_simulation(model, make_padmm_dense_config(), _TARGETS)
        with wp.ScopedCapture() as discarded:
            _step_twice(solver, state_0, state_1, control)
        with wp.ScopedCapture() as capture:
            _step_twice(solver, state_0, state_1, control)
        # Destroy the first graph: nothing the solver or the states keep may be owned by it.
        del discarded
        gc.collect()
        wp.synchronize()
        for _ in range(2):
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_0.joint_qd.numpy(), expected_qd, rtol=1.0e-4, atol=1.0e-6)
        np.testing.assert_allclose(state_0.joint_lambdas_tau.numpy(), expected_effort, rtol=1.0e-4, atol=1.0e-6)

        # Release everything inside the test so invalid frees are reported here.
        del capture, solver, state_0, state_1, control
        gc.collect()
        wp.synchronize()


if __name__ == "__main__":
    newton.utils.setup_console_logging()
    setup_tests()
    unittest.main(verbosity=2)
