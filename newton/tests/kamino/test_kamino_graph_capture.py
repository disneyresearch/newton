# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph capture of :meth:`SolverKamino.step` on never-stepped solvers and states."""

import gc
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_test_output, setup_tests, test_context
from newton.tests.kamino.utils.solver_configs import make_padmm_dense_config
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


def _make_simulation(model: newton.Model):
    """Create a never-stepped solver with fresh ping-pong states."""
    solver = SolverKamino(model, make_padmm_dense_config())
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    control.joint_target_q.assign(_TARGETS)
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

    def test_graph_replacement_on_fresh_states(self):
        """Replay a recaptured graph after the first, never-launched capture is destroyed."""
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = _build_model()

        solver, state_0, state_1, control = _make_simulation(model)
        for _ in range(2):
            _step_twice(solver, state_0, state_1, control)
        expected_qd = state_0.joint_qd.numpy()
        expected_effort = state_0.joint_lambdas_tau.numpy()
        self.assertTrue(np.all(np.abs(expected_qd) > 1.0e-3))

        solver, state_0, state_1, control = _make_simulation(model)
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
