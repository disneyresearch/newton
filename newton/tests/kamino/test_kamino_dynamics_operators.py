# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: DYNAMICS: MASS AND JACOBIAN OPERATORS
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.operators import (
    compute_delassus_diagonal,
    jacobian_gemv,
    jacobian_transpose_gemv,
    mass_gemv,
)
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


def _make_problem(device: wp.DeviceLike, sparse: bool):
    builder = make_basics_heterogeneous_builder()
    for i in range(len(builder.body_qd)):
        builder.body_qd[i] = wp.spatial_vectorf(0.5, 0.0, -0.2, 0.0, 0.3, 0.0)
    model = ModelKamino.from_newton(builder.finalize(device=device))
    model, data, state, limits, detector, jacobians = make_containers(model=model, max_world_contacts=12, sparse=sparse)
    update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)
    return model, data, limits, detector.contacts, jacobians


def _world_blocks(model: ModelKamino, data, x: np.ndarray, y: np.ndarray):
    """Splits flat body-space and constraint-space vectors into per-world active blocks."""
    bio = model.info.bodies_offset.numpy()
    vio = model.info.total_cts_offset.numpy()
    dim = data.info.num_total_cts.numpy()
    xs = [x[6 * bio[w] : 6 * bio[w + 1]] for w in range(model.size.num_worlds)]
    ys = [y[vio[w] : vio[w] + dim[w]] for w in range(model.size.num_worlds)]
    return xs, ys


###
# Tests
###


class TestDynamicsOperators(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.rng = np.random.default_rng(seed=7)

    def tearDown(self):
        self.default_device = None

    def test_01_mass_gemv(self):
        model, data, _limits, _contacts, _jacobians = _make_problem(self.default_device, sparse=False)
        mask = wp.ones(model.size.num_worlds, dtype=wp.bool, device=model.device)
        n_u = model.size.sum_of_num_body_dofs
        x_np = self.rng.standard_normal(n_u).astype(np.float32)
        y_np = self.rng.standard_normal(n_u).astype(np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=model.device)
        y = wp.array(y_np, dtype=wp.float32, device=model.device)
        mass_gemv(model, data, x, y, mask, alpha=2.0, beta=0.5)

        M_np = make_generalized_mass_matrices(model, data)
        xs, _ = _world_blocks(model, data, x_np, np.zeros(model.size.sum_of_max_total_cts))
        ys, _ = _world_blocks(model, data, y_np, np.zeros(model.size.sum_of_max_total_cts))
        ws, _ = _world_blocks(model, data, y.numpy(), np.zeros(model.size.sum_of_max_total_cts))
        for w in range(model.size.num_worlds):
            np.testing.assert_allclose(ws[w], 2.0 * M_np[w] @ xs[w] + 0.5 * ys[w], rtol=1e-5, atol=1e-5)

    def test_02_jacobian_products_and_delassus_diagonal(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                model, data, limits, contacts, jacobians = _make_problem(self.default_device, sparse=sparse)
                device = model.device
                mask = wp.ones(model.size.num_worlds, dtype=wp.bool, device=device)
                dim = data.info.num_total_cts
                vio = model.info.total_cts_offset
                n_u = model.size.sum_of_num_body_dofs
                n_c = model.size.sum_of_max_total_cts

                # Extract the dense reference Jacobians of the active constraints
                J_np = extract_cts_jacobians(model, limits, contacts, jacobians, only_active_cts=True)
                dim_np = dim.numpy()
                nbd_np = model.info.num_body_dofs.numpy()
                J_np = [J_np[w][: dim_np[w], : nbd_np[w]] for w in range(model.size.num_worlds)]
                invM_np = make_inverse_generalized_mass_matrices(model, data)

                x_np = self.rng.standard_normal(n_u).astype(np.float32)
                y_np = self.rng.standard_normal(n_c).astype(np.float32)
                xs, ys = _world_blocks(model, data, x_np, y_np)

                # y = 1.5 * J x - 0.5 * y
                y = wp.array(y_np, dtype=wp.float32, device=device)
                jacobian_gemv(model, jacobians, dim, vio, wp.array(x_np, device=device), y, mask, alpha=1.5, beta=-0.5)
                _, Jx = _world_blocks(model, data, x_np, y.numpy())

                # x = 2.0 * J^T y + 1.0 * x
                x = wp.array(x_np, dtype=wp.float32, device=device)
                jacobian_transpose_gemv(
                    model, jacobians, dim, vio, wp.array(y_np, device=device), x, mask, alpha=2.0, beta=1.0
                )
                JTy, _ = _world_blocks(model, data, x.numpy(), y_np)

                # diag(J M^-1 J^T)
                diag = wp.zeros(n_c, dtype=wp.float32, device=device)
                compute_delassus_diagonal(model, data, jacobians, dim, vio, diag)
                _, D_diag = _world_blocks(model, data, x_np, diag.numpy())

                for w in range(model.size.num_worlds):
                    np.testing.assert_allclose(Jx[w], 1.5 * J_np[w] @ xs[w] - 0.5 * ys[w], rtol=1e-4, atol=1e-4)
                    np.testing.assert_allclose(JTy[w], 2.0 * J_np[w].T @ ys[w] + xs[w], rtol=1e-4, atol=1e-4)
                    D_ref = np.diag(J_np[w] @ invM_np[w] @ J_np[w].T)
                    np.testing.assert_allclose(D_diag[w], D_ref, rtol=1e-4, atol=1e-5)


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
