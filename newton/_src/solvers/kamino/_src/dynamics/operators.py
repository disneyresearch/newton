# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides operations with the generalized mass matrix ``M`` and the constraint Jacobian ``J`` of the
system, which are shared by the formulations of the constrained forward dynamics problem that operate
on generalized velocities, e.g. the primal and KKT formulations.

Generalized (i.e. body-space) vectors are represented as flat arrays of type :class:`float32` with
shape ``(sum_of_num_body_dofs,)``, where the block of world ``w`` starts at ``6 * bodies_offset[w]``
and each body contributes its twist coordinates ``[v; omega]`` in world frame. Constraint-space vectors
use the layout of :class:`ConstrainedDynamicsProblemData`, i.e. the block of world ``w`` starts at
``total_cts_offset[w]`` and spans the active number of constraints ``num_total_cts[w]``.
"""

from __future__ import annotations

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..core.types import vec6f
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..linalg.blas import block_sparse_gemv, block_sparse_transpose_gemv
from .delassus import _compute_block_sparse_delassus_diagonal

###
# Module interface
###

__all__ = [
    "compute_delassus_diagonal",
    "inverse_mass_gemv",
    "jacobian_gemv",
    "jacobian_transpose_gemv",
    "mass_gemv",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Kernels
###


@wp.kernel
def _mass_gemv(
    # Inputs:
    model_bodies_wid: wp.array[wp.int32],
    bodies_m_i: wp.array[wp.float32],
    bodies_I_i: wp.array[wp.mat33f],
    world_mask: wp.array[wp.bool],
    alpha: wp.float32,
    beta: wp.float32,
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Skip if the world of the body is inactive
    if not world_mask[model_bodies_wid[bid]]:
        return

    # Compute the product with the body block of the (inverse) mass matrix: [m * v; I @ omega]
    k = 6 * bid
    m_i = bodies_m_i[bid]
    I_i = bodies_I_i[bid]
    f = m_i * wp.vec3f(x[k], x[k + 1], x[k + 2])
    tau = I_i @ wp.vec3f(x[k + 3], x[k + 4], x[k + 5])
    for d in range(3):
        y[k + d] = alpha * f[d] + beta * y[k + d]
        y[k + 3 + d] = alpha * tau[d] + beta * y[k + 3 + d]


@wp.kernel
def _dense_jacobian_gemv(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    jacobians_J_cts_offsets: wp.array[wp.int32],
    jacobians_J_cts_data: wp.array[wp.float32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    alpha: wp.float32,
    beta: wp.float32,
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the world and constraint row indices as the thread indices
    wid, row = wp.tid()

    # Skip if the world is inactive or the row index exceeds the problem size
    if not world_mask[wid] or row >= problem_dim[wid]:
        return

    # Retrieve the body-space dimensions and offsets of the world
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    J_row = jacobians_J_cts_offsets[wid] + nbd * row
    x_start = 6 * bio

    # Compute the row-vector product of the Jacobian
    acc = wp.float32(0.0)
    for k in range(nbd):
        acc += jacobians_J_cts_data[J_row + k] * x[x_start + k]

    # Store the generalized matrix-vector product
    v_i = problem_vio[wid] + row
    y[v_i] = alpha * acc + beta * y[v_i]


@wp.kernel
def _dense_jacobian_transpose_gemv(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    jacobians_J_cts_offsets: wp.array[wp.int32],
    jacobians_J_cts_data: wp.array[wp.float32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    alpha: wp.float32,
    beta: wp.float32,
    y: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and body-space column indices as the thread indices
    wid, col = wp.tid()

    # Retrieve the body-space dimensions and offsets of the world
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)

    # Skip if the world is inactive or the column index exceeds the number of body DoFs
    if not world_mask[wid] or col >= nbd:
        return

    # Compute the column-vector product of the Jacobian over the active constraints
    J_start = jacobians_J_cts_offsets[wid]
    vio = problem_vio[wid]
    acc = wp.float32(0.0)
    for row in range(problem_dim[wid]):
        acc += jacobians_J_cts_data[J_start + nbd * row + col] * y[vio + row]

    # Store the generalized matrix-transpose-vector product
    k = 6 * bio + col
    x[k] = alpha * acc + beta * x[k]


@wp.kernel
def _dense_delassus_diagonal(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    model_bodies_inv_m_i: wp.array[wp.float32],
    data_bodies_inv_I_i: wp.array[wp.mat33f],
    jacobians_J_cts_offsets: wp.array[wp.int32],
    jacobians_J_cts_data: wp.array[wp.float32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the world and constraint row indices as the thread indices
    wid, row = wp.tid()

    # Skip if the row index exceeds the problem size
    if row >= problem_dim[wid]:
        return

    # Retrieve the body-space dimensions and offsets of the world
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    J_row = jacobians_J_cts_offsets[wid] + 6 * nb * row

    # Accumulate the contributions of each body: D_ii = sum_k J_ik @ M_kk^-1 @ J_ik^T
    J_k = vec6f(0.0)
    d_ii = wp.float32(0.0)
    for k in range(nb):
        for d in range(6):
            J_k[d] = jacobians_J_cts_data[J_row + 6 * k + d]
        Jv = wp.vec3f(J_k[0], J_k[1], J_k[2])
        Jw = wp.vec3f(J_k[3], J_k[4], J_k[5])
        d_ii += model_bodies_inv_m_i[bio + k] * wp.dot(Jv, Jv) + wp.dot(Jw, data_bodies_inv_I_i[bio + k] @ Jw)

    # Store the diagonal entry of the Delassus matrix
    diag[problem_vio[wid] + row] = d_ii


###
# Operations
###


def mass_gemv(
    model: ModelKamino,
    data: DataKamino,
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: float = 1.0,
    beta: float = 0.0,
):
    """
    Computes the generalized matrix-vector product ``y = alpha * M @ x + beta * y`` with the
    block-diagonal generalized mass matrix ``M`` of the system.

    Args:
        model: The model providing the body masses.
        data: The model data providing the current (world-frame) body inertias.
        x: The input array of generalized vectors.
        y: The input-output array of generalized vectors.
        world_mask: The mask selecting the worlds to operate on.
        alpha: The scaling of the matrix-vector product.
        beta: The scaling of the input values of `y`.
    """
    if model.size.sum_of_num_bodies == 0:
        return
    wp.launch(
        _mass_gemv,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            model.bodies.wid,
            model.bodies.m_i,
            data.bodies.I_i,
            world_mask,
            wp.float32(alpha),
            wp.float32(beta),
            x,
            y,
        ],
        device=model.device,
    )


def inverse_mass_gemv(
    model: ModelKamino,
    data: DataKamino,
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: float = 1.0,
    beta: float = 0.0,
):
    """
    Computes the generalized matrix-vector product ``y = alpha * M^{-1} @ x + beta * y`` with the
    inverse of the block-diagonal generalized mass matrix ``M`` of the system.

    Args:
        model: The model providing the inverse body masses.
        data: The model data providing the current (world-frame) inverse body inertias.
        x: The input array of generalized vectors.
        y: The input-output array of generalized vectors.
        world_mask: The mask selecting the worlds to operate on.
        alpha: The scaling of the matrix-vector product.
        beta: The scaling of the input values of `y`.
    """
    if model.size.sum_of_num_bodies == 0:
        return
    wp.launch(
        _mass_gemv,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            model.bodies.wid,
            model.bodies.inv_m_i,
            data.bodies.inv_I_i,
            world_mask,
            wp.float32(alpha),
            wp.float32(beta),
            x,
            y,
        ],
        device=model.device,
    )


def jacobian_gemv(
    model: ModelKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: float = 1.0,
    beta: float = 0.0,
):
    """
    Computes the generalized matrix-vector product ``y = alpha * J @ x + beta * y`` with the
    constraint Jacobian ``J`` of the system, mapping generalized to constraint-space vectors.

    Args:
        model: The model providing the body-space layout.
        jacobians: The current constraint Jacobians of the system.
        dim: The active number of constraints of each world.
        vio: The constraint-space vector index offset of each world.
        x: The input array of generalized vectors.
        y: The input-output array of constraint-space vectors.
        world_mask: The mask selecting the worlds to operate on.
        alpha: The scaling of the matrix-vector product.
        beta: The scaling of the input values of `y`.
    """
    if isinstance(jacobians, SparseSystemJacobians):
        block_sparse_gemv(jacobians._J_cts.bsm, x, y, wp.float32(alpha), wp.float32(beta), world_mask)
    else:
        wp.launch(
            _dense_jacobian_gemv,
            dim=(model.size.num_worlds, model.size.max_of_max_total_cts),
            inputs=[
                model.info.bodies_offset,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                dim,
                vio,
                world_mask,
                wp.float32(alpha),
                wp.float32(beta),
                x,
                y,
            ],
            device=model.device,
        )


def jacobian_transpose_gemv(
    model: ModelKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    y: wp.array[wp.float32],
    x: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: float = 1.0,
    beta: float = 0.0,
):
    """
    Computes the generalized matrix-transpose-vector product ``x = alpha * J^T @ y + beta * x`` with
    the constraint Jacobian ``J`` of the system, mapping constraint-space to generalized vectors.

    Args:
        model: The model providing the body-space layout.
        jacobians: The current constraint Jacobians of the system.
        dim: The active number of constraints of each world.
        vio: The constraint-space vector index offset of each world.
        y: The input array of constraint-space vectors.
        x: The input-output array of generalized vectors.
        world_mask: The mask selecting the worlds to operate on.
        alpha: The scaling of the matrix-vector product.
        beta: The scaling of the input values of `x`.
    """
    if isinstance(jacobians, SparseSystemJacobians):
        block_sparse_transpose_gemv(jacobians._J_cts.bsm, y, x, wp.float32(alpha), wp.float32(beta), world_mask)
    else:
        wp.launch(
            _dense_jacobian_transpose_gemv,
            dim=(model.size.num_worlds, model.size.max_of_num_body_dofs),
            inputs=[
                model.info.bodies_offset,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                dim,
                vio,
                world_mask,
                wp.float32(alpha),
                wp.float32(beta),
                y,
                x,
            ],
            device=model.device,
        )


def compute_delassus_diagonal(
    model: ModelKamino,
    data: DataKamino,
    jacobians: DenseSystemJacobians | SparseSystemJacobians,
    dim: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    diag: wp.array[wp.float32],
):
    """
    Computes the diagonal of the Delassus matrix ``D = J M^{-1} J^T`` row-wise from the
    constraint Jacobian, without assembling the matrix. Inactive rows are set to zero.

    Args:
        model: The model providing the body masses and body-space layout.
        data: The model data providing the current inverse body inertias.
        jacobians: The current constraint Jacobians of the system.
        dim: The active number of constraints of each world.
        vio: The constraint-space vector index offset of each world.
        diag: The output array of constraint-space diagonal entries.
    """
    diag.zero_()
    if isinstance(jacobians, SparseSystemJacobians):
        J_cts = jacobians._J_cts.bsm
        wp.launch(
            _compute_block_sparse_delassus_diagonal,
            dim=(model.size.num_worlds, J_cts.max_of_num_nzb),
            inputs=[
                model.info.bodies_offset,
                model.bodies.inv_m_i,
                data.bodies.inv_I_i,
                J_cts.nzb_start,
                J_cts.num_nzb,
                J_cts.nzb_coords,
                J_cts.nzb_values,
                J_cts.row_start,
                diag,
            ],
            device=model.device,
        )
    else:
        wp.launch(
            _dense_delassus_diagonal,
            dim=(model.size.num_worlds, model.size.max_of_max_total_cts),
            inputs=[
                model.info.bodies_offset,
                model.bodies.inv_m_i,
                data.bodies.inv_I_i,
                jacobians.data.J_cts_offsets,
                jacobians.data.J_cts_data,
                dim,
                vio,
                diag,
            ],
            device=model.device,
        )
