# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides containers to represent and operate the constraint-augmented generalized mass matrix

.. code-block:: text

    M_W := M + J^T W J

where ``M`` is the block-diagonal generalized mass matrix, ``J`` the constraint Jacobian and
``W >= 0`` a diagonal matrix of per-constraint effective stiffnesses. ``M_W`` is the primal
counterpart of the regularized Delassus matrix ``D + W^{-1}``, to which it is related by the
Woodbury identity, and is symmetric positive definite for any ``W >= 0``, irrespective of the
rank of ``J``. Its dimension is the fixed number of generalized velocities of each world.

The :class:`AugmentedMassOperator` assembles ``M_W`` explicitly and factorizes it with a direct
solver, while the :class:`BlockSparseMatrixFreeAugmentedMassOperator` only realizes products with
``M_W`` and solves linear systems with an iterative solver. Both can apply a symmetric diagonal or
body-block-diagonal preconditioner (see :data:`AugmentedMassPreconditioner`).
"""

from __future__ import annotations

from typing import Any, Literal

import warp as wp

from ..core.data import DataKamino
from ..core.math import FLOAT32_EPS
from ..core.model import ModelKamino
from ..core.types import mat66f, vec6f
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LinearSolverType
from ..linalg.conjugate import BatchedLinearOperator
from ..linalg.linear import DirectSolver, IterativeSolver
from .operators import jacobian_gemv, jacobian_transpose_gemv, mass_gemv

###
# Module interface
###

__all__ = [
    "AugmentedMassOperator",
    "AugmentedMassPreconditioner",
    "BlockSparseMatrixFreeAugmentedMassOperator",
]


###
# Types
###

AugmentedMassPreconditioner = Literal["none", "jacobi", "block_jacobi"]
"""
The symmetric preconditioners ``S`` of the constraint-augmented generalized mass matrix
``A := M + J^T W J``, with which linear systems are solved via the scaled matrix ``S^T A S``:

- ``"none"``: No preconditioning, i.e. ``S = I``.
- ``"jacobi"``: Diagonal scaling ``S = diag(A)^{-1/2}``, so that ``S^T A S`` has a unit diagonal.
- ``"block_jacobi"``: Block-diagonal scaling ``S_k = L_k^{-T}`` from the Cholesky factors
  ``A_kk = L_k L_k^T`` of the 6x6 diagonal blocks of the bodies, so that ``S^T A S`` has identity
  diagonal blocks. This normalizes the mass and inertia of each body, i.e. it operates in the
  kinetic metric of the generalized mass matrix ``M``.
"""


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Kernels
###


@wp.kernel
def _assemble_augmented_mass_stiffness_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    jacobians_J_cts_offsets: wp.array[wp.int32],
    jacobians_J_cts_data: wp.array[wp.float32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    W: wp.array[wp.float32],
    operator_mio: wp.array[wp.int32],
    # Outputs:
    operator_M: wp.array[wp.float32],
):
    # Retrieve the world index and the flat matrix element index as the thread indices
    wid, tid = wp.tid()

    # Retrieve the body-space dimensions and offsets of the world
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)

    # Compute the row and column indices of the matrix element
    a = tid // nbd
    b = tid % nbd

    # Skip if the element exceeds the matrix size
    if a >= nbd:
        return

    # Accumulate the constraint stiffness contribution: (J^T W J)_ab = sum_i J_ia W_i J_ib
    J_start = jacobians_J_cts_offsets[wid]
    vio = problem_vio[wid]
    m_ab = wp.float32(0.0)
    for i in range(problem_dim[wid]):
        J_i = J_start + nbd * i
        m_ab += jacobians_J_cts_data[J_i + a] * W[vio + i] * jacobians_J_cts_data[J_i + b]

    # Add the contribution to the matrix element
    operator_M[operator_mio[wid] + nbd * a + b] += m_ab


@wp.kernel
def _assemble_mass_dense(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    operator_mio: wp.array[wp.int32],
    # Outputs:
    operator_M: wp.array[wp.float32],
):
    # Retrieve the world index and the flat matrix element index as the thread indices
    wid, tid = wp.tid()

    # Retrieve the body-space dimensions and offsets of the world
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)

    # Compute the row and column indices of the matrix element
    a = tid // nbd
    b = tid % nbd

    # Skip if the element exceeds the matrix size
    if a >= nbd:
        return

    # Store the element of the block-diagonal mass matrix
    m_ab = wp.float32(0.0)
    k = a // 6
    if k == b // 6:
        la = a % 6
        lb = b % 6
        if la < 3 and lb < 3:
            if la == lb:
                m_ab = model_bodies_m_i[bio + k]
        elif la >= 3 and lb >= 3:
            m_ab = data_bodies_I_i[bio + k][la - 3, lb - 3]
    operator_M[operator_mio[wid] + nbd * a + b] = m_ab


@wp.kernel
def _assemble_augmented_mass_stiffness_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    jac_num_nzb: wp.array[wp.int32],
    jac_nzb_start: wp.array[wp.int32],
    jac_nzb_coords: wp.array2d[wp.int32],
    jac_nzb_values: wp.array[vec6f],
    jac_row_start: wp.array[wp.int32],
    W: wp.array[wp.float32],
    operator_mio: wp.array[wp.int32],
    # Outputs:
    operator_M: wp.array[wp.float32],
):
    # Retrieve the world index and the flat index of a pair of non-zero blocks as the thread indices
    wid, tid = wp.tid()

    # Skip if the pair exceeds the number of non-zero blocks of the world
    num_nzb = jac_num_nzb[wid]
    if tid >= num_nzb * num_nzb:
        return

    # Compute the local indices of the pair of non-zero blocks
    k = tid // num_nzb
    l = tid % num_nzb

    # Only blocks of the same constraint row contribute: (J^T W J)_kl = J_ik^T W_i J_il
    nzb_k = jac_nzb_start[wid] + k
    nzb_l = jac_nzb_start[wid] + l
    row = jac_nzb_coords[nzb_k, 0]
    if row != jac_nzb_coords[nzb_l, 0]:
        return

    # Accumulate the outer product of the two blocks into the matrix
    nbd = 6 * (model_info_bodies_offset[wid + 1] - model_info_bodies_offset[wid])
    w_i = W[jac_row_start[wid] + row]
    J_k = jac_nzb_values[nzb_k]
    J_l = jac_nzb_values[nzb_l]
    m_start = operator_mio[wid] + nbd * jac_nzb_coords[nzb_k, 1] + jac_nzb_coords[nzb_l, 1]
    for d_k in range(6):
        for d_l in range(6):
            wp.atomic_add(operator_M, m_start + nbd * d_k + d_l, w_i * J_k[d_k] * J_l[d_l])


@wp.kernel
def _compute_augmented_mass_diagonal_mass(
    # Inputs:
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Store the diagonal of the body block of the mass matrix
    k = 6 * bid
    m_i = model_bodies_m_i[bid]
    I_i = data_bodies_I_i[bid]
    for d in range(3):
        diag[k + d] = m_i
        diag[k + 3 + d] = I_i[d, d]


@wp.kernel
def _compute_augmented_mass_diagonal_stiffness_sparse(
    # Inputs:
    jac_num_nzb: wp.array[wp.int32],
    jac_nzb_start: wp.array[wp.int32],
    jac_nzb_coords: wp.array2d[wp.int32],
    jac_nzb_values: wp.array[vec6f],
    jac_row_start: wp.array[wp.int32],
    jac_col_start: wp.array[wp.int32],
    W: wp.array[wp.float32],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the world index and the local non-zero block index as the thread indices
    wid, nzb_id = wp.tid()

    # Skip if the block index exceeds the number of non-zero blocks of the world
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve the Jacobian block and its coordinates
    nzb = jac_nzb_start[wid] + nzb_id
    coords = jac_nzb_coords[nzb]
    J_block = jac_nzb_values[nzb]

    # Accumulate the constraint stiffness contribution: (J^T W J)_kk = sum_i W_i J_ik^2
    w_i = W[jac_row_start[wid] + coords[0]]
    col = jac_col_start[wid] + coords[1]
    for d in range(6):
        wp.atomic_add(diag, col + d, w_i * J_block[d] * J_block[d])


@wp.kernel
def _compute_jacobi_scaling(
    # Inputs:
    diag: wp.array[wp.float32],
    # Outputs:
    scaling: wp.array[wp.float32],
):
    # Retrieve the generalized coordinate index as the thread index
    k = wp.tid()

    # Compute the symmetric Jacobi scaling `1 / sqrt(diag)` of the positive definite matrix
    scaling[k] = 1.0 / wp.sqrt(diag[k])


@wp.kernel
def _scale_body_vector(
    # Inputs:
    model_bodies_wid: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    scaling: wp.array[wp.float32],
    inverse: wp.bool,
    alpha: wp.float32,
    beta: wp.float32,
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the generalized coordinate index as the thread index
    k = wp.tid()

    # Skip if the world of the body is inactive
    if not world_mask[model_bodies_wid[k // 6]]:
        return

    # Compute the (inversely) scaled generalized vector
    s_k = scaling[k]
    if inverse:
        s_k = 1.0 / s_k
    y[k] = alpha * s_k * x[k] + beta * y[k]


@wp.kernel
def _scale_constraint_vector(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    W: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the world is inactive or the row index exceeds the problem size
    if not world_mask[wid] or tid >= problem_dim[wid]:
        return

    # Scale the constraint-space vector in-place by the effective stiffness
    v_i = problem_vio[wid] + tid
    x[v_i] = W[v_i] * x[v_i]


###
# Kernels: Preconditioning
###


@wp.func
def _load_body_block(blocks: wp.array[wp.float32], bid: wp.int32) -> mat66f:
    """Loads the 6x6 block of a body stored in row-major order in a flat array."""
    B = mat66f(0.0)
    base = 36 * bid
    for i in range(6):
        for j in range(6):
            B[i, j] = blocks[base + 6 * i + j]
    return B


@wp.func
def _store_body_block(blocks: wp.array[wp.float32], bid: wp.int32, B: mat66f):
    """Stores the 6x6 block of a body in row-major order in a flat array."""
    base = 36 * bid
    for i in range(6):
        for j in range(6):
            blocks[base + 6 * i + j] = B[i, j]


@wp.func
def _cholesky_6x6(A: mat66f) -> mat66f:
    """Computes the lower-triangular Cholesky factor ``L`` of a symmetric positive definite 6x6 matrix."""
    L = mat66f(0.0)
    for j in range(6):
        s = A[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        d = wp.sqrt(wp.max(s, FLOAT32_EPS * A[j, j]))
        L[j, j] = d
        for i in range(j + 1, 6):
            t = A[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t / d
    return L


@wp.func
def _solve_lower_6x6(L: mat66f, b: vec6f) -> vec6f:
    """Solves ``L x = b`` by forward substitution, i.e. computes ``x = L^{-1} b``."""
    x = vec6f(0.0)
    for i in range(6):
        s = b[i]
        for k in range(i):
            s -= L[i, k] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def _solve_upper_6x6(L: mat66f, b: vec6f) -> vec6f:
    """Solves ``L^T x = b`` by backward substitution, i.e. computes ``x = L^{-T} b``."""
    x = vec6f(0.0)
    for r in range(6):
        i = 5 - r
        s = b[i]
        for k in range(i + 1, 6):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


@wp.func
def _mul_transpose_6x6(L: mat66f, b: vec6f) -> vec6f:
    """Computes ``L^T b``."""
    x = vec6f(0.0)
    for i in range(6):
        s = wp.float32(0.0)
        for k in range(6):
            s += L[k, i] * b[k]
        x[i] = s
    return x


@wp.func
def _solve_lower_6x6_mat(L: mat66f, A: mat66f) -> mat66f:
    """Computes ``L^{-1} A`` by column-wise forward substitution."""
    X = mat66f(0.0)
    for c in range(6):
        for i in range(6):
            s = A[i, c]
            for k in range(i):
                s -= L[i, k] * X[k, c]
            X[i, c] = s / L[i, i]
    return X


@wp.func
def _solve_lower_6x6_mat_transposed(L: mat66f, A: mat66f) -> mat66f:
    """Computes ``L^{-1} A^T`` by column-wise forward substitution."""
    X = mat66f(0.0)
    for c in range(6):
        for i in range(6):
            s = A[c, i]
            for k in range(i):
                s -= L[i, k] * X[k, c]
            X[i, c] = s / L[i, i]
    return X


@wp.kernel
def _extract_dense_diagonal(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    operator_mio: wp.array[wp.int32],
    operator_M: wp.array[wp.float32],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the world index and the local generalized coordinate index as the thread indices
    wid, a = wp.tid()

    # Skip if the coordinate exceeds the body-space dimension of the world
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    if a >= nbd:
        return

    # Store the diagonal element of the assembled matrix
    diag[6 * bio + a] = operator_M[operator_mio[wid] + nbd * a + a]


@wp.kernel
def _scale_dense_matrix_diagonal(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    operator_mio: wp.array[wp.int32],
    scaling: wp.array[wp.float32],
    # Outputs:
    operator_M: wp.array[wp.float32],
):
    # Retrieve the world index and the flat matrix element index as the thread indices
    wid, tid = wp.tid()

    # Compute the row and column indices of the matrix element
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    a = tid // nbd
    b = tid % nbd

    # Skip if the element exceeds the matrix size
    if a >= nbd:
        return

    # Apply the symmetric diagonal scaling in-place: A_ab <- s_a A_ab s_b
    m_ab = operator_mio[wid] + nbd * a + b
    operator_M[m_ab] = scaling[6 * bio + a] * operator_M[m_ab] * scaling[6 * bio + b]


@wp.kernel
def _extract_dense_body_blocks(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    operator_mio: wp.array[wp.int32],
    operator_M: wp.array[wp.float32],
    # Outputs:
    blocks: wp.array[wp.float32],
):
    # Retrieve the world index and the flat index of a body-block element as the thread indices
    wid, tid = wp.tid()

    # Compute the local body index and the element indices within its block
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    k = tid // 36
    if k >= nb:
        return
    i = (tid % 36) // 6
    j = tid % 6

    # Store the element of the 6x6 diagonal block of the body
    nbd = 6 * nb
    blocks[36 * (bio + k) + 6 * i + j] = operator_M[operator_mio[wid] + nbd * (6 * k + i) + 6 * k + j]


@wp.kernel
def _scale_dense_matrix_blocks(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    operator_mio: wp.array[wp.int32],
    factors: wp.array[wp.float32],
    # Outputs:
    operator_M: wp.array[wp.float32],
):
    # Retrieve the world index and the flat index of a pair of body blocks as the thread indices
    wid, tid = wp.tid()

    # Compute the local indices of the pair of bodies
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    k = tid // nb
    l = tid % nb
    if k >= nb:
        return

    # Load the 6x6 block of the matrix coupling the two bodies
    nbd = 6 * nb
    m_kl = operator_mio[wid] + nbd * 6 * k + 6 * l
    A = mat66f(0.0)
    for i in range(6):
        for j in range(6):
            A[i, j] = operator_M[m_kl + nbd * i + j]

    # Apply the symmetric block scaling in-place: A_kl <- L_k^{-1} A_kl L_l^{-T}
    X = _solve_lower_6x6_mat(_load_body_block(factors, bio + k), A)
    Y = _solve_lower_6x6_mat_transposed(_load_body_block(factors, bio + l), X)
    for i in range(6):
        for j in range(6):
            operator_M[m_kl + nbd * i + j] = Y[j, i]


@wp.kernel
def _compute_mass_body_blocks(
    # Inputs:
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    # Outputs:
    blocks: wp.array[wp.float32],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Store the block of the mass matrix of the body
    B = mat66f(0.0)
    m_i = model_bodies_m_i[bid]
    I_i = data_bodies_I_i[bid]
    for d in range(3):
        B[d, d] = m_i
        for e in range(3):
            B[3 + d, 3 + e] = I_i[d, e]
    _store_body_block(blocks, bid, B)


@wp.kernel
def _accumulate_stiffness_body_blocks_sparse(
    # Inputs:
    jac_num_nzb: wp.array[wp.int32],
    jac_nzb_start: wp.array[wp.int32],
    jac_nzb_coords: wp.array2d[wp.int32],
    jac_nzb_values: wp.array[vec6f],
    jac_row_start: wp.array[wp.int32],
    jac_col_start: wp.array[wp.int32],
    W: wp.array[wp.float32],
    # Outputs:
    blocks: wp.array[wp.float32],
):
    # Retrieve the world index and the local non-zero block index as the thread indices
    wid, nzb_id = wp.tid()

    # Skip if the block index exceeds the number of non-zero blocks of the world
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve the Jacobian block and its coordinates
    nzb = jac_nzb_start[wid] + nzb_id
    coords = jac_nzb_coords[nzb]
    J_block = jac_nzb_values[nzb]

    # Accumulate the constraint stiffness contribution to the body block: W_i J_ik^T J_ik
    w_i = W[jac_row_start[wid] + coords[0]]
    base = 36 * ((jac_col_start[wid] + coords[1]) // 6)
    for i in range(6):
        for j in range(6):
            wp.atomic_add(blocks, base + 6 * i + j, w_i * J_block[i] * J_block[j])


@wp.kernel
def _factorize_body_blocks(
    # Outputs:
    blocks: wp.array[wp.float32],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Replace the 6x6 block of the body by its lower-triangular Cholesky factor
    _store_body_block(blocks, bid, _cholesky_6x6(_load_body_block(blocks, bid)))


@wp.kernel
def _scale_body_vector_blocks(
    # Inputs:
    model_bodies_wid: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    factors: wp.array[wp.float32],
    mode: wp.int32,
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

    # Load the twist of the body
    k = 6 * bid
    x_k = vec6f(0.0)
    for d in range(6):
        x_k[d] = x[k + d]

    # Apply the block scaling ``S = L^{-T}`` (mode 0), its transpose ``S^T = L^{-1}``
    # (mode 1), or its inverse ``S^{-1} = L^T`` (mode 2), with ``L`` the body-block factor
    L = _load_body_block(factors, bid)
    s_k = vec6f(0.0)
    if mode == 0:
        s_k = _solve_upper_6x6(L, x_k)
    elif mode == 1:
        s_k = _solve_lower_6x6(L, x_k)
    else:
        s_k = _mul_transpose_6x6(L, x_k)
    for d in range(6):
        y[k + d] = alpha * s_k[d] + beta * y[k + d]


###
# Interfaces
###


def _check_preconditioner(preconditioner: AugmentedMassPreconditioner):
    """Raises if the given augmented mass matrix preconditioner is not supported."""
    supported = ("none", "jacobi", "block_jacobi")
    if preconditioner not in supported:
        raise ValueError(f"Invalid augmented mass preconditioner: {preconditioner}. Must be one of {supported}.")


def _allocate_scaling(model: ModelKamino, preconditioner: AugmentedMassPreconditioner) -> wp.array | None:
    """Allocates the data of the symmetric scaling of the given preconditioner."""
    if preconditioner == "jacobi":
        return wp.ones(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32, device=model.device)
    if preconditioner == "block_jacobi":
        return wp.zeros(shape=(36 * model.size.sum_of_num_bodies,), dtype=wp.float32, device=model.device)
    return None


def _apply_scaling(
    model: ModelKamino,
    preconditioner: AugmentedMassPreconditioner,
    scaling: wp.array[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
    inverse: bool = False,
):
    """
    Computes ``y = alpha * op(S) @ x + beta * y`` for the symmetric scaling ``S`` of a preconditioner,
    with ``op(S) = S``, ``S^T`` (if `transpose`) or ``S^{-1}`` (if `inverse`). Operates in-place if
    ``x`` and ``y`` are the same array.
    """
    if preconditioner == "jacobi":
        wp.launch(
            _scale_body_vector,
            dim=model.size.sum_of_num_body_dofs,
            inputs=[
                model.bodies.wid,
                world_mask,
                scaling,
                wp.bool(inverse),
                wp.float32(alpha),
                wp.float32(beta),
                x,
                y,
            ],
            device=model.device,
        )
    elif preconditioner == "block_jacobi":
        mode = 2 if inverse else (1 if transpose else 0)
        wp.launch(
            _scale_body_vector_blocks,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                model.bodies.wid,
                world_mask,
                scaling,
                wp.int32(mode),
                wp.float32(alpha),
                wp.float32(beta),
                x,
                y,
            ],
            device=model.device,
        )
    else:
        raise ValueError(f"The augmented mass preconditioner `{preconditioner}` has no scaling.")


def _make_body_space_info(model: ModelKamino) -> DenseSquareMultiLinearInfo[wp.float32, wp.int32]:
    """Creates the multi-linear info describing the (fixed) body-space dimensions of each world."""
    info = DenseSquareMultiLinearInfo[wp.float32, wp.int32]()
    info.finalize(
        dimensions=[int(n) for n in model.info.num_body_dofs.numpy().tolist()],
        dtype=wp.float32,
        itype=wp.int32,
        device=model.device,
    )
    return info


class AugmentedMassOperator:
    """
    A dense representation of the constraint-augmented generalized mass matrix ``M + J^T W J``.

    The matrix of each world is assembled explicitly from the (dense or sparse) constraint
    Jacobian and factorized with a direct linear solver, e.g. :class:`LLTBlockedSolver`.

    Optionally, the matrix is symmetrically preconditioned before the factorization, i.e. the
    scaled matrix ``S^T (M + J^T W J) S`` is factorized and linear systems are solved as
    ``x = S (S^T (M + J^T W J) S)^{-1} S^T b`` (see :data:`AugmentedMassPreconditioner`).
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        preconditioner: AugmentedMassPreconditioner = "none",
    ):
        """
        Creates a dense constraint-augmented generalized mass matrix operator.

        Args:
            model: The model for which the operator is allocated.
            solver: The direct linear solver type used to factorize the matrix.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
            preconditioner: The symmetric preconditioner applied to the matrix before its factorization.
        """
        self._device: wp.DeviceLike = None
        self._model: ModelKamino | None = None
        self._num_worlds: int = 0
        self._max_dim: int = 0
        self._operator: DenseLinearOperatorData[wp.float32, wp.int32] | None = None
        self._solver: DirectSolver | None = None
        self._preconditioner: AugmentedMassPreconditioner = "none"
        self._scaling: wp.array[wp.float32] | None = None
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._u_tmp: wp.array[wp.float32] | None = None
        if model is not None:
            self.finalize(model=model, solver=solver, solver_kwargs=solver_kwargs, preconditioner=preconditioner)

    @property
    def info(self) -> DenseSquareMultiLinearInfo[wp.float32, wp.int32]:
        """Returns the multi-linear info describing the matrix blocks of each world."""
        return self._operator.info

    @property
    def matrix(self) -> wp.array[wp.float32]:
        """Returns the flat array of matrix blocks."""
        return self._operator.mat

    @property
    def solver(self) -> DirectSolver | None:
        """Returns the direct linear solver used to factorize the matrix."""
        return self._solver

    @property
    def preconditioner(self) -> AugmentedMassPreconditioner:
        """Returns the symmetric preconditioner applied to the matrix before its factorization."""
        return self._preconditioner

    @property
    def scaling(self) -> wp.array[wp.float32] | None:
        """
        Returns the data of the symmetric scaling ``S`` of the preconditioner, if any, i.e. the
        diagonal of ``S`` for ``"jacobi"``, or the row-major 6x6 lower-triangular Cholesky factors
        ``L_k`` of the body blocks, with ``S_k = L_k^{-T}``, for ``"block_jacobi"``.
        """
        return self._scaling

    def finalize(
        self,
        model: ModelKamino,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        preconditioner: AugmentedMassPreconditioner = "none",
    ):
        """
        Allocates the matrix blocks and the linear solver.

        Args:
            model: The model for which the operator is allocated.
            solver: The direct linear solver type used to factorize the matrix.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
            preconditioner: The symmetric preconditioner applied to the matrix before its factorization.
        """
        if solver is None or not issubclass(solver, DirectSolver):
            raise ValueError("The dense augmented mass operator requires a direct linear solver.")
        _check_preconditioner(preconditioner)
        self._device = model.device
        self._model = model
        self._num_worlds = model.size.num_worlds
        self._max_dim = model.size.max_of_num_body_dofs
        self._preconditioner = preconditioner
        info = _make_body_space_info(model)
        self._operator = DenseLinearOperatorData[wp.float32, wp.int32](
            info=info, mat=wp.zeros(shape=(info.total_mat_size,), dtype=wp.float32, device=self._device)
        )
        self._solver = solver(operator=self._operator, device=self._device, **(solver_kwargs or {}))
        self._scaling = _allocate_scaling(model, preconditioner)
        if preconditioner != "none":
            with wp.ScopedDevice(self._device):
                self._all_worlds_mask = wp.ones(shape=(model.size.num_worlds,), dtype=wp.bool)
                self._u_tmp = wp.zeros(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32)

    def compute(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: DenseSystemJacobians | SparseSystemJacobians,
        dim: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        W: wp.array[wp.float32],
    ):
        """
        Assembles the matrix ``M + J^T W J`` of each world and factorizes it.

        Args:
            model: The model providing the body masses and body-space layout.
            data: The model data providing the current body inertias.
            jacobians: The current constraint Jacobians of the system.
            dim: The active number of constraints of each world.
            vio: The constraint-space vector index offset of each world.
            W: The per-constraint effective stiffness. Shape of ``(sum_of_max_total_cts,)``.
        """
        wp.launch(
            _assemble_mass_dense,
            dim=(self._num_worlds, self._max_dim * self._max_dim),
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                model.bodies.m_i,
                data.bodies.I_i,
                self._operator.info.mio,
                # Outputs:
                self._operator.mat,
            ],
            device=self._device,
        )
        if isinstance(jacobians, SparseSystemJacobians):
            J_cts = jacobians._J_cts.bsm
            wp.launch(
                _assemble_augmented_mass_stiffness_sparse,
                dim=(self._num_worlds, J_cts.max_of_num_nzb * J_cts.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    J_cts.num_nzb,
                    J_cts.nzb_start,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    J_cts.row_start,
                    W,
                    self._operator.info.mio,
                    # Outputs:
                    self._operator.mat,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                _assemble_augmented_mass_stiffness_dense,
                dim=(self._num_worlds, self._max_dim * self._max_dim),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    jacobians.data.J_cts_offsets,
                    jacobians.data.J_cts_data,
                    dim,
                    vio,
                    W,
                    self._operator.info.mio,
                    # Outputs:
                    self._operator.mat,
                ],
                device=self._device,
            )
        self._apply_preconditioner()
        self._solver.reset()
        self._solver.compute(A=self._operator.mat)

    def solve(self, b: wp.array[wp.float32], x: wp.array[wp.float32]):
        """
        Solves the linear system ``(M + J^T W J) x = b`` of each world using the factorization.

        Args:
            b: The flat array of generalized right-hand-side vectors.
            x: The flat array to hold the generalized solution vectors.
        """
        if self._preconditioner == "none":
            self._solver.solve(b=b, x=x)
            return

        # Solve the scaled system: x = S (S^T A S)^{-1} S^T b
        mask = self._all_worlds_mask
        _apply_scaling(self._model, self._preconditioner, self._scaling, b, self._u_tmp, mask, transpose=True)
        self._solver.solve(b=self._u_tmp, x=x)
        _apply_scaling(self._model, self._preconditioner, self._scaling, x, x, mask)

    ###
    # Internals
    ###

    def _apply_preconditioner(self):
        """Computes the symmetric scaling ``S`` from the assembled matrix and scales the matrix in-place."""
        if self._preconditioner == "none":
            return
        model = self._model
        info = self._operator.info
        if self._preconditioner == "jacobi":
            wp.launch(
                _extract_dense_diagonal,
                dim=(self._num_worlds, self._max_dim),
                inputs=[model.info.bodies_offset, info.mio, self._operator.mat, self._scaling],
                device=self._device,
            )
            wp.launch(
                _compute_jacobi_scaling,
                dim=model.size.sum_of_num_body_dofs,
                inputs=[self._scaling, self._scaling],
                device=self._device,
            )
            wp.launch(
                _scale_dense_matrix_diagonal,
                dim=(self._num_worlds, self._max_dim * self._max_dim),
                inputs=[model.info.bodies_offset, info.mio, self._scaling, self._operator.mat],
                device=self._device,
            )
        else:
            max_num_bodies = model.size.max_of_num_bodies
            wp.launch(
                _extract_dense_body_blocks,
                dim=(self._num_worlds, 36 * max_num_bodies),
                inputs=[model.info.bodies_offset, info.mio, self._operator.mat, self._scaling],
                device=self._device,
            )
            wp.launch(
                _factorize_body_blocks,
                dim=model.size.sum_of_num_bodies,
                inputs=[self._scaling],
                device=self._device,
            )
            wp.launch(
                _scale_dense_matrix_blocks,
                dim=(self._num_worlds, max_num_bodies * max_num_bodies),
                inputs=[model.info.bodies_offset, info.mio, self._scaling, self._operator.mat],
                device=self._device,
            )


class BlockSparseMatrixFreeAugmentedMassOperator:
    """
    A matrix-free representation of the constraint-augmented generalized mass matrix ``M + J^T W J``.

    Products are realized with the block-sparse constraint Jacobian, and linear systems are solved
    with an iterative solver, e.g. :class:`ConjugateResidualSolver`, applied to the symmetrically
    scaled system ``S^T (M + J^T W J) S z = S^T b`` with ``x = S z``, where ``S`` is the symmetric
    scaling of the preconditioner (see :data:`AugmentedMassPreconditioner`).
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        preconditioner: AugmentedMassPreconditioner = "jacobi",
    ):
        """
        Creates a matrix-free constraint-augmented generalized mass matrix operator.

        Args:
            model: The model for which the operator is allocated.
            data: The model data providing the current body inertias.
            jacobians: The sparse constraint Jacobians of the system.
            solver: The iterative linear solver type used to solve linear systems.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
            preconditioner: The symmetric preconditioner applied to the linear systems.
        """
        self._model: ModelKamino | None = None
        self._data: DataKamino | None = None
        self._jacobians: SparseSystemJacobians | None = None
        self._dim: wp.array[wp.int32] | None = None
        self._vio: wp.array[wp.int32] | None = None
        self._W: wp.array[wp.float32] | None = None
        self._info: DenseSquareMultiLinearInfo[wp.float32, wp.int32] | None = None
        self._operator: BatchedLinearOperator[wp.float32, wp.int32] | None = None
        self._solver: IterativeSolver | None = None
        self._preconditioner: AugmentedMassPreconditioner = "jacobi"
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._scaling: wp.array[wp.float32] | None = None
        self._u_rhs: wp.array[wp.float32] | None = None
        self._u_tmp: wp.array[wp.float32] | None = None
        self._u_prod: wp.array[wp.float32] | None = None
        self._c_tmp: wp.array[wp.float32] | None = None
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                jacobians=jacobians,
                solver=solver,
                solver_kwargs=solver_kwargs,
                preconditioner=preconditioner,
            )

    @property
    def solver(self) -> IterativeSolver | None:
        """Returns the iterative linear solver used to solve linear systems."""
        return self._solver

    @property
    def preconditioner(self) -> AugmentedMassPreconditioner:
        """Returns the symmetric preconditioner applied to the linear systems."""
        return self._preconditioner

    @property
    def scaling(self) -> wp.array[wp.float32] | None:
        """
        Returns the data of the symmetric scaling ``S`` of the preconditioner, if any, i.e. the
        diagonal of ``S`` for ``"jacobi"``, or the row-major 6x6 lower-triangular Cholesky factors
        ``L_k`` of the body blocks, with ``S_k = L_k^{-T}``, for ``"block_jacobi"``.
        """
        return self._scaling

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        preconditioner: AugmentedMassPreconditioner = "jacobi",
    ):
        """
        Allocates the temporaries and the iterative linear solver of the operator.

        Args:
            model: The model for which the operator is allocated.
            data: The model data providing the current body inertias.
            jacobians: The sparse constraint Jacobians of the system.
            solver: The iterative linear solver type used to solve linear systems.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
            preconditioner: The symmetric preconditioner applied to the linear systems.
        """
        if solver is None or not issubclass(solver, IterativeSolver):
            raise ValueError("The matrix-free augmented mass operator requires an iterative linear solver.")
        if getattr(solver, "uses_raw_jacobian", False):
            raise ValueError(f"The linear solver `{solver.__name__}` only supports the Delassus operator.")
        _check_preconditioner(preconditioner)
        self._model = model
        self._data = data
        self._jacobians = jacobians
        self._info = _make_body_space_info(model)
        self._preconditioner = preconditioner

        device = model.device
        num_body_dofs = model.size.sum_of_num_body_dofs
        with wp.ScopedDevice(device):
            self._all_worlds_mask = wp.ones(shape=(model.size.num_worlds,), dtype=wp.bool)
            self._u_rhs = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._u_tmp = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._u_prod = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._c_tmp = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)
        self._scaling = _allocate_scaling(model, preconditioner)

        self._operator = BatchedLinearOperator[wp.float32, wp.int32](
            gemv_fn=self._gemv_scaled if preconditioner != "none" else self._gemv_unscaled,
            n_worlds=model.size.num_worlds,
            max_dim=self._info.max_dimension,
            active_dims=self._info.dim,
            device=device,
            dtype=wp.float32,
            vio=self._info.vio,
            total_vec_size=num_body_dofs,
        )
        # NOTE: In float32, Krylov solvers typically require more iterations than the
        # dimension of the system to converge, due to the loss of orthogonality
        solver_kwargs = dict(solver_kwargs or {})
        solver_kwargs.setdefault("maxiter", 4 * self._info.max_dimension)
        self._solver = solver(device=device, **solver_kwargs)
        self._solver.finalize(self._operator)

    def compute(
        self,
        dim: wp.array[wp.int32],
        vio: wp.array[wp.int32],
        W: wp.array[wp.float32],
    ):
        """
        Captures the effective stiffness of the operator and updates the scaling of the preconditioner.

        Args:
            dim: The active number of constraints of each world.
            vio: The constraint-space vector index offset of each world.
            W: The per-constraint effective stiffness. Shape of ``(sum_of_max_total_cts,)``.
        """
        self._dim = dim
        self._vio = vio
        self._W = W
        if self._preconditioner == "none":
            return

        model = self._model
        J_cts = self._jacobians._J_cts.bsm
        jac_inputs = [J_cts.num_nzb, J_cts.nzb_start, J_cts.nzb_coords, J_cts.nzb_values, J_cts.row_start]
        if self._preconditioner == "jacobi":
            # Compute the diagonal of `M + J^T W J` into the scaling array and invert its square-root in-place
            wp.launch(
                _compute_augmented_mass_diagonal_mass,
                dim=model.size.sum_of_num_bodies,
                inputs=[model.bodies.m_i, self._data.bodies.I_i, self._scaling],
                device=model.device,
            )
            wp.launch(
                _compute_augmented_mass_diagonal_stiffness_sparse,
                dim=(model.size.num_worlds, J_cts.max_of_num_nzb),
                inputs=[*jac_inputs, J_cts.col_start, W, self._scaling],
                device=model.device,
            )
            wp.launch(
                _compute_jacobi_scaling,
                dim=model.size.sum_of_num_body_dofs,
                inputs=[self._scaling, self._scaling],
                device=model.device,
            )
        else:
            # Compute the 6x6 body blocks of `M + J^T W J` into the scaling array and factorize them in-place
            wp.launch(
                _compute_mass_body_blocks,
                dim=model.size.sum_of_num_bodies,
                inputs=[model.bodies.m_i, self._data.bodies.I_i, self._scaling],
                device=model.device,
            )
            wp.launch(
                _accumulate_stiffness_body_blocks_sparse,
                dim=(model.size.num_worlds, J_cts.max_of_num_nzb),
                inputs=[*jac_inputs, J_cts.col_start, W, self._scaling],
                device=model.device,
            )
            wp.launch(
                _factorize_body_blocks,
                dim=model.size.sum_of_num_bodies,
                inputs=[self._scaling],
                device=model.device,
            )

    def solve(self, b: wp.array[wp.float32], x: wp.array[wp.float32]):
        """
        Solves the linear system ``(M + J^T W J) x = b`` of each world iteratively.

        Args:
            b: The flat array of generalized right-hand-side vectors.
            x: The flat array holding the initial guess on input and the generalized solution on output.
        """
        if self._preconditioner == "none":
            self._solver.solve(b=b, x=x)
            return

        # Solve the scaled system `S^T A S z = S^T b`, warm-started from `z = S^{-1} x`, and recover `x = S z`
        mask = self._all_worlds_mask
        self._scale(b, self._u_rhs, mask, transpose=True)
        self._scale(x, x, mask, inverse=True)
        self._solver.solve(b=self._u_rhs, x=x)
        self._scale(x, x, mask)

    def gemv(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        world_mask: wp.array[wp.bool],
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        """
        Computes the generalized matrix-vector product ``y = alpha * (M + J^T W J) @ x + beta * y``.

        Args:
            x: The input array of generalized vectors.
            y: The input-output array of generalized vectors.
            world_mask: The mask selecting the worlds to operate on.
            alpha: The scaling of the matrix-vector product.
            beta: The scaling of the input values of `y`.
        """
        jacobian_gemv(self._model, self._jacobians, self._dim, self._vio, x, self._c_tmp, world_mask)
        wp.launch(
            _scale_constraint_vector,
            dim=(self._model.size.num_worlds, self._model.size.max_of_max_total_cts),
            inputs=[self._dim, self._vio, world_mask, self._W, self._c_tmp],
            device=self._model.device,
        )
        mass_gemv(self._model, self._data, x, y, world_mask, alpha=alpha, beta=beta)
        jacobian_transpose_gemv(
            self._model, self._jacobians, self._dim, self._vio, self._c_tmp, y, world_mask, alpha=alpha, beta=1.0
        )

    ###
    # Internals
    ###

    def _scale(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        world_mask: wp.array[wp.bool],
        alpha: float = 1.0,
        beta: float = 0.0,
        transpose: bool = False,
        inverse: bool = False,
    ):
        """Computes ``y = alpha * op(S) @ x + beta * y`` with ``op(S) = S``, ``S^T`` or ``S^{-1}``."""
        _apply_scaling(
            self._model,
            self._preconditioner,
            self._scaling,
            x,
            y,
            world_mask,
            alpha=alpha,
            beta=beta,
            transpose=transpose,
            inverse=inverse,
        )

    def _gemv_unscaled(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        alpha: float,
        beta: float,
    ):
        """Computes ``y = alpha * (M + J^T W J) @ x + beta * y`` for the iterative solver."""
        self.gemv(x, y, world_active, alpha=alpha, beta=beta)

    def _gemv_scaled(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        alpha: float,
        beta: float,
    ):
        """Computes ``y = alpha * S^T (M + J^T W J) S @ x + beta * y`` for the iterative solver."""
        self._scale(x, self._u_tmp, world_active)
        self.gemv(self._u_tmp, self._u_prod, world_active)
        self._scale(self._u_prod, y, world_active, alpha=alpha, beta=beta, transpose=True)
