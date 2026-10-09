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
``M_W`` and solves linear systems with an iterative solver, applying a symmetric Jacobi scaling.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..core.types import vec6f
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
    "BlockSparseMatrixFreeAugmentedMassOperator",
]


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
# Interfaces
###


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
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Creates a dense constraint-augmented generalized mass matrix operator.

        Args:
            model: The model for which the operator is allocated.
            solver: The direct linear solver type used to factorize the matrix.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
        """
        self._device: wp.DeviceLike = None
        self._num_worlds: int = 0
        self._max_dim: int = 0
        self._operator: DenseLinearOperatorData[wp.float32, wp.int32] | None = None
        self._solver: DirectSolver | None = None
        if model is not None:
            self.finalize(model=model, solver=solver, solver_kwargs=solver_kwargs)

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

    def finalize(
        self,
        model: ModelKamino,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the matrix blocks and the linear solver.

        Args:
            model: The model for which the operator is allocated.
            solver: The direct linear solver type used to factorize the matrix.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
        """
        if solver is None or not issubclass(solver, DirectSolver):
            raise ValueError("The dense augmented mass operator requires a direct linear solver.")
        self._device = model.device
        self._num_worlds = model.size.num_worlds
        self._max_dim = model.size.max_of_num_body_dofs
        info = _make_body_space_info(model)
        self._operator = DenseLinearOperatorData[wp.float32, wp.int32](
            info=info, mat=wp.zeros(shape=(info.total_mat_size,), dtype=wp.float32, device=self._device)
        )
        self._solver = solver(operator=self._operator, device=self._device, **(solver_kwargs or {}))

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
        self._solver.reset()
        self._solver.compute(A=self._operator.mat)

    def solve(self, b: wp.array[wp.float32], x: wp.array[wp.float32]):
        """
        Solves the linear system ``(M + J^T W J) x = b`` of each world using the factorization.

        Args:
            b: The flat array of generalized right-hand-side vectors.
            x: The flat array to hold the generalized solution vectors.
        """
        self._solver.solve(b=b, x=x)


class BlockSparseMatrixFreeAugmentedMassOperator:
    """
    A matrix-free representation of the constraint-augmented generalized mass matrix ``M + J^T W J``.

    Products are realized with the block-sparse constraint Jacobian, and linear systems are solved
    with an iterative solver, e.g. :class:`ConjugateResidualSolver`, applied to the symmetrically
    scaled system ``S (M + J^T W J) S z = S b`` with ``x = S z`` and the Jacobi scaling
    ``S = diag(M + J^T W J)^{-1/2}``.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Creates a matrix-free constraint-augmented generalized mass matrix operator.

        Args:
            model: The model for which the operator is allocated.
            data: The model data providing the current body inertias.
            jacobians: The sparse constraint Jacobians of the system.
            solver: The iterative linear solver type used to solve linear systems.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
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
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._scaling: wp.array[wp.float32] | None = None
        self._u_rhs: wp.array[wp.float32] | None = None
        self._u_tmp: wp.array[wp.float32] | None = None
        self._u_prod: wp.array[wp.float32] | None = None
        self._c_tmp: wp.array[wp.float32] | None = None
        if model is not None:
            self.finalize(model=model, data=data, jacobians=jacobians, solver=solver, solver_kwargs=solver_kwargs)

    @property
    def solver(self) -> IterativeSolver | None:
        """Returns the iterative linear solver used to solve linear systems."""
        return self._solver

    @property
    def scaling(self) -> wp.array[wp.float32]:
        """Returns the symmetric Jacobi scaling ``S`` applied to the linear system."""
        return self._scaling

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SparseSystemJacobians,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        """
        Allocates the temporaries and the iterative linear solver of the operator.

        Args:
            model: The model for which the operator is allocated.
            data: The model data providing the current body inertias.
            jacobians: The sparse constraint Jacobians of the system.
            solver: The iterative linear solver type used to solve linear systems.
            solver_kwargs: Additional keyword arguments to pass to the solver constructor.
        """
        if solver is None or not issubclass(solver, IterativeSolver):
            raise ValueError("The matrix-free augmented mass operator requires an iterative linear solver.")
        if getattr(solver, "uses_raw_jacobian", False):
            raise ValueError(f"The linear solver `{solver.__name__}` only supports the Delassus operator.")
        self._model = model
        self._data = data
        self._jacobians = jacobians
        self._info = _make_body_space_info(model)

        device = model.device
        num_body_dofs = model.size.sum_of_num_body_dofs
        with wp.ScopedDevice(device):
            self._all_worlds_mask = wp.ones(shape=(model.size.num_worlds,), dtype=wp.bool)
            self._scaling = wp.ones(shape=(num_body_dofs,), dtype=wp.float32)
            self._u_rhs = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._u_tmp = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._u_prod = wp.zeros(shape=(num_body_dofs,), dtype=wp.float32)
            self._c_tmp = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)

        self._operator = BatchedLinearOperator[wp.float32, wp.int32](
            gemv_fn=self._gemv_scaled,
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
        Captures the effective stiffness of the operator and updates the Jacobi scaling.

        Args:
            dim: The active number of constraints of each world.
            vio: The constraint-space vector index offset of each world.
            W: The per-constraint effective stiffness. Shape of ``(sum_of_max_total_cts,)``.
        """
        self._dim = dim
        self._vio = vio
        self._W = W

        # Compute the diagonal of `M + J^T W J` into the scaling array and invert its square-root in-place
        model = self._model
        J_cts = self._jacobians._J_cts.bsm
        wp.launch(
            _compute_augmented_mass_diagonal_mass,
            dim=model.size.sum_of_num_bodies,
            inputs=[model.bodies.m_i, self._data.bodies.I_i, self._scaling],
            device=model.device,
        )
        wp.launch(
            _compute_augmented_mass_diagonal_stiffness_sparse,
            dim=(model.size.num_worlds, J_cts.max_of_num_nzb),
            inputs=[
                J_cts.num_nzb,
                J_cts.nzb_start,
                J_cts.nzb_coords,
                J_cts.nzb_values,
                J_cts.row_start,
                J_cts.col_start,
                W,
                self._scaling,
            ],
            device=model.device,
        )
        wp.launch(
            _compute_jacobi_scaling,
            dim=model.size.sum_of_num_body_dofs,
            inputs=[self._scaling, self._scaling],
            device=model.device,
        )

    def solve(self, b: wp.array[wp.float32], x: wp.array[wp.float32]):
        """
        Solves the linear system ``(M + J^T W J) x = b`` of each world iteratively.

        Args:
            b: The flat array of generalized right-hand-side vectors.
            x: The flat array holding the initial guess on input and the generalized solution on output.
        """
        mask = self._all_worlds_mask
        self._scale(b, self._u_rhs, mask)
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
        inverse: bool = False,
    ):
        """Computes ``y = alpha * S @ x + beta * y`` (or with ``S^{-1}`` if `inverse`)."""
        wp.launch(
            _scale_body_vector,
            dim=self._model.size.sum_of_num_body_dofs,
            inputs=[
                self._model.bodies.wid,
                world_mask,
                self._scaling,
                wp.bool(inverse),
                wp.float32(alpha),
                wp.float32(beta),
                x,
                y,
            ],
            device=self._model.device,
        )

    def _gemv_scaled(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        world_active: wp.array[wp.bool],
        alpha: float,
        beta: float,
    ):
        """Computes ``y = alpha * S (M + J^T W J) S @ x + beta * y`` for the iterative solver."""
        self._scale(x, self._u_tmp, world_active)
        self.gemv(self._u_tmp, self._u_prod, world_active)
        self._scale(self._u_prod, y, world_active, alpha=alpha, beta=beta)
