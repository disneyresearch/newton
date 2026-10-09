# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a data container and relevant operations to
represent and construct a dual forward dynamics problem.

The dual forward dynamics problem arises from the formulation of
the equations of motion in terms of constraint reactions.

`lambdas = argmin_{x} 1/2 * x^T D x + lambda^T (v_f + Gamma(v_plus(x)))`


This module thus provides the :class:`DualProblem` formulation of the
:class:`ConstrainedDynamicsProblem`, which realizes the linear operator of the problem
using a :class:`DelassusOperator` (dense) or :class:`BlockSparseMatrixFreeDelassusOperator`
(sparse). These provide methods to allocate the necessary data arrays, build the
Delassus matrix given the current state of the model and the active constraints,
add diagonal regularization, and solve linear systems of the form `D @ x = v` given
arrays holding the right-hand-side (rhs) vectors v.

Typical usage example:
    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = ModelKamino.from_newton(builder.finalize())
    data = model.data()
    limits = LimitsKamino(model)
    contacts = ContactsKamino(model)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Define a linear solver type to use as a back-end for the
    # Delassus operator computations such as factorization and
    # solving the linear system when a rhs vector is provided
    linear_solver = LLTBlockedSolver
    ...

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a dual forward dynamics problem and build it using the current model
    # data and active unilateral constraints (i.e. for limits and contacts).
    dual = DualProblem(model, data, limits, contacts, jacobians, linear_solver, sparse=False)
    dual.build(model, data, jacobians, limits, contacts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..dynamics.delassus import BlockSparseMatrixFreeDelassusOperator, DelassusOperator
from ..geometry.contacts import ContactsKamino
from ..kinematics.limits import LimitsKamino
from ..linalg import LinearSolverType
from .config import ProblemFormulation
from .problem import ConstrainedDynamicsProblem, ConstrainedDynamicsProblemData, SystemJacobiansType

###
# Module interface
###

__all__ = [
    "DualProblem",
    "DualProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Types
###


@dataclass
class DualProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the the dual forward dynamics problem data over multiple worlds.
    """

    mio: wp.array[wp.int32] | None = None
    """
    The matrix index offset of each Delassus matrix block.
    This is applicable to `D` as well as to its (optional) factorizations.
    Only allocated for dense problems.
    Shape of `(num_worlds,)`.
    """

    D: wp.array[wp.float32] | None = None
    """
    The flat array of Delassus matrix blocks (constraint-space apparent inertia).
    Only allocated for dense problems.
    Shape of `(sum_of_max_total_delassus_size,)`.
    """


###
# Kernels
###


@wp.kernel
def _extract_dense_diagonal(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_mio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_D: wp.array[wp.float32],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Skip if row index exceed the problem size
    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    # Store the diagonal element of the Delassus matrix
    diag[problem_vio[wid] + tid] = problem_D[problem_mio[wid] + ncts * tid + tid]


@wp.kernel
def _apply_dual_preconditioner_to_matrix(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_mio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    # Outputs:
    X: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if there no constraints ar active
    if ncts == 0:
        return

    # Compute i (row) and j (col) indices from the tid
    i = tid // ncts
    j = tid % ncts

    # Skip if indices exceed the problem size
    if i >= ncts or j >= ncts:
        return

    # Retrieve the matrix index offset of the world
    mio = problem_mio[wid]

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the matrix entry
    m_ij = mio + ncts * i + j

    # Retrieve the i,j-th entry of the target matrix
    X_ij = X[m_ij]

    # Retrieve the i,j-th entries of the diagonal preconditioner
    P_i = problem_P[vio + i]
    P_j = problem_P[vio + j]

    # Store the preconditioned i,j-th entry of the matrix
    X[m_ij] = P_i * (P_j * X_ij)


@wp.kernel
def _apply_dense_regularization(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_mio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_eta: wp.array[wp.float32],
    # Outputs:
    eta_applied: wp.array[wp.float32],
    problem_D: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Skip if row index exceed the problem size
    ncts = problem_dim[wid]
    if tid >= ncts:
        return

    # Replace the regularization currently applied to the diagonal element
    v_i = problem_vio[wid] + tid
    eta_i = problem_eta[v_i]
    problem_D[problem_mio[wid] + ncts * tid + tid] += eta_i - eta_applied[v_i]
    eta_applied[v_i] = eta_i


@wp.kernel
def _gemv_dense_unregularized(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_mio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_D: wp.array[wp.float32],
    problem_P: wp.array[wp.float32],
    eta_applied: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: wp.float32,
    beta: wp.float32,
    preconditioned: wp.bool,
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    """
    Computes ``y = alpha * A @ x + beta * y`` with ``A = P D P`` or ``A = D`` from the stored
    regularized and preconditioned Delassus matrix ``P D P + diag(eta_applied)``.
    """
    # Retrieve the thread index
    wid, i = wp.tid()

    # Skip if the world is inactive or the row index exceed the problem size
    ncts = problem_dim[wid]
    if not world_mask[wid] or i >= ncts:
        return

    # Retrieve the matrix and vector index offsets of the world
    mio = problem_mio[wid]
    vio = problem_vio[wid]

    # Accumulate the row-vector product with the preconditioned Delassus matrix
    acc = wp.float32(0.0)
    for j in range(ncts):
        x_j = x[vio + j]
        if not preconditioned:
            x_j /= problem_P[vio + j]
        acc += problem_D[mio + ncts * i + j] * x_j

    # Remove the regularization and optionally the preconditioning of the row
    v_i = vio + i
    x_i = x[v_i]
    if not preconditioned:
        x_i /= problem_P[v_i]
    acc -= eta_applied[v_i] * x_i
    if not preconditioned:
        acc /= problem_P[v_i]

    # Store the generalized matrix-vector product
    y[v_i] = alpha * acc + beta * y[v_i]


###
# Interfaces
###


class DualProblem(ConstrainedDynamicsProblem):
    """
    The dual formulation of the constrained forward dynamics problem.

    Realizes the linear operator of the problem in constraint space using the Delassus matrix
    ``D = J M^{-1} J^T``, which is either stored explicitly and factorized (dense) or represented
    as a matrix-free operator solved iteratively (sparse).
    """

    _data_type = DualProblemData

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[DualProblem.Config] | DualProblem.Config | None = None,
        compute_h: bool = False,
        sparse: bool = True,
    ):
        """
        Constructs a dual problem interface container.

        See :class:`ConstrainedDynamicsProblem` for a description of the arguments.
        """
        self._delassus: DelassusOperator | BlockSparseMatrixFreeDelassusOperator | None = None
        """The Delassus operator interface container."""

        self._eta_applied: wp.array[wp.float32] | None = None
        """The regularization currently applied in-place to the diagonal of the dense Delassus matrix."""

        super().__init__(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobians=jacobians,
            solver=solver,
            solver_kwargs=solver_kwargs,
            config=config,
            compute_h=compute_h,
            sparse=sparse,
        )

    ###
    # Properties
    ###

    @property
    def formulation(self) -> ProblemFormulation:
        """
        Returns the formulation of the problem, i.e. :attr:`ProblemFormulation.DUAL`.
        """
        return ProblemFormulation.DUAL

    @property
    def delassus(self) -> DelassusOperator | BlockSparseMatrixFreeDelassusOperator:
        """
        Returns the Delassus operator interface.
        """
        if self._delassus is None:
            raise ValueError("Delassus operator is not allocated. Call `finalize()` first.")
        return self._delassus

    @property
    def linear_solver(self) -> LinearSolverType | None:
        """
        Returns the linear solver of the Delassus operator, if any.
        """
        return self.delassus.solver

    @property
    def data(self) -> DualProblemData:
        """
        Returns the dual problem data container.
        """
        return self._data

    ###
    # Implementation
    ###

    def _finalize_impl(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians: SystemJacobiansType,
        solver: LinearSolverType | None,
        solver_kwargs: dict[str, Any] | None,
    ):
        if self._sparse:
            self._delassus = BlockSparseMatrixFreeDelassusOperator(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                solver=solver,
                solver_kwargs=solver_kwargs,
            )
            # Connect the operator to the regularization and preconditioner arrays of the problem
            self._delassus.set_regularization(self._data.eta)
            self._delassus.set_preconditioner(self._data.P)
        else:
            self._delassus = DelassusOperator(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                solver=solver,
                solver_kwargs=solver_kwargs,
            )
            self._data.mio = self._delassus.info.mio
            self._data.D = self._delassus.D
            self._eta_applied = wp.zeros_like(self._data.eta)

    def _zero_impl(self):
        if self._sparse:
            self._delassus.set_needs_update()

    def _build_impl(self, model: ModelKamino, data: DataKamino, jacobians: SystemJacobiansType, reset_to_zero: bool):
        # The sparse Delassus operator is matrix-free and only needs to be flagged for an update
        if self._sparse:
            return

        # Build the dense Delassus matrix, which is free of any regularization afterwards
        self._delassus.build(model=model, data=data, jacobians=jacobians, reset_to_zero=reset_to_zero)
        self._eta_applied.zero_()

    def _compute_operator_diagonal_impl(self, diag: wp.array[wp.float32]):
        if self._sparse:
            self._delassus.diagonal(diag)
        else:
            wp.launch(
                _extract_dense_diagonal,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[self._data.dim, self._data.mio, self._data.vio, self._data.D, diag],
                device=self.device,
            )

    def _apply_preconditioner_to_operator_impl(self):
        # The sparse operator references the preconditioner array of the problem directly
        if self._sparse:
            return
        wp.launch(
            _apply_dual_preconditioner_to_matrix,
            dim=(self._size.num_worlds, self._delassus._max_of_max_total_D_size),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.mio,
                self._data.vio,
                self._data.P,
                # Outputs:
                self._data.D,
            ],
            device=self.device,
        )

    def _update_regularization_impl(self):
        # The sparse operator references the regularization array of the problem directly. Only the
        # regularization changed (not the Jacobian sparsity), so we flag a regularization-only refresh.
        if self._sparse:
            self._delassus.set_regularization_needs_update()
            return

        # Replace the regularization applied in-place to the dense Delassus matrix and re-factorize it
        wp.launch(
            _apply_dense_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.mio,
                self._data.vio,
                self._data.eta,
                # Outputs:
                self._eta_applied,
                self._data.D,
            ],
            device=self.device,
        )
        self._delassus.compute(reset_to_zero=True)

    def _solve_impl(self, v: wp.array[wp.float32], x: wp.array[wp.float32]):
        self._delassus.solve(v=v, x=x)

    def _gemv_impl(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        alpha: float,
        beta: float,
        preconditioned: bool,
        world_mask: wp.array[wp.bool],
    ):
        if self._sparse:
            # Temporarily remove the regularization and optionally the preconditioning
            eta = self._delassus._eta
            self._delassus.set_regularization(None)
            if not preconditioned:
                self._delassus.set_preconditioner(None)
            if alpha == 1.0 and beta == 0.0:
                self._delassus.matvec(x=x, y=y, world_mask=world_mask)
            else:
                self._delassus.gemv(x=x, y=y, world_mask=world_mask, alpha=alpha, beta=beta)
            self._delassus.set_regularization(eta)
            if not preconditioned:
                self._delassus.set_preconditioner(self._data.P)
        else:
            wp.launch(
                _gemv_dense_unregularized,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    self._data.dim,
                    self._data.mio,
                    self._data.vio,
                    self._data.D,
                    self._data.P,
                    self._eta_applied,
                    world_mask,
                    wp.float32(alpha),
                    wp.float32(beta),
                    wp.bool(preconditioned),
                    x,
                    # Outputs:
                    y,
                ],
                device=self.device,
            )
