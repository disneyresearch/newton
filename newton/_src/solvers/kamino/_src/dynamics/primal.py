# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a data container and relevant operations to
represent and construct a primal forward dynamics problem.

The primal forward dynamics problem arises from the formulation of the equations of motion in
terms of the post-event generalized velocities ``u+``, i.e. Gauss' principle of least constraint
at velocity level in the kinetic metric of the generalized mass matrix ``M``:

.. code-block:: text

    u+ = argmin_u  1/2 ||u - u_f||^2_M + phi_R(J u + v* + Gamma(v+))

where ``phi_R`` is the (compliant) indicator of the dual cone ``K^*`` of the constraints.

Splitting methods such as PADMM solve this problem with iterations whose linear step solves
the compliant constrained-dynamics saddle point:

.. code-block:: text

    [ M    J^T    ] [  u     ]   [ p_f ]
    [ J   -R_eff  ] [ -lambda ] = [ b   ],     R_eff := R + (eta + rho^{-1}) I

where ``p_f = M u_f`` and ``b`` is a constraint-space target velocity. Taking the Schur complement
onto the generalized velocities yields the linear system of the constraint-augmented generalized
mass matrix ``M_W := M + J^T W J`` with the effective constraint stiffness ``W := R_eff^{-1}``:

.. code-block:: text

    M_W u = p_f + J^T W b,     lambda = W (b - J u)

The :class:`PrimalProblem` realizes the operator contract of :class:`ConstrainedDynamicsProblem`,
i.e. the regularized constraint-space solve ``x = (P (D + R) P + diag(eta))^{-1} v``, using the
Woodbury identity, without ever forming the Delassus matrix ``D = J M^{-1} J^T``. With
``v' := P^{-1} v`` and ``W = (R + diag(eta) P^{-2})^{-1}`` it computes

.. code-block:: text

    du = M_W^{-1} J^T W v',     x = P^{-1} W (v' - J du)

which is the generalized-velocity step of the primal Proximal-ADMM with ``b = v' + J u_f`` and
the generalized velocity iterate ``u = u_f + du``. The iterates of PADMM operating on this problem
are thus identical to those of PADMM operating on the :class:`DualProblem` in exact arithmetic.

Typical usage example:
    # Create a model, data, limits, contacts and Jacobians containers
    ...

    # Create a primal forward dynamics problem and build it using the current model
    # data and active unilateral constraints (i.e. for limits and contacts).
    primal = PrimalProblem(model, data, limits, contacts, jacobians, LLTBlockedSolver, sparse=False)
    primal.build(model, data, jacobians, limits, contacts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino
from ..linalg import LinearSolverType
from .augmented_mass import AugmentedMassOperator, BlockSparseMatrixFreeAugmentedMassOperator
from .config import ProblemFormulation
from .operators import compute_delassus_diagonal, inverse_mass_gemv, jacobian_gemv, jacobian_transpose_gemv
from .problem import ConstrainedDynamicsProblem, ConstrainedDynamicsProblemData, SystemJacobiansType

###
# Module interface
###

__all__ = [
    "PrimalProblem",
    "PrimalProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Types
###


@dataclass
class PrimalProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the primal forward dynamics problem data over multiple worlds.
    """

    W: wp.array[wp.float32] | None = None
    """
    The per-constraint effective stiffness ``W = (R + diag(eta) P^{-2})^{-1}`` in physical coordinates.
    Zero for inactive constraints. Shape of `(sum_of_max_total_cts,)`.
    """

    du: wp.array[wp.float32] | None = None
    """
    The generalized velocity correction ``du = u - u_f`` computed by the last call to :meth:`PrimalProblem.solve`.
    Stored as a flat array of per-body twists. Shape of `(sum_of_num_body_dofs,)`.
    """


###
# Kernels
###


@wp.kernel
def _compute_effective_stiffness(
    # Inputs:
    problem_maxdim: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    problem_R: wp.array[wp.float32],
    problem_eta: wp.array[wp.float32],
    # Outputs:
    problem_W: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the row index exceeds the maximum problem size
    if tid >= problem_maxdim[wid]:
        return

    # Inactive constraints do not couple to the generalized velocities
    v_i = problem_vio[wid] + tid
    if tid >= problem_dim[wid]:
        problem_W[v_i] = 0.0
        return

    # Compute the effective compliance in physical coordinates: R_eff = R + eta / P^2
    p_i = problem_P[v_i]
    r_eff = problem_eta[v_i] / (p_i * p_i)
    if problem_R:
        r_eff += problem_R[v_i]

    # Rows without any compliance (e.g. of converged worlds) are decoupled
    problem_W[v_i] = wp.where(r_eff > 0.0, 1.0 / r_eff, 0.0)


@wp.kernel
def _add_compliance_to_diagonal(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_R: wp.array[wp.float32],
    # Outputs:
    diag: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the row index exceeds the problem size
    if tid >= problem_dim[wid]:
        return

    # Add the compliance to the diagonal of the Delassus matrix
    v_i = problem_vio[wid] + tid
    diag[v_i] += problem_R[v_i]


@wp.kernel
def _compute_weighted_target_velocity(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    problem_W: wp.array[wp.float32],
    v: wp.array[wp.float32],
    # Outputs:
    c: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the row index exceeds the problem size
    if tid >= problem_dim[wid]:
        return

    # Compute the stiffness-weighted target velocity in physical coordinates: c = W P^{-1} v
    v_i = problem_vio[wid] + tid
    c[v_i] = problem_W[v_i] * v[v_i] / problem_P[v_i]


@wp.kernel
def _compute_trial_reactions(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    problem_W: wp.array[wp.float32],
    v: wp.array[wp.float32],
    J_du: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the row index exceeds the problem size
    if tid >= problem_dim[wid]:
        return

    # Compute the trial reactions in preconditioned coordinates: x = P^{-1} W (P^{-1} v - J du)
    v_i = problem_vio[wid] + tid
    inv_p_i = 1.0 / problem_P[v_i]
    x[v_i] = inv_p_i * problem_W[v_i] * (inv_p_i * v[v_i] - J_du[v_i])


@wp.kernel
def _scale_by_preconditioner(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    preconditioned: wp.bool,
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the world is inactive or the row index exceeds the problem size
    if not world_mask[wid] or tid >= problem_dim[wid]:
        return

    # Map from preconditioned to physical coordinates if required
    v_i = problem_vio[wid] + tid
    y[v_i] = wp.where(preconditioned, problem_P[v_i] * x[v_i], x[v_i])


@wp.kernel
def _finalize_compliant_gemv(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    problem_R: wp.array[wp.float32],
    world_mask: wp.array[wp.bool],
    alpha: wp.float32,
    beta: wp.float32,
    preconditioned: wp.bool,
    x_phys: wp.array[wp.float32],
    D_x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the world is inactive or the row index exceeds the problem size
    if not world_mask[wid] or tid >= problem_dim[wid]:
        return

    # Add the compliance to the Delassus product: (D + R) x
    v_i = problem_vio[wid] + tid
    a_x = D_x[v_i]
    if problem_R:
        a_x += problem_R[v_i] * x_phys[v_i]

    # Map back to preconditioned coordinates if required
    if preconditioned:
        a_x *= problem_P[v_i]

    # Store the generalized matrix-vector product
    y[v_i] = alpha * a_x + beta * y[v_i]


###
# Interfaces
###


class PrimalProblem(ConstrainedDynamicsProblem):
    """
    The primal formulation of the constrained forward dynamics problem.

    Realizes the linear operator of the problem in the generalized velocities using the
    constraint-augmented generalized mass matrix ``M + J^T W J``, which is either assembled
    explicitly and factorized (dense) or represented as a matrix-free operator solved
    iteratively (sparse). See the module documentation for details.
    """

    _data_type = PrimalProblemData

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[PrimalProblem.Config] | PrimalProblem.Config | None = None,
        compute_h: bool = False,
        sparse: bool = True,
    ):
        """
        Constructs a primal problem interface container.

        See :class:`ConstrainedDynamicsProblem` for a description of the arguments. The linear
        solver must be a direct solver for dense problems and an iterative solver for sparse ones.
        """
        self._operator: AugmentedMassOperator | BlockSparseMatrixFreeAugmentedMassOperator | None = None
        """The constraint-augmented generalized mass matrix operator."""

        self._model: ModelKamino | None = None
        self._system_data: DataKamino | None = None
        self._jacobians: SystemJacobiansType | None = None

        self._u_rhs: wp.array[wp.float32] | None = None
        """Temporary generalized vector holding right-hand-sides and intermediate products."""

        self._c_tmp: wp.array[wp.float32] | None = None
        """Temporary constraint-space vector holding intermediate products."""

        self._c_phys: wp.array[wp.float32] | None = None
        """Temporary constraint-space vector holding inputs in physical coordinates."""

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
        Returns the formulation of the problem, i.e. :attr:`ProblemFormulation.PRIMAL`.
        """
        return ProblemFormulation.PRIMAL

    @property
    def operator(self) -> AugmentedMassOperator | BlockSparseMatrixFreeAugmentedMassOperator:
        """
        Returns the constraint-augmented generalized mass matrix operator.
        """
        if self._operator is None:
            raise ValueError("The augmented mass operator is not allocated. Call `finalize()` first.")
        return self._operator

    @property
    def linear_solver(self) -> LinearSolverType | None:
        """
        Returns the linear solver of the constraint-augmented generalized mass matrix operator.
        """
        return self.operator.solver

    @property
    def data(self) -> PrimalProblemData:
        """
        Returns the primal problem data container.
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
        self._model = model
        self._system_data = data
        self._jacobians = jacobians
        if self._sparse:
            self._operator = BlockSparseMatrixFreeAugmentedMassOperator(
                model=model, data=data, jacobians=jacobians, solver=solver, solver_kwargs=solver_kwargs
            )
        else:
            if not isinstance(jacobians, DenseSystemJacobians):
                raise TypeError("A dense primal problem requires dense jacobians.")
            self._operator = AugmentedMassOperator(model=model, solver=solver, solver_kwargs=solver_kwargs)

        with wp.ScopedDevice(self._device):
            self._data.W = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)
            self._data.du = wp.zeros(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32)
            self._u_rhs = wp.zeros(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32)
            self._c_tmp = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)
            self._c_phys = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)

    def _zero_impl(self):
        self._data.du.zero_()

    def _build_impl(self, model: ModelKamino, data: DataKamino, jacobians: SystemJacobiansType, reset_to_zero: bool):
        if self._sparse != isinstance(jacobians, SparseSystemJacobians):
            raise TypeError("The type of the jacobians does not match the sparsity of the primal problem.")
        self._model = model
        self._system_data = data
        self._jacobians = jacobians

    def _compute_operator_diagonal_impl(self, diag: wp.array[wp.float32]):
        compute_delassus_diagonal(self._model, self._system_data, self._jacobians, self._data.dim, self._data.vio, diag)
        if self._data.R is not None:
            wp.launch(
                _add_compliance_to_diagonal,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[self._data.dim, self._data.vio, self._data.R, diag],
                device=self.device,
            )

    def _apply_preconditioner_to_operator_impl(self):
        # The preconditioner only enters the effective stiffness, computed in `update_regularization()`
        return

    def _update_regularization_impl(self):
        wp.launch(
            _compute_effective_stiffness,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.maxdim,
                self._data.dim,
                self._data.vio,
                self._data.P,
                self._data.R,
                self._data.eta,
                # Outputs:
                self._data.W,
            ],
            device=self.device,
        )
        if self._sparse:
            self._operator.compute(self._data.dim, self._data.vio, self._data.W)
        else:
            self._operator.compute(
                self._model, self._system_data, self._jacobians, self._data.dim, self._data.vio, self._data.W
            )

    def _solve_impl(self, v: wp.array[wp.float32], x: wp.array[wp.float32]):
        cts_dim = (self._size.num_worlds, self._size.max_of_max_total_cts)
        mask = self._all_worlds_mask

        # Compute the generalized right-hand-side: J^T W P^{-1} v
        wp.launch(
            _compute_weighted_target_velocity,
            dim=cts_dim,
            inputs=[self._data.dim, self._data.vio, self._data.P, self._data.W, v, self._c_tmp],
            device=self.device,
        )
        jacobian_transpose_gemv(
            self._model, self._jacobians, self._data.dim, self._data.vio, self._c_tmp, self._u_rhs, mask
        )

        # Solve for the generalized velocity correction: (M + J^T W J) du = J^T W P^{-1} v
        self._operator.solve(b=self._u_rhs, x=self._data.du)

        # Recover the trial reactions: x = P^{-1} W (P^{-1} v - J du)
        jacobian_gemv(self._model, self._jacobians, self._data.dim, self._data.vio, self._data.du, self._c_tmp, mask)
        wp.launch(
            _compute_trial_reactions,
            dim=cts_dim,
            inputs=[self._data.dim, self._data.vio, self._data.P, self._data.W, v, self._c_tmp, x],
            device=self.device,
        )

    def _gemv_impl(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        alpha: float,
        beta: float,
        preconditioned: bool,
        world_mask: wp.array[wp.bool],
    ):
        cts_dim = (self._size.num_worlds, self._size.max_of_max_total_cts)

        # Map the input to physical coordinates
        wp.launch(
            _scale_by_preconditioner,
            dim=cts_dim,
            inputs=[self._data.dim, self._data.vio, self._data.P, world_mask, wp.bool(preconditioned), x, self._c_phys],
            device=self.device,
        )

        # Compute the Delassus product matrix-free: D x = J M^{-1} J^T x
        jacobian_transpose_gemv(
            self._model, self._jacobians, self._data.dim, self._data.vio, self._c_phys, self._u_rhs, world_mask
        )
        inverse_mass_gemv(self._model, self._system_data, self._u_rhs, self._u_rhs, world_mask)
        jacobian_gemv(
            self._model, self._jacobians, self._data.dim, self._data.vio, self._u_rhs, self._c_tmp, world_mask
        )

        # Add the compliance and map back to the coordinates of the output
        wp.launch(
            _finalize_compliant_gemv,
            dim=cts_dim,
            inputs=[
                self._data.dim,
                self._data.vio,
                self._data.P,
                self._data.R,
                world_mask,
                wp.float32(alpha),
                wp.float32(beta),
                wp.bool(preconditioned),
                self._c_phys,
                self._c_tmp,
                y,
            ],
            device=self.device,
        )
