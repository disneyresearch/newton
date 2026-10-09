# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a data container and relevant operations to represent
the KKT formulation of the constrained forward dynamics problem.

The KKT formulation retains both the post-event generalized velocities ``u`` and the constraint
reactions ``lambda`` as unknowns. The linear step of splitting methods such as PADMM is then the
compliant constrained-dynamics saddle-point system

.. code-block:: text

    [ M    J^T    ] [  u      ]   [ p_f ]
    [ J   -R_eff  ] [ -lambda ] = [ b   ],     R_eff := R + diag(eta) P^{-2}

where ``M`` is the generalized mass matrix, ``J`` the constraint Jacobian, ``p_f = M u_f`` the free
generalized momentum, ``b`` a constraint-space target velocity, ``R`` the (optional) constraint
compliance, ``eta`` the regularization held in preconditioned coordinates and ``P`` the diagonal
preconditioner. Eliminating ``u`` (Schur complement onto ``lambda``) yields the system of the
:class:`DualProblem`, while eliminating ``lambda`` yields that of the :class:`PrimalProblem`.

The KKT matrix ``K`` is symmetric quasi-definite, i.e. indefinite, so it cannot be solved with
the Cholesky-based direct solvers or the conjugate gradient/residual solvers of Kamino. Its linear
solve is therefore not implemented yet, and requires a batched symmetric quasi-definite LDL^T
factorization or a MINRES-type iterative solver. The formulation is nevertheless fully defined:
its constraint-space data, stacked layout, right-hand-side and matrix-vector products with ``K``
are available, as well as the constraint-space operator products of the common problem API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..geometry.contacts import ContactsKamino
from ..kinematics.limits import LimitsKamino
from ..linalg import LinearSolverType
from .config import ProblemFormulation
from .operators import (
    compliant_delassus_gemv,
    compute_compliant_delassus_diagonal,
    jacobian_gemv,
    jacobian_transpose_gemv,
    mass_gemv,
)
from .problem import ConstrainedDynamicsProblem, ConstrainedDynamicsProblemData, SystemJacobiansType

###
# Module interface
###

__all__ = [
    "KKTProblem",
    "KKTProblemData",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Types
###


@dataclass
class KKTProblemData(ConstrainedDynamicsProblemData):
    """
    A container to hold the KKT forward dynamics problem data over multiple worlds.

    The stacked vectors of the KKT system hold the block of each world at offset `kkt_vio`, where
    the first `6 * num_bodies` entries are the generalized part and the remaining `max_total_cts`
    entries the constraint part, of which only the first `num_total_cts` are active.
    """

    kkt_maxdim: wp.array[wp.int32] | None = None
    """
    The maximum dimension of the KKT system of each world, i.e. `6 * num_bodies + max_total_cts`.
    Shape of `(num_worlds,)`.
    """

    kkt_dim: wp.array[wp.int32] | None = None
    """
    The active dimension of the KKT system of each world, i.e. `6 * num_bodies + num_total_cts`.
    Shape of `(num_worlds,)`.
    """

    kkt_vio: wp.array[wp.int32] | None = None
    """
    The vector index offset of the stacked KKT vector block of each world.
    Shape of `(num_worlds,)`.
    """

    k: wp.array[wp.float32] | None = None
    """
    The stacked right-hand-side vectors ``k = [p_f; b]`` of the KKT system.
    Shape of `(sum_of_max_kkt_dims,)`.
    """

    ux: wp.array[wp.float32] | None = None
    """
    The stacked solution vectors ``ux = [u; -lambda]`` of the KKT system.
    Shape of `(sum_of_max_kkt_dims,)`.
    """


###
# Kernels
###


@wp.kernel
def _update_kkt_dims(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    # Outputs:
    kkt_dim: wp.array[wp.int32],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Compute the active dimension of the KKT system of the world
    nbd = 6 * (model_info_bodies_offset[wid + 1] - model_info_bodies_offset[wid])
    kkt_dim[wid] = nbd + problem_dim[wid]


@wp.kernel
def _build_kkt_momentum_rhs(
    # Inputs:
    model_bodies_wid: wp.array[wp.int32],
    model_info_bodies_offset: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    problem_u_f: wp.array[wp.spatial_vectorf],
    kkt_vio: wp.array[wp.int32],
    # Outputs:
    kkt_k: wp.array[wp.float32],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Compute the free generalized momentum of the body: p_f = M u_f
    wid = model_bodies_wid[bid]
    u_f = problem_u_f[bid]
    f = model_bodies_m_i[bid] * wp.spatial_top(u_f)
    tau = data_bodies_I_i[bid] @ wp.spatial_bottom(u_f)

    # Store the momentum in the generalized part of the stacked vector of the world
    k = kkt_vio[wid] + 6 * (bid - model_info_bodies_offset[wid])
    for d in range(3):
        kkt_k[k + d] = f[d]
        kkt_k[k + 3 + d] = tau[d]


@wp.kernel
def _build_kkt_constraint_rhs(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    problem_maxdim: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    kkt_vio: wp.array[wp.int32],
    b: wp.array[wp.float32],
    # Outputs:
    kkt_k: wp.array[wp.float32],
):
    # Retrieve the world and constraint indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the row index exceeds the maximum problem size
    if tid >= problem_maxdim[wid]:
        return

    # Store the target velocity in the constraint part of the stacked vector of the world
    nbd = 6 * (model_info_bodies_offset[wid + 1] - model_info_bodies_offset[wid])
    b_i = wp.float32(0.0)
    if tid < problem_dim[wid]:
        b_i = b[problem_vio[wid] + tid]
    kkt_k[kkt_vio[wid] + nbd + tid] = b_i


@wp.kernel
def _gather_kkt_vector(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    kkt_maxdim: wp.array[wp.int32],
    kkt_vio: wp.array[wp.int32],
    x: wp.array[wp.float32],
    # Outputs:
    x_u: wp.array[wp.float32],
    x_c: wp.array[wp.float32],
):
    # Retrieve the world and stacked indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the index exceeds the maximum size of the KKT system
    if tid >= kkt_maxdim[wid]:
        return

    # Split the stacked vector into its generalized and constraint parts
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    x_k = x[kkt_vio[wid] + tid]
    if tid < nbd:
        x_u[6 * bio + tid] = x_k
    else:
        row = tid - nbd
        x_c[problem_vio[wid] + row] = wp.where(row < problem_dim[wid], x_k, 0.0)


@wp.kernel
def _scatter_kkt_vector(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    problem_R: wp.array[wp.float32],
    problem_eta: wp.array[wp.float32],
    kkt_maxdim: wp.array[wp.int32],
    kkt_vio: wp.array[wp.int32],
    x_c: wp.array[wp.float32],
    y_u: wp.array[wp.float32],
    J_x_u: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the world and stacked indices as the thread indices
    wid, tid = wp.tid()

    # Skip if the index exceeds the maximum size of the KKT system
    if tid >= kkt_maxdim[wid]:
        return

    # Store the generalized part: M x_u + J^T x_c
    bio = model_info_bodies_offset[wid]
    nbd = 6 * (model_info_bodies_offset[wid + 1] - bio)
    k = kkt_vio[wid] + tid
    if tid < nbd:
        y[k] = y_u[6 * bio + tid]
        return

    # Store the constraint part of active constraints: J x_u - R_eff x_c
    row = tid - nbd
    y_k = wp.float32(0.0)
    if row < problem_dim[wid]:
        v_i = problem_vio[wid] + row
        p_i = problem_P[v_i]
        r_eff = problem_eta[v_i] / (p_i * p_i)
        if problem_R:
            r_eff += problem_R[v_i]
        y_k = J_x_u[v_i] - r_eff * x_c[v_i]
    y[k] = y_k


###
# Interfaces
###


class KKTProblem(ConstrainedDynamicsProblem):
    """
    The KKT formulation of the constrained forward dynamics problem.

    Realizes the stacked saddle-point system of the generalized velocities and constraint reactions
    with the quasi-definite KKT matrix ``K = [M, J^T; J, -R_eff]``. The linear solve of this system
    is not implemented yet, i.e. :meth:`update_regularization` and :meth:`solve` raise a
    :class:`NotImplementedError`, while all other operations of the problem are available, as well as
    the KKT-specific operations :meth:`build_rhs` and :meth:`kkt_matvec`. See the module documentation
    for details.
    """

    _data_type = KKTProblemData

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[KKTProblem.Config] | KKTProblem.Config | None = None,
        compute_h: bool = False,
        sparse: bool = True,
    ):
        """
        Constructs a KKT problem interface container.

        See :class:`ConstrainedDynamicsProblem` for a description of the arguments. The `solver`
        and `solver_kwargs` arguments are currently unused, since the linear solve of the KKT
        system is not implemented yet.
        """
        self._model: ModelKamino | None = None
        self._system_data: DataKamino | None = None
        self._jacobians: SystemJacobiansType | None = None

        self._max_kkt_dim: int = 0
        """The largest maximum dimension of the KKT system across all worlds."""

        self._x_u: wp.array[wp.float32] | None = None
        """Temporary generalized vector holding the generalized part of inputs."""

        self._y_u: wp.array[wp.float32] | None = None
        """Temporary generalized vector holding intermediate products."""

        self._x_c: wp.array[wp.float32] | None = None
        """Temporary constraint-space vector holding the constraint part of inputs."""

        self._c_tmp: wp.array[wp.float32] | None = None
        """Temporary constraint-space vector holding intermediate products."""

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
        Returns the formulation of the problem, i.e. :attr:`ProblemFormulation.KKT`.
        """
        return ProblemFormulation.KKT

    @property
    def linear_solver(self) -> LinearSolverType | None:
        """
        Returns `None`, since the linear solve of the KKT system is not implemented yet.
        """
        return None

    @property
    def data(self) -> KKTProblemData:
        """
        Returns the KKT problem data container.
        """
        return self._data

    ###
    # Operations
    ###

    def build_rhs(self, b: wp.array[wp.float32]):
        """
        Builds the stacked right-hand-side vectors ``k = [M u_f; b]`` of the KKT system.

        Args:
            b: The constraint-space target velocities in physical coordinates.
                Shape of ``(sum_of_max_total_cts,)``.
        """
        self._assert_is_finalized()
        model = self._model
        wp.launch(
            _build_kkt_momentum_rhs,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.bodies.wid,
                model.info.bodies_offset,
                model.bodies.m_i,
                self._system_data.bodies.I_i,
                self._data.u_f,
                self._data.kkt_vio,
                # Outputs:
                self._data.k,
            ],
            device=self.device,
        )
        wp.launch(
            _build_kkt_constraint_rhs,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                self._data.maxdim,
                self._data.dim,
                self._data.vio,
                self._data.kkt_vio,
                b,
                # Outputs:
                self._data.k,
            ],
            device=self.device,
        )

    def kkt_matvec(self, x: wp.array[wp.float32], y: wp.array[wp.float32]):
        """
        Computes the matrix-vector product ``y = K @ x`` with the KKT matrix ``K = [M, J^T; J, -R_eff]``
        on stacked vectors, where ``R_eff = R + diag(eta) P^{-2}`` in physical coordinates.

        The entries of inactive constraints are ignored in `x` and set to zero in `y`.

        Args:
            x: The input array of stacked vectors. Shape of ``(sum_of_max_kkt_dims,)``.
            y: The output array of stacked vectors. Shape of ``(sum_of_max_kkt_dims,)``.
        """
        self._assert_is_finalized()
        model = self._model
        mask = self._all_worlds_mask
        stacked_dim = (self._size.num_worlds, self._max_kkt_dim)

        # Split the input into its generalized and constraint parts
        wp.launch(
            _gather_kkt_vector,
            dim=stacked_dim,
            inputs=[
                model.info.bodies_offset,
                self._data.dim,
                self._data.vio,
                self._data.kkt_maxdim,
                self._data.kkt_vio,
                x,
                self._x_u,
                self._x_c,
            ],
            device=self.device,
        )

        # Compute the generalized part: M x_u + J^T x_c
        mass_gemv(model, self._system_data, self._x_u, self._y_u, mask)
        jacobian_transpose_gemv(
            model, self._jacobians, self._data.dim, self._data.vio, self._x_c, self._y_u, mask, beta=1.0
        )

        # Compute the constraint part: J x_u - R_eff x_c
        jacobian_gemv(model, self._jacobians, self._data.dim, self._data.vio, self._x_u, self._c_tmp, mask)
        wp.launch(
            _scatter_kkt_vector,
            dim=stacked_dim,
            inputs=[
                # Inputs:
                model.info.bodies_offset,
                self._data.dim,
                self._data.vio,
                self._data.P,
                self._data.R,
                self._data.eta,
                self._data.kkt_maxdim,
                self._data.kkt_vio,
                self._x_c,
                self._y_u,
                self._c_tmp,
                # Outputs:
                y,
            ],
            device=self.device,
        )

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

        # Compute the stacked layout of the KKT system of each world
        kkt_maxdims = model.info.num_body_dofs.numpy() + model.info.max_total_cts.numpy()
        kkt_vio = np.concatenate(([0], np.cumsum(kkt_maxdims)[:-1]))
        self._max_kkt_dim = int(np.max(kkt_maxdims))
        sum_of_max_kkt_dims = int(np.sum(kkt_maxdims))

        with wp.ScopedDevice(self._device):
            self._data.kkt_maxdim = wp.array(kkt_maxdims, dtype=wp.int32)
            self._data.kkt_dim = wp.array(kkt_maxdims, dtype=wp.int32)
            self._data.kkt_vio = wp.array(kkt_vio, dtype=wp.int32)
            self._data.k = wp.zeros(shape=(sum_of_max_kkt_dims,), dtype=wp.float32)
            self._data.ux = wp.zeros(shape=(sum_of_max_kkt_dims,), dtype=wp.float32)
            self._x_u = wp.zeros(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32)
            self._y_u = wp.zeros(shape=(model.size.sum_of_num_body_dofs,), dtype=wp.float32)
            self._x_c = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)
            self._c_tmp = wp.zeros(shape=(model.size.sum_of_max_total_cts,), dtype=wp.float32)

    def _zero_impl(self):
        self._data.k.zero_()
        self._data.ux.zero_()

    def _build_impl(self, model: ModelKamino, data: DataKamino, jacobians: SystemJacobiansType, reset_to_zero: bool):
        self._model = model
        self._system_data = data
        self._jacobians = jacobians
        wp.launch(
            _update_kkt_dims,
            dim=self._size.num_worlds,
            inputs=[model.info.bodies_offset, self._data.dim, self._data.kkt_dim],
            device=self.device,
        )

    def _compute_operator_diagonal_impl(self, diag: wp.array[wp.float32]):
        compute_compliant_delassus_diagonal(
            self._model, self._system_data, self._jacobians, self._data.dim, self._data.vio, self._data.R, diag
        )

    def _apply_preconditioner_to_operator_impl(self):
        # The KKT matrix is never stored, and the preconditioner only enters its effective compliance
        return

    def _update_regularization_impl(self):
        raise NotImplementedError(
            "The linear solve of the KKT system requires a batched symmetric quasi-definite LDL^T "
            "factorization or a MINRES-type iterative solver, which are not implemented yet."
        )

    def _solve_impl(self, v: wp.array[wp.float32], x: wp.array[wp.float32]):
        raise NotImplementedError(
            "The linear solve of the KKT system requires a batched symmetric quasi-definite LDL^T "
            "factorization or a MINRES-type iterative solver, which are not implemented yet."
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
        compliant_delassus_gemv(
            self._model,
            self._system_data,
            self._jacobians,
            self._data.dim,
            self._data.vio,
            self._data.P,
            self._data.R,
            x,
            y,
            world_mask,
            alpha,
            beta,
            preconditioned,
            self._x_c,
            self._c_tmp,
            self._y_u,
        )
