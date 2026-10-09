# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a common interface for the formulations of the constrained forward dynamics (CFD) problem.

A single time-step of the constrained forward dynamics amounts to the nonlinear complementarity
problem (NCP) in the post-event generalized velocities ``u+`` and constraint reactions ``lambda``:

.. code-block:: text

    M (u+ - u_f) = J^T lambda
    K^* ∋ v+ + Gamma(v+)  ⊥  lambda ∈ K,     v+ = J u+ + v* + R lambda

where ``M`` is the generalized mass matrix, ``u_f`` the unconstrained generalized velocity, ``J``
the constraint Jacobian, ``v*`` the constraint-space bias velocity, ``R`` the (optional, diagonal)
constraint compliance, ``K`` the composite cone of admissible reactions and ``Gamma`` the De Saxce
correction. Eliminating ``u+`` yields the dual problem in the Delassus matrix ``D = J M^{-1} J^T``
and the free velocity ``v_f = J u_f + v*``, i.e. ``v+ = (D + R) lambda + v_f``.

All formulations share the same constraint-space data (layout, ``v_f``, ``mu``, bounds, ``R``) as
well as an optional diagonal preconditioner ``P``. Splitting methods such as PADMM iterate in the
preconditioned constraint space and only require the problem to realize the regularized linear
solve ``x = (P (D + R) P + diag(eta))^{-1} v``. Each formulation realizes this operation in its
own space, e.g. the dual problem factorizes the Delassus matrix while the primal problem solves
with the constraint-augmented mass matrix ``M + J^T W J``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..core.data import DataKamino
from ..core.math import FLOAT32_EPS, compute_gyroscopic_torque
from ..core.model import ModelKamino
from ..core.size import SizeKamino
from ..core.types import vec6f
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino
from ..linalg import LinearSolverType
from ..linalg.linear import IterativeSolver
from .config import (
    ConstrainedDynamicsConfigStruct,
    ConstrainedDynamicsProblemConfig,
    ProblemFormulation,
    assert_containers_are_valid,
)

###
# Module interface
###

__all__ = [
    "ConstrainedDynamicsProblem",
    "ConstrainedDynamicsProblemData",
    "SystemJacobiansType",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


###
# Types
###


SystemJacobiansType = DenseSystemJacobians | SparseSystemJacobians
"""A utility type alias for the supported Jacobians containers."""


@dataclass
class ConstrainedDynamicsProblemData:
    """
    A container to hold the constraint-space data of a constrained forward dynamics problem over multiple worlds.
    """

    num_worlds: int = 0
    """The number of worlds represented in the problem."""

    max_of_maxdims: int = 0
    """The largest maximum number of constraints across all worlds."""

    ###
    # Problem configurations
    ###

    config: wp.array[ConstrainedDynamicsConfigStruct] | None = None
    """
    Problem configuration parameters for each world.
    Shape of `(num_worlds,)`.
    """

    ###
    # Constraints info
    ###

    njc: wp.array[wp.int32] | None = None
    """
    The number of active bilateral joint constraints in each world.
    Shape of `(num_worlds,)`.
    """

    nbc: wp.array[wp.int32] | None = None
    """
    The number of active bounded-multiplier constraints in each world.
    Shape of `(num_worlds,)`.
    """

    nl: wp.array[wp.int32] | None = None
    """
    The number of active limit constraints in each world.
    Shape of `(num_worlds,)`.
    """

    nc: wp.array[wp.int32] | None = None
    """
    The number of active contact constraints in each world.
    Shape of `(num_worlds,)`.
    """

    bcio: wp.array[wp.int32] | None = None
    """
    The bounded-multiplier index offset of each world.
    Used to index into `bound_lower` and `bound_upper`.
    Shape of `(num_worlds,)`.
    """

    lio: wp.array[wp.int32] | None = None
    """
    The limit index offset of each world.
    Shape of `(num_worlds,)`.
    """

    cio: wp.array[wp.int32] | None = None
    """
    The contact index offset of each world.
    Shape of `(num_worlds,)`.
    """

    iio: wp.array[wp.int32] | None = None
    """
    The inequality index offset of each world.
    Used to index flat arrays spanning bounded-multiplier, limit, and contact entities.
    Shape of `(num_worlds,)`.
    """

    bcgo: wp.array[wp.int32] | None = None
    """
    The bounded-multiplier constraint group offset of each world.
    Shape of `(num_worlds,)`.
    """

    lcgo: wp.array[wp.int32] | None = None
    """
    The limit constraint group offset of each world.
    Shape of `(num_worlds,)`.
    """

    ccgo: wp.array[wp.int32] | None = None
    """
    The contact constraint group offset of each world.
    Shape of `(num_worlds,)`.
    """

    ###
    # Constraint-space dimensions
    ###

    maxdim: wp.array[wp.int32] | None = None
    """
    The maximum number of constraints of each world.
    Shape of `(num_worlds,)`.
    """

    dim: wp.array[wp.int32] | None = None
    """
    The active number of constraints of each world.
    Shape of `(num_worlds,)`.
    """

    vio: wp.array[wp.int32] | None = None
    """
    The vector index offset of each constraint-space vector block.
    This is applicable to all constraint-space vectors, e.g. `v_b`, `v_i`, `v_f`, `P`, `R` and `eta`.
    Shape of `(num_worlds,)`.
    """

    ###
    # Problem vectors
    ###

    h: wp.array[wp.spatial_vectorf] | None = None
    """
    Stack of non-linear generalized forces vectors of each world.

    Computed as:
    `h = dt * (w_e + w_gc + w_a)`

    where:
    - `dt` is the simulation time step
    - `w_e` is the stack of per-body purely external wrenches
    - `w_gc` is the stack of per-body gravitational + Coriolis wrenches
    - `w_a` is the stack of per-body joint actuation wrenches

    Construction of this term is optional, as it's contributions are already
    incorporated in the computation of the generalized free-velocity `u_f`.
    It is can be optionally built for analysis or debugging purposes.

    Shape of `(sum_of_num_body_dofs,)`.
    """

    u_f: wp.array[wp.spatial_vectorf] | None = None
    """
    Stack of unconstrained generalized velocity vectors.

    Computed as:
    `u_f = u_minus + dt * M^{-1} @ h`

    where:
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step
    - `M^{-1}` is the block-diagonal inverse generalized mass matrix
    - `h` is the stack of non-linear generalized forces vectors

    Shape of `(sum_of_num_body_dofs,)`.
    """

    v_b: wp.array[wp.float32] | None = None
    """
    Stack of free-velocity constraint bias vectors (in constraint-space).

    Computed as:
    `v_b = [ v_b_dynamics;
             alpha * inv_dt * r_joints;
             beta * inv_dt * r_limits;
             gamma * inv_dt * r_contacts ]`

    where:
    - `v_b_dynamics` is the joint dynamics velocity bias.
    - `dt` and `inv_dt` is the simulation time step and it inverse
    - `r_joints` is the stack of joint constraint residuals
    - `r_limits` is the stack of limit constraint residuals
    - `r_contacts` is the stack of contact constraint residuals
    - `alpha`, `beta`, `gamma` are the Baumgarte stabilization
        parameters for joints, limits and contacts, respectively

    Shape of `(sum_of_max_total_cts,)`.
    """

    v_i: wp.array[wp.float32] | None = None
    """
    The stack of free-velocity impact biases vector (in constraint-space).

    Computed as:
    `v_i = epsilon @ (J_cts @ u_minus)`

    where:
    - `epsilon` is the stack of per-contact restitution coefficients
    - `J_cts` is the constraint Jacobian matrix
    - `u_minus` is the stack of per-body generalized velocities at the beginning of the time step

    Shape of `(sum_of_max_total_cts,)`.
    """

    v_f: wp.array[wp.float32] | None = None
    """
    Stack of free-velocity vectors (constraint-space unconstrained velocity).

    Computed as:
    `v_f = J_cts @ u_f + v_b + v_i`

    where:
    - `J_cts` is the constraint Jacobian matrix
    - `u_f` is the stack of unconstrained generalized velocity vectors
    - `v_b` is the stack of free-velocity stabilization biases vectors
    - `v_i` is the stack of free-velocity impact biases vectors

    If preconditioning is enabled, this vector is stored in preconditioned coordinates, i.e. `P @ v_f`.

    Shape of `(sum_of_max_total_cts,)`.
    """

    bound_lower: wp.array[wp.float32] | None = None
    """
    Lower impulse bound for each bounded-multiplier row in preconditioned coordinates.

    Shape of `(sum_of_num_bounded_joint_cts,)`.
    """

    bound_upper: wp.array[wp.float32] | None = None
    """
    Upper impulse bound for each bounded-multiplier row in preconditioned coordinates.

    Shape of `(sum_of_num_bounded_joint_cts,)`.
    """

    mu: wp.array[wp.float32] | None = None
    """
    Stack of per-contact constraint friction coefficient vectors.
    Shape of `(sum_of_max_contacts,)`.
    """

    P: wp.array[wp.float32] | None = None
    """
    The flat array of diagonal preconditioner blocks.
    Shape of `(sum_of_max_total_cts,)`.
    """

    R: wp.array[wp.float32] | None = None
    """
    The flat array of diagonal constraint compliance blocks, in physical (i.e. unpreconditioned) coordinates.

    This is `None` if all constraints of all worlds are rigid, i.e. the compliance is zero.

    Shape of `(sum_of_max_total_cts,)`.
    """

    eta: wp.array[wp.float32] | None = None
    """
    The flat array of diagonal (proximal) regularization blocks in preconditioned coordinates.

    These values are written by the constraint solver operating on the problem, and are
    applied to the problem's linear system by :meth:`ConstrainedDynamicsProblem.update_regularization`.

    Shape of `(sum_of_max_total_cts,)`.
    """


###
# Functions
###


@wp.func
def gravity_plus_coriolis_wrench(
    g: wp.vec3f,
    m_i: wp.float32,
    I_i: wp.mat33f,
    omega_i: wp.vec3f,
    dt: wp.float32,
) -> wp.spatial_vectorf:
    """
    Compute the gravitational + Coriolis wrench acting on a body.
    """
    f_gi_i, tau_gi_i = gravity_plus_coriolis_wrench_split(g, m_i, I_i, omega_i, dt)
    return wp.spatial_vectorf(*f_gi_i, *tau_gi_i)


@wp.func
def gravity_plus_coriolis_wrench_split(
    g: wp.vec3f,
    m_i: wp.float32,
    I_i: wp.mat33f,
    omega_i: wp.vec3f,
    dt: wp.float32,
) -> tuple[wp.vec3f, wp.vec3f]:
    """
    Compute the gravitational+inertial wrench on a body.
    """
    f_gi_i = m_i * g
    tau_gi_i = compute_gyroscopic_torque(dt, I_i, omega_i)
    return f_gi_i, tau_gi_i


@wp.func
def precondition_scalar(d: wp.float32) -> wp.float32:
    """Compute a scalar preconditioner from a diagonal matrix entry.

    Produces an entry of the diagonal preconditioner P used to scale the
    system as P D Pᵀ. Clamping bounds the scale factor to preserve float32
    solver precision during scaling and unscaling.

    The upper bound of 50.0 limits matrix scaling to approximately 10³,
    retaining roughly four significant digits for the solver.

    Args:
        d: Diagonal matrix entry.
    """
    inv_sqrt_d = 1.0 / wp.sqrt(wp.abs(d) + FLOAT32_EPS)
    return wp.clamp(inv_sqrt_d, wp.float32(1.0 / 50.0), wp.float32(50.0))


@wp.func
def _build_preconditioner_entry(
    tid: wp.int32,
    njlc: wp.int32,
    vio: wp.int32,
    problem_P: wp.array[wp.float32],
):
    """Build one scalar or three-dimensional contact preconditioner entries in-place from the operator diagonal."""
    # First handle joint, bounded, and limit constraints, then contact constraints.
    if tid < njlc:
        problem_P[vio + tid] = precondition_scalar(problem_P[vio + tid])
    else:
        ccid = tid - njlc
        # Only the first contact-dimension thread computes the preconditioner.
        if ccid % 3 == 0:
            # Retrieve the operator diagonal entries for the contact set.
            d_kk_0 = problem_P[vio + tid]
            d_kk_1 = problem_P[vio + tid + 1]
            d_kk_2 = problem_P[vio + tid + 2]
            # Compute the effective diagonal entry.
            # Possible options are mean, min, max.
            # d_kk = (d_kk_0 + d_kk_1 + d_kk_2) / 3.0
            # d_kk = wp.min(wp.vec3f(d_kk_0, d_kk_1, d_kk_2))
            d_kk = wp.max(wp.vec3f(d_kk_0, d_kk_1, d_kk_2))
            p_k = precondition_scalar(d_kk)
            problem_P[vio + tid] = p_k
            problem_P[vio + tid + 1] = p_k
            problem_P[vio + tid + 2] = p_k


###
# Kernels
###


@wp.kernel
def _build_nonlinear_generalized_force(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_gravity_vector: wp.array[wp.vec3f],
    model_bodies_wid: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    state_bodies_u_i: wp.array[wp.spatial_vectorf],
    state_bodies_I_i: wp.array[wp.mat33f],
    state_bodies_w_e_i: wp.array[wp.spatial_vectorf],
    state_bodies_w_a_i: wp.array[wp.spatial_vectorf],
    # Outputs:
    problem_h: wp.array[wp.spatial_vectorf],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    g = model_gravity_vector[wid]

    # Extract the linear and angular components of the generalized velocity
    omega_i = wp.spatial_bottom(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i, dt)

    # Store the generalized free-velocity vector
    problem_h[bid] = dt * h_i


@wp.kernel
def _build_generalized_free_velocity(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_gravity_vector: wp.array[wp.vec3f],
    model_bodies_wid: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    model_bodies_inv_m_i: wp.array[wp.float32],
    state_bodies_u_i: wp.array[wp.spatial_vectorf],
    state_bodies_I_i: wp.array[wp.mat33f],
    state_bodies_inv_I_i: wp.array[wp.mat33f],
    state_bodies_w_e_i: wp.array[wp.spatial_vectorf],
    state_bodies_w_a_i: wp.array[wp.spatial_vectorf],
    # Outputs:
    problem_u_f: wp.array[wp.spatial_vectorf],
):
    # Retrieve the body index as the thread index
    bid = wp.tid()

    # Retrieve the body model and data
    wid = model_bodies_wid[bid]
    m_i = model_bodies_m_i[bid]
    I_i = state_bodies_I_i[bid]
    inv_m_i = model_bodies_inv_m_i[bid]
    inv_I_i = state_bodies_inv_I_i[bid]
    u_i = state_bodies_u_i[bid]
    w_e_i = state_bodies_w_e_i[bid]
    w_a_i = state_bodies_w_a_i[bid]

    # Get world data
    dt = model_time_dt[wid]
    g = model_gravity_vector[wid]

    # Extract the linear and angular components of the generalized velocity
    v_i = wp.spatial_top(u_i)
    omega_i = wp.spatial_bottom(u_i)

    # Compute the net external wrench on the body
    h_i = w_e_i + w_a_i + gravity_plus_coriolis_wrench(g, m_i, I_i, omega_i, dt)
    f_h_i = wp.spatial_top(h_i)
    tau_h_i = wp.spatial_bottom(h_i)

    # Compute the generalized free-velocity vector components
    v_f_i = v_i + dt * (inv_m_i * f_h_i)
    omega_f_i = omega_i + dt * (inv_I_i @ tau_h_i)

    # Store the generalized free-velocity vector
    problem_u_f[bid] = wp.spatial_vectorf(*v_f_i, *omega_f_i)


@wp.kernel
def _build_free_velocity_bias_joint_dynamics(
    # Inputs:
    model_joints_wid: wp.array[wp.int32],
    model_joints_dynamic_cts_offset: wp.array[wp.int32],
    model_joints_dynamic_cts_offset_total_cts: wp.array[wp.int32],
    data_joints_dq_b_j: wp.array[wp.float32],
    # Outputs:
    problem_v_b: wp.array[wp.float32],
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the joint constraints size + index offset into the dynamic-only constraints array
    bias_row_start_j = model_joints_dynamic_cts_offset[jid]
    num_dyn_cts_j = model_joints_dynamic_cts_offset[jid + 1] - bias_row_start_j

    # Skip operation if the joint has no dynamic constraints
    if num_dyn_cts_j == 0:
        return

    # Retrieve the joint constraints index offset into the full constraints array
    cts_row_start_j = model_joints_dynamic_cts_offset_total_cts[jid]

    # Compute the free-velocity bias for the joint
    for j in range(num_dyn_cts_j):
        problem_v_b[cts_row_start_j + j] = -data_joints_dq_b_j[bias_row_start_j + j]


@wp.kernel
def _build_free_velocity_bias_joint_kinematics(
    # Inputs:
    model_time_inv_dt: wp.array[wp.float32],
    model_joints_wid: wp.array[wp.int32],
    model_joints_kinematic_cts_offset: wp.array[wp.int32],
    model_joints_kinematic_cts_offset_total_cts: wp.array[wp.int32],
    data_joints_r_j: wp.array[wp.float32],
    problem_config: wp.array[ConstrainedDynamicsConfigStruct],
    # Outputs:
    problem_v_b: wp.array[wp.float32],
):
    # Retrieve the joint index as the thread index
    jid = wp.tid()

    # Retrieve the joint constraints size + index offset into the kinematic-only constraints array
    res_row_start_j = model_joints_kinematic_cts_offset[jid]
    num_kin_cts_j = model_joints_kinematic_cts_offset[jid + 1] - res_row_start_j

    # Retrieve the joint constraints index offset into the full constraints array
    cts_row_start_j = model_joints_kinematic_cts_offset_total_cts[jid]

    # Retrieve the world index
    wid = model_joints_wid[jid]

    # Retrieve the model time step
    inv_dt = model_time_inv_dt[wid]

    # Retrieve the dual problem config
    config = problem_config[wid]

    # Compute baumgarte constraint stabilization coefficient
    c_b = config.alpha * inv_dt

    # Compute the free-velocity bias for the joint
    for j in range(num_kin_cts_j):
        problem_v_b[cts_row_start_j + j] = c_b * data_joints_r_j[res_row_start_j + j]


@wp.kernel
def _build_joint_friction_bounds(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_joints_wid: wp.array[wp.int32],
    model_joints_dofs_offset: wp.array[wp.int32],
    model_joints_num_friction_cts: wp.array[wp.int32],
    model_joints_friction_cts_offset: wp.array[wp.int32],
    model_joints_friction_cts_axis: wp.array[wp.int32],
    model_info_joint_friction_cts_offset: wp.array[wp.int32],
    model_info_joint_bounded_cts_offset: wp.array[wp.int32],
    model_joints_f_j: wp.array[wp.float32],
    # Outputs:
    problem_bound_lower: wp.array[wp.float32],
    problem_bound_upper: wp.array[wp.float32],
):
    """Build per-row Coulomb joint friction impulse bounds."""
    jid = wp.tid()
    wid = model_joints_wid[jid]
    num_rows = model_joints_num_friction_cts[jid]
    dof_start = model_joints_dofs_offset[jid]
    friction_start = model_joints_friction_cts_offset[jid]
    friction_cts_offset_world = friction_start - model_info_joint_friction_cts_offset[wid]
    row_start = model_info_joint_bounded_cts_offset[wid] + friction_cts_offset_world
    dt = model_time_dt[wid]
    for j in range(num_rows):
        axis = model_joints_friction_cts_axis[friction_start + j]
        bound = dt * model_joints_f_j[dof_start + axis]
        problem_bound_lower[row_start + j] = -bound
        problem_bound_upper[row_start + j] = bound


@wp.kernel
def _build_joint_effort_bounds(
    # Inputs:
    model_joints_wid: wp.array[wp.int32],
    model_joints_num_effort_cts: wp.array[wp.int32],
    model_joints_effort_cts_offset: wp.array[wp.int32],
    model_info_num_friction_cts: wp.array[wp.int32],
    model_info_joint_effort_cts_offset: wp.array[wp.int32],
    model_info_joint_bounded_cts_offset: wp.array[wp.int32],
    data_joints_bound_a: wp.array[wp.float32],
    # Outputs:
    problem_bound_lower: wp.array[wp.float32],
    problem_bound_upper: wp.array[wp.float32],
):
    """Scatter precomputed effort-limit implicit-PD impulse bounds."""
    jid = wp.tid()
    wid = model_joints_wid[jid]
    effort_start = model_joints_effort_cts_offset[jid]
    effort_cts_offset_world = effort_start - model_info_joint_effort_cts_offset[wid]
    row_start = model_info_joint_bounded_cts_offset[wid] + model_info_num_friction_cts[wid] + effort_cts_offset_world
    num_rows = model_joints_num_effort_cts[jid]
    for j in range(num_rows):
        bound = data_joints_bound_a[effort_start + j]
        problem_bound_lower[row_start + j] = -bound
        problem_bound_upper[row_start + j] = bound


@wp.kernel
def _build_joint_effort_bias(
    # Inputs:
    model_joints_num_effort_cts: wp.array[wp.int32],
    model_joints_effort_cts_offset: wp.array[wp.int32],
    model_joints_effort_cts_offset_total_cts: wp.array[wp.int32],
    data_joints_dq_b_a: wp.array[wp.float32],
    # Outputs:
    problem_v_b: wp.array[wp.float32],
):
    """Scatter precomputed effort-row velocity biases into the dual problem."""
    jid = wp.tid()
    effort_start = model_joints_effort_cts_offset[jid]
    row_start = model_joints_effort_cts_offset_total_cts[jid]
    num_rows = model_joints_num_effort_cts[jid]
    for j in range(num_rows):
        problem_v_b[row_start + j] = -data_joints_dq_b_a[effort_start + j]


@wp.kernel
def _build_free_velocity_bias_limits(
    # Inputs:
    model_time_inv_dt: wp.array[wp.float32],
    data_info_limit_cts_group_offset: wp.array[wp.int32],
    limits_model_max: wp.int32,
    limits_model_num: wp.array[wp.int32],
    limits_wid: wp.array[wp.int32],
    limits_lid: wp.array[wp.int32],
    limits_r_q: wp.array[wp.float32],
    problem_config: wp.array[ConstrainedDynamicsConfigStruct],
    problem_vio: wp.array[wp.int32],
    # Outputs:
    problem_v_b: wp.array[wp.float32],
):
    # Retrieve the limit index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nl = wp.min(limits_model_num[0], limits_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nl:
        return

    # Retrieve the limit entity data
    wid = limits_wid[tid]
    lid = limits_lid[tid]
    r_q = limits_r_q[tid]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid]
    config = problem_config[wid]
    vio = problem_vio[wid]
    lcio = data_info_limit_cts_group_offset[wid]

    # Compute the total constraint index offset of the current contact
    lcio_l = vio + lcio + lid

    # Compute the contact constraint stabilization bias
    problem_v_b[lcio_l] = config.beta * inv_dt * wp.min(0.0, r_q)


@wp.kernel
def _build_free_velocity_bias_contacts(
    # Inputs:
    model_time_inv_dt: wp.array[wp.float32],
    model_info_contacts_offset: wp.array[wp.int32],
    data_info_contact_cts_group_offset: wp.array[wp.int32],
    contacts_model_max: wp.int32,
    contacts_model_num: wp.array[wp.int32],
    contacts_wid: wp.array[wp.int32],
    contacts_cid: wp.array[wp.int32],
    contacts_gapfunc: wp.array[wp.vec4f],
    contacts_material: wp.array[wp.vec2f],
    problem_config: wp.array[ConstrainedDynamicsConfigStruct],
    problem_vio: wp.array[wp.int32],
    # Outputs:
    problem_v_b: wp.array[wp.float32],
    problem_v_i: wp.array[wp.float32],
    problem_mu: wp.array[wp.float32],
):
    # Retrieve the contact index as the thread index
    tid = wp.tid()

    # Retrieve the number of contacts active in the model
    model_nc = wp.min(contacts_model_num[0], contacts_model_max)

    # Skip if cid is greater than the number of contacts active in the world
    if tid >= model_nc:
        return

    # Retrieve the contact entity data
    wid_k = contacts_wid[tid]
    cid_k = contacts_cid[tid]
    material_k = contacts_material[tid]
    distance_k = contacts_gapfunc[tid][3]

    # Retrieve the world-specific data
    inv_dt = model_time_inv_dt[wid_k]
    cio = model_info_contacts_offset[wid_k]
    ccio = data_info_contact_cts_group_offset[wid_k]
    vio = problem_vio[wid_k]
    config = problem_config[wid_k]

    # Compute the total constraint index offset of the current contact
    ccio_k = vio + ccio + 3 * cid_k

    # Compute the total contact index offset of the current contact
    cio_k = cio + cid_k

    # Retrieve the contact material properties
    mu_k = material_k.x  # Friction coefficient
    epsilon_k = material_k.y  # Restitution coefficient

    # The gap-function value (penetration_k) is the margin-shifted
    # signed distance: negative means penetration past the resting
    # separation, zero means at rest, positive means within the
    # detection gap.
    # A dead-zone of config.delta on either side filters out floating-point
    # noise on nearly-touching contacts. Outside the dead-zone, we shift
    # penetration_k by delta to preserve continuity w.r.t. distance_k.
    penetration_k = wp.sign(distance_k) * wp.max(0.0, wp.abs(distance_k) - config.delta)

    # Compute the per-contact penetration error reduction term
    # NOTE#1: Penetrations are represented as penetration_k < 0
    # NOTE#2: xi corresponds to one-sided Baumgarte-like stabilization
    xi = inv_dt * penetration_k
    xi_relaxed = config.gamma * wp.min(0.0, xi) + wp.max(0.0, xi)

    # Gate contact stabilization for restitutive impacts with
    # critical restitution coefficients (i.e. epsilon_k >= 1.0)
    # NOTE: Otherwise the bias would be too large and destabilize the solver
    alpha = wp.where(epsilon_k >= 1.0, 0.0, 1.0)

    # Store the contact constraint stabilization bias in the output vector
    # NOTE: We still write zeros to overwrite previous values
    problem_v_b[ccio_k] = 0.0
    problem_v_b[ccio_k + 1] = 0.0
    problem_v_b[ccio_k + 2] = alpha * xi_relaxed

    # Initialize the restitutive Newton-type impact model term
    # NOTE: We still write zeros to overwrite previous values
    problem_v_i[ccio_k] = 0.0
    problem_v_i[ccio_k + 1] = 0.0
    problem_v_i[ccio_k + 2] = epsilon_k

    # Store the contact friction coefficient in the output vector
    problem_mu[cio_k] = mu_k


@wp.kernel
def _build_free_velocity(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    data_bodies_u_i: wp.array[wp.spatial_vectorf],
    jacobians_J_cts_offsets: wp.array[wp.int32],
    jacobians_J_cts_data: wp.array[wp.float32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_u_f: wp.array[wp.spatial_vectorf],
    problem_v_b: wp.array[wp.float32],
    problem_v_i: wp.array[wp.float32],
    # Outputs:
    problem_v_f: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the problem dimensions and matrix block index offset
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    nb = model_info_bodies_offset[wid + 1] - bio
    cjmio = jacobians_J_cts_offsets[wid]
    vio = problem_vio[wid]

    # Compute the number of Jacobian rows, i.e. the number of body DoFs
    nbd = 6 * nb

    # Compute the thread-specific index offset
    cts_offset = vio + tid

    # Append the column offset to the Jacobian index
    cjmio += nbd * tid

    # Extract the cached impact bias scaling (i.e. restitution coefficient)
    # NOTE: This is a quick hack to avoid multiple kernels. The
    # proper way would be to perform this op only for contacts
    epsilon_j = problem_v_i[cts_offset]

    # Retrieve the cached velocity bias term for the constraint
    v_b_j = problem_v_b[cts_offset]

    # Buffers
    J_i = vec6f(0.0)
    v_f_j = wp.float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    for i in range(nb):
        # Compute the Jacobian block index
        m_ji = cjmio + 6 * i

        # Extract the twist and unconstrained velocity of the body
        u_i = data_bodies_u_i[bio + i]
        u_f_i = problem_u_f[bio + i]

        # Extract the Jacobian block J_ji
        # TODO: use slicing operation when available
        for d in range(6):
            J_i[d] = jacobians_J_cts_data[m_ji + d]

        # Accumulate J_i @ u_i
        v_f_j += wp.dot(J_i, u_f_i)

        # Accumulate the impact bias term
        v_f_j += epsilon_j * wp.dot(J_i, u_i)

    # Store sum of velocity bias terms
    problem_v_f[cts_offset] = v_f_j + v_b_j


@wp.kernel
def _build_free_velocity_sparse(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    state_bodies_u_i: wp.array[wp.spatial_vectorf],
    jac_num_nzb: wp.array[wp.int32],
    jac_nzb_start: wp.array[wp.int32],
    jac_nzb_coords: wp.array2d[wp.int32],
    jac_nzb_values: wp.array[vec6f],
    problem_vio: wp.array[wp.int32],
    problem_u_f: wp.array[wp.spatial_vectorf],
    problem_v_i: wp.array[wp.float32],
    # Outputs:
    problem_v_f: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, nzb_id = wp.tid()

    # Skip if block index exceed the number of blocks
    if nzb_id >= jac_num_nzb[wid]:
        return

    # Retrieve block data
    global_block_idx = jac_nzb_start[wid] + nzb_id
    jac_block_coord = jac_nzb_coords[global_block_idx]
    jac_block = jac_nzb_values[global_block_idx]

    # Retrieve the world-specific data
    bio = model_info_bodies_offset[wid]
    vio = problem_vio[wid]

    # Compute the thread-specific index offset
    thread_offset = vio + jac_block_coord[0]

    # Restitution is nonzero only for contact rows. Reading it for every row
    # keeps this shared kernel single-pass; non-contact rows have epsilon_j = 0.0.
    epsilon_j = problem_v_i[thread_offset]

    # Buffers
    v_f_j = wp.float32(0.0)

    # Iterate over each body to accumulate velocity contributions
    bid = jac_block_coord[1] // 6

    # Extract the twist and unconstrained velocity of the body
    u_i = state_bodies_u_i[bio + bid]
    u_f_i = problem_u_f[bio + bid]

    # Accumulate J_i @ u_i
    v_f_j += wp.dot(jac_block, u_f_i)

    # Accumulate the impact bias term
    v_f_j += epsilon_j * wp.dot(jac_block, u_i)

    # Store sum of velocity bias terms
    wp.atomic_add(problem_v_f, thread_offset, v_f_j)


@wp.kernel
def _build_preconditioner_from_diagonal(
    # Inputs:
    problem_config: wp.array[ConstrainedDynamicsConfigStruct],
    problem_maxdim: wp.array[wp.int32],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_njc: wp.array[wp.int32],
    problem_nbc: wp.array[wp.int32],
    problem_nl: wp.array[wp.int32],
    # Inputs-Outputs:
    problem_P: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Skip if row index exceed the maximum problem size
    if tid >= problem_maxdim[wid]:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Set identity scaling for inactive constraints and worlds without preconditioning
    if tid >= problem_dim[wid] or not problem_config[wid].preconditioning:
        problem_P[vio + tid] = 1.0
        return

    # Infer the start of the contact constraints for this world
    njlc = problem_njc[wid] + problem_nbc[wid] + problem_nl[wid]

    # Convert the operator diagonal into the preconditioner entries in-place
    _build_preconditioner_entry(tid, njlc, vio, problem_P)


@wp.kernel
def _apply_preconditioner_to_vector(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the thread index
    wid, tid = wp.tid()

    # Retrieve the number of active constraints in the world
    ncts = problem_dim[wid]

    # Skip if row index exceed the problem size
    if tid >= ncts:
        return

    # Retrieve the vector index offset of the world
    vio = problem_vio[wid]

    # Compute the global index of the vector entry
    v_i = vio + tid

    # Retrieve the i-th entry of the target vector
    x_i = x[v_i]

    # Retrieve the i-th entry of the diagonal preconditioner
    P_i = problem_P[v_i]

    # Store the preconditioned i-th entry of the vector
    x[v_i] = P_i * x_i


@wp.kernel
def _apply_preconditioner_to_bounds(
    # Inputs:
    problem_nbc: wp.array[wp.int32],
    problem_bcgo: wp.array[wp.int32],
    problem_bcio: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_P: wp.array[wp.float32],
    # Outputs:
    problem_bound_lower: wp.array[wp.float32],
    problem_bound_upper: wp.array[wp.float32],
):
    """Scale bounded-multiplier constraint impulse bounds into preconditioned coordinates."""
    wid, bid = wp.tid()
    if bid >= problem_nbc[wid]:
        return

    row = problem_vio[wid] + problem_bcgo[wid] + bid
    bio = problem_bcio[wid] + bid
    inv_p = 1.0 / problem_P[row]
    problem_bound_lower[bio] *= inv_p
    problem_bound_upper[bio] *= inv_p


###
# Interfaces
###


class ConstrainedDynamicsProblem(ABC):
    """
    Provides a base class to define a common interface for constrained forward dynamics (CFD) problems.

    This class manages the constraint-space data shared by all problem formulations and builds it from
    the current state of the model, limits, contacts and Jacobians. Derived classes realize the inner
    linear operator of the problem in their respective space (see :class:`ProblemFormulation`) and
    implement the following operator contract in preconditioned constraint-space coordinates:

    - :meth:`solve` computes ``x = (P (D + R) P + diag(eta))^{-1} v``
    - :meth:`gemv` computes ``y = alpha * P (D + R) P x + beta * y`` (or without ``P`` if requested)

    where ``eta`` denotes the diagonal regularization held in :attr:`ConstrainedDynamicsProblemData.eta`,
    which is applied to the linear system by :meth:`update_regularization`.
    """

    Config = ConstrainedDynamicsProblemConfig
    """
    Defines a type alias of the per-world problem configurations container.

    See :class:`ConstrainedDynamicsProblemConfig` for the full list of configuration options.
    """

    _data_type: ClassVar[type[ConstrainedDynamicsProblemData]] = ConstrainedDynamicsProblemData
    """The type of the problem data container allocated by the formulation."""

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[ConstrainedDynamicsProblemConfig] | ConstrainedDynamicsProblemConfig | None = None,
        compute_h: bool = False,
        sparse: bool = True,
    ):
        """
        Constructs a constrained forward dynamics problem.

        If a `model` is provided, all data members are allocated immediately, otherwise
        allocation is deferred until :meth:`finalize` is called. The problem always includes
        bilateral (equality) joint constraints and any bounded-multiplier (inequality) constraints
        defined in the model (e.g. joint friction). Joint-limit constraints require the `limits`
        container, and contact constraints require the `contacts` container.

        Args:
            model: The model to build the problem for.
            data: The model data container holding the time-varying state.
            limits: The joint-limits container to use for the problem.
            contacts: The contacts container to use for the problem.
            jacobians: The constraints Jacobians for this model. Must be provided if model is provided.
            solver: The linear solver type to use as back-end of the problem's linear operator.
            solver_kwargs: Additional keyword arguments to pass to the linear solver constructor.
            config: The config for the problem.
                If a single config object is provided, it will be replicated for all worlds.
                Defaults to `None`, indicating that default config will be used for all worlds.
            compute_h: Set to `True` to enable the computation of the nonlinear
                generalized forces vectors in construction of the problem.
            sparse: Set to `True` to use matrix-free sparse operators.
        """
        # Ensure Jacobians are given if model is provided.
        if model is not None and jacobians is None:
            raise ValueError("`jacobians` parameter must be provided if `model` parameter is specified.")

        self._device: wp.DeviceLike = None
        """The device on which the problem data is allocated."""

        self._size: SizeKamino | None = None
        """Host-side cache of the model size meta-data."""

        self._config: list[ConstrainedDynamicsProblemConfig] = []
        """Host-side cache of the list of per world problem configs."""

        self._data: ConstrainedDynamicsProblemData | None = None
        """The problem data container bundling all relevant memory allocations."""

        self._sparse: bool = sparse
        """Flag to indicate whether the problem uses sparse operators."""

        self._compute_h: bool = compute_h
        """Flag to indicate whether the nonlinear generalized forces are computed during builds."""

        self._all_worlds_mask: wp.array[wp.bool] | None = None
        """An array of ones used as a world mask when operating over all worlds."""

        # Finalize the problem data if a model is provided
        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                solver=solver,
                solver_kwargs=solver_kwargs,
                config=config,
                compute_h=compute_h,
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device the problem is allocated on.
        """
        return self._device

    @property
    def size(self) -> SizeKamino:
        """
        Returns the model size of the problem.
        This is the size of the model that the problem is built for.
        """
        if self._size is None:
            raise ValueError("ModelKamino size is not allocated. Call `finalize()` first.")
        return self._size

    @property
    def config(self) -> list[ConstrainedDynamicsProblemConfig]:
        """
        Returns the list of per world problem configs.
        """
        return self._config

    @config.setter
    def config(self, value: list[ConstrainedDynamicsProblemConfig] | ConstrainedDynamicsProblemConfig):
        """
        Sets the list of per world problem configs.
        If a single config object is provided, it will be replicated for all worlds.
        """
        self._config = ConstrainedDynamicsProblemConfig.check_configs(value, self._data.num_worlds)

    @property
    def data(self) -> ConstrainedDynamicsProblemData:
        """
        Returns the problem data container.
        """
        return self._data

    @property
    def sparse(self) -> bool:
        """
        Returns whether the problem is using sparse operators.
        """
        return self._sparse

    @property
    def all_worlds_mask(self) -> wp.array[wp.bool]:
        """
        Returns a world mask selecting all worlds.
        """
        self._assert_is_finalized()
        return self._all_worlds_mask

    @property
    @abstractmethod
    def formulation(self) -> ProblemFormulation:
        """
        Returns the formulation of the constrained forward dynamics problem.
        """
        raise NotImplementedError("The `formulation` property is not implemented.")

    @property
    @abstractmethod
    def linear_solver(self) -> LinearSolverType | None:
        """
        Returns the linear solver used as back-end of the problem's linear operator, if any.
        """
        raise NotImplementedError("The `linear_solver` property is not implemented.")

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        jacobians: SystemJacobiansType,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        solver: LinearSolverType | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        config: list[ConstrainedDynamicsProblemConfig] | ConstrainedDynamicsProblemConfig | None = None,
        compute_h: bool | None = None,
    ):
        """
        Finalizes all memory allocations of the problem data
        for the given model, limits, contacts and Jacobians.

        Args:
            model: The model to build the problem for.
            jacobians: The constraints Jacobians for this model.
            data: The model data container holding the time-varying state.
            limits: The joint-limits container to use for the problem.
            contacts: The contacts container to use for the problem.
            solver: The linear solver type to use as back-end of the problem's linear operator.
            solver_kwargs: Additional keyword arguments to pass to the linear solver constructor.
            config: The config for the problem.
                If a single config object is provided, it will be replicated for all worlds.
                Defaults to `None`, indicating that default config will be used for all worlds.
            compute_h: Set to `True` to enable the computation of the nonlinear generalized forces
                vectors in construction of the problem. If `None`, the current setting is kept.
        """
        # Ensure the provided containers are valid
        assert_containers_are_valid(model=model, data=data, limits=limits, contacts=contacts, data_required=True)
        if jacobians is None:
            raise ValueError("The system Jacobians must be provided to allocate the problem.")
        if self._sparse and not isinstance(jacobians, SparseSystemJacobians):
            raise TypeError("A problem in sparse configuration requires sparse jacobians.")

        # Use the model's device and capture reference to the model size
        self._device = model.device
        self._size = model.size

        # Check config validity and update cache
        self._config = ConstrainedDynamicsProblemConfig.check_configs(config, model.info.num_worlds)
        if compute_h is not None:
            self._compute_h = compute_h

        # Determine the maximum number of contacts supported by the model
        # in order to allocate corresponding per-friction-cone parameters
        model_max_contacts_host = contacts.model_max_contacts_host if contacts is not None else 0

        # Allocate the problem data container holding the shared constraint-space data
        sum_of_max_total_cts = model.size.sum_of_max_total_cts
        with wp.ScopedDevice(self._device):
            self._data = self._data_type(
                # Set the host-side caches of the maximal problem dimensions
                num_worlds=model.size.num_worlds,
                max_of_maxdims=model.size.max_of_max_total_cts,
                # Capture references to the model and data info arrays
                njc=model.info.num_joint_bilateral_cts,
                nbc=model.info.num_joint_bounded_cts,
                nl=data.info.num_limits,
                nc=data.info.num_contacts,
                bcio=model.info.joint_bounded_cts_offset,
                lio=model.info.limits_offset,
                cio=model.info.contacts_offset,
                iio=model.info.inequalities_offset,
                bcgo=model.info.joint_bounded_cts_group_offset,
                lcgo=data.info.limit_cts_group_offset,
                ccgo=data.info.contact_cts_group_offset,
                maxdim=model.info.max_total_cts,
                dim=data.info.num_total_cts,
                vio=model.info.total_cts_offset,
                # Allocate new memory for the remaining problem quantities
                config=wp.array([c.to_struct() for c in self._config], dtype=ConstrainedDynamicsConfigStruct),
                h=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=wp.spatial_vectorf)
                if self._compute_h
                else None,
                u_f=wp.zeros(shape=(model.size.sum_of_num_bodies,), dtype=wp.spatial_vectorf),
                v_b=wp.zeros(shape=(sum_of_max_total_cts,), dtype=wp.float32),
                v_i=wp.zeros(shape=(sum_of_max_total_cts,), dtype=wp.float32),
                v_f=wp.zeros(shape=(sum_of_max_total_cts,), dtype=wp.float32),
                mu=wp.zeros(shape=(model_max_contacts_host,), dtype=wp.float32),
                bound_lower=wp.zeros(shape=(model.size.sum_of_num_bounded_joint_cts,), dtype=wp.float32),
                bound_upper=wp.zeros(shape=(model.size.sum_of_num_bounded_joint_cts,), dtype=wp.float32),
                P=wp.ones(shape=(sum_of_max_total_cts,), dtype=wp.float32),
                eta=wp.zeros(shape=(sum_of_max_total_cts,), dtype=wp.float32),
            )
            self._all_worlds_mask = wp.ones(shape=(model.size.num_worlds,), dtype=wp.bool)

        # Allocate the formulation-specific linear operator and data
        self._finalize_impl(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobians=jacobians,
            solver=solver,
            solver_kwargs=solver_kwargs,
        )

    def zero(self):
        """
        Resets all problem vectors to zero and the preconditioner to identity.
        """
        self._assert_is_finalized()
        if self._compute_h:
            self._data.h.zero_()
        self._data.u_f.zero_()
        self._data.v_b.zero_()
        self._data.v_i.zero_()
        self._data.v_f.zero_()
        self._data.mu.zero_()
        self._data.bound_lower.zero_()
        self._data.bound_upper.zero_()
        self._data.P.fill_(1.0)
        self._zero_impl()

    def build(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SystemJacobiansType,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        reset_to_zero: bool = True,
    ):
        """
        Builds the problem for the given model, data, limits and contacts data.

        Args:
            model: The model to build the problem for.
            data: The model data container holding the time-varying state.
            jacobians: The current constraint Jacobians of the model.
            limits: The joint-limits container holding the active limit constraints.
            contacts: The contacts container holding the active contact constraints.
            reset_to_zero: Whether to reset the problem data before building.
        """
        self._assert_is_finalized()
        if self._sparse and not isinstance(jacobians, SparseSystemJacobians):
            raise TypeError("A problem in sparse configuration requires sparse jacobians.")

        # Initialize problem data
        if reset_to_zero:
            self.zero()

        # Build the formulation-specific linear operator data
        self._build_impl(model=model, data=data, jacobians=jacobians, reset_to_zero=reset_to_zero)

        # Optionally also build the non-linear generalized force vector
        if self._compute_h:
            self._build_nonlinear_generalized_force(model, data)

        # Build the generalized free-velocity vector
        self._build_generalized_free_velocity(model, data)

        # Build the free-velocity bias terms
        self._build_free_velocity_bias(model, data, limits, contacts)

        # Build the free-velocity vector
        self._build_free_velocity(model, data, jacobians)

        # Optionally build and apply the diagonal preconditioner
        if any(s.dynamics.preconditioning for s in self._config):
            self._build_preconditioner()
            self._apply_preconditioner_to_operator_impl()
            self._apply_preconditioner_to_vector(self._data.v_f)
            if model.size.sum_of_num_bounded_joint_cts > 0:
                wp.launch(
                    _apply_preconditioner_to_bounds,
                    dim=(self._size.num_worlds, self._size.max_of_num_bounded_joint_cts),
                    inputs=[
                        # Inputs:
                        self._data.nbc,
                        self._data.bcgo,
                        self._data.bcio,
                        self._data.vio,
                        self._data.P,
                        # Outputs:
                        self._data.bound_lower,
                        self._data.bound_upper,
                    ],
                    device=self.device,
                )

    ###
    # Linear Operator API
    ###

    def update_regularization(self):
        """
        Applies the current diagonal regularization held in :attr:`ConstrainedDynamicsProblemData.eta`
        to the linear system of the problem, and performs any pre-computations (e.g. factorizations)
        that are required by subsequent calls to :meth:`solve`.
        """
        self._assert_is_finalized()
        self._update_regularization_impl()

    def prepare_solve(self):
        """
        Runs any per-step preparation of the linear solver before a (possibly graph-captured) solve loop.

        This is a no-op for linear solvers that do not require any preparation.
        """
        self._assert_is_finalized()
        solver = self.linear_solver
        if isinstance(solver, IterativeSolver):
            solver.prepare_solve()

    def solve(self, v: wp.array[wp.float32], x: wp.array[wp.float32]):
        """
        Solves the regularized linear system ``(P (D + R) P + diag(eta)) x = v`` of each world.

        Args:
            v: The input array holding the right-hand-side vectors in preconditioned coordinates.
                Shape of ``(sum_of_max_total_cts,)``.
            x: The output array to hold the solution vectors in preconditioned coordinates.
                Shape of ``(sum_of_max_total_cts,)``.
        """
        self._assert_is_finalized()
        self._solve_impl(v=v, x=x)

    def gemv(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        alpha: float = 1.0,
        beta: float = 0.0,
        preconditioned: bool = True,
        world_mask: wp.array[wp.bool] | None = None,
    ):
        """
        Computes the generalized matrix-vector product ``y = alpha * A @ x + beta * y`` with the
        compliant constraint-space operator of the problem, excluding the regularization ``eta``.

        Args:
            x: The input array of constraint-space vectors.
            y: The input-output array of constraint-space vectors.
            alpha: The scaling of the matrix-vector product.
            beta: The scaling of the input values of `y`.
            preconditioned: If `True`, the operator is ``A = P (D + R) P`` and the vectors are in
                preconditioned coordinates, otherwise ``A = D + R`` in physical coordinates.
            world_mask: An optional mask selecting the worlds to operate on. Defaults to all worlds.
        """
        self._assert_is_finalized()
        mask = world_mask if world_mask is not None else self._all_worlds_mask
        self._gemv_impl(x=x, y=y, alpha=alpha, beta=beta, preconditioned=preconditioned, world_mask=mask)

    ###
    # Internals
    ###

    def _assert_is_finalized(self):
        """
        Asserts that the problem has been finalized.

        Raises:
            RuntimeError: If the problem data has not been allocated.
        """
        if self._data is None:
            raise RuntimeError("The problem data is not allocated. Call `finalize()` first.")

    def _build_nonlinear_generalized_force(self, model: ModelKamino, data: DataKamino):
        """
        Builds the nonlinear generalized force vector `h`.
        """
        wp.launch(
            _build_nonlinear_generalized_force,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                data.bodies.u_i,
                data.bodies.I_i,
                data.bodies.w_e_i,
                data.bodies.w_a_i,
                # Outputs:
                self._data.h,
            ],
            device=self.device,
        )

    def _build_generalized_free_velocity(self, model: ModelKamino, data: DataKamino):
        """
        Builds the generalized free-velocity vector (i.e. unconstrained) `u_f`.
        """
        wp.launch(
            _build_generalized_free_velocity,
            dim=model.size.sum_of_num_bodies,
            inputs=[
                # Inputs:
                model.time.dt,
                model.gravity.vector,
                model.bodies.wid,
                model.bodies.m_i,
                model.bodies.inv_m_i,
                data.bodies.u_i,
                data.bodies.I_i,
                data.bodies.inv_I_i,
                data.bodies.w_e_i,
                data.bodies.w_a_i,
                # Outputs:
                self._data.u_f,
            ],
            device=self.device,
        )

    def _build_free_velocity_bias(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Assemble per-constraint problem inputs for the current step.

        Primarily builds the free-velocity bias vector ``v_b`` (joint dynamics and
        kinematics, limits, contacts). Also fills auxiliary inequality data that is
        zeroed in :meth:`zero` and consumed later by the solver: joint-friction
        and effort-limit impulse bounds (``bound_lower``, ``bound_upper``),
        effort-row velocity biases in ``v_b``, and contact friction coefficients
        (``mu``).
        """

        if model.size.sum_of_num_joints > 0:
            if model.size.sum_of_num_friction_joint_cts > 0:
                wp.launch(
                    _build_joint_friction_bounds,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.time.dt,
                        model.joints.wid,
                        model.joints.dofs_offset,
                        model.joints.num_friction_cts,
                        model.joints.friction_cts_offset,
                        model.joints.friction_cts_axis,
                        model.info.joint_friction_cts_offset,
                        model.info.joint_bounded_cts_offset,
                        model.joints.f_j,
                        # Outputs:
                        self._data.bound_lower,
                        self._data.bound_upper,
                    ],
                    device=self.device,
                )
            if model.size.sum_of_num_effort_joint_cts > 0:
                wp.launch(
                    _build_joint_effort_bounds,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.joints.wid,
                        model.joints.num_effort_cts,
                        model.joints.effort_cts_offset,
                        model.info.num_joint_friction_cts,
                        model.info.joint_effort_cts_offset,
                        model.info.joint_bounded_cts_offset,
                        data.joints.bound_a,
                        # Outputs:
                        self._data.bound_lower,
                        self._data.bound_upper,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _build_joint_effort_bias,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.joints.num_effort_cts,
                        model.joints.effort_cts_offset,
                        model.joints.effort_cts_offset_total_cts,
                        data.joints.dq_b_a,
                        # Outputs:
                        self._data.v_b,
                    ],
                    device=self.device,
                )
            if model.size.sum_of_num_dynamic_joints > 0:
                wp.launch(
                    _build_free_velocity_bias_joint_dynamics,
                    dim=model.size.sum_of_num_joints,
                    inputs=[
                        # Inputs:
                        model.joints.wid,
                        model.joints.dynamic_cts_offset,
                        model.joints.dynamic_cts_offset_total_cts,
                        data.joints.dq_b_j,
                        # Outputs:
                        self._data.v_b,
                    ],
                    device=self.device,
                )
            wp.launch(
                _build_free_velocity_bias_joint_kinematics,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    model.joints.wid,
                    model.joints.kinematic_cts_offset,
                    model.joints.kinematic_cts_offset_total_cts,
                    data.joints.r_j,
                    self._data.config,
                    # Outputs:
                    self._data.v_b,
                ],
                device=self.device,
            )

        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _build_free_velocity_bias_limits,
                dim=limits.model_max_limits_host,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    data.info.limit_cts_group_offset,
                    limits.model_max_limits_host,
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.r_q,
                    self._data.config,
                    self._data.vio,
                    # Outputs:
                    self._data.v_b,
                ],
                device=self.device,
            )

        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                _build_free_velocity_bias_contacts,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    # Inputs:
                    model.time.inv_dt,
                    model.info.contacts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_max_contacts_host,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.gapfunc,
                    contacts.material,
                    self._data.config,
                    self._data.vio,
                    # Outputs:
                    self._data.v_b,
                    self._data.v_i,
                    self._data.mu,
                ],
                device=self.device,
            )

    def _build_free_velocity(self, model: ModelKamino, data: DataKamino, jacobians: SystemJacobiansType):
        """
        Builds the free-velocity vector `v_f`.
        """
        if isinstance(jacobians, SparseSystemJacobians):
            wp.copy(self._data.v_f, self._data.v_b)
            J_cts = jacobians._J_cts.bsm
            wp.launch(
                _build_free_velocity_sparse,
                dim=(self._size.num_worlds, J_cts.max_of_num_nzb),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    data.bodies.u_i,
                    J_cts.num_nzb,
                    J_cts.nzb_start,
                    J_cts.nzb_coords,
                    J_cts.nzb_values,
                    self._data.vio,
                    self._data.u_f,
                    self._data.v_i,
                    # Outputs:
                    self._data.v_f,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _build_free_velocity,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    # Inputs:
                    model.info.bodies_offset,
                    data.bodies.u_i,
                    jacobians.data.J_cts_offsets,
                    jacobians.data.J_cts_data,
                    self._data.dim,
                    self._data.vio,
                    self._data.u_f,
                    self._data.v_b,
                    self._data.v_i,
                    # Outputs:
                    self._data.v_f,
                ],
                device=self.device,
            )

    def _build_preconditioner(self):
        """
        Builds the diagonal preconditioner `P` from the diagonal of the compliant operator `D + R`.
        """
        self._compute_operator_diagonal_impl(self._data.P)
        wp.launch(
            _build_preconditioner_from_diagonal,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.config,
                self._data.maxdim,
                self._data.dim,
                self._data.vio,
                self._data.njc,
                self._data.nbc,
                self._data.nl,
                # Outputs:
                self._data.P,
            ],
            device=self.device,
        )

    def _apply_preconditioner_to_vector(self, x: wp.array[wp.float32]):
        """
        Applies the diagonal preconditioner `P` to a given constraint-space vector.
        """
        wp.launch(
            _apply_preconditioner_to_vector,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                self._data.dim,
                self._data.vio,
                self._data.P,
                # Outputs:
                x,
            ],
            device=self.device,
        )

    ###
    # Implementation API
    ###

    @abstractmethod
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
        """Allocates the formulation-specific linear operator and data members."""
        raise NotImplementedError("The `_finalize_impl` operation is not implemented.")

    @abstractmethod
    def _zero_impl(self):
        """Resets any formulation-specific data."""
        raise NotImplementedError("The `_zero_impl` operation is not implemented.")

    @abstractmethod
    def _build_impl(self, model: ModelKamino, data: DataKamino, jacobians: SystemJacobiansType, reset_to_zero: bool):
        """Builds the formulation-specific linear operator data for the current state."""
        raise NotImplementedError("The `_build_impl` operation is not implemented.")

    @abstractmethod
    def _compute_operator_diagonal_impl(self, diag: wp.array[wp.float32]):
        """Stores the diagonal of the unpreconditioned compliant operator `D + R` in the given array."""
        raise NotImplementedError("The `_compute_operator_diagonal_impl` operation is not implemented.")

    @abstractmethod
    def _apply_preconditioner_to_operator_impl(self):
        """Applies the preconditioner `P` to any explicitly stored operator data."""
        raise NotImplementedError("The `_apply_preconditioner_to_operator_impl` operation is not implemented.")

    @abstractmethod
    def _update_regularization_impl(self):
        """Applies the regularization `eta` to the linear system and pre-computes any solve data."""
        raise NotImplementedError("The `_update_regularization_impl` operation is not implemented.")

    @abstractmethod
    def _solve_impl(self, v: wp.array[wp.float32], x: wp.array[wp.float32]):
        """Solves the regularized linear system in preconditioned constraint-space coordinates."""
        raise NotImplementedError("The `_solve_impl` operation is not implemented.")

    @abstractmethod
    def _gemv_impl(
        self,
        x: wp.array[wp.float32],
        y: wp.array[wp.float32],
        alpha: float,
        beta: float,
        preconditioned: bool,
        world_mask: wp.array[wp.bool],
    ):
        """Computes the generalized matrix-vector product with the unregularized compliant operator."""
        raise NotImplementedError("The `_gemv_impl` operation is not implemented.")
