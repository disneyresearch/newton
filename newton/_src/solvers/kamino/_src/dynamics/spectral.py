# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides batched linear operators of the forward dynamics problems for spectral estimates, e.g.
with :class:`PowerIteration`:

- :func:`make_problem_operator`: The compliant constraint-space operator ``P (D + R) P`` (or
  ``D + R``) of any :class:`ConstrainedDynamicsProblem`, realized with :meth:`ConstrainedDynamicsProblem.gemv`,
  i.e. using the (dense or matrix-free) Delassus matrix of a :class:`DualProblem`, or the matrix-free
  product ``J M^{-1} J^T`` of a :class:`PrimalProblem`.
- :func:`make_problem_solve_fn`: The regularized solve ``x = (P (D + R) P + diag(eta))^{-1} v`` of a
  problem, e.g. for inverse iteration with an existing factorization.
- :func:`make_mass_operator`: The block-diagonal generalized mass matrix ``M`` of a model.
- :func:`make_augmented_mass_operator`: The constraint-augmented generalized mass matrix
  ``M + J^T W J`` of a :class:`PrimalProblem`.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..linalg.conjugate import BatchedLinearOperator
from .operators import mass_gemv
from .primal import PrimalProblem
from .problem import ConstrainedDynamicsProblem

###
# Module interface
###

__all__ = [
    "make_augmented_mass_operator",
    "make_mass_operator",
    "make_problem_operator",
    "make_problem_solve_fn",
]


###
# Functions
###


def make_problem_operator(
    problem: ConstrainedDynamicsProblem, preconditioned: bool = True
) -> BatchedLinearOperator[wp.float32, wp.int32]:
    """
    Creates the batched constraint-space operator of a forward dynamics problem.

    Args:
        problem: The finalized forward dynamics problem.
        preconditioned: Whether the operator is ``P (D + R) P`` (`True`) or ``D + R`` (`False`).

    Returns:
        The batched operator realizing products with the compliant constraint-space operator of the
        problem, excluding its diagonal regularization ``eta``.
    """
    data = problem.data

    def gemv_fn(x, y, world_active, alpha, beta):
        problem.gemv(x=x, y=y, alpha=alpha, beta=beta, preconditioned=preconditioned, world_mask=world_active)

    return BatchedLinearOperator[wp.float32, wp.int32](
        gemv_fn=gemv_fn,
        n_worlds=data.num_worlds,
        max_dim=data.max_of_maxdims,
        active_dims=data.dim,
        device=problem.device,
        dtype=wp.float32,
        vio=data.vio,
        total_vec_size=problem.size.sum_of_max_total_cts,
    )


def make_problem_solve_fn(
    problem: ConstrainedDynamicsProblem,
) -> Callable[[wp.array[wp.float32], wp.array[wp.float32]], None]:
    """
    Creates a function solving linear systems with the regularized constraint-space operator of a problem.

    Args:
        problem: The finalized forward dynamics problem, whose regularization has been applied with
            :meth:`ConstrainedDynamicsProblem.update_regularization`.

    Returns:
        The function ``solve_fn(b, x)`` computing ``x = (P (D + R) P + diag(eta))^{-1} b``.
    """

    def solve_fn(b: wp.array[wp.float32], x: wp.array[wp.float32]):
        problem.solve(v=b, x=x)

    return solve_fn


def make_mass_operator(model: ModelKamino, data: DataKamino) -> BatchedLinearOperator[wp.float32, wp.int32]:
    """
    Creates the batched block-diagonal generalized mass matrix operator of a model.

    Args:
        model: The model providing the body masses and body-space layout.
        data: The model data providing the current body inertias.

    Returns:
        The batched operator realizing products with the generalized mass matrix ``M``.
    """

    def gemv_fn(x, y, world_active, alpha, beta):
        mass_gemv(model, data, x, y, world_active, alpha=alpha, beta=beta)

    return BatchedLinearOperator[wp.float32, wp.int32](
        gemv_fn=gemv_fn,
        n_worlds=model.size.num_worlds,
        max_dim=model.size.max_of_num_body_dofs,
        active_dims=model.info.num_body_dofs,
        device=model.device,
        dtype=wp.float32,
        vio=model.info.body_dofs_offset,
        total_vec_size=model.size.sum_of_num_body_dofs,
    )


def make_augmented_mass_operator(problem: PrimalProblem) -> BatchedLinearOperator[wp.float32, wp.int32]:
    """
    Creates the batched constraint-augmented generalized mass matrix operator of a primal problem.

    Args:
        problem: The finalized primal problem, whose regularization has been applied with
            :meth:`PrimalProblem.update_regularization`.

    Returns:
        The batched operator realizing products with ``M + J^T W J``.
    """
    if not isinstance(problem, PrimalProblem):
        raise TypeError(f"Expected a `PrimalProblem`, got {type(problem)}.")
    model = problem._model

    def gemv_fn(x, y, world_active, alpha, beta):
        problem.augmented_mass_gemv(x=x, y=y, world_mask=world_active, alpha=alpha, beta=beta)

    return BatchedLinearOperator[wp.float32, wp.int32](
        gemv_fn=gemv_fn,
        n_worlds=model.size.num_worlds,
        max_dim=model.size.max_of_num_body_dofs,
        active_dims=model.info.num_body_dofs,
        device=model.device,
        dtype=wp.float32,
        vio=model.info.body_dofs_offset,
        total_vec_size=model.size.sum_of_num_body_dofs,
    )
