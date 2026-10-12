# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a batched power-iteration algorithm to estimate the extreme eigenvalues of per-world
symmetric positive semi-definite linear operators, e.g. the Delassus matrix ``D = J M^{-1} J^T``
or the generalized mass matrix ``M`` of each world.

The operators are given as :class:`BatchedLinearOperator` instances, which can wrap dense or
block-sparse matrices as well as matrix-free matrix-vector products, e.g. ``x -> D @ x``.

- The largest eigenvalue ``lambda_max(A)`` is estimated by power iteration on ``A``.
- The smallest eigenvalue ``lambda_min(A)`` is estimated either by power iteration on the shifted
  operator ``sigma I - A`` with ``sigma >= lambda_max(A)`` (no factorization required), or by
  inverse iteration, i.e. power iteration on ``(A + s I)^{-1}``, given a function that solves
  linear systems with ``A + s I`` for a known diagonal shift ``s`` (e.g. using an existing
  factorization of the regularized operator).

Each estimate is the Rayleigh quotient of the normalized iterate. The iterates are initialized with
deterministic pseudo-random vectors, and are kept between calls to serve as the initial guesses of
subsequent estimations, which is effective when the operators change slowly, e.g. across steps.

Typical usage example:
    # Create the power iteration for an operator, e.g. the Delassus matrix of a dual problem
    pi = PowerIteration(operator=BatchedLinearOperator.from_dense(delassus.operator))

    # Estimate the extreme eigenvalues of each world
    lambda_max = pi.compute_max()
    lambda_min = pi.compute_min(method="shifted")
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import warp as wp

from .conjugate import BatchedLinearOperator, dot_sequential, make_dot_kernel

###
# Module interface
###

__all__ = [
    "PowerIteration",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


INITIAL_VECTOR_SEED = wp.constant(wp.int32(0x5EED))
"""
The seed of the pseudo-random initial iterates. Unlike structured vectors, e.g. of all ones, which are
often orthogonal to the extreme eigenvectors of physical operators (e.g. the null space of a hyperstatic
Delassus matrix), pseudo-random iterates have non-zero components along all eigenvectors.
"""


@wp.kernel
def _reset_vector(
    # Inputs:
    vio: wp.array[wp.int32],
    dim: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and entry indices as the thread indices
    wid, i = wp.tid()

    # Skip if the world is inactive or the entry exceeds the active dimension
    if not world_mask[wid] or i >= dim[wid]:
        return

    # Reset the entry to a deterministic pseudo-random value in [0.5, 1.5)
    v_i = vio[wid] + i
    state = wp.rand_init(INITIAL_VECTOR_SEED, v_i)
    x[v_i] = 0.5 + wp.randf(state)


@wp.kernel
def _normalize_vector(
    # Inputs:
    vio: wp.array[wp.int32],
    dim: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    norm_sq: wp.array2d[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and entry indices as the thread indices
    wid, i = wp.tid()

    # Skip if the world is inactive or the entry exceeds the active dimension
    n = dim[wid]
    if not world_mask[wid] or i >= n:
        return

    # Normalize the vector, or restart from a pseudo-random vector if it vanishes
    v_i = vio[wid] + i
    nsq = norm_sq[0, wid]
    if nsq > 0.0:
        x[v_i] = x[v_i] / wp.sqrt(nsq)
    else:
        state = wp.rand_init(INITIAL_VECTOR_SEED, v_i)
        x[v_i] = (0.5 + wp.randf(state)) / wp.sqrt(wp.float32(n))


@wp.kernel
def _scale_vector(
    # Inputs:
    vio: wp.array[wp.int32],
    dim: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    scale: wp.array[wp.float32],
    x: wp.array[wp.float32],
    # Outputs:
    y: wp.array[wp.float32],
):
    # Retrieve the world and entry indices as the thread indices
    wid, i = wp.tid()

    # Skip if the world is inactive or the entry exceeds the active dimension
    if not world_mask[wid] or i >= dim[wid]:
        return

    # Compute the per-world scaled vector: y = s * x
    v_i = vio[wid] + i
    y[v_i] = scale[wid] * x[v_i]


@wp.kernel
def _update_estimate(
    # Inputs:
    iteration: wp.int32,
    tolerance: wp.float32,
    dim: wp.array[wp.int32],
    dots: wp.array2d[wp.float32],
    # Outputs:
    estimate: wp.array[wp.float32],
    inv_norm: wp.array[wp.float32],
    iterations: wp.array[wp.int32],
    active: wp.array[wp.bool],
    stepped: wp.array[wp.bool],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Skip if the world has converged or is inactive
    stepped[wid] = active[wid]
    if not active[wid]:
        return

    # Worlds with an empty operator have no eigenvalues, which are reported as zero
    if dim[wid] == 0:
        estimate[wid] = 0.0
        inv_norm[wid] = 0.0
        active[wid] = False
        stepped[wid] = False
        return

    # The Rayleigh quotient of the normalized iterate `x` is `x^T y` with `y = A x`
    value = dots[0, wid]
    y_norm_sq = dots[1, wid]
    inv_norm[wid] = wp.where(y_norm_sq > 0.0, 1.0 / wp.sqrt(y_norm_sq), 0.0)

    # Check convergence of the estimate w.r.t. its relative change
    if iteration > 0 and wp.abs(value - estimate[wid]) <= tolerance * wp.abs(value):
        active[wid] = False
    estimate[wid] = value
    iterations[wid] = iteration + 1


@wp.kernel
def _normalize_iterate(
    # Inputs:
    vio: wp.array[wp.int32],
    dim: wp.array[wp.int32],
    stepped: wp.array[wp.bool],
    inv_norm: wp.array[wp.float32],
    y: wp.array[wp.float32],
    # Outputs:
    x: wp.array[wp.float32],
):
    # Retrieve the world and entry indices as the thread indices
    wid, i = wp.tid()

    # Skip if the world has not been stepped or the entry exceeds the active dimension
    if not stepped[wid] or i >= dim[wid]:
        return

    # Update the iterate with the normalized product: x = y / ||y||
    v_i = vio[wid] + i
    x[v_i] = inv_norm[wid] * y[v_i]


@wp.kernel
def _finalize_shifted_estimate(
    # Inputs:
    world_mask: wp.array[wp.bool],
    shift: wp.array[wp.float32],
    estimate: wp.array[wp.float32],
    # Outputs:
    result: wp.array[wp.float32],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Skip if the world is inactive
    if not world_mask[wid]:
        return

    # The largest eigenvalue of `sigma I - A` is `sigma - lambda_min(A)`
    result[wid] = shift[wid] - estimate[wid]


@wp.kernel
def _finalize_inverse_estimate(
    # Inputs:
    world_mask: wp.array[wp.bool],
    shift: wp.array[wp.float32],
    estimate: wp.array[wp.float32],
    # Outputs:
    result: wp.array[wp.float32],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Skip if the world is inactive
    if not world_mask[wid]:
        return

    # The largest eigenvalue of `(A + s I)^{-1}` is `1 / (lambda_min(A) + s)`
    mu = estimate[wid]
    result[wid] = wp.where(mu > 0.0, 1.0 / mu, 0.0) - shift[wid]


@wp.kernel
def _copy_estimate(
    # Inputs:
    world_mask: wp.array[wp.bool],
    estimate: wp.array[wp.float32],
    # Outputs:
    result: wp.array[wp.float32],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Skip if the world is inactive
    if not world_mask[wid]:
        return

    # Store the estimate of the largest eigenvalue
    result[wid] = estimate[wid]


###
# Interfaces
###


class PowerIteration:
    """
    A batched power-iteration algorithm estimating the extreme eigenvalues of per-world symmetric
    positive semi-definite linear operators.

    All operations are parallelized over the worlds and the entries of the per-world vectors, and
    use a fixed maximum number of iterations, so that they can be captured in CUDA graphs. Worlds
    whose estimates have converged, i.e. whose relative change falls below the tolerance, skip all
    subsequent iterations.
    """

    def __init__(
        self,
        operator: BatchedLinearOperator[wp.float32, wp.int32] | None = None,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        early_exit: bool = True,
    ):
        """
        Creates a batched power iteration.

        Args:
            operator: The batched symmetric positive semi-definite operator of which to estimate the eigenvalues.
            max_iterations: The default maximum number of iterations of each estimation.
            tolerance: The relative tolerance on the change of the estimates between iterations.
            early_exit: Whether to stop iterating once all worlds have converged. This requires a
                device-to-host synchronization per iteration, and is skipped during graph capture.
        """
        if max_iterations <= 0:
            raise ValueError(f"Invalid max_iterations: {max_iterations}. Must be greater than zero.")
        if tolerance < 0.0:
            raise ValueError(f"Invalid tolerance: {tolerance}. Must be non-negative.")
        self._max_iterations: int = int(max_iterations)
        self._tolerance: float = float(tolerance)
        self._early_exit: bool = bool(early_exit)
        self._operator: BatchedLinearOperator[wp.float32, wp.int32] | None = None
        self._device: wp.DeviceLike = None
        self._num_worlds: int = 0
        self._max_dim: int = 0
        self._x_max: wp.array[wp.float32] | None = None
        self._x_min: wp.array[wp.float32] | None = None
        self._y: wp.array[wp.float32] | None = None
        self._lambda_max: wp.array[wp.float32] | None = None
        self._lambda_min: wp.array[wp.float32] | None = None
        self._estimate: wp.array[wp.float32] | None = None
        self._shift: wp.array[wp.float32] | None = None
        self._inv_norm: wp.array[wp.float32] | None = None
        self._iterations: wp.array[wp.int32] | None = None
        self._active: wp.array[wp.bool] | None = None
        self._stepped: wp.array[wp.bool] | None = None
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._dots: wp.array2d[wp.float32] | None = None
        self._dot_partial_sums: wp.array3d[wp.float32] | None = None
        self._dot_tile_size: int = 0
        self._dot_kernel: wp.Kernel | None = None
        if operator is not None:
            self.finalize(operator)

    ###
    # Properties
    ###

    @property
    def operator(self) -> BatchedLinearOperator[wp.float32, wp.int32]:
        """Returns the batched operator of which the eigenvalues are estimated."""
        return self._operator

    @property
    def max_iterations(self) -> int:
        """Returns the default maximum number of iterations of each estimation."""
        return self._max_iterations

    @property
    def tolerance(self) -> float:
        """Returns the relative tolerance on the change of the estimates between iterations."""
        return self._tolerance

    @property
    def lambda_max(self) -> wp.array[wp.float32]:
        """Returns the per-world estimates of the largest eigenvalue. Shape of ``(num_worlds,)``."""
        return self._lambda_max

    @property
    def lambda_min(self) -> wp.array[wp.float32]:
        """Returns the per-world estimates of the smallest eigenvalue. Shape of ``(num_worlds,)``."""
        return self._lambda_min

    @property
    def eigenvector_max(self) -> wp.array[wp.float32]:
        """Returns the normalized per-world iterates of the eigenvectors of the largest eigenvalues."""
        return self._x_max

    @property
    def eigenvector_min(self) -> wp.array[wp.float32]:
        """Returns the normalized per-world iterates of the eigenvectors of the smallest eigenvalues."""
        return self._x_min

    @property
    def iterations(self) -> wp.array[wp.int32]:
        """Returns the per-world number of iterations performed by the last estimation."""
        return self._iterations

    ###
    # Operations
    ###

    def finalize(self, operator: BatchedLinearOperator[wp.float32, wp.int32]):
        """
        Allocates the data of the power iteration for the given operator.

        Args:
            operator: The batched symmetric positive semi-definite operator of which to estimate the eigenvalues.
        """
        if not isinstance(operator, BatchedLinearOperator):
            raise TypeError(f"Expected a `BatchedLinearOperator`, got {type(operator)}.")
        if operator.vio is None or operator.total_vec_size <= 0:
            raise ValueError("The operator must define the vector index offsets and the total vector size.")
        self._operator = operator
        self._device = operator.device
        self._num_worlds = operator.n_worlds
        self._max_dim = operator.max_dim
        n_worlds = self._num_worlds
        with wp.ScopedDevice(self._device):
            self._x_max = wp.zeros(shape=(operator.total_vec_size,), dtype=wp.float32)
            self._x_min = wp.zeros(shape=(operator.total_vec_size,), dtype=wp.float32)
            self._y = wp.zeros(shape=(operator.total_vec_size,), dtype=wp.float32)
            self._lambda_max = wp.zeros(shape=(n_worlds,), dtype=wp.float32)
            self._lambda_min = wp.zeros(shape=(n_worlds,), dtype=wp.float32)
            self._estimate = wp.zeros(shape=(n_worlds,), dtype=wp.float32)
            self._shift = wp.zeros(shape=(n_worlds,), dtype=wp.float32)
            self._inv_norm = wp.zeros(shape=(n_worlds,), dtype=wp.float32)
            self._iterations = wp.zeros(shape=(n_worlds,), dtype=wp.int32)
            self._active = wp.zeros(shape=(n_worlds,), dtype=wp.bool)
            self._stepped = wp.zeros(shape=(n_worlds,), dtype=wp.bool)
            self._all_worlds_mask = wp.ones(shape=(n_worlds,), dtype=wp.bool)
            if self._tiled_dot_product:
                self._dot_tile_size = min(2048, 2 ** math.ceil(math.log(max(self._max_dim, 2), 2)))
                self._dot_kernel = make_dot_kernel(self._dot_tile_size, self._max_dim)
                self._dots = wp.zeros(shape=(2, n_worlds), dtype=wp.float32)
            else:
                self._dot_partial_sums = wp.zeros(shape=(2, n_worlds, (self._max_dim + 1) // 2), dtype=wp.float32)
                self._dots = self._dot_partial_sums[:, :, 0]
        self.reset()

    def reset(self, world_mask: wp.array[wp.bool] | None = None):
        """
        Resets the iterates of the selected worlds to deterministic pseudo-random vectors.

        Args:
            world_mask: The optional mask selecting the worlds to reset. Defaults to all worlds.
        """
        mask = self._all_worlds_mask if world_mask is None else world_mask
        for x in (self._x_max, self._x_min):
            self._launch_entries(_reset_vector, inputs=[mask], outputs=[x])

    def compute_max(
        self,
        world_mask: wp.array[wp.bool] | None = None,
        max_iterations: int | None = None,
    ) -> wp.array[wp.float32]:
        """
        Estimates the largest eigenvalue of the operator of each selected world by power iteration.

        Args:
            world_mask: The optional mask selecting the worlds to operate on. Defaults to all worlds.
            max_iterations: The maximum number of iterations. Defaults to :attr:`max_iterations`.

        Returns:
            The per-world estimates of the largest eigenvalue, i.e. :attr:`lambda_max`.
        """
        mask = self._all_worlds_mask if world_mask is None else world_mask

        def apply(x: wp.array[wp.float32], y: wp.array[wp.float32], active: wp.array[wp.bool]):
            self._operator.gemv(x, y, active, 1.0, 0.0)

        self._iterate(apply, self._x_max, mask, max_iterations)
        wp.launch(
            _copy_estimate,
            dim=self._num_worlds,
            inputs=[mask, self._estimate],
            outputs=[self._lambda_max],
            device=self._device,
        )
        return self._lambda_max

    def compute_min(
        self,
        method: Literal["shifted", "inverse"] = "shifted",
        shift: wp.array[wp.float32] | None = None,
        solve_fn: Callable[[wp.array[wp.float32], wp.array[wp.float32]], None] | None = None,
        world_mask: wp.array[wp.bool] | None = None,
        max_iterations: int | None = None,
    ) -> wp.array[wp.float32]:
        """
        Estimates the smallest eigenvalue of the operator of each selected world.

        Args:
            method: The estimation method:\n
                - ``"shifted"``: Power iteration on ``sigma I - A``, with the per-world ``shift``
                  ``sigma >= lambda_max(A)``, defaulting to the current :attr:`lambda_max` estimates.\n
                - ``"inverse"``: Power iteration on ``(A + s I)^{-1}``, applied with `solve_fn`,
                  with the per-world diagonal ``shift`` ``s`` already included in the solves,
                  defaulting to zero.
            shift: The optional per-world shift of the method. Shape of ``(num_worlds,)``.
            solve_fn: The function ``solve_fn(b, x)`` computing ``x = (A + s I)^{-1} b`` for all
                worlds, required by the ``"inverse"`` method.
            world_mask: The optional mask selecting the worlds to operate on. Defaults to all worlds.
            max_iterations: The maximum number of iterations. Defaults to :attr:`max_iterations`.

        Returns:
            The per-world estimates of the smallest eigenvalue, i.e. :attr:`lambda_min`.
        """
        mask = self._all_worlds_mask if world_mask is None else world_mask
        if method == "shifted":
            wp.copy(self._shift, self._lambda_max if shift is None else shift)

            def apply(x: wp.array[wp.float32], y: wp.array[wp.float32], active: wp.array[wp.bool]):
                self._launch_entries(_scale_vector, inputs=[active, self._shift, x], outputs=[y])
                self._operator.gemv(x, y, active, -1.0, 1.0)

            finalize_kernel = _finalize_shifted_estimate
        elif method == "inverse":
            if solve_fn is None:
                raise ValueError("The `inverse` method requires a `solve_fn`.")
            if shift is None:
                self._shift.zero_()
            else:
                wp.copy(self._shift, shift)

            def apply(x: wp.array[wp.float32], y: wp.array[wp.float32], active: wp.array[wp.bool]):
                solve_fn(x, y)

            finalize_kernel = _finalize_inverse_estimate
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of ('shifted', 'inverse').")

        self._iterate(apply, self._x_min, mask, max_iterations)
        wp.launch(
            finalize_kernel,
            dim=self._num_worlds,
            inputs=[mask, self._shift, self._estimate],
            outputs=[self._lambda_min],
            device=self._device,
        )
        return self._lambda_min

    ###
    # Internals
    ###

    @property
    def _tiled_dot_product(self) -> bool:
        return wp.get_device(self._device).is_cuda

    def _launch_entries(self, kernel: wp.Kernel, inputs: list, outputs: list):
        """Launches a kernel over the worlds and the entries of the per-world vectors."""
        op = self._operator
        wp.launch(
            kernel,
            dim=(self._num_worlds, self._max_dim),
            inputs=[op.vio, op.active_dims, *inputs],
            outputs=outputs,
            device=self._device,
        )

    def _dot(self, a: wp.array[wp.float32], b: wp.array[wp.float32], col: int, world_mask: wp.array[wp.bool]):
        """Computes the per-world dot products ``a^T b`` into the column `col` of the dot-product results."""
        op = self._operator
        a2 = a.reshape((1, a.shape[0]))
        b2 = b.reshape((1, b.shape[0]))
        if self._tiled_dot_product:
            wp.launch_tiled(
                self._dot_kernel,
                dim=(1, self._num_worlds),
                block_dim=max(1, min(256, self._dot_tile_size // 8)),
                inputs=[a2, b2, op.vio, op.active_dims, world_mask],
                outputs=[self._dots[col:]],
                device=self._device,
            )
        else:
            wp.launch(
                dot_sequential,
                dim=(1, self._num_worlds),
                inputs=[a2, b2, op.vio, op.active_dims, world_mask],
                outputs=[self._dot_partial_sums[col:]],
                device=self._device,
            )

    def _iterate(
        self,
        apply: Callable[[wp.array[wp.float32], wp.array[wp.float32], wp.array[wp.bool]], None],
        x: wp.array[wp.float32],
        world_mask: wp.array[wp.bool],
        max_iterations: int | None,
    ):
        """Runs the power iteration with the operator realized by `apply`, starting from the iterate `x`."""
        max_iterations = self._max_iterations if max_iterations is None else int(max_iterations)

        # Normalize the initial iterates, restarting from pseudo-random vectors if they vanish,
        # which are only normalized by the second pass
        for _ in range(2):
            self._dot(x, x, 0, world_mask)
            self._launch_entries(_normalize_vector, inputs=[world_mask, self._dots], outputs=[x])

        # Initialize the per-world iteration state
        wp.copy(self._active, world_mask)
        self._iterations.zero_()
        self._estimate.zero_()

        early_exit = self._early_exit and not wp.get_device(self._device).is_capturing
        for k in range(max_iterations):
            # Apply the operator: y = A x
            apply(x, self._y, self._active)

            # Compute the Rayleigh quotient and the norm of the product, and check convergence
            self._dot(x, self._y, 0, self._active)
            self._dot(self._y, self._y, 1, self._active)
            wp.launch(
                _update_estimate,
                dim=self._num_worlds,
                inputs=[wp.int32(k), wp.float32(self._tolerance), self._operator.active_dims, self._dots],
                outputs=[self._estimate, self._inv_norm, self._iterations, self._active, self._stepped],
                device=self._device,
            )

            # Update the iterates with the normalized products
            self._launch_entries(_normalize_iterate, inputs=[self._stepped, self._inv_norm, self._y], outputs=[x])

            # Stop once all worlds have converged
            if early_exit and not self._active.numpy().any():
                break
