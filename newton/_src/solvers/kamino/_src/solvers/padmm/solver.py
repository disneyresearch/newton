# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the Proximal-ADMM solver class.

This is the highest-level interface for the Proximal-ADMM solver for constrained rigid multi-body systems.

See the :mod:`newton._src.solvers.kamino.solvers.padmm` module for a detailed description and usage example.
"""

from __future__ import annotations

import math

import warp as wp

from ....config import PADMMPrimalSolverConfig, PADMMSolverConfig
from ...core.data import DataKamino
from ...core.math import FLOAT32_MAX
from ...core.model import ModelKamino
from ...core.size import SizeKamino
from ...dynamics.problem import ConstrainedDynamicsProblem
from ...dynamics.spectral import make_problem_operator, make_problem_solve_fn
from ...geometry.contacts import ContactsKamino
from ...kinematics.limits import LimitsKamino
from ...linalg.eigen import PowerIteration
from ...linalg.linear import IterativeSolver
from ...utils.tile import get_block_dim, get_tile_size
from .kernels import (
    _apply_dual_preconditioner_to_solution,
    _apply_dual_preconditioner_to_state,
    _compute_complementarity_residuals,
    _compute_desaxce_correction,
    _compute_final_desaxce_correction,
    _compute_projection_argument,
    _compute_solution_vectors,
    _compute_spectral_lower_bound,
    _compute_spectral_shift,
    _compute_velocity_bias,
    _initialize_spectral_penalty,
    _make_compute_infnorm_residuals_kernel,
    _make_project_dual_convergence_accel_kernel,
    _project_to_feasible_cone,
    _project_warmstart_velocity,
    _refine_spectral_lower_bound,
    _reset_penalty,
    _reset_solver_data,
    _scale_warmstart_forces,
    _update_problem_proximal_regularization,
    _update_problem_spectral_regularization,
    _update_spectral_penalty,
    _warmstart_contact_constraints,
    _warmstart_desaxce_correction,
    _warmstart_joint_constraints,
    _warmstart_limit_constraints,
    make_collect_solver_info_kernel,
    make_desaxce_correction_and_velocity_bias_kernel,
    make_initialize_solver_kernel,
    make_update_dual_variables_and_compute_primal_dual_residuals,
)
from .types import (
    PADMMConfigStruct,
    PADMMData,
    PADMMPenaltyUpdate,
    PADMMWarmStartMode,
    convert_config_to_struct,
)

###
# Module interface
###

__all__ = ["PADMMSolver"]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class PADMMSolver:
    """
    The Proximal-ADMM (PADMM) forward dynamics solver for constrained rigid multi-body systems.

    This solver implements the Proximal-ADMM algorithm to solve the constrained forward
    dynamics problem in constraint reactions (i.e. impulses) and post-event constraint-space
    velocities.

    Notes:
    - operates on any :class:`ConstrainedDynamicsProblem` formulation, whose linear operator
      realizes the regularized solve `x = (P (D + R) P + (eta + rho) I)^{-1} v` in its own space.
    - the iterates are those of the Lagrange dual of the constrained forward dynamics problem,
      which are identical to those of the primal PADMM with penalty `1 / rho`.
    - is based on the Proximal-ADMM algorithm, which introduces a proximal regularization term
    - supports multiple penalty update methods, including fixed, balanced-residuals, and spectral updates.
    - can be configured with various tolerances, penalty parameters, and other settings.
    """

    Config = PADMMSolverConfig
    """
    Defines a type alias of the PADMM solver configurations container, including convergence
    criteria, maximum iterations, and options for the linear solver and preconditioning.

    See :class:`PADMMSolverConfig` for the full list of configuration options and their descriptions.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        config: list[PADMMSolver.Config] | PADMMSolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        use_graph_conditionals: bool = True,
        collect_info: bool = False,
    ):
        """
        Initializes a PADMM solver.

        If a model is provided, it will perform all necessary memory allocations on the
        target device, otherwise the user must call `finalize()` before using the solver.

        Args:
            model: The model for which to allocate the solver data.
            limits: The limits container associated with the model.
            contacts: The contacts container associated with the model.
            config: The solver config to use.
            warmstart: The warm-start mode to use for the solver.
            use_acceleration: Set to `True` to enable Nesterov acceleration.
            use_graph_conditionals: Set to `False` to disable CUDA graph conditional nodes.
                When disabled, replaces `wp.capture_while` with an unrolled for-loop over max iterations.
            collect_info: Set to `True` to enable collection of solver convergence info.
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
        """

        # Declare the internal solver config cache
        self._config: list[PADMMSolver.Config] = []
        self._warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE
        self._use_acceleration: bool = True
        self._use_adaptive_penalty: bool = False
        self._use_graph_conditionals: bool = True
        self._collect_info: bool = False

        # Declare the spectral penalty data
        self._use_spectral_penalty: bool = False
        self._spectral_update_freq: int = 0
        self._spectral_max_iterations: int = 0
        self._spectral_tolerance: float = 0.0
        self._spectral_mask: wp.array[wp.bool] | None = None
        self._spectral_lower_bound: wp.array[wp.float32] | None = None
        self._spectral_power_iteration: PowerIteration | None = None
        self._spectral_problem: ConstrainedDynamicsProblem | None = None
        self._use_spectral_inverse: bool = False
        self._spectral_inverse_iterations: int = 0
        self._spectral_inverse_mask: wp.array[wp.bool] | None = None
        self._spectral_shift: wp.array[wp.float32] | None = None

        # Declare the model size cache
        self._size: SizeKamino | None = None

        # Declare the solver data container
        self._data: PADMMData | None = None
        self._all_worlds_mask: wp.array[wp.bool] | None = None

        # Declare the data of the velocity-consistent warm-start
        self._use_consistent_warmstart_velocity: bool = False
        self._consistent_warmstart_mask: wp.array[wp.bool] | None = None
        self._warmstart_sigma: wp.array[wp.float32] | None = None

        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Perform memory allocations if a model is provided
        if model is not None:
            self.finalize(
                model=model,
                config=config,
                warmstart=warmstart,
                use_acceleration=use_acceleration,
                use_graph_conditionals=use_graph_conditionals,
                collect_info=collect_info,
            )

    ###
    # Properties
    ###

    @property
    def config(self) -> list[PADMMSolver.Config]:
        """
        Returns the host-side cache of the solver config.
        They are used to construct the warp array of type :class:`PADMMSolver.Config` on the target device.
        """
        return self._config

    @property
    def size(self) -> SizeKamino:
        """
        Returns the host-side cache of the solver allocation sizes.
        """
        return self._size

    @property
    def data(self) -> PADMMData:
        """
        Returns a reference to the high-level solver data container.
        """
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `finalize()` first.")
        return self._data

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device on which the solver data is allocated.
        """
        return self._device

    ###
    # Public API
    ###

    def finalize(
        self,
        model: ModelKamino | None = None,
        config: list[PADMMSolver.Config] | PADMMSolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        use_graph_conditionals: bool = True,
        collect_info: bool = False,
    ):
        """
        Allocates the solver data structures on the specified device.

        Args:
            model: The model for which to allocate the solver data.
            limits: The limits container associated with the model.
            contacts: The contacts container associated with the model.
            config: The solver config to use.
            warmstart: The warm-start mode to use for the solver.
            use_acceleration: Set to `True` to enable Nesterov acceleration.
            use_graph_conditionals: Set to `False` to disable CUDA graph conditional nodes.
                When disabled, replaces `wp.capture_while` with an unrolled for-loop over max iterations.
            collect_info: Set to `True` to enable collection of solver convergence info.
                This setting is intended only for analysis and debugging purposes, as it
                will increase memory consumption and reduce wall-clock time.
        """

        # Ensure the model is valid
        if model is None:
            raise ValueError("A model of type `ModelKamino` must be provided to allocate the solver.")
        elif not isinstance(model, ModelKamino):
            raise ValueError("Invalid model provided. Must be an instance of `ModelKamino`.")

        # Cache a reference to the model size meta-data container
        self._size = model.size

        # Use the model's device
        self._device = model.device

        # Cache solver configs and validate them against the model size
        # NOTE: These are configurations which could potentially be different across
        # worlds, so we cache them in a list and write to device memory as an array
        if config is not None:
            self._config = self._check_config(model, config)
        elif len(self._config) == 0:
            self._config = self._check_config(model, None)

        # Cache high-level solver options shared across all worlds
        self._warmstart = warmstart
        self._use_acceleration = use_acceleration
        self._use_graph_conditionals = use_graph_conditionals
        self._collect_info = collect_info

        # Check if any world uses in-loop adaptive penalty updates (requiring per-iteration regularization updates)
        methods = [PADMMPenaltyUpdate.from_string(c.penalty_update_method) for c in self._config]
        self._use_adaptive_penalty = any(m == PADMMPenaltyUpdate.BALANCED for m in methods)

        # Check if any world uses the spectral penalty, which is updated after blocks of iterations
        spectral_configs = [c for c, m in zip(self._config, methods, strict=True) if m == PADMMPenaltyUpdate.SPECTRAL]
        self._use_spectral_penalty = len(spectral_configs) > 0
        self._spectral_update_freq = 0
        self._spectral_max_iterations = 0
        self._spectral_tolerance = 0.0
        self._spectral_mask = None
        self._spectral_lower_bound = None
        self._spectral_power_iteration = None
        self._spectral_problem = None
        self._use_spectral_inverse = False
        self._spectral_inverse_iterations = 0
        self._spectral_inverse_mask = None
        self._spectral_shift = None
        if self._use_spectral_penalty:
            update_freqs = {c.penalty_update_freq for c in spectral_configs}
            if len(update_freqs) > 1:
                raise ValueError(
                    f"All worlds using the spectral penalty must use the same `penalty_update_freq`, got {update_freqs}."
                )
            self._spectral_update_freq = update_freqs.pop()
            self._spectral_max_iterations = max(c.spectral_max_iterations for c in spectral_configs)
            self._spectral_tolerance = min(c.spectral_tolerance for c in spectral_configs)
            inverse = [
                m == PADMMPenaltyUpdate.SPECTRAL and c.spectral_min_eigenvalue == "inverse"
                for c, m in zip(self._config, methods, strict=True)
            ]
            self._use_spectral_inverse = any(inverse)
            self._spectral_inverse_iterations = max(
                (c.spectral_inverse_iterations for c, i in zip(self._config, inverse, strict=True) if i), default=0
            )
            with wp.ScopedDevice(self._device):
                self._spectral_mask = wp.array([m == PADMMPenaltyUpdate.SPECTRAL for m in methods], dtype=wp.bool)
                self._spectral_lower_bound = wp.zeros(shape=(self._size.num_worlds,), dtype=wp.float32)
                if self._use_spectral_inverse:
                    self._spectral_inverse_mask = wp.array(inverse, dtype=wp.bool)
                    self._spectral_shift = wp.zeros(shape=(self._size.num_worlds,), dtype=wp.float32)

        # Compute the largest max iterations across all worlds
        # NOTE: This is needed to allocate the solver
        # info arrays if `collect_info` is enabled
        max_of_max_iters = max([c.max_iterations for c in self._config])
        self._max_of_max_iters = max_of_max_iters

        # Allocate memory in device global memory
        self._data = PADMMData(
            size=self._size,
            max_iters=max_of_max_iters,
            use_acceleration=self._use_acceleration,
            collect_info=self._collect_info,
            device=self._device,
        )

        # Write algorithm configs into device memory
        configs = [convert_config_to_struct(c) for c in self._config]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=PADMMConfigStruct)
            self._all_worlds_mask = wp.ones(shape=(self._size.num_worlds,), dtype=wp.bool)

        # Initialize the ALM penalty, which may be warm-started across solves
        self._reset_penalty(self._all_worlds_mask)

        # Allocate the data of the velocity-consistent warm-start of the augmented constraint velocities
        consistent = [c.warmstart_velocity == "consistent" for c in self._config]
        self._use_consistent_warmstart_velocity = any(consistent)
        self._consistent_warmstart_mask = None
        self._warmstart_sigma = None
        if self._use_consistent_warmstart_velocity:
            with wp.ScopedDevice(self._device):
                self._consistent_warmstart_mask = wp.array(consistent, dtype=wp.bool)
                self._warmstart_sigma = wp.zeros(shape=(self._size.sum_of_max_total_cts,), dtype=wp.float32)

        # Specialize certain solver kernels depending on whether acceleration is enabled
        self._initialize_solver_kernel = make_initialize_solver_kernel(self._use_acceleration)
        self._collect_solver_info_kernel = make_collect_solver_info_kernel(self._use_acceleration)
        self._update_dual_variables_and_compute_primal_dual_residuals_kernel = (
            make_update_dual_variables_and_compute_primal_dual_residuals(self._use_acceleration)
        )
        tile_size = get_tile_size(self._size.max_of_max_total_cts)
        block_dim = get_block_dim(tile_size, ratio=2, min_size=1)
        self._project_dual_convergence_accel_kernel = _make_project_dual_convergence_accel_kernel(block_dim)

    def reset(self, problem: ConstrainedDynamicsProblem | None = None, world_mask: wp.array[wp.bool] | None = None):
        """
        Resets the persistent solution cache used for internal warm-starting, as well as the
        (possibly warm-started) ALM penalty, for all worlds or the subset selected by `world_mask`.

        This does not modify the scratch solver state (`self._data.state`), since that is
        unconditionally reinitialized by `coldstart()`/`warmstart()`.
        """
        # Reset the ALM penalty of the selected worlds
        self._reset_penalty(self._all_worlds_mask if world_mask is None else world_mask)

        # If no world mask is provided, reset the solution cache of all worlds
        if world_mask is None:
            self._data.solution.zero()

        # Otherwise, only the solution cache of the specified worlds
        else:
            if problem is None:
                raise ValueError("A `ConstrainedDynamicsProblem` instance must be provided when a world mask is used.")
            wp.launch(
                kernel=_reset_solver_data,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    world_mask,
                    problem.data.vio,
                    problem.data.maxdim,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

    def prepare(self, problem: ConstrainedDynamicsProblem):
        """
        Allocates the solver data that depends on the problem to be solved, e.g. the power iteration
        estimating the spectrum of its operator for the spectral penalty.

        This is performed lazily by the first call to :meth:`solve` for a given problem, but should be
        performed beforehand if the solves are captured in CUDA graphs, since it allocates memory.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        if not self._use_spectral_penalty or self._spectral_problem is problem:
            return
        self._spectral_power_iteration = PowerIteration(
            operator=make_problem_operator(problem, preconditioned=True),
            max_iterations=self._spectral_max_iterations,
            tolerance=self._spectral_tolerance,
        )
        self._spectral_problem = problem

    def coldstart(self):
        """
        Initializes the internal solver state to perform a cold-start solve.
        This method sets all solver state variables to zeros.
        """
        # Initialize state arrays to zero
        self._data.state.reset(use_acceleration=self._use_acceleration)

    def warmstart(
        self,
        problem: ConstrainedDynamicsProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Warm-starts the internal solver state based on the selected warm-start mode.

        Supported warm-start modes:
        - `PADMMWarmStartMode.NONE`: No warm-starting is performed.
        - `PADMMWarmStartMode.INTERNAL`: Warm-starts from the internal solution state.
        - `PADMMWarmStartMode.CONTAINERS`: Warm-starts from the provided limits and contacts containers.

        Args:
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            model: The model associated with the problem.
            data: The model data associated with the problem.
            limits: The limits container associated with the model.
                If `None`, no warm-starting from limits is performed.
            contacts: The contacts container associated with the model.
                If `None`, no warm-starting from contacts is performed.
        """
        # Reset the internal solver state first. Warmstarting will only populate
        # the primal/dual iterate variables, all remaining fields need a clean
        # initialization.
        self._data.state.reset(use_acceleration=self._use_acceleration)

        # Warm-start based on the selected mode
        match self._warmstart:
            case PADMMWarmStartMode.NONE:
                return
            case PADMMWarmStartMode.INTERNAL:
                self._warmstart_from_solution(problem)
            case PADMMWarmStartMode.CONTAINERS:
                self._warmstart_from_containers(problem, model, data, limits, contacts)
            case _:
                raise ValueError(f"Invalid warmstart mode: {self._warmstart}")

        self._scale_warmstart_forces(problem)

        # Optionally replace the warm-started augmented constraint velocities by those consistent with the reactions
        if self._use_consistent_warmstart_velocity:
            self._warmstart_consistent_velocity(problem)

    def _warmstart_consistent_velocity(self, problem: ConstrainedDynamicsProblem):
        """
        Warm-starts the augmented constraint velocities with those consistent with the warm-started reactions,
        i.e. `z_0 = Proj_{K*}(sigma_0 + Gamma(sigma_0))` with `sigma_0 = P (D + R) P y_0 + P v_f`, for the worlds
        configured with the `consistent` warm-start velocity.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        y_0 = self._data.state.y_hat if self._use_acceleration else self._data.state.y_p
        z_0 = self._data.state.z_hat if self._use_acceleration else self._data.state.z_p
        sigma = self._warmstart_sigma

        # Compute the (preconditioned) constraint velocities consistent with the reactions: sigma = P (D + R) P y + P v_f
        wp.copy(sigma, problem.data.v_f)
        problem.gemv(
            x=y_0, y=sigma, alpha=1.0, beta=1.0, preconditioned=True, world_mask=self._consistent_warmstart_mask
        )

        # Apply the De Saxce correction and project to the dual cone of the constraints
        self._warmstart_desaxce_correction(problem, z=sigma)
        data = problem.data
        wp.launch(
            kernel=_project_warmstart_velocity,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                self._consistent_warmstart_mask,
                data.dim,
                data.vio,
                data.nbc,
                data.nl,
                data.nc,
                data.bcio,
                data.cio,
                data.bcgo,
                data.lcgo,
                data.ccgo,
                data.mu,
                data.bound_lower,
                data.bound_upper,
                y_0,
                sigma,
            ],
            outputs=[z_0],
            device=self.device,
        )

    def _scale_warmstart_forces(self, problem: ConstrainedDynamicsProblem):
        """Scales the warm-started primal and slack force iterates."""
        x_0 = self._data.state.x_p
        y_0 = self._data.state.y_hat if self._use_acceleration else self._data.state.y_p
        wp.launch(
            kernel=_scale_warmstart_forces,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                # Outputs:
                x_0,
                y_0,
            ],
            device=self.device,
        )

    def solve(self, problem: ConstrainedDynamicsProblem):
        """
        Solves the given dual problem using PADMM.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Pass the PADMM-owned tolerance array to the iterative linear solver (if present), so the
        # inexact-ADMM tolerance schedule (set in the convergence kernel) drives the inner solve.
        linear_solver = problem.linear_solver
        inner = getattr(linear_solver, "solver", None)
        if inner is not None:
            inner.atol = self._data.linear_solver_atol
        elif isinstance(linear_solver, IterativeSolver):
            # The fused single-kernel CR has no wrapped ``solver``; it reads its own ``.atol`` array
            # live each solve (see ConjugateResidualSolverFused._solve_impl).
            linear_solver.atol = self._data.linear_solver_atol

        # Initialize the solver status, ALM penalty, and iterative solver tolerance
        self._initialize()

        # Initialize the spectral penalty from the extreme eigenvalues of the problem operator
        if self._use_spectral_penalty:
            self._initialize_spectral_penalty(problem)

        # Add the diagonal proximal regularization to the linear system of the problem, e.g.
        # D_{eta,rho} := D + (eta + rho) * I_{ncts} for the dual formulation
        self._update_regularization(problem)

        # Reset the solver info to zero if collection is enabled
        if self._collect_info:
            self._data.info.zero()

        # Iterate until convergence or maximum number of iterations is reached
        # NOTE: With the spectral penalty, the iterations are performed in blocks, after each of which the
        # penalty is updated and the linear system of the problem is refreshed (e.g. re-factorized)
        step_fn = self._step_accel if self._use_acceleration else self._step
        loop_fn, num_loops = step_fn, self._max_of_max_iters
        if self._use_spectral_penalty and self._spectral_update_freq > 0:
            block_size = self._spectral_update_freq

            def loop_fn(problem: ConstrainedDynamicsProblem):
                for _ in range(block_size):
                    step_fn(problem)
                self._update_spectral_penalty(problem)

            num_loops = math.ceil(self._max_of_max_iters / block_size)
        if self._use_graph_conditionals:
            wp.capture_while(self._data.state.done, while_body=loop_fn, problem=problem)
        else:
            for _ in range(num_loops):
                loop_fn(problem)

        # Update the final solution from the terminal PADMM state
        self._update_solution(problem)

    ###
    # Internals - High-Level Operations
    ###

    @staticmethod
    def _check_config(
        model: ModelKamino | None = None, config: list[PADMMSolver.Config] | PADMMSolver.Config | None = None
    ) -> list[PADMMSolver.Config]:
        """
        Checks and validates the provided solver config, returning a list
        of config objects corresponding to each world in the model.

        Args:
            model: The model for which to validate the config.
            config: The solver configurations container to validate.
        """
        # If no config is provided, use defaults
        if config is None:
            # If no model is provided, use a single default config object
            if model is None:
                config = [PADMMSolver.Config()]

            # If a model is provided, create a list of default config
            # objects based on the number of worlds in the model
            else:
                num_worlds = model.info.num_worlds
                config = [PADMMSolver.Config()] * num_worlds

        # If a single config object is provided, convert it to a list
        elif isinstance(config, PADMMSolver.Config):
            config = [config] * (model.info.num_worlds if model else 1)

        # If a list of configs is provided, ensure it matches the number
        # of worlds and that all configs are instances of PADMMSolver.Config
        elif isinstance(config, list):
            if model is not None and len(config) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} configs, got {len(config)}")
            if not all(isinstance(s, PADMMSolver.Config) for s in config):
                raise TypeError("All configs must be instances of PADMMSolver.Config")
        else:
            raise TypeError(f"Expected a single object or list of `PADMMSolver.Config`, got {type(config)}")

        # Configurations expressed in the units of the primal algorithm must first be converted
        if any(isinstance(c, PADMMPrimalSolverConfig) for c in config):
            raise TypeError("Primal PADMM configs must be used with `PADMMPrimalSolver`.")

        # Return the validated config
        return config

    def _initialize(self):
        """
        Launches a kernel to initialize the internal solver state before starting a new solve.
        The kernel is parallelized over the number of worlds.
        """
        # Initialize solver status, penalty parameters, and iterative solver tolerance
        wp.launch(
            kernel=self._initialize_solver_kernel,
            dim=self._size.num_worlds,
            inputs=[
                self._data.config,
                self._data.status,
                self._data.penalty,
                self._data.state.sigma,
                self._data.state.a_p,
                self._data.linear_solver_atol,
            ],
            device=self.device,
        )

        # Initialize the global while condition flag
        # NOTE: We use a single-element array that is initialized
        # to number of worlds and decremented by each world that
        # converges or reaches the maximum number of iterations
        self._data.state.done.fill_(self._size.num_worlds)

    def _reset_penalty(self, world_mask: wp.array[wp.bool]):
        """Resets the ALM penalty of the worlds selected by `world_mask` to its initial value."""
        wp.launch(
            kernel=_reset_penalty,
            dim=self._size.num_worlds,
            inputs=[world_mask, self._data.config],
            outputs=[self._data.penalty],
            device=self.device,
        )

    def _update_problem_regularization(self, problem: ConstrainedDynamicsProblem):
        """Propagate eta + rho to the diagonal regularization of the problem's linear system."""
        wp.launch(
            kernel=_update_problem_proximal_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.penalty,
                self._data.status,
                problem.data.eta,
            ],
            device=self.device,
        )
        # For sparse problems only eta changed (not the Jacobian sparsity), so this triggers a
        # regularization-only refresh: a raw-Jacobian solver (fused CR) refreshes its combined
        # regularization and skips the index rebuild + segmented sort, which is both wasted work
        # per PADMM iteration and unsafe inside wp.capture_while (the sort allocates).
        problem.update_regularization()

    def _update_regularization(self, problem: ConstrainedDynamicsProblem):
        """
        Updates the diagonal regularization of the problem's linear system with the proximal regularization
        terms and performs any necessary pre-computations, e.g. matrix factorizations for direct solvers.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        self._update_problem_regularization(problem)
        # Let a raw-Jacobian linear solver (e.g. the fused single-kernel CR) rebuild its
        # per-step index structures here, before the (possibly graph-captured) iteration loop,
        # so that one-off work stays out of the replayed graph. No-op for other solvers.
        problem.prepare_solve()

    def _update_adaptive_regularization(self, problem: ConstrainedDynamicsProblem):
        """
        Propagates an in-loop (adaptive-penalty) update of rho to the problem's linear system.

        Under graph-conditional capture the iteration body is traced once: the lazy refresh inside the
        linear solve runs *before* the eta update in that single trace, so it might skip the update
        because the (host-side) flags that signal an update have been cleared before the loop start.
        The replayed graph then never picks up the per-iteration eta update (-> divergence/NaN).
        Calling ``prepare_solve`` here, *after* the update, records the refresh in the body.
        """
        self._update_problem_regularization(problem)
        problem.prepare_solve()

    def _initialize_spectral_penalty(self, problem: ConstrainedDynamicsProblem):
        """
        Initializes the spectral penalty `rho = sqrt(m * L) * kappa^p` of the worlds using it, from the extreme
        eigenvalues `m` and `L` of the regularized preconditioned operator `P (D + R) P + eta I` of the problem.

        The largest eigenvalue is estimated by power iteration, warm-started from the previous solve,
        while the smallest one is bounded from below by `eta + min(P R P)`, since `D` is positive semi-definite
        and in general singular, e.g. for hyperstatic constraint configurations. Optionally, the smallest
        eigenvalue is then refined by inverse iteration, using the linear system of the problem regularized
        with the penalty initialized from the lower bound, e.g. using the existing factorization of the
        augmented Delassus matrix for the dual problem, or of the augmented mass matrix via the Woodbury
        identity for the primal problem.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Estimate the largest eigenvalue of the operator, warm-started from the previous solve of the same problem
        self.prepare(problem)
        lambda_max = self._spectral_power_iteration.compute_max(world_mask=self._spectral_mask)

        # Compute the lower bound of the smallest eigenvalue from the preconditioned compliance
        self._spectral_lower_bound.fill_(FLOAT32_MAX)
        wp.launch(
            kernel=_compute_spectral_lower_bound,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[problem.data.dim, problem.data.vio, problem.data.P, problem.data.R, self._spectral_mask],
            outputs=[self._spectral_lower_bound],
            device=self.device,
        )

        # Initialize the spectral penalty and the total proximal regularization
        wp.launch(
            kernel=_initialize_spectral_penalty,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.dim,
                self._data.config,
                self._spectral_mask,
                lambda_max,
                self._spectral_lower_bound,
            ],
            outputs=[self._data.penalty, self._data.state.sigma],
            device=self.device,
        )

        # Optionally refine the smallest eigenvalue by inverse iteration with the regularized operator,
        # factorized (or updated) with the penalty initialized from the lower bound, and re-initialize the penalty
        if self._use_spectral_inverse:
            self._update_regularization(problem)
            wp.launch(
                kernel=_compute_spectral_shift,
                dim=self._size.num_worlds,
                inputs=[self._data.config, self._data.penalty],
                outputs=[self._spectral_shift],
                device=self.device,
            )
            lambda_min = self._spectral_power_iteration.compute_min(
                method="inverse",
                shift=self._spectral_shift,
                solve_fn=make_problem_solve_fn(problem),
                world_mask=self._spectral_inverse_mask,
                max_iterations=self._spectral_inverse_iterations,
            )
            wp.launch(
                kernel=_refine_spectral_lower_bound,
                dim=self._size.num_worlds,
                inputs=[self._spectral_inverse_mask, lambda_min],
                outputs=[self._spectral_lower_bound],
                device=self.device,
            )
            wp.launch(
                kernel=_initialize_spectral_penalty,
                dim=self._size.num_worlds,
                inputs=[
                    problem.data.dim,
                    self._data.config,
                    self._spectral_inverse_mask,
                    lambda_max,
                    self._spectral_lower_bound,
                ],
                outputs=[self._data.penalty, self._data.state.sigma],
                device=self.device,
            )

    def _update_spectral_penalty(self, problem: ConstrainedDynamicsProblem):
        """
        Adapts the exponent of the spectral penalty from the current primal and dual residuals,
        and refreshes the linear system of the problem (e.g. re-factorizes it) with the updated penalty.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        wp.launch(
            kernel=_update_spectral_penalty,
            dim=self._size.num_worlds,
            inputs=[self._data.config, self._data.status],
            outputs=[self._data.penalty, self._data.state.sigma],
            device=self.device,
        )
        wp.launch(
            kernel=_update_problem_spectral_regularization,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[problem.data.dim, problem.data.vio, self._data.config, self._data.penalty],
            outputs=[problem.data.eta],
            device=self.device,
        )
        problem.update_regularization()
        problem.prepare_solve()

    def _step(self, problem: ConstrainedDynamicsProblem):
        """
        Performs a single PADMM solver iteration.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Compute De Saxce correction and velocity bias in one launch.
        self._update_desaxce_and_velocity_bias(problem, self._data.state.y_p, self._data.state.z_p)

        # Compute the unconstrained solution and store in the primal variables
        self._update_unconstrained_solution(problem)

        # Compute the argument to the projection operator with over-relaxation
        self._update_projection_argument(problem, self._data.state.z_p)

        # Project the over-relaxed primal variables to the feasible set
        self._update_projection_to_feasible_set(problem)

        # Update the dual variables and compute residuals from the current state
        self._update_dual_variables_and_residuals(problem)

        # Compute infinity-norm of all residuals and check for convergence
        self._update_convergence_check(problem)

        # Update the sparse problem regularization if the penalty was updated adaptively
        if problem.sparse and self._use_adaptive_penalty:
            self._update_adaptive_regularization(problem)

        # Optionally record internal solver info
        if self._collect_info:
            self._update_solver_info(problem)

        # Update caches of previous state variables
        self._update_previous_state()

    def _step_accel(self, problem: ConstrainedDynamicsProblem):
        """
        Performs a single PADMM solver iteration with Nesterov acceleration.

        Uses multi-stage kernels to reduce kernel launch overhead:
        - _compute_desaxce_correction_and_velocity_bias computes De Saxce correction and velocity bias
        - _project_dual_convergence_accel_kernel advances projection, dual update,
          residual reduction, convergence, acceleration, and previous-state caching

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Compute De Saxce correction and velocity bias in one launch.
        self._update_desaxce_and_velocity_bias(problem, self._data.state.y_hat, self._data.state.z_hat)

        # Compute the unconstrained solution and store in the primal variables
        self._update_unconstrained_solution(problem)

        # Advance projection, dual update, residual status, and acceleration state.
        self._update_projection_dual_convergence_accel(problem)

        # Update the sparse problem regularization if the penalty was updated adaptively
        if problem.sparse and self._use_adaptive_penalty:
            self._update_adaptive_regularization(problem)

        # Optionally record internal solver info from the fused status/state.
        if self._collect_info:
            self._update_solver_info(problem)

        # Nesterov acceleration and previous-state caching are handled above.

    ###
    # Internals - Warm-starting
    ###

    def _warmstart_desaxce_correction(self, problem: ConstrainedDynamicsProblem, z: wp.array[wp.float32]):
        """
        Applies the De Saxce correction to the provided post-event constraint-space velocity warm-start.

        Args:
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            z: The post-event constraint-space velocity warm-start variable.
                This can either be `z_p` or `z_hat` depending on whether acceleration is used.
        """
        wp.launch(
            kernel=_warmstart_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                # Outputs:
                z,
            ],
            device=self.device,
        )

    def _warmstart_joint_constraints(
        self,
        model: ModelKamino,
        data: DataKamino,
        problem: ConstrainedDynamicsProblem,
        x_0: wp.array[wp.float32],
        y_0: wp.array[wp.float32],
        z_0: wp.array[wp.float32],
    ):
        """
        Warm-starts the bilateral joint constraint variables from the model data container.

        Args:
            model: The model associated with the problem.
            data: The model data associated with the problem.
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0: The output primal variables array to be warm-started.
            y_0: The output slack variables array to be warm-started.
            z_0: The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_joint_constraints,
            dim=model.size.sum_of_num_joints,
            inputs=[
                # Inputs:
                model.time.dt,
                model.joints.wid,
                model.joints.num_dynamic_cts,
                model.joints.num_kinematic_cts,
                model.joints.num_friction_cts,
                model.joints.num_effort_cts,
                model.joints.dofs_offset,
                model.joints.dynamic_cts_offset,
                model.joints.kinematic_cts_offset,
                model.joints.friction_cts_offset,
                model.joints.effort_cts_offset,
                model.joints.friction_cts_axis,
                model.joints.effort_cts_axis,
                model.joints.dynamic_cts_offset_total_cts,
                model.joints.kinematic_cts_offset_total_cts,
                model.joints.friction_cts_offset_total_cts,
                model.joints.effort_cts_offset_total_cts,
                data.joints.lambda_dyn_j,
                data.joints.lambda_kin_j,
                data.joints.lambda_f_j,
                data.joints.lambda_tau_j,
                data.joints.dq_j,
                data.joints.inv_m_a,
                data.joints.dq_b_a,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
            device=self.device,
        )

    def _warmstart_limit_constraints(
        self,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino,
        problem: ConstrainedDynamicsProblem,
        x_0: wp.array[wp.float32],
        y_0: wp.array[wp.float32],
        z_0: wp.array[wp.float32],
    ):
        """
        Warm-starts the unilateral limit constraint variables from the limits data container.

        Args:
            model: The model associated with the problem.
            data: The model data associated with the problem.
            limits: The limits container associated with the model.
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0: The output primal variables array to be warm-started.
            y_0: The output slack variables array to be warm-started.
            z_0: The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_limit_constraints,
            dim=limits.model_max_limits_host,
            inputs=[
                # Inputs:
                model.time.dt,
                model.info.total_cts_offset,
                data.info.limit_cts_group_offset,
                limits.model_active_limits,
                limits.wid,
                limits.lid,
                limits.reaction,
                limits.velocity,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
            device=self.device,
        )

    def _warmstart_contact_constraints(
        self,
        model: ModelKamino,
        data: DataKamino,
        contacts: ContactsKamino,
        problem: ConstrainedDynamicsProblem,
        x_0: wp.array[wp.float32],
        y_0: wp.array[wp.float32],
        z_0: wp.array[wp.float32],
    ):
        """
        Warm-starts the unilateral contact constraint variables from the contacts data container.

        Args:
            model: The model associated with the problem.
            data: The model data associated with the problem.
            contacts: The contacts container associated with the model.
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            x_0: The output primal variables array to be warm-started.
            y_0: The output slack variables array to be warm-started.
            z_0: The output dual variables array to be warm-started.
        """
        wp.launch(
            kernel=_warmstart_contact_constraints,
            dim=contacts.model_max_contacts_host,
            inputs=[
                # Inputs:
                model.time.dt,
                model.info.total_cts_offset,
                data.info.contact_cts_group_offset,
                contacts.model_active_contacts,
                contacts.wid,
                contacts.cid,
                contacts.material,
                contacts.reaction,
                contacts.velocity,
                problem.data.P,
                # Outputs:
                x_0,
                y_0,
                z_0,
            ],
            device=self.device,
        )

    def _warmstart_from_solution(self, problem: ConstrainedDynamicsProblem):
        """
        Warm-starts the internal solver state from the stored solution variables.

        Args:
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
        """
        # Apply the dual-problem preconditioner to the stored solution
        # in order to project to solution space of the PADMM variables
        wp.launch(
            kernel=_apply_dual_preconditioner_to_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

        # Capture references to the warm-start variables
        # depending on whether acceleration is used or not
        if self._use_acceleration:
            y_0 = self._data.state.y_hat
            z_0 = self._data.state.z_hat
        else:
            y_0 = self._data.state.y_p
            z_0 = self._data.state.z_p

        # Copy the last solution into the warm-start variables
        wp.copy(self._data.state.x_p, self._data.solution.lambdas)
        wp.copy(y_0, self._data.solution.lambdas)
        wp.copy(z_0, self._data.solution.v_plus)
        self._warmstart_desaxce_correction(problem, z=z_0)

    def _warmstart_from_containers(
        self,
        problem: ConstrainedDynamicsProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """
        Warm-starts the internal solver state from the provided model data and limits and contacts containers.

        Args:
            problem: The forward dynamics problem to be solved.
                This is needed during warm-starts in order to access the problem preconditioning.
            model: The model associated with the problem.
            data: The model data associated with the problem.
            limits: The limits container associated with the model.
                If `None`, no warm-starting from limits is performed.
            contacts: The contacts container associated with the model.
                If `None`, no warm-starting from contacts is performed.
        """
        # Capture references to the warm-start variables
        # depending on whether acceleration is used or not
        x_0 = self._data.state.x_p
        if self._use_acceleration:
            y_0 = self._data.state.y_hat
            z_0 = self._data.state.z_hat
        else:
            y_0 = self._data.state.y_p
            z_0 = self._data.state.z_p

        # Warm-start each constraint group from constraint states cached in the data containers
        if model.size.sum_of_num_joints > 0:
            self._warmstart_joint_constraints(model, data, problem, x_0, y_0, z_0)
        if limits is not None and limits.model_max_limits_host > 0:
            self._warmstart_limit_constraints(model, data, limits, problem, x_0, y_0, z_0)
        if contacts is not None and contacts.model_max_contacts_host > 0:
            self._warmstart_contact_constraints(model, data, contacts, problem, x_0, y_0, z_0)

    ###
    # Internals - Per-Step Operations
    ###

    def _update_desaxce_correction(self, problem: ConstrainedDynamicsProblem, z: wp.array[wp.float32]):
        """
        Launches a kernel to compute the De Saxce correction velocity using the previous dual variables.
        The kernel is parallelized over the number of worlds and the maximum number of contacts.

        Args:
            problem: The forward dynamics problem to be solved.
            z: The dual variable array from the previous iteration.
                This can either be the acceleration variable `z_hat` or the standard dual variable `z_p`.
        """
        wp.launch(
            kernel=_compute_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.status,
                z,
                # Outputs:
                self._data.state.s,
            ],
            device=self.device,
        )

    def _update_velocity_bias(
        self, problem: ConstrainedDynamicsProblem, y: wp.array[wp.float32], z: wp.array[wp.float32]
    ):
        """
        Launches a kernel to compute the total bias velocity vector using the previous state variables.
        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem: The forward dynamics problem to be solved.
            y: The primal variable array from the previous iteration.
                This can either be the acceleration variable `y_hat` or the standard primal variable `y_p`.
            z: The dual variable array from the previous iteration.
                This can either be the acceleration variable `z_hat` or the standard dual variable `z_p`.
        """
        wp.launch(
            kernel=_compute_velocity_bias,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.s,
                self._data.state.x_p,
                y,
                z,
                # Outputs:
                self._data.state.v,
            ],
            device=self.device,
        )

    def _update_desaxce_and_velocity_bias(
        self, problem: ConstrainedDynamicsProblem, y: wp.array[wp.float32], z: wp.array[wp.float32]
    ):
        """Fused De Saxce correction + velocity bias in a single kernel launch.

        Computes the De Saxce correction inline for contact constraints and the velocity
        bias for all constraints.  Uses compile-time specialization: when no contacts
        are present, the De Saxce branch is eliminated entirely.  When ``collect_info``
        is disabled the intermediate De Saxce vector is kept as a register-only local;
        when enabled it is also persisted to ``solver_s`` so that the info kernel can
        read the original value.

        Args:
            problem: The forward dynamics problem to be solved.
            y: The primal variable array from the previous iteration.
            z: The dual variable array from the previous iteration.
        """
        has_contacts = self._size.max_of_max_contacts > 0
        kernel = make_desaxce_correction_and_velocity_bias_kernel(has_contacts, self._collect_info)
        wp.launch(
            kernel=kernel,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                problem.data.v_f,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x_p,
                y,
                z,
                self._data.state.v,
                self._data.state.s,
            ],
            device=self.device,
        )

    def _update_unconstrained_solution(self, problem: ConstrainedDynamicsProblem):
        """
        Solves the unconstrained sub-problem for the primal variables using the linear operator of the problem.

        For the dual formulation, this corresponds to solving a linear system with the Delassus matrix,
        while for the primal formulation, it corresponds to solving with the constraint-augmented mass matrix.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        problem.solve(v=self._data.state.v, x=self._data.state.x)

    def _update_projection_argument(self, problem: ConstrainedDynamicsProblem, z: wp.array[wp.float32]):
        """
        Launches a kernel to compute the argument for the projection operator onto the
        feasible set using the accelerated state variables and the unconstrained solution.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem: The forward dynamics problem to be solved.
            z: The dual variable array from the previous iteration.
                This can either be the acceleration variable `z_hat` or the standard dual variable `z
        """
        # Apply over-relaxation and compute the argument to the projection operator
        wp.launch(
            kernel=_compute_projection_argument,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.penalty,
                self._data.status,
                z,
                self._data.state.x,
                # Outputs:
                self._data.state.y,
            ],
            device=self.device,
        )

    def _update_projection_to_feasible_set(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a kernel to project the current primal variables
        onto the feasible set defined by the constraint cone K.

        The kernel is parallelized over the number of worlds and the maximum
        number of bounded-multiplier, limit, and contact entities.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Project each bounded, limit, and contact entity onto its feasible set.
        wp.launch(
            kernel=_project_to_feasible_cone,
            dim=(self._size.num_worlds, self._size.max_of_max_inequalities),
            inputs=[
                # Inputs:
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcio,
                problem.data.cio,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                problem.data.bound_lower,
                problem.data.bound_upper,
                self._data.status,
                # Outputs:
                self._data.state.y,
            ],
            device=self.device,
        )

    def _update_complementarity_residuals(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a kernel to compute the complementarity residuals from the current state variables.
        The kernel is parallelized over the number of worlds and the maximum number of inequality constraints.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Compute complementarity residual from the current state.
        wp.launch(
            kernel=_compute_complementarity_residuals,
            dim=(self._size.num_worlds, self._size.max_of_max_inequalities),
            inputs=[
                # Inputs:
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.vio,
                problem.data.bcio,
                problem.data.iio,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                self._data.status,
                self._data.state.x,
                self._data.state.z,
                problem.data.bound_lower,
                problem.data.bound_upper,
                # Outputs:
                self._data.residuals.r_compl,
            ],
            device=self.device,
        )

    def _update_dual_variables_and_residuals(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a kernel to update the dual variables and compute the
        PADMM residuals from the current and previous state variables.

        The kernel is parallelized over the number of worlds and the maximum number of total constraints.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Update the dual variables and compute primal-dual residuals from the current state
        # NOTE: These are combined into a single kernel to reduce kernel launch overhead
        wp.launch(
            kernel=self._update_dual_variables_and_compute_primal_dual_residuals_kernel,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.status,
                self._data.state.x,
                self._data.state.y,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.z,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_dx,
                self._data.residuals.r_dy,
                self._data.residuals.r_dz,
            ],
            device=self.device,
        )

        # Compute complementarity residual from the current state.
        self._update_complementarity_residuals(problem)

    def _update_projection_dual_convergence_accel(self, problem: ConstrainedDynamicsProblem):
        """Advance accelerated PADMM projection, residual status, and state cache."""
        tile_size = get_tile_size(self._size.max_of_max_total_cts)
        block_dim = get_block_dim(tile_size, ratio=2, min_size=1)
        wp.launch_tiled(
            kernel=self._project_dual_convergence_accel_kernel,
            dim=self._size.num_worlds,
            block_dim=block_dim,
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcio,
                problem.data.cio,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                problem.data.bound_lower,
                problem.data.bound_upper,
                problem.data.P,
                self._data.config,
                self._data.penalty,
                self._data.state.a_p,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y_hat,
                self._data.state.z_hat,
                self._data.state.y_p,
                self._data.state.z_p,
                # Outputs:
                self._data.state.y,
                self._data.state.z,
                self._data.state.done,
                self._data.state.a,
                self._data.state.a_factor,
                self._data.status,
                self._data.penalty,
                self._data.linear_solver_atol,
                self._data.state.y_hat,
                self._data.state.z_hat,
                self._data.state.x_p,
                self._data.state.y_p,
                self._data.state.z_p,
                self._data.state.a_p,
            ],
            device=self.device,
        )

    def _update_convergence_check(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a kernel to compute the infinity-norm of the PADMM residuals
        using the current and previous state variables and check for convergence.

        The kernel is parallelized over the number of worlds.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Compute infinity-norm of all residuals and check for convergence
        tile_size = get_tile_size(self._size.max_of_max_total_cts)
        block_dim = get_block_dim(tile_size, min_size=1)
        wp.launch_tiled(
            kernel=_make_compute_infnorm_residuals_kernel(
                tile_size,
                self._size.max_of_max_total_cts,
                self._size.max_of_max_inequalities,
            ),
            dim=self._size.num_worlds,
            block_dim=block_dim,
            inputs=[
                # Inputs:
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.iio,
                problem.data.dim,
                problem.data.vio,
                self._data.config,
                self._data.residuals.r_primal,
                self._data.residuals.r_dual,
                self._data.residuals.r_compl,
                # Outputs:
                self._data.state.done,
                self._data.status,
                self._data.penalty,
                self._data.linear_solver_atol,
            ],
            device=self.device,
        )

    def _update_solver_info(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a kernel to update the solver info history from the current solver data.

        The kernel is parallelized over the number of worlds.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # First reset the internal buffer arrays to zero
        # to ensure we do not accumulate values across iterations
        self.data.info.v_plus.zero_()
        self.data.info.v_aug.zero_()
        self._data.info.s.zero_()

        # Compute the preconditioned post-event constraint-space velocity: v_plus = v_f + P (D + R) P @ lambda
        wp.copy(self._data.info.v_plus, problem.data.v_f)
        problem.gemv(x=self._data.state.y, y=self._data.info.v_plus, alpha=1.0, beta=1.0)

        # Collect convergence information from the current state
        wp.launch(
            kernel=self._collect_solver_info_kernel,
            dim=self._size.num_worlds,
            inputs=[
                # Inputs:
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcio,
                problem.data.cio,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.dim,
                problem.data.vio,
                problem.data.mu,
                problem.data.v_f,
                problem.data.P,
                problem.data.bound_lower,
                problem.data.bound_upper,
                self._data.state.s,
                self._data.state.x,
                self._data.state.x_p,
                self._data.state.y,
                self._data.state.y_p,
                self._data.state.z,
                self._data.state.z_p,
                self._data.state.a,
                self._data.penalty,
                self._data.status,
                # Outputs:
                self._data.info.lambdas,
                self._data.info.v_plus,
                self._data.info.v_aug,
                self._data.info.s,
                self._data.info.offsets,
                self._data.info.num_restarts,
                self._data.info.num_rho_updates,
                self._data.info.a,
                self._data.info.norm_s,
                self._data.info.norm_x,
                self._data.info.norm_y,
                self._data.info.norm_z,
                self._data.info.f_ccp,
                self._data.info.f_ncp,
                self._data.info.r_dx,
                self._data.info.r_dy,
                self._data.info.r_dz,
                self._data.info.r_primal,
                self._data.info.r_dual,
                self._data.info.r_compl,
                self._data.info.r_pd,
                self._data.info.r_dp,
                self._data.info.r_comb,
                self._data.info.r_comb_ratio,
                self._data.info.r_ncp_primal,
                self._data.info.r_ncp_dual,
                self._data.info.r_ncp_compl,
                self._data.info.r_ncp_natmap,
            ],
            device=self.device,
        )

    def _update_previous_state(self):
        """
        Updates the cached previous state variables with the current.
        This function uses on-device memory copy operations.
        """
        wp.copy(self._data.state.x_p, self._data.state.x)
        wp.copy(self._data.state.y_p, self._data.state.y)
        wp.copy(self._data.state.z_p, self._data.state.z)

    ###
    # Internals - Post-Solve Operations
    ###

    def _update_solution(self, problem: ConstrainedDynamicsProblem):
        """
        Launches a set of kernels to extract and post-process
        the final solution from the internal PADMM state data.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        # Apply the dual preconditioner to recover the final PADMM state
        wp.launch(
            kernel=_apply_dual_preconditioner_to_state,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                # Outputs:
                self._data.state.x,
                self._data.state.y,
                self._data.state.z,
            ],
            device=self.device,
        )

        # Update the De Saxce correction from terminal PADMM dual variables
        wp.launch(
            kernel=_compute_final_desaxce_correction,
            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
            inputs=[
                # Inputs:
                problem.data.nc,
                problem.data.cio,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.mu,
                self._data.state.z,
                # Outputs:
                self._data.state.s,
            ],
            device=self.device,
        )

        # Update solution vectors from the terminal PADMM state
        wp.launch(
            kernel=_compute_solution_vectors,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                # Inputs:
                problem.data.dim,
                problem.data.vio,
                self._data.state.s,
                self._data.state.y,
                self._data.state.z,
                # Outputs:
                self._data.solution.v_plus,
                self._data.solution.lambdas,
            ],
            device=self.device,
        )
