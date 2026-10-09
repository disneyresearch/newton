# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines the primal Proximal-ADMM solver class.

See the :mod:`newton._src.solvers.kamino._src.solvers.padmm_primal` module for a detailed description and usage example.
"""

from __future__ import annotations

import warp as wp

from ....config import PADMMPrimalSolverConfig
from ...core.model import ModelKamino
from ...dynamics.problem import ConstrainedDynamicsProblem
from ..padmm import PADMMSolver, PADMMWarmStartMode
from ..padmm.types import PADMMStatus

###
# Module interface
###

__all__ = ["PADMMPrimalSolver"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _swap_status_residuals(
    # Inputs-Outputs:
    solver_status: wp.array[PADMMStatus],
):
    # Retrieve the world index as the thread index
    wid = wp.tid()

    # Swap the primal and dual residuals
    status = solver_status[wid]
    r_p = status.r_p
    status.r_p = status.r_d
    status.r_d = r_p
    solver_status[wid] = status


@wp.kernel
def _swap_arrays(
    # Inputs-Outputs:
    a: wp.array[wp.float32],
    b: wp.array[wp.float32],
):
    # Retrieve the element index as the thread index
    i = wp.tid()

    # Swap the elements of the two arrays
    a_i = a[i]
    a[i] = b[i]
    b[i] = a_i


###
# Interfaces
###


class PADMMPrimalSolver(PADMMSolver):
    """
    The primal Proximal-ADMM (PADMM) forward dynamics solver for constrained rigid multi-body systems.

    This solver implements the primal Proximal-ADMM algorithm, which operates on the generalized
    velocities of the system and is iterate-wise equivalent to the (dual) Proximal-ADMM of
    :class:`PADMMSolver` with the inverse penalty ``rho_dual = 1 / rho``. It therefore shares the
    iteration kernels of :class:`PADMMSolver`, and differs only in:

    - its configurations, which are expressed in the units and conventions of the primal
      algorithm (see :class:`PADMMPrimalSolverConfig`), and
    - the residuals reported in :attr:`PADMMData.status` and :attr:`PADMMData.info`, which are
      swapped to the primal convention after each solve, i.e. `r_p` is the constraint-velocity
      consistency and `r_d` the consistency of the trial and admissible constraint reactions.

    The solver is designed to operate on a :class:`PrimalProblem`, whose linear operator solves the
    generalized-velocity step with the constraint-augmented mass matrix ``M + J^T W J``, but it accepts
    any :class:`ConstrainedDynamicsProblem` formulation. The internal state holds the iterates in the
    variables of the dual algorithm: ``x``, ``y`` and ``z`` respectively correspond to the trial reactions,
    the admissible reactions and the augmented constraint velocities of the primal algorithm, and the
    penalty held in :attr:`PADMMData.penalty` is the dual penalty ``1 / rho``.
    """

    Config = PADMMPrimalSolverConfig
    """
    Defines a type alias of the primal PADMM solver configurations container.

    See :class:`PADMMPrimalSolverConfig` for the full list of configuration options and their descriptions.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        config: list[PADMMPrimalSolver.Config] | PADMMPrimalSolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        use_graph_conditionals: bool = True,
        collect_info: bool = False,
    ):
        """
        Initializes a primal PADMM solver.

        See :class:`PADMMSolver` for a description of the arguments.
        """
        self._primal_config: list[PADMMPrimalSolver.Config] = []
        super().__init__(
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
    def config(self) -> list[PADMMPrimalSolver.Config]:
        """
        Returns the host-side cache of the solver config in the units of the primal algorithm.
        """
        return self._primal_config

    ###
    # Public API
    ###

    def finalize(
        self,
        model: ModelKamino | None = None,
        config: list[PADMMPrimalSolver.Config] | PADMMPrimalSolver.Config | None = None,
        warmstart: PADMMWarmStartMode = PADMMWarmStartMode.NONE,
        use_acceleration: bool = True,
        use_graph_conditionals: bool = True,
        collect_info: bool = False,
    ):
        """
        Allocates the solver data structures on the specified device.

        See :meth:`PADMMSolver.finalize` for a description of the arguments.
        """
        if config is not None or len(self._primal_config) == 0:
            self._primal_config = self._check_primal_config(model, config)
        super().finalize(
            model=model,
            config=[c.to_dual_units() for c in self._primal_config],
            warmstart=warmstart,
            use_acceleration=use_acceleration,
            use_graph_conditionals=use_graph_conditionals,
            collect_info=collect_info,
        )

    def solve(self, problem: ConstrainedDynamicsProblem):
        """
        Solves the given forward dynamics problem using the primal PADMM.

        Args:
            problem: The forward dynamics problem to be solved.
        """
        super().solve(problem)

        # Report the residuals in the conventions of the primal algorithm
        wp.launch(_swap_status_residuals, dim=self._size.num_worlds, inputs=[self._data.status], device=self.device)
        if self._collect_info:
            info = self._data.info
            for a, b in ((info.r_primal, info.r_dual), (info.r_pd, info.r_dp)):
                wp.launch(_swap_arrays, dim=a.size, inputs=[a, b], device=self.device)

    ###
    # Internals
    ###

    @staticmethod
    def _check_primal_config(
        model: ModelKamino | None = None,
        config: list[PADMMPrimalSolver.Config] | PADMMPrimalSolver.Config | None = None,
    ) -> list[PADMMPrimalSolver.Config]:
        """
        Checks and validates the provided solver config, returning a list
        of config objects corresponding to each world in the model.

        Args:
            model: The model for which to validate the config.
            config: The solver configurations container to validate.
        """
        num_worlds = model.info.num_worlds if model is not None else 1
        if config is None:
            return [PADMMPrimalSolver.Config()] * num_worlds
        elif isinstance(config, PADMMPrimalSolver.Config):
            return [config] * num_worlds
        elif isinstance(config, list):
            if model is not None and len(config) != num_worlds:
                raise ValueError(f"Expected {num_worlds} configs, got {len(config)}")
            if not all(isinstance(c, PADMMPrimalSolver.Config) for c in config):
                raise TypeError("All configs must be instances of PADMMPrimalSolver.Config")
            return config
        else:
            raise TypeError(f"Expected a single object or list of `PADMMPrimalSolver.Config`, got {type(config)}")
