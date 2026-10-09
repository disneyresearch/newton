# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Primal Proximal-ADMM Solver

Provides a forward dynamics solver for constrained rigid multi-body systems that realizes the
primal variant of the Proximal-ADMM algorithm of J. Carpentier et al. in [1], operating on the
generalized velocities ``u`` of the system instead of the constraint reactions. Starting from the
velocity-level primal forward dynamics problem (i.e. Gauss' principle in the kinetic metric), each
iteration consists of:

1. the generalized-velocity step, i.e. a linear solve with the constraint-augmented generalized
   mass matrix ``M + J^T W J``, where ``W = (R + (eta + 1/rho) I)^{-1}`` is the effective constraint
   stiffness combining the physical compliance ``R``, the proximal parameter ``eta`` and the
   penalty ``rho``, which renders the trial reactions ``lambda~ = W (b - J u)``,
2. the cone step, i.e. a Moreau decomposition onto the cones ``K`` and ``K^*``, which renders
   admissible reactions ``lambda`` and augmented constraint velocities ``v^`` that are exactly
   complementary at every iteration, and
3. the De Saxce update of the cascade formulation of the Coulomb friction cone NCP.

With ``rho = 1 / rho_d``, the primal iterates coincide with those of the dual Proximal-ADMM of
:class:`PADMMSolver` (an instance of the self-duality of ADMM), i.e. ``(lambda~, lambda, v^)`` are
the dual iterates ``(x, y, z)``, including the proximal regularization, the compliance, the residuals
and the penalty updates. The solver therefore shares the iteration kernels of :class:`PADMMSolver`
and only differs in the linear operator of the problem it operates on, i.e. :class:`PrimalProblem`.

The primal variant never requires the Delassus matrix and its linear system has the fixed dimension
of the generalized velocities, independent of the number of active constraints, which makes it
preferable for contact-rich and hyperstatic systems with many more constraints than DoFs.

References
----
- [1] https://arxiv.org/pdf/2405.17020
- [2] https://arxiv.org/abs/2504.19771

Usage
----
A typical example for using this module is:

    # Create a model, data, limits, contacts and Jacobians containers
    ...

    # Create a forward-dynamics PrimalProblem to be solved
    primal = PrimalProblem(model, data, limits, contacts, jacobians, LLTBlockedSolver, sparse=False)
    primal.build(model, data, jacobians, limits, contacts)

    # Create a forward-dynamics primal PADMM solver
    solver = PADMMPrimalSolver(model)

    # Solve the primal forward dynamics problem
    solver.coldstart()
    solver.solve(problem=primal)

    # Extract the resulting constraint reactions
    multipliers = solver.data.solution.lambdas
"""

from .solver import PADMMPrimalSolver

###
# Module interface
###

__all__ = [
    "PADMMPrimalSolver",
]
