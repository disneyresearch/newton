.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Kamino
======

:class:`~newton.solvers.SolverKamino` simulates constrained rigid multi-body
systems in maximal coordinates. It is designed for mechanical assemblies with
kinematic loops, under- or overactuation, joint limits, hard frictional
contacts, and restitutive impacts.

Unlike the other maximal-coordinate solvers, Kamino focuses on constrained
rigid mechanical assemblies rather than particle or deformable simulation.
Kamino is currently in BETA 1, and Newton users are discouraged from depending
on it. Evaluate it only when kinematic loops and hard contact constraints are
primary requirements and an experimental solver is acceptable.

.. experimental::

   :class:`~newton.solvers.SolverKamino` is experimental. Its public API,
   behavior, feature support, performance, and implementation may change
   without prior notice.

See the :class:`~newton.solvers.SolverKamino` API reference for construction
and configuration details. Runnable workflows are available in the
`Kamino examples <https://github.com/newton-physics/newton/tree/main/newton/examples/kamino>`_.

Body acceleration
-----------------

Kamino populates :attr:`~newton.State.body_qdd` when the extended state
attribute is requested. Values are center-of-mass spatial accelerations in the
world frame, with linear acceleration followed by angular acceleration. For a
step of duration ``dt``, Kamino reports the discrete step average
``(body_qd_out - body_qd_in) / dt``. Consequently, a contact impact reports its
velocity impulse divided by ``dt`` rather than a continuous midpoint
acceleration. Resetting a state clears the requested acceleration in each reset
world.

Choosing a dynamics solver
--------------------------

Kamino provides three forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
* ``"padmm_primal"`` (opt-in): the primal variant of proximal ADMM, which solves
  each iteration's linear system in the generalized velocities, using the
  constraint-augmented mass matrix ``M + J^T W J`` instead of the Delassus
  matrix. Its iterations are equivalent to those of ``"padmm"`` with the
  inverse penalty, so it is equally robust, while its linear system keeps the
  fixed size of the generalized velocities regardless of the number of active
  constraints. It is therefore preferable for contact-rich and hyperstatic
  systems. Its penalty parameters in ``config.padmm_primal`` are expressed in
  primal units, i.e. as the inverses of those of ``"padmm"``.
* ``"dvi"`` (opt-in): projected dual iterations, sparse Jacobians, dense dynamics
  with the RCM-reordered blocked LLT solver, and the Euler integrator. It is
  generally faster, but approximates the coupled problem by alternating between
  a direct solve for equality constraints and projected iterations for
  inequality constraints. As a rule of thumb, DVI solves inequality constraints
  less accurately than PADMM, particularly as the number of active inequalities
  grows. Dual preconditioning is not supported.

Select the backend when constructing the configuration so dependent defaults
initialize consistently:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(dynamics_solver="dvi")
   solver = newton.solvers.SolverKamino(model, config=config)

DVI is best suited to performance-sensitive rigid mechanisms with relatively
few active contacts; PADMM remains the safer and more broadly validated choice.
Set ``sparse_jacobian=False`` for fully dense DVI, or set
``sparse_dynamics=True`` to use sparse dynamics with the Conjugate Residual
solver.

For large bilateral systems, opt into RCM-reordered factorization explicitly:

.. code-block:: python

   config.dvi.bilateral_solver_type = "LLTBRCM"
   config.dvi.bilateral_solver_kwargs = {
       "block_size": 32,
       "reuse_permutation": True,
       "parallel_factorization": True,
   }

The cached permutation remains mathematically valid when matrix values or
sparsity change and is recomputed automatically if the active dimension
changes. Keep the default ``"LLTB"`` solver for small systems.

DVI unilateral subsolvers
^^^^^^^^^^^^^^^^^^^^^^^^^

DVI offers two subsolvers for bounded joint rows, limits, and contacts:

* **PGS** (default) uses projected Gauss-Seidel updates with less work per
  iteration, making it a simple choice for modest contact problems.
* **APGD** offers a faster convergence rate for each frozen-correction
  quadratic problem, at the cost of objective-gradient evaluations and an
  inner backtracking loop.
  CUDA execution requires conditional-graph support.

Both support dense and sparse operators and bilateral Schur elimination.
See the :class:`~newton.solvers.SolverKamino` configuration reference for
subsolver selection and iteration controls.

PADMM convergence options
^^^^^^^^^^^^^^^^^^^^^^^^^

Both PADMM backends share the following opt-in options of ``config.padmm``
(or ``config.padmm_primal``), all of which are disabled by default:

* **Spectral penalty:** ``penalty_update_method="spectral"`` sets the penalty
  to ``rho = sqrt(m L) (L / m)^p``, where ``m`` and ``L`` are the extreme
  eigenvalues of the regularized, preconditioned Delassus operator, which are
  estimated by power iteration at the beginning of each step. The exponent
  ``p`` starts at ``spectral_power_0`` and is adapted from the ratio of the
  primal and dual residuals within ``[spectral_power_min, spectral_power_max]``,
  i.e. ``sqrt(m L) <= rho <= L`` by default. With ``penalty_update_freq=N``,
  the iterations are performed in blocks of ``N``, after each of which the
  penalty is updated and the linear system is re-factorized, so larger values
  reduce the cost of the updates. Unlike ``"balanced"``, the spectral penalty
  also supports dense dynamics.
* **Smallest eigenvalue:** By default, ``m`` is the lower bound
  ``eta + min(P R P)``, which is exact for hyperstatic rigid constraint
  configurations. ``spectral_min_eigenvalue="inverse"`` refines it by inverse
  iteration for non-singular configurations, at the cost of
  ``spectral_inverse_iterations`` linear solves and one additional
  factorization per step.
* **Penalty warm-start:** ``penalty_warmstart=True`` starts each step from the
  last spectral exponent, or the last ``"balanced"`` penalty, of the previous
  step instead of the initial values.
* **Velocity warm-start:** ``warmstart_velocity="consistent"`` initializes the
  augmented constraint velocities from the warm-started impulses of the
  current step, i.e. projects the De Saxce-corrected velocity
  ``(D + R) lambda_0 + v_f`` onto the dual cone, at the cost of one
  matrix-vector product, instead of reusing the velocities cached in the
  previous step.
* **Relative tolerances:** ``primal_tolerance_rel``, ``dual_tolerance_rel``,
  and ``compl_tolerance_rel`` add tolerances proportional to the
  infinity-norms of the physical impulses, the augmented constraint
  velocities, and their product, respectively, to the absolute tolerances.
  They only affect the stopping criterion, not the reported residuals.

For ``"padmm_primal"``, the spectral penalty is the inverse of the dual one,
and ``primal_tolerance_rel`` and ``dual_tolerance_rel`` are relative to the
constraint velocities and impulses, respectively. Additionally,
``config.dynamics.augmented_mass_preconditioner`` selects a symmetric
``"jacobi"`` or ``"block_jacobi"`` (per-body 6x6 Cholesky) scaling of the
constraint-augmented mass matrix ``M + J^T W J`` for its linear solves. The
default ``None`` selects ``"none"`` for dense and ``"jacobi"`` for sparse
dynamics. The scaling only changes the conditioning of the linear solves, not
the PADMM iterates.

.. code-block:: python

   config = newton.solvers.SolverKamino.Config()
   config.padmm.penalty_update_method = "spectral"
   config.padmm.penalty_update_freq = 10
   config.padmm.primal_tolerance_rel = 1e-4
   config.padmm.dual_tolerance_rel = 1e-4
   config.padmm.compl_tolerance_rel = 1e-4

Constraint compliance
---------------------

By default, joint, limit, and contact constraints are rigid. A diagonal
constraint compliance ``R`` (also known as constraint force mixing) can soften
them, so that the post-event constraint velocity becomes
``v+ = J u+ + v* + R lambda`` for the constraint impulses ``lambda``.
Compliance has units of velocity per impulse, and is configured globally per
constraint type:

.. code-block:: python

   config = newton.solvers.SolverKamino.Config()
   config.compliance.limits = 1e-4
   config.compliance.contacts_normal = 1e-4
   config.compliance.contacts_tangential = 1e-4

Joint friction constraints are never compliant, and the rows of joints with
armature, damping, or implicit PD actuation always use the compliance implied
by their implicit joint dynamics. All forward-dynamics backends support
compliance, and it incurs no cost for fully rigid systems.

Inspecting terminal status
--------------------------

After each step, :attr:`~newton.solvers.SolverKamino.status` provides one
device-resident terminal status record per world. All backends provide
``converged``, ``iterations``, ``r_p``, ``r_d``, and ``r_c`` fields. Their
residual definitions are backend-specific:

* **PADMM:** Let ``x`` and ``y`` be the current preconditioned impulse iterates,
  ``x_prev`` and ``y_prev`` their previous values, ``P`` the diagonal dual
  preconditioner, and ``eta`` and ``rho`` the proximal and penalty parameters.
  ``r_p = ||P (x - y)||_inf`` is the primal consensus residual
  [N·s or N·m·s].
  ``r_d = ||P^-1 (eta (x - x_prev) + rho (y - y_prev))||_inf`` is the ADMM dual
  residual [m/s or rad/s]. ``r_c`` is the maximum absolute impulse-velocity
  inner product over inequality blocks [J]. The ``P`` factors
  convert the first two residuals from solver scaling to physical constraint
  units.
* **Primal PADMM:** The same residuals as PADMM, reported in the conventions of
  the primal algorithm, i.e. with ``r_p`` and ``r_d`` swapped: ``r_p`` is the
  constraint-velocity consistency [m/s or rad/s] and ``r_d`` the consistency of
  the trial and admissible impulses [N·s or N·m·s].
* **DVI:** With physical impulse ``lambda`` and augmented constraint velocity
  ``v``, ``r_p`` is the maximum infinity-norm projection distance of unilateral
  impulses from the nonnegative limit cone or Coulomb contact cone
  [N·s or N·m·s]. ``r_d`` is the maximum of the corresponding velocity distance
  from the dual cone and the bilateral velocity violation [m/s or rad/s].
  ``r_c = max |lambda_k dot v_k|`` is the maximum inequality complementarity
  violation [J].

These are absolute maxima: neither backend divides them by a reference norm,
constraint count, or tolerance, including when the PADMM relative tolerances
are enabled. Additional fields are not portable between backends.

.. code-block:: python

   status = solver.status
   assert status.device == model.device
   assert status.shape == (model.world_count,)

   # Host inspection is explicit and synchronizes the device-to-host copy.
   status_host = status.numpy()
   unconverged_worlds = (~status_host["converged"].astype(bool)).nonzero()[0]

Terminal status is always maintained. ``collect_solver_info=True`` enables
additional solver diagnostics and adds runtime and memory overhead; it is not
required to access ``status``.

Actuation and forward kinematics
--------------------------------

Kamino dynamics routes actuation independently for each joint DoF. A DoF can
use explicit effort, unbounded implicit PD, or effort-limited implicit PD;
passive armature, damping, and Coulomb friction are likewise configured per
DoF. Implicit-PD target modes require a non-zero applicable gain: velocity
mode requires derivative gain, while position-based modes require proportional
or derivative gain. Coulomb friction supports all non-free joint types, while
joint dynamics and implicit PD currently support revolute, prismatic, and
gimbal joint types only.

The forward-kinematics solver still partitions each joint as entirely passive
or entirely actuated. Different non-passive target modes are allowed within a
joint, but mixing passive and actuated DoFs within one joint is not yet
supported. The ``fk_actuation_flag`` model attribute provides an explicit
joint-level override for this FK partition.
