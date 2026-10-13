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

Kamino provides two forward-dynamics backends:

* ``"padmm"`` (default): proximal ADMM, dense Jacobians/dynamics, and the Euler
  integrator. It is the slower, more robust option because it solves equality
  and inequality constraints together.
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

Contact capacity
----------------

Kamino allocates its contact buffers once, when the solver is constructed, with
a fixed capacity for each world. The same rule applies to PADMM and DVI, and to
both contact sources: Kamino's internal collision detection
(``use_collision_detector=True``) and contacts from a
:class:`~newton.CollisionPipeline`.

By default, the capacity of a world is the sum of the per-shape-pair contact
limits of its colliding shape pairs, capped at the larger of 1000 and the
:class:`~newton.CollisionPipeline` capacity estimate evaluated for that world.
Pairs that involve a mesh or heightfield reserve the 255 contacts that Newton's
contact reduction can keep for one pair, so worlds with several such pairs
usually reach the cap. Two settings of ``config.collision_detector`` override
this default:

* ``max_contacts_per_world`` gives every world exactly this capacity. Use it to
  bound memory in contact-rich scenes; ``0`` allocates no contacts.
* ``max_contacts`` caps the total. The per-world capacities are scaled down
  proportionally so that they sum to at most this value.

Dense dynamics, the PADMM default, allocate per-world matrices whose size grows
with the square of the per-world capacity. For many worlds with mesh or
heightfield contacts, set ``max_contacts_per_world`` or use
``sparse_dynamics=True``.

.. code-block:: python

   config = newton.solvers.SolverKamino.Config(use_collision_detector=True)
   config.collision_detector.max_contacts_per_world = 64
   solver = newton.solvers.SolverKamino(model, config=config)

``Model.rigid_contact_max`` links this capacity to Newton's contact buffers. If
it is ``None`` when the solver is constructed, Kamino sets it to its own total,
so a :class:`~newton.CollisionPipeline` created afterwards allocates a matching
buffer. If it is already set, by the user or by an existing pipeline, Kamino
treats it as an upper bound and scales the per-world capacities down to fit.
Kamino warns only if this leaves a world that has colliding shape pairs without
any capacity. Construct the solver before the pipeline to keep Kamino's full
capacity, or set ``model.rigid_contact_max`` explicitly to choose one memory
budget for both.

When a step generates more contacts than a world can hold, Kamino keeps the
contacts that fit, drops the rest, and prints a warning for that step. Raise
``max_contacts_per_world`` if this happens regularly.

Inspecting terminal status
--------------------------

After each step, :attr:`~newton.solvers.SolverKamino.status` provides one
device-resident terminal status record per world. PADMM and DVI both provide
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
* **DVI:** With physical impulse ``lambda`` and augmented constraint velocity
  ``v``, ``r_p`` is the maximum infinity-norm projection distance of unilateral
  impulses from the nonnegative limit cone or Coulomb contact cone
  [N·s or N·m·s]. ``r_d`` is the maximum of the corresponding velocity distance
  from the dual cone and the bilateral velocity violation [m/s or rad/s].
  ``r_c = max |lambda_k dot v_k|`` is the maximum inequality complementarity
  violation [J].

These are absolute maxima: neither backend divides them by a reference norm,
constraint count, or tolerance. Additional fields are not portable between
backends.

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
