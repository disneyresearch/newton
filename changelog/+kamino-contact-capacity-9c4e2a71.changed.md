Unify `SolverKamino` contact-buffer sizing behind a single resolver:

  - Kamino previously used three separate estimation pipelines (geometry-pair
    metadata, Newton's `_estimate_rigid_contact_max`, and a DVI-specific
    blended pre-fill). All construction paths now go through
    `resolve_contact_capacity`, which returns an immutable `ContactCapacity`
    with literal per-world buffers whose sum equals the model total.
  - `Model.rigid_contact_max` no longer influences the Kamino model
    conversion: `ModelKamino.from_newton` always computes
    `model.geoms.model_minimum_contacts` / `world_minimum_contacts` from
    the shape-pair geometry. `rigid_contact_max` is an input only for
    external Newton contacts (`use_collision_detector=False`) and an
    output otherwise, so construction order no longer matters.
  - External Newton contact totals are preserved exactly instead of being
    rounded up to a multiple of `world_count` (previously
    `rigid_contact_max=1000` became `1002` for a three-world model).
  - The DVI internal policy now returns heterogeneous per-world budgets
    instead of a single uniform maximum, while still bounding dense scenes
    with the same Newton heuristic used before.

Configuration precedence for internal collision detection is unchanged:
`CollisionDetectorConfig.max_contacts_per_world` overrides everything,
`CollisionDetectorConfig.max_contacts` proportionally caps the model total
via largest-remainder rounding, and `CollisionDetectorConfig.max_contacts_per_pair`
narrows the geometry estimate. For external Newton contacts, set
`Model.rigid_contact_max` directly; the internal knobs are ignored in that
mode.
