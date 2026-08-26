# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unified resolver for Kamino contact-buffer capacity.

This module centralizes the logic that determines how many contacts should
be allocated for a given simulation, replacing three previously divergent
estimation pipelines (Kamino geometry-pair metadata, Newton's rigid contact
heuristic, and the DVI blended pre-fill) with a single result type and
policy-driven entry point.

Precedence, applied in order for every policy:

1. ``CollisionDetectorConfig.max_contacts_per_world`` when set. This is a
   deterministic, uniform per-world override and takes precedence over
   every other input.
2. Policy-specific geometry / heuristic sizing:
   * :attr:`ContactCapacity.Policy.INTERNAL_FULL` uses
     :attr:`~newton._src.solvers.kamino._src.core.geometry.GeometriesModel.world_minimum_contacts`
     verbatim (with a narrow geometry fallback used only when pair metadata
     is unavailable, e.g. for standalone ``ModelKamino`` instances built
     without pair enumeration).
   * :attr:`ContactCapacity.Policy.INTERNAL_BOUNDED` returns per-world
     ``min(geometry, bounded Newton heuristic)``. This intentional
     divergence keeps DVI's dense-storage requirements bounded.
   * :attr:`ContactCapacity.Policy.EXTERNAL_NEWTON` honors a nonzero
     ``model.rigid_contact_max`` exactly, otherwise it uses
     :func:`newton._src.sim.collide._estimate_rigid_contact_max` and
     distributes the total across worlds using geometry weights via
     largest-remainder distribution.
3. ``CollisionDetectorConfig.max_contacts`` proportionally scales the
   per-world budget down when the sum exceeds it (largest-remainder rounded
   for exact preservation of the total).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from .....geometry.types import GeoType
from .....sim.collide import (
    _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE,
    _RIGID_CONTACT_MIN_CAPACITY,
    _RIGID_CONTACTS_PER_MESH_PAIR,
    _RIGID_CONTACTS_PER_PRIMITIVE_PAIR,
    _estimate_rigid_contact_max,
)

if TYPE_CHECKING:
    from .....sim.model import Model
    from ...config import CollisionDetectorConfig
    from ..core.model import ModelKamino

###
# Module interface
###

__all__ = [
    "ContactCapacity",
]


###
# Constants
###

# Conservative heuristics for the pair-based fallback path used only when
# pair metadata is unavailable on a standalone ``ModelKamino``.
_EXPLICIT_CONTACTS_PER_PAIR: int = 10
"""Contacts assumed per explicit shape-pair in the geometry fallback."""

_DYNAMIC_CONTACTS_PER_COLLIDABLE: int = 20
"""Contacts assumed per collidable shape in the dynamic fallback."""


###
# Interfaces
###


@dataclass(frozen=True)
class ContactCapacity:
    """An immutable resolved contact-buffer capacity.

    :attr:`world_max_contacts` is a tuple of literal per-world capacities.
    :attr:`model_max_contacts` is derived and always equals their sum.
    """

    ###
    # Types
    ###

    class Policy(IntEnum):
        """Policies driving how :meth:`ContactCapacity.resolve_from` sizes contact buffers."""

        INTERNAL_FULL = 0
        """
        Full internal collision detection (default, e.g. PADMM).

        Per-world capacity equals the Kamino geometry-pair estimate. The model
        total is capped by ``CollisionDetectorConfig.max_contacts`` (proportional
        to the pre-cap per-world weights) unless
        ``CollisionDetectorConfig.max_contacts_per_world`` is set, which takes
        precedence.
        """

        INTERNAL_BOUNDED = 1
        """
        Bounded internal collision detection (i.e. used by DVI).

        Per-world capacity is ``min(geometry, bounded Newton heuristic)``, an
        intentional divergence from :attr:`INTERNAL_FULL` that keeps DVI's dense
        storage requirements finite for scenes whose geometry-pair count grows
        quadratically. ``max_contacts_per_world`` and ``max_contacts`` follow
        the standard precedence.
        """

        EXTERNAL_NEWTON = 2
        """
        External Newton collision detection (``use_collision_detector=False``).

        Honors ``model.rigid_contact_max`` when nonzero; otherwise defers to
        :func:`newton._src.sim.collide._estimate_rigid_contact_max`. The
        resulting model total is distributed across worlds using geometry-pair
        weights via largest-remainder distribution so per-world sums equal the
        model total exactly.
        """

    world_max_contacts: tuple[int, ...]
    """Per-world contact-buffer capacities (host, non-negative integers)."""

    def __post_init__(self) -> None:
        if not isinstance(self.world_max_contacts, tuple):
            object.__setattr__(self, "world_max_contacts", tuple(self.world_max_contacts))
        if len(self.world_max_contacts) == 0:
            raise ValueError("ContactCapacity requires at least one world entry")
        for i, value in enumerate[int](self.world_max_contacts):
            if not isinstance(value, int):
                raise TypeError(f"ContactCapacity.world_max_contacts[{i}] must be int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"ContactCapacity.world_max_contacts[{i}] must be non-negative, got {value}")

    ###
    # Properties
    ###

    @property
    def num_worlds(self) -> int:
        """Number of worlds described by this capacity."""
        return len(self.world_max_contacts)

    @property
    def model_max_contacts(self) -> int:
        """Model-wide contact-buffer capacity, i.e. the sum of per-world entries."""
        return sum(self.world_max_contacts)

    def as_list(self) -> list[int]:
        """Return per-world capacities as a mutable ``list`` for legacy call sites."""
        return list[int](self.world_max_contacts)

    ###
    # Public API
    ###

    @classmethod
    def resolve_from(
        cls: type[ContactCapacity],
        model: ModelKamino,
        config: CollisionDetectorConfig,
        *,
        policy: Policy,
    ) -> ContactCapacity:
        """Resolve a :class:`ContactCapacity` for the given model and policy.

        Args:
            model: The Kamino simulation model whose geometry drives per-world sizing.
            config: The collision-detector configuration providing
                ``max_contacts_per_world`` and ``max_contacts`` precedence knobs.
            policy: The policy to apply for sizing the contact capacity.

        Returns:
            An immutable :class:`ContactCapacity` describing per-world budgets
            with an exact model-wide total sum.
        """
        num_worlds = model.size.num_worlds
        if num_worlds <= 0:
            raise ValueError("Cannot resolve contact capacity for a model with zero worlds")

        # Capture a reference to the Newton model for the bounded policy.
        newton_model = model._model

        # 1. Highest-precedence internal override.
        if config.max_contacts_per_world is not None:
            per_world = int(config.max_contacts_per_world)
            return cls(world_max_contacts=tuple(per_world for _ in range(num_worlds)))

        # 2. Policy-specific sizing.
        match policy:
            # Internal CD, with allocation based on full geometry-pair metadata.
            case cls.Policy.INTERNAL_FULL:
                world_max_contacts = cls._compute_world_weights_from_geometry(model, config)

            # Internal CD, with allocation based on a bounded Newton heuristic.
            case cls.Policy.INTERNAL_BOUNDED:
                if newton_model is None:
                    raise ValueError("INTERNAL_BOUNDED policy requires a Newton model")
                geometry_weights = list(model.geoms.world_minimum_contacts or [0] * num_worlds)
                heuristic = cls._estimate_bounded_world_max_contacts(model, newton_model)

                # If a world has no possible contacts (zero geometry weight) the per-world
                # budget must be zero regardless of the bounded heuristic minimum. Otherwise
                # we return ``min(geometry, heuristic)`` so dense scenes stay bounded but
                # heterogeneous worlds keep their literal per-world sizing.
                world_max_contacts = [
                    min(geom, bound) if geom > 0 else 0 for geom, bound in zip(geometry_weights, heuristic, strict=True)
                ]

            # External CD, with allocation based on the Newton heuristic.
            case cls.Policy.EXTERNAL_NEWTON:
                if newton_model is None:
                    raise ValueError("EXTERNAL_NEWTON policy requires a Newton model")
                newton_total = int(getattr(newton_model, "rigid_contact_max", 0) or 0)
                if newton_total <= 0:
                    newton_total = int(_estimate_rigid_contact_max(newton_model))
                weights = cls._compute_world_weights_from_geometry(model, config)
                world_max_contacts = cls._distribute_total_by_weights(weights, newton_total)

            # Unsupported policies are errors.
            # NOTE: This path currently cannot be reached because the policy
            # enum is exhaustive, but we keep it here for future extensibility.
            case _:
                raise ValueError(f"Unsupported ContactCapacity.Policy: {policy!r}")

        # 3. Optional model-wide cap.
        world_max_contacts = cls._apply_max_contacts_cap(world_max_contacts, config.max_contacts)

        # Return the resolved capacity.
        return cls(world_max_contacts=tuple(int(v) for v in world_max_contacts))

    ###
    # Internals
    ###

    @staticmethod
    def _estimate_fallback_world_max_contacts(
        model: ModelKamino,
        config: CollisionDetectorConfig,
    ) -> list[int]:
        """Estimate per-world contact capacity from geometry when pair metadata is unavailable.

        This narrow fallback path is only intended for standalone
        :class:`ModelKamino` instances built without pair-tables. Any
        Newton->Kamino conversion always populates
        :attr:`~newton._src.solvers.kamino._src.core.geometry.GeometriesModel.model_minimum_contacts`.
        """
        num_worlds = model.size.num_worlds
        world_max_contacts = [0] * num_worlds

        if config.broadphase == "explicit" and model.geoms.collidable_pairs is not None:
            pairs = model.geoms.collidable_pairs.numpy()
            wid = model.geoms.wid.numpy()
            for pair in pairs:
                g0, g1 = int(pair[0]), int(pair[1])
                world_id = int(wid[g0]) if wid[g0] >= 0 else int(wid[g1])
                if 0 <= world_id < num_worlds:
                    world_max_contacts[world_id] += _EXPLICIT_CONTACTS_PER_PAIR
        else:
            wid = model.geoms.wid.numpy()
            group = model.geoms.group.numpy()
            for geom_id in range(len(wid)):
                world_id = int(wid[geom_id])
                if 0 <= world_id < num_worlds and group[geom_id] > 0:
                    world_max_contacts[world_id] += _DYNAMIC_CONTACTS_PER_COLLIDABLE

        return world_max_contacts

    @staticmethod
    def _estimate_bounded_world_max_contacts(model: ModelKamino, newton_model: Model) -> list[int]:
        """Per-world Newton-style bound reused by the bounded policy.

        Returns a list of length ``model.size.num_worlds``. Global (world ``-1``)
        shapes are counted into every world following the same accounting used
        by Newton's ``_estimate_rigid_contact_max``. Single-world models simply
        call the Newton estimator directly.
        """
        num_worlds = model.size.num_worlds
        if num_worlds == 1:
            return [max(_RIGID_CONTACT_MIN_CAPACITY, _estimate_rigid_contact_max(newton_model))]

        geom_world = model.geoms.wid.numpy()
        geom_group = model.geoms.group.numpy()
        geom_type = model.geoms.type.numpy()
        collidable = geom_group > 0
        if not np.any(collidable):
            return [0] * num_worlds

        mesh = collidable & (
            (geom_type == int(GeoType.MESH))
            | (geom_type == int(GeoType.CONVEX_MESH))
            | (geom_type == int(GeoType.HFIELD))
        )
        plane = collidable & (geom_type == int(GeoType.PLANE))
        non_plane = collidable & ~plane
        local = collidable & (geom_world >= 0)

        def count_per_world(mask: np.ndarray) -> np.ndarray:
            global_count = int(np.count_nonzero(mask & (geom_world < 0)))
            local_worlds = geom_world[mask & local]
            return np.bincount(local_worlds, minlength=num_worlds) + global_count

        non_plane_count = count_per_world(non_plane)
        mesh_count = count_per_world(mesh)
        primitive_count = non_plane_count - mesh_count
        plane_count = count_per_world(plane)
        non_plane_contacts = (
            primitive_count * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR
            + mesh_count * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE * _RIGID_CONTACTS_PER_MESH_PAIR
        ) // 2
        plane_contacts = plane_count * (
            primitive_count * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + mesh_count * _RIGID_CONTACTS_PER_MESH_PAIR
        )
        per_world = non_plane_contacts + plane_contacts
        return [max(_RIGID_CONTACT_MIN_CAPACITY, int(v)) for v in per_world]

    @staticmethod
    def _compute_world_weights_from_geometry(model: ModelKamino, config: CollisionDetectorConfig) -> list[int]:
        """Return per-world geometry weights used for distribution and fallbacks."""
        num_worlds = model.size.num_worlds
        world_minimum = list(model.geoms.world_minimum_contacts or [0] * num_worlds)
        if sum(world_minimum) > 0:
            return world_minimum
        return ContactCapacity._estimate_fallback_world_max_contacts(model, config)

    @staticmethod
    def _distribute_total_by_weights(weights: list[int], total: int) -> list[int]:
        """Distribute ``total`` across worlds using largest-remainder rounding.

        Preserves ``sum(result) == total`` exactly and gives the largest
        fractional remainders the +1 rounding, breaking ties by world index.
        When every weight is zero, the total is spread evenly with any remainder
        assigned to the leading worlds.
        """
        if total < 0:
            raise ValueError(f"'total' must be non-negative, got {total}")
        num_worlds = len(weights)
        if num_worlds == 0:
            raise ValueError("'weights' must not be empty")
        if total == 0:
            return [0] * num_worlds

        weight_sum = sum(weights)
        if weight_sum == 0:
            base, remainder = divmod(total, num_worlds)
            return [base + (1 if i < remainder else 0) for i in range(num_worlds)]

        assigned = [0] * num_worlds
        remainders: list[tuple[float, int]] = []
        running = 0
        for i, w in enumerate(weights):
            scaled = w * total / weight_sum
            floor = int(scaled)
            assigned[i] = floor
            running += floor
            remainders.append((scaled - floor, i))
        for _, i in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if running >= total:
                break
            assigned[i] += 1
            running += 1
        return assigned

    @staticmethod
    def _apply_max_contacts_cap(world_max_contacts: list[int], max_contacts: int | None) -> list[int]:
        """Scale per-world budgets down so their sum does not exceed ``max_contacts``."""
        if max_contacts is None:
            return world_max_contacts
        total = sum(world_max_contacts)
        if total <= max_contacts:
            return world_max_contacts
        return ContactCapacity._distribute_total_by_weights(world_max_contacts, max_contacts)
