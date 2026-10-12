# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Resolves the contact-buffer capacity of Kamino simulations.

:class:`ContactCapacity` holds the per-world contact capacities that size a
:class:`~newton._src.solvers.kamino._src.geometry.contacts.ContactsKamino`
container. :meth:`ContactCapacity.resolve_from` derives them with a single rule
shared by every allocation path, i.e. internal collision detection with any
dynamics solver as well as external Newton contacts:

1. ``CollisionDetectorConfig.max_contacts_per_world``, when set, gives every
   world the same capacity.
2. Otherwise each world receives ``min(T_w, max(1000, H_w))``, where ``T_w`` is
   the pair bound of the world (see
   :attr:`~newton._src.solvers.kamino._src.core.geometry.GeometriesModel.world_minimum_contacts`)
   and ``H_w`` is Newton's neighbor-budget heuristic evaluated for that world.
   ``CollisionDetectorConfig.max_contacts`` then scales the total down.
3. A known Newton ``rigid_contact_max`` scales the total down, but never up.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .....geometry import ShapeFlags
from .....geometry.types import GeoType
from .....sim.collide import (
    _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE,
    _RIGID_CONTACT_MIN_CAPACITY,
    _RIGID_CONTACTS_PER_MESH_PAIR,
    _RIGID_CONTACTS_PER_PRIMITIVE_PAIR,
)
from ...config import CollisionDetectorConfig
from ..utils import logger as msg

if TYPE_CHECKING:
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

_INT32_MAX: int = 2**31 - 1
"""Largest model-wide capacity representable by the int32 device counters and indices."""


###
# Interfaces
###


@dataclass(frozen=True)
class ContactCapacity:
    """
    An immutable set of per-world contact-buffer capacities.

    Entries are literal: a zero gives that world no contact slots, and
    :attr:`model_max_contacts` is always the sum of all entries.
    """

    world_max_contacts: tuple[int, ...]
    """The contact capacity of each world, of length :attr:`num_worlds`."""

    def __post_init__(self) -> None:
        values: list[int] = []
        for i, entry in enumerate(self.world_max_contacts):
            if isinstance(entry, bool):
                raise TypeError(f"ContactCapacity.world_max_contacts[{i}] must be an integer, got bool")
            try:
                value = operator.index(entry)
            except TypeError as err:
                raise TypeError(
                    f"ContactCapacity.world_max_contacts[{i}] must be an integer, got {type(entry).__name__}"
                ) from err
            if value < 0:
                raise ValueError(f"ContactCapacity.world_max_contacts[{i}] must be non-negative, got {value}")
            values.append(value)
        if len(values) == 0:
            raise ValueError("ContactCapacity requires at least one world")
        if sum(values) > _INT32_MAX:
            raise ValueError(f"ContactCapacity total of {sum(values)} contacts exceeds the int32 limit of {_INT32_MAX}")
        object.__setattr__(self, "world_max_contacts", tuple(values))

    ###
    # Properties
    ###

    @property
    def num_worlds(self) -> int:
        """The number of worlds described by this capacity."""
        return len(self.world_max_contacts)

    @property
    def model_max_contacts(self) -> int:
        """The model-wide contact capacity, i.e. the sum of all per-world capacities."""
        return sum(self.world_max_contacts)

    ###
    # Constructors
    ###

    @classmethod
    def uniform(cls, per_world: int, num_worlds: int) -> ContactCapacity:
        """
        Creates a capacity that gives every world the same number of contacts.

        Args:
            per_world: The contact capacity of each world.
            num_worlds: The number of worlds.

        Returns:
            The uniform :class:`ContactCapacity`.
        """
        return cls(world_max_contacts=(per_world,) * num_worlds)

    @classmethod
    def resolve_from(
        cls,
        model: ModelKamino,
        *,
        config: CollisionDetectorConfig | None = None,
        rigid_contact_max: int | None = None,
    ) -> ContactCapacity:
        """
        Resolves the contact capacity of a model using the rule described in :mod:`.capacity`.

        Args:
            model: The Kamino model whose geometry determines the per-world capacities.
            config: The collision-detector configuration providing the
                ``max_contacts_per_world`` override and the ``max_contacts`` cap.
                If ``None``, the default configuration is used.
            rigid_contact_max: The rigid-contact capacity of the Newton model, if known.
                It caps the resolved total, but never raises it.

        Returns:
            The resolved :class:`ContactCapacity`.
        """
        if config is None:
            config = CollisionDetectorConfig()
        if rigid_contact_max is not None and rigid_contact_max < 0:
            raise ValueError(f"`rigid_contact_max` must be non-negative, got {rigid_contact_max}")

        # Per-world budget, before any model-wide cap
        pair_bounds = _world_pair_bounds(model)
        if config.max_contacts_per_world is not None:
            uncapped = [int(config.max_contacts_per_world)] * len(pair_bounds)
            budget = uncapped
        else:
            neighbor_bounds = _world_neighbor_bounds(model)
            uncapped = [
                min(pair, max(_RIGID_CONTACT_MIN_CAPACITY, neighbor))
                for pair, neighbor in zip(pair_bounds, neighbor_bounds, strict=True)
            ]
            budget = uncapped
            if config.max_contacts is not None:
                budget = _scale_down_to_total(uncapped, config.max_contacts)

        # The Newton buffer, when it already exists, bounds what Kamino can ever receive or export
        world_max_contacts = budget
        if rigid_contact_max is not None:
            world_max_contacts = _scale_down_to_total(budget, rigid_contact_max)
            if sum(world_max_contacts) < sum(budget):
                msg.warning(
                    "ContactCapacity: Model.rigid_contact_max (%d) reduces the Kamino contact capacity from %d to %d. "
                    "Create the Kamino solver before the CollisionPipeline, or raise rigid_contact_max, "
                    "to keep the full capacity.",
                    rigid_contact_max,
                    sum(budget),
                    sum(world_max_contacts),
                )

        starved = [
            wid
            for wid, (pair, before, after) in enumerate(zip(pair_bounds, uncapped, world_max_contacts, strict=True))
            if pair > 0 and before > 0 and after == 0
        ]
        if len(starved) > 0:
            msg.warning(
                "ContactCapacity: %d world(s) with contact pairs have zero contact capacity after capping, "
                "starting with world(s) %s.",
                len(starved),
                starved[:8],
            )

        capacity = cls(world_max_contacts=tuple(world_max_contacts))
        msg.debug(
            "ContactCapacity: resolved %d contacts over %d worlds (pair bound: %d, rigid_contact_max: %s).",
            capacity.model_max_contacts,
            capacity.num_worlds,
            sum(pair_bounds),
            rigid_contact_max,
        )
        return capacity


###
# Internals
###


def _world_pair_bounds(model: ModelKamino) -> list[int]:
    """Returns the pair bound of each world, i.e. the sum of the per-pair contact limits of its shape pairs."""
    num_worlds = model.size.num_worlds
    pair_bounds = model.geoms.world_minimum_contacts
    if pair_bounds is None:
        return [0] * num_worlds
    pair_bounds = [int(bound) for bound in pair_bounds]
    if len(pair_bounds) != num_worlds:
        raise ValueError(f"Expected {num_worlds} per-world pair bounds, got {len(pair_bounds)}")
    return pair_bounds


def _world_neighbor_bounds(model: ModelKamino) -> list[int]:
    """
    Evaluates Newton's neighbor-budget contact heuristic for each world.

    Each colliding non-plane shape is assumed to touch at most
    ``_RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE`` other shapes, at
    ``_RIGID_CONTACTS_PER_PRIMITIVE_PAIR`` contacts per primitive pair and
    ``_RIGID_CONTACTS_PER_MESH_PAIR`` contacts per mesh or heightfield pair,
    halved because each pair is counted from both sides. Each plane is then
    assumed to touch every non-plane shape. Global shapes count in every world.
    """
    num_worlds = model.size.num_worlds
    if model.geoms.num_geoms == 0:
        return [0] * num_worlds

    geom_world = model.geoms.wid.numpy()
    geom_flags = model.geoms.flags.numpy()
    geom_group = model.geoms.group.numpy()
    geom_type = model.geoms.type.numpy()

    # Group zero collides with nothing, while negative groups collide with all other groups
    colliding = ((geom_flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0) & (geom_group != 0)
    mesh = colliding & np.isin(geom_type, (int(GeoType.MESH), int(GeoType.HFIELD)))
    plane = colliding & (geom_type == int(GeoType.PLANE))
    primitive = colliding & ~mesh & ~plane

    def count_per_world(mask: np.ndarray) -> np.ndarray:
        local = geom_world[mask & (geom_world >= 0)]
        num_global = np.count_nonzero(mask & (geom_world < 0))
        return np.bincount(local, minlength=num_worlds)[:num_worlds].astype(np.int64) + num_global

    num_primitives = count_per_world(primitive)
    num_meshes = count_per_world(mesh)
    num_planes = count_per_world(plane)
    contacts_per_neighbor = (
        num_primitives * _RIGID_CONTACTS_PER_PRIMITIVE_PAIR + num_meshes * _RIGID_CONTACTS_PER_MESH_PAIR
    )
    neighbor_bounds = contacts_per_neighbor * _RIGID_CONTACT_MAX_NEIGHBORS_PER_SHAPE // 2
    neighbor_bounds += num_planes * contacts_per_neighbor
    return [int(bound) for bound in neighbor_bounds]


def _scale_down_to_total(values: Sequence[int], total: int) -> list[int]:
    """
    Scales non-negative per-world values down so that their sum equals ``total``.

    Uses exact integer largest-remainder rounding, breaking ties by world index.
    Values are returned unchanged when their sum does not exceed ``total``,
    so the result never exceeds any input value.
    """
    current = sum(values)
    if total >= current:
        return list(values)
    if total <= 0:
        return [0] * len(values)
    scaled = [divmod(value * total, current) for value in values]
    result = [quotient for quotient, _ in scaled]
    shortfall = total - sum(result)
    for i in sorted(range(len(values)), key=lambda i: (-scaled[i][1], i))[:shortfall]:
        result[i] += 1
    return result
