# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified Kamino contact-capacity resolver."""

from __future__ import annotations

import logging
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction import MAX_CONTACTS_PER_PAIR
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.capacity import (
    ContactCapacity,
    _scale_down_to_total,
    _world_neighbor_bounds,
    _world_pair_count_bounds,
)
from newton._src.solvers.kamino.config import CollisionDetectorConfig
from newton.tests.kamino import setup_tests, test_context
from newton.tests.utils import basics

###
# Helpers
###

_UNIT_INERTIA = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _make_nunchaku_newton_model(device: wp.DeviceLike, world_count: int) -> newton.Model:
    """Build a Newton model of ``world_count`` replicated boxes-nunchaku worlds."""
    builder = newton.ModelBuilder()
    builder.replicate(builder=basics.build_boxes_nunchaku(), world_count=world_count)
    return builder.finalize(device=device)


def _make_nunchaku_worlds(device: wp.DeviceLike, world_count: int) -> ModelKamino:
    """Build a Kamino model of ``world_count`` replicated boxes-nunchaku worlds."""
    return ModelKamino.from_newton(_make_nunchaku_newton_model(device, world_count))


def _make_box_worlds(device: wp.DeviceLike, box_counts: list[int]) -> ModelKamino:
    """Build one world per entry of ``box_counts`` above a shared global ground plane.

    A zero entry adds a single free body without shapes, so that world has no contact pairs.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for count in box_counts:
        builder.begin_world()
        if count == 0:
            builder.add_body(mass=1.0, inertia=_UNIT_INERTIA)
        for index in range(count):
            body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * index, 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.end_world()
    return ModelKamino.from_newton(builder.finalize(device=device))


###
# Tests
###


class TestContactCapacityValue(unittest.TestCase):
    """Verify the invariants of the immutable :class:`ContactCapacity` value."""

    def test_total_is_sum_of_worlds(self):
        """Report the model total as the sum of the per-world capacities."""
        capacity = ContactCapacity(world_max_contacts=(10, 20, 0, 5))
        self.assertEqual(capacity.model_max_contacts, 35)
        self.assertEqual(capacity.num_worlds, 4)

    def test_rejects_negative_empty_and_bool(self):
        """Reject negative, missing, boolean, and non-integer per-world capacities."""
        with self.assertRaises(ValueError):
            ContactCapacity(world_max_contacts=(10, -1))
        with self.assertRaises(ValueError):
            ContactCapacity(world_max_contacts=())
        with self.assertRaises(TypeError):
            ContactCapacity(world_max_contacts=(True, 2))
        with self.assertRaises(TypeError):
            ContactCapacity(world_max_contacts=(1.5,))

    def test_accepts_numpy_integers(self):
        """Normalize NumPy integer capacities to Python integers."""
        capacity = ContactCapacity(world_max_contacts=(np.int32(3), np.int64(4)))
        self.assertEqual(capacity.world_max_contacts, (3, 4))
        self.assertTrue(all(type(value) is int for value in capacity.world_max_contacts))

    def test_rejects_totals_beyond_int32(self):
        """Reject capacities whose total exceeds the int32 range of device counters."""
        self.assertEqual(ContactCapacity(world_max_contacts=(2**31 - 1,)).model_max_contacts, 2**31 - 1)
        with self.assertRaisesRegex(ValueError, "max_contacts_per_world or max_contacts"):
            ContactCapacity(world_max_contacts=(2**31 - 1, 1))

    def test_is_immutable(self):
        """Store the per-world capacities as an immutable tuple."""
        capacity = ContactCapacity(world_max_contacts=[1, 2, 3])
        self.assertEqual(capacity.world_max_contacts, (1, 2, 3))
        with self.assertRaises(FrozenInstanceError):
            capacity.world_max_contacts = (4, 5, 6)  # type: ignore[misc]

    def test_uniform(self):
        """Build the same capacity for every world."""
        self.assertEqual(ContactCapacity.uniform(7, 3).world_max_contacts, (7, 7, 7))
        self.assertEqual(ContactCapacity.uniform(0, 2).world_max_contacts, (0, 0))
        with self.assertRaises(ValueError):
            ContactCapacity.uniform(1, 0)
        with self.assertRaises(ValueError):
            ContactCapacity.uniform(-1, 2)


class TestScaleDownToTotal(unittest.TestCase):
    """Verify the integer largest-remainder scaling used by every capacity cap."""

    def test_scale_down_preserves_total(self):
        """Scale per-world budgets down so they sum exactly to the requested total."""
        self.assertEqual(_scale_down_to_total([100, 50, 50], 120), [60, 30, 30])
        result = _scale_down_to_total([7, 11, 13], 20)
        self.assertEqual(sum(result), 20)
        self.assertTrue(all(scaled <= value for scaled, value in zip(result, [7, 11, 13], strict=True)))

    def test_scale_down_never_scales_up(self):
        """Return the budgets unchanged when they already fit the requested total."""
        self.assertEqual(_scale_down_to_total([1, 2, 3], 100), [1, 2, 3])
        self.assertEqual(_scale_down_to_total([1, 2, 3], 6), [1, 2, 3])
        self.assertEqual(_scale_down_to_total([1, 2, 3], 0), [0, 0, 0])

    def test_scale_down_breaks_ties_by_world_index(self):
        """Give leftover slots to the lowest world indices when remainders tie."""
        self.assertEqual(_scale_down_to_total([1, 1, 1], 2), [1, 1, 0])
        self.assertEqual(_scale_down_to_total([5, 5, 5, 5], 6), [2, 2, 1, 1])

    def test_scale_down_is_exact_for_large_values(self):
        """Keep the total exact for magnitudes where floating-point scaling rounds."""
        self.assertEqual(_scale_down_to_total([2**60, 1, 1], 2**60), [2**60 - 2, 1, 1])


class TestContactCapacityResolve(unittest.TestCase):
    """Verify the single sizing rule of :meth:`ContactCapacity.resolve_from`."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_small_pair_bound_is_used_verbatim(self):
        """Use the per-world pair bound unchanged when it is below the dense-scene floor."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)
        pair_bound = list(model.geoms.world_minimum_contacts)
        self.assertTrue(all(0 < bound <= 1000 for bound in pair_bound))

        capacity = ContactCapacity.resolve_from(model)
        self.assertEqual(list(capacity.world_max_contacts), pair_bound)

    def test_dense_world_is_bounded(self):
        """Bound a dense world by the neighbor heuristic instead of its quadratic pair bound."""
        model = _make_box_worlds(self.default_device, [16])
        self.assertEqual(list(model.geoms.world_minimum_contacts), [1024])

        capacity = ContactCapacity.resolve_from(model)
        self.assertEqual(capacity.world_max_contacts, (1000,))

    def test_heterogeneous_worlds_stay_literal(self):
        """Size each world on its own, including a literal zero for a world without contact pairs."""
        model = _make_box_worlds(self.default_device, [16, 1, 0])

        capacity = ContactCapacity.resolve_from(model)
        self.assertEqual(capacity.world_max_contacts, (1000, 4, 0))

    def test_max_contacts_per_world_overrides_and_ignores_max_contacts(self):
        """Apply the uniform per-world override ahead of geometry and the model-wide cap."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)

        override = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts_per_world=37))
        self.assertEqual(override.world_max_contacts, (37, 37, 37))

        ignored_cap = ContactCapacity.resolve_from(
            model, config=CollisionDetectorConfig(max_contacts=1, max_contacts_per_world=25)
        )
        self.assertEqual(ignored_cap.world_max_contacts, (25, 25, 25))

        disabled = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts_per_world=0))
        self.assertEqual(disabled.model_max_contacts, 0)

    def test_max_contacts_caps_total_exactly(self):
        """Scale the geometry budget down to the model-wide cap without exceeding any world's bound."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)
        pair_bound = list(model.geoms.world_minimum_contacts)
        cap = sum(pair_bound) // 3

        capacity = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts=cap))
        self.assertEqual(capacity.model_max_contacts, cap)
        self.assertTrue(all(c <= p for c, p in zip(capacity.world_max_contacts, pair_bound, strict=True)))

    def test_rigid_contact_max_caps_without_warning(self):
        """Scale Kamino down to a known Newton capacity, never up, and report the cap only at info level."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)
        pair_bound = list(model.geoms.world_minimum_contacts)
        total = sum(pair_bound)

        with self.assertNoLogs(level="WARNING"):
            unchanged = ContactCapacity.resolve_from(model, rigid_contact_max=total + 100)
        self.assertEqual(list(unchanged.world_max_contacts), pair_bound)

        with self.assertLogs(level="INFO") as logs:
            capped = ContactCapacity.resolve_from(model, rigid_contact_max=total // 2)
        self.assertEqual(capped.model_max_contacts, total // 2)
        self.assertTrue(all(c <= p for c, p in zip(capped.world_max_contacts, pair_bound, strict=True)))
        self.assertTrue(any("rigid_contact_max" in record.getMessage() for record in logs.records))
        self.assertTrue(any("rigid_contact_max is None" in record.getMessage() for record in logs.records))
        self.assertTrue(all(record.levelno < logging.WARNING for record in logs.records))

    def test_float_max_contacts_caps_like_an_integer(self):
        """Cap with an integral float max_contacts exactly as with the equivalent integer."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)
        cap = sum(model.geoms.world_minimum_contacts) // 3

        as_float = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts=float(cap)))
        as_int = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts=cap))
        self.assertEqual(as_float, as_int)

    def test_rigid_contact_max_zero_gives_empty_capacity(self):
        """Resolve to zero contact capacity when Newton publishes no rigid contacts."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)

        with self.assertLogs(level="WARNING"):
            capacity = ContactCapacity.resolve_from(model, rigid_contact_max=0)
        self.assertEqual(capacity.world_max_contacts, (0, 0, 0))

        with self.assertRaises(ValueError):
            ContactCapacity.resolve_from(model, rigid_contact_max=-1)

    def test_cap_zeroing_world_with_pairs_warns(self):
        """Warn when a cap leaves worlds that have contact pairs without any contact capacity."""
        model = _make_nunchaku_worlds(self.default_device, world_count=3)

        with self.assertLogs(level="WARNING") as logs:
            capacity = ContactCapacity.resolve_from(model, config=CollisionDetectorConfig(max_contacts=1))
        self.assertEqual(capacity.model_max_contacts, 1)
        self.assertEqual(capacity.world_max_contacts.count(0), 2)
        self.assertTrue(any("zero contact capacity" in line for line in logs.output))

    def test_neighbor_bound_ignores_non_colliding_shapes(self):
        """Exclude shapes without shape collision from the neighbor heuristic."""
        visual_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        bounds = []
        for visual_count in (0, 3):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            builder.begin_world()
            body = builder.add_body()
            builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
            for _ in range(visual_count):
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=visual_cfg)
            builder.end_world()
            model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
            bounds.append(_world_neighbor_bounds(model))
        self.assertEqual(bounds[0], bounds[1])

    def test_global_shapes_count_in_every_world(self):
        """Count a shared global ground plane in the neighbor bound of every world."""
        model = _make_box_worlds(self.default_device, [2, 3])
        # Each box adds 20 neighbors x 5 contacts / 2 = 50, and the global plane adds 5 per box.
        self.assertEqual(_world_neighbor_bounds(model), [2 * 50 + 2 * 5, 3 * 50 + 3 * 5])

    def test_pair_count_bound_counts_pairs_per_world(self):
        """Weight the contact pairs of each world, including pairs with global shapes, by contacts per pair."""
        model = _make_box_worlds(self.default_device, [2, 3])
        # World 0 has 2 box-plane and 1 box-box pairs, world 1 has 3 box-plane and 3 box-box pairs.
        self.assertEqual(_world_pair_count_bounds(model), [3 * 5, 6 * 5])

    def test_mesh_world_is_bounded_by_pipeline_estimate(self):
        """Bound a world of mesh shapes on a plane by the pipeline estimate rather than the per-pair reduction cap."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        mesh = newton.Mesh.create_box(0.25, 0.25, 0.25)
        for index in range(12):
            builder.add_shape_mesh(body, xform=wp.transform(wp.vec3(index, 0.0, 0.0), wp.quat_identity()), mesh=mesh)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Twelve mesh-plane pairs reserve 12 x 255 contacts, while the pipeline estimate is 12 x 40.
        self.assertEqual(list(model.geoms.world_minimum_contacts), [12 * MAX_CONTACTS_PER_PAIR])
        self.assertEqual(_world_pair_count_bounds(model), [12 * 40])
        self.assertEqual(ContactCapacity.resolve_from(model).world_max_contacts, (1000,))

    def test_heightfield_world_reserves_reduction_cap(self):
        """Reserve the per-pair reduction cap for a capsule resting on a heightfield."""
        builder = newton.ModelBuilder()
        heightfield = newton.Heightfield(data=np.zeros((8, 8), dtype=np.float32), nrow=8, ncol=8, hx=4.0, hy=4.0)
        builder.add_shape_heightfield(heightfield=heightfield)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_capsule(body, radius=0.1, half_height=0.2)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # One primitive and one mesh-like shape: (5 + 40) contacts x 20 neighbors / 2.
        self.assertEqual(_world_neighbor_bounds(model), [450])
        self.assertEqual(ContactCapacity.resolve_from(model).world_max_contacts, (MAX_CONTACTS_PER_PAIR,))


class TestConversionMetadataInvariance(unittest.TestCase):
    """Verify that the Kamino pair bounds depend only on the model geometry."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_pair_metadata_ignores_preset_rigid_contact_max(self):
        """Derive the pair bounds from geometry even when the Newton model already has a contact capacity."""
        base = ModelKamino.from_newton(_make_nunchaku_newton_model(self.default_device, world_count=3))

        primed_newton_model = _make_nunchaku_newton_model(self.default_device, world_count=3)
        primed_newton_model.rigid_contact_max = 12345
        primed = ModelKamino.from_newton(primed_newton_model)

        self.assertEqual(list(primed.geoms.world_minimum_contacts), list(base.geoms.world_minimum_contacts))
        self.assertEqual(primed.geoms.model_minimum_contacts, base.geoms.model_minimum_contacts)
        self.assertEqual(primed.geoms.model_minimum_contacts, sum(primed.geoms.world_minimum_contacts))


###
# Test execution
###


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
