# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified Kamino contact-capacity resolver."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import warp as wp

import newton
from newton._src.sim.collide import _estimate_rigid_contact_max
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.capacity import (
    ContactCapacity,
    ContactCapacityPolicy,
    _distribute_total_by_weights,
    resolve_contact_capacity,
)
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.config import CollisionDetectorConfig
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.utils import basics as public_basics


def _make_boxes_nunchaku_kamino(device: wp.DeviceLike, num_worlds: int = 3) -> ModelKamino:
    """Build a Kamino model of ``num_worlds`` boxes_nunchaku worlds."""
    builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=basics.build_boxes_nunchaku)
    return builder.finalize(device=device)


def _make_stacked_boxes_newton_model(device: wp.DeviceLike, num_boxes: int = 16) -> newton.Model:
    """Build a Newton model of ``num_boxes`` boxes above a ground plane."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.add_ground_plane()
    for box_index in range(num_boxes):
        body = builder.add_body(xform=wp.transform(wp.vec3(float(box_index), 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    return builder.finalize(device=device)


def _make_three_world_sphere_on_plane(device: wp.DeviceLike) -> newton.Model:
    """Build a Newton model with three replicated sphere-on-plane worlds."""
    source = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverKamino.register_custom_attributes(source)
    public_basics.build_sphere_on_plane(builder=source, z_offset=0.5)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverKamino.register_custom_attributes(builder)
    builder.replicate(source, world_count=3)
    return builder.finalize(device=device, skip_validation_joints=True)


class TestContactCapacityDataclass(unittest.TestCase):
    """Verify structural invariants of the immutable :class:`ContactCapacity`."""

    def test_model_total_is_sum_of_world_capacities(self):
        """The model total is always the sum of the per-world tuple entries."""
        capacity = ContactCapacity(world_max_contacts=(10, 20, 0, 5))
        self.assertEqual(capacity.model_max_contacts, 35)
        self.assertEqual(capacity.num_worlds, 4)

    def test_negative_capacity_is_rejected(self):
        """Negative per-world budgets are rejected at construction time."""
        with self.assertRaises(ValueError):
            ContactCapacity(world_max_contacts=(10, -1))

    def test_zero_worlds_is_rejected(self):
        """An empty per-world tuple is not a valid capacity result."""
        with self.assertRaises(ValueError):
            ContactCapacity(world_max_contacts=())

    def test_capacity_is_immutable(self):
        """The per-world capacities must be an immutable tuple of ints."""
        capacity = ContactCapacity(world_max_contacts=(1, 2, 3))
        self.assertIsInstance(capacity.world_max_contacts, tuple)
        with self.assertRaises(FrozenInstanceError):
            capacity.world_max_contacts = (4, 5, 6)  # type: ignore[misc]


class TestDistributeTotalByWeights(unittest.TestCase):
    """Verify the largest-remainder distribution of model totals across worlds."""

    def test_distribution_preserves_total(self):
        """Exact sum preservation for both scale-up and scale-down cases."""
        self.assertEqual(sum(_distribute_total_by_weights([2, 3, 5], 100)), 100)
        self.assertEqual(sum(_distribute_total_by_weights([1, 1, 1], 10)), 10)
        self.assertEqual(sum(_distribute_total_by_weights([100, 50, 50], 120)), 120)

    def test_scale_down_matches_legacy_cap(self):
        """Legacy proportional cap behavior is preserved for down-scaling."""
        result = _distribute_total_by_weights([100, 50, 50], 120)
        self.assertEqual(sum(result), 120)
        self.assertEqual(result[0], 60)

    def test_zero_total_zeroes_all_worlds(self):
        """A zero model total leaves every world at zero."""
        self.assertEqual(_distribute_total_by_weights([1, 2, 3], 0), [0, 0, 0])

    def test_all_zero_weights_produce_equal_split(self):
        """When weights are all zero, the total is distributed evenly across worlds."""
        result = _distribute_total_by_weights([0, 0, 0], 9)
        self.assertEqual(sum(result), 9)
        self.assertEqual(result, [3, 3, 3])

    def test_all_zero_weights_with_remainder(self):
        """Remainders from equal splits are distributed deterministically."""
        result = _distribute_total_by_weights([0, 0, 0], 10)
        self.assertEqual(sum(result), 10)
        # The remainder (1) is assigned to the first world.
        self.assertEqual(result[0], 4)


class TestResolveContactCapacityInternalFull(unittest.TestCase):
    """PADMM / standalone detector: geometry-pair metadata drives capacity."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_geometry_metadata_is_canonical(self):
        """Kamino geometry metadata dictates model and per-world budgets."""
        model = _make_boxes_nunchaku_kamino(self.default_device, num_worlds=3)
        capacity = resolve_contact_capacity(
            model,
            newton_model=None,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.INTERNAL_FULL,
        )
        self.assertEqual(
            list(capacity.world_max_contacts),
            list(model.geoms.world_minimum_contacts),
        )
        self.assertEqual(capacity.model_max_contacts, model.geoms.model_minimum_contacts)

    def test_max_contacts_caps_model_total(self):
        """A model-wide cap proportionally reduces the geometry estimate."""
        model = _make_boxes_nunchaku_kamino(self.default_device, num_worlds=3)
        uncapped = sum(model.geoms.world_minimum_contacts)
        cap = max(15, uncapped // 3)
        self.assertLess(cap, uncapped)
        capacity = resolve_contact_capacity(
            model,
            newton_model=None,
            config=CollisionDetectorConfig(max_contacts=cap),
            policy=ContactCapacityPolicy.INTERNAL_FULL,
        )
        self.assertEqual(capacity.model_max_contacts, cap)
        self.assertEqual(sum(capacity.world_max_contacts), cap)

    def test_max_contacts_per_world_is_uniform_override(self):
        """``max_contacts_per_world`` produces a uniform per-world allocation."""
        model = _make_boxes_nunchaku_kamino(self.default_device, num_worlds=3)
        capacity = resolve_contact_capacity(
            model,
            newton_model=None,
            config=CollisionDetectorConfig(max_contacts_per_world=37),
            policy=ContactCapacityPolicy.INTERNAL_FULL,
        )
        self.assertEqual(list(capacity.world_max_contacts), [37, 37, 37])
        self.assertEqual(capacity.model_max_contacts, 3 * 37)

    def test_max_contacts_per_world_ignores_max_contacts(self):
        """The explicit per-world override bypasses ``max_contacts``."""
        model = _make_boxes_nunchaku_kamino(self.default_device, num_worlds=3)
        capacity = resolve_contact_capacity(
            model,
            newton_model=None,
            config=CollisionDetectorConfig(max_contacts=1, max_contacts_per_world=25),
            policy=ContactCapacityPolicy.INTERNAL_FULL,
        )
        self.assertEqual(list(capacity.world_max_contacts), [25, 25, 25])

    def test_heterogeneous_worlds_receive_geometry_budgets(self):
        """Heterogeneous per-world budgets follow the geometry, not an equal split."""
        builder = ModelBuilderKamino(default_world=False)
        basics.build_boxes_nunchaku(builder=builder)
        builder.add_world(name="empty_world")
        model = builder.finalize(self.default_device)

        capacity = resolve_contact_capacity(
            model,
            newton_model=None,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.INTERNAL_FULL,
        )
        self.assertEqual(len(capacity.world_max_contacts), 2)
        self.assertGreater(capacity.world_max_contacts[0], 0)
        self.assertEqual(capacity.world_max_contacts[1], 0)


class TestResolveContactCapacityInternalDviBounded(unittest.TestCase):
    """DVI: per-world ``min(geometry, bounded Newton heuristic)``."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_dvi_caps_dense_scene_below_full_geometry(self):
        """Dense scenes get bounded below the full geometry estimate."""
        newton_model = _make_stacked_boxes_newton_model(self.default_device, num_boxes=16)
        kamino_model = ModelKamino.from_newton(newton_model)

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.INTERNAL_DVI_BOUNDED,
        )
        # The bounded DVI heuristic must not exceed the pair-based geometry estimate.
        self.assertLessEqual(
            capacity.model_max_contacts,
            kamino_model.geoms.model_minimum_contacts,
        )
        # And it must be strictly smaller for a dense stacked-box scene.
        self.assertLess(
            capacity.model_max_contacts,
            kamino_model.geoms.model_minimum_contacts,
        )

    def test_dvi_single_world_matches_min_geometry_heuristic(self):
        """Single-world DVI returns ``min(theoretical, Newton heuristic)``."""
        newton_model = _make_stacked_boxes_newton_model(self.default_device, num_boxes=16)
        kamino_model = ModelKamino.from_newton(newton_model)
        expected = min(kamino_model.geoms.world_minimum_contacts[0], _estimate_rigid_contact_max(newton_model))

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.INTERNAL_DVI_BOUNDED,
        )
        self.assertEqual(capacity.world_max_contacts, (expected,))

    def test_dvi_max_contacts_per_world_still_overrides(self):
        """Explicit ``max_contacts_per_world`` takes precedence over the bounded estimate."""
        newton_model = _make_stacked_boxes_newton_model(self.default_device, num_boxes=16)
        kamino_model = ModelKamino.from_newton(newton_model)

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(max_contacts_per_world=37),
            policy=ContactCapacityPolicy.INTERNAL_DVI_BOUNDED,
        )
        self.assertEqual(list(capacity.world_max_contacts), [37])

    def test_dvi_max_contacts_caps_bounded_total(self):
        """``max_contacts`` proportionally caps the bounded DVI estimate."""
        newton_model = _make_stacked_boxes_newton_model(self.default_device, num_boxes=16)
        kamino_model = ModelKamino.from_newton(newton_model)

        uncapped = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.INTERNAL_DVI_BOUNDED,
        )
        cap = max(1, uncapped.model_max_contacts // 2)
        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(max_contacts=cap),
            policy=ContactCapacityPolicy.INTERNAL_DVI_BOUNDED,
        )
        self.assertEqual(capacity.model_max_contacts, cap)


class TestResolveContactCapacityExternal(unittest.TestCase):
    """External Newton contacts: ``rigid_contact_max`` -> exact per-world split."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_explicit_rigid_contact_max_is_honored_exactly(self):
        """Explicit Newton totals are preserved exactly, without ceil rounding."""
        newton_model = _make_three_world_sphere_on_plane(self.default_device)
        newton_model.rigid_contact_max = 1000
        kamino_model = ModelKamino.from_newton(newton_model)

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.EXTERNAL_NEWTON,
        )
        self.assertEqual(capacity.model_max_contacts, 1000)
        self.assertEqual(sum(capacity.world_max_contacts), 1000)

    def test_zero_rigid_contact_max_uses_newton_estimator(self):
        """A zero Newton total defers to :func:`_estimate_rigid_contact_max`."""
        newton_model = _make_three_world_sphere_on_plane(self.default_device)
        self.assertEqual(newton_model.rigid_contact_max, 0)
        kamino_model = ModelKamino.from_newton(newton_model)
        estimate = _estimate_rigid_contact_max(newton_model)

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=newton_model,
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.EXTERNAL_NEWTON,
        )
        self.assertEqual(capacity.model_max_contacts, estimate)

    def test_external_distribution_is_geometry_proportional(self):
        """External per-world budgets scale with per-world geometry weights."""
        builder = ModelBuilderKamino(default_world=False)
        basics.build_boxes_nunchaku(builder=builder)
        builder.add_world(name="empty_world")
        kamino_model = builder.finalize(self.default_device)

        # Fabricate a Newton-style total; the resolver only reads
        # ``rigid_contact_max`` and geometry weights, so a stub model suffices.
        class _Stub:
            rigid_contact_max = 30
            world_count = 2

        capacity = resolve_contact_capacity(
            kamino_model,
            newton_model=_Stub(),
            config=CollisionDetectorConfig(),
            policy=ContactCapacityPolicy.EXTERNAL_NEWTON,
        )
        # World 0 has all the geometry; world 1 has none.
        self.assertEqual(capacity.model_max_contacts, 30)
        self.assertEqual(capacity.world_max_contacts[1], 0)
        self.assertEqual(capacity.world_max_contacts[0], 30)


class TestConversionMetadataInvariance(unittest.TestCase):
    """Newton->Kamino conversion always populates geometry metadata."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_pair_metadata_ignores_preset_rigid_contact_max(self):
        """Geometry metadata is independent of pre-populated ``rigid_contact_max``."""
        base = _make_three_world_sphere_on_plane(self.default_device)
        base_model = ModelKamino.from_newton(base)

        primed = _make_three_world_sphere_on_plane(self.default_device)
        primed.rigid_contact_max = 12345
        primed_model = ModelKamino.from_newton(primed)

        self.assertEqual(
            list(base_model.geoms.world_minimum_contacts),
            list(primed_model.geoms.world_minimum_contacts),
        )
        self.assertEqual(base_model.geoms.model_minimum_contacts, primed_model.geoms.model_minimum_contacts)


class TestSolverKaminoCapacityRouting(unittest.TestCase):
    """Integration: SolverKamino uses the shared resolver across all modes."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_padmm_matches_full_internal_policy(self):
        """PADMM with internal CD allocates directly from geometry metadata."""
        model = _make_three_world_sphere_on_plane(self.default_device)
        solver = SolverKamino(
            model,
            config=SolverKamino.Config(dynamics_solver="padmm", use_collision_detector=True),
        )
        self.assertIsNotNone(solver._contacts_kamino)
        self.assertEqual(
            list(solver._contacts_kamino.world_max_contacts_host),
            list(solver._model_kamino.geoms.world_minimum_contacts),
        )
        self.assertEqual(model.rigid_contact_max, solver._contacts_kamino.model_max_contacts_host)

    def test_dvi_matches_bounded_policy(self):
        """DVI with internal CD stays within the pair-based geometry bound."""
        model = _make_stacked_boxes_newton_model(self.default_device, num_boxes=16)
        solver = SolverKamino(
            model,
            config=SolverKamino.Config(dynamics_solver="dvi", sparse_jacobian=False, use_collision_detector=True),
        )
        self.assertLessEqual(
            solver._contacts_kamino.model_max_contacts_host,
            solver._model_kamino.geoms.model_minimum_contacts,
        )

    def test_external_preserves_explicit_total(self):
        """External-CD mode preserves an explicit ``rigid_contact_max`` exactly."""
        model = _make_three_world_sphere_on_plane(self.default_device)
        model.rigid_contact_max = 1000
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        self.assertEqual(model.rigid_contact_max, 1000)
        self.assertEqual(solver._contacts_kamino.model_max_contacts_host, 1000)
        self.assertEqual(sum(solver._contacts_kamino.world_max_contacts_host), 1000)

    def test_external_construction_order_is_symmetric(self):
        """Pipeline-before vs pipeline-after Kamino produce the same buffer size."""
        model_a = _make_three_world_sphere_on_plane(self.default_device)
        solver_a = SolverKamino(model_a, config=SolverKamino.Config(use_collision_detector=False))
        pipeline_a = newton.CollisionPipeline(model_a)

        model_b = _make_three_world_sphere_on_plane(self.default_device)
        pipeline_b = newton.CollisionPipeline(model_b)
        solver_b = SolverKamino(model_b, config=SolverKamino.Config(use_collision_detector=False))

        self.assertEqual(
            solver_a._contacts_kamino.model_max_contacts_host,
            pipeline_a.rigid_contact_max,
        )
        self.assertEqual(
            solver_b._contacts_kamino.model_max_contacts_host,
            pipeline_b.rigid_contact_max,
        )
        self.assertEqual(
            solver_a._contacts_kamino.model_max_contacts_host,
            solver_b._contacts_kamino.model_max_contacts_host,
        )


###
# Test execution
###


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
