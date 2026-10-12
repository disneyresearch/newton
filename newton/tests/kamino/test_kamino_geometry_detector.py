# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `geometry/detector.py`"""

import unittest
import warnings

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry import (
    CollisionDetector,
)
from newton._src.solvers.kamino._src.geometry.capacity import ContactCapacity
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.test_kamino_geometry_primitive import check_contacts
from newton.tests.utils import basics

###
# Tests
###


class TestCollisionDetectorConfig(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        """Test making default collision detector config."""
        config = CollisionDetector.Config()
        self.assertEqual(config.pipeline, "unified")
        self.assertEqual(config.broadphase, "explicit")
        self.assertEqual(config.bvtype, "aabb")

    def test_01_make_with_string_args(self):
        """Test making collision detector config with string arguments."""
        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit", bvtype="aabb")
        self.assertEqual(config.pipeline, "primitive")
        self.assertEqual(config.broadphase, "explicit")
        self.assertEqual(config.bvtype, "aabb")

    def test_02_max_contacts_per_pair_is_deprecated(self):
        """Warn that max_contacts_per_pair is deprecated and keep the resolved capacity unchanged."""
        model = ModelKamino.from_newton(basics.build_boxes_nunchaku().finalize(self.default_device))
        with self.assertWarnsRegex(DeprecationWarning, "max_contacts_per_pair") as caught:
            config = CollisionDetector.Config(max_contacts_per_pair=1)
        self.assertEqual(caught.filename, __file__)
        self.assertEqual(
            ContactCapacity.resolve_from(model, config=config),
            ContactCapacity.resolve_from(model, config=CollisionDetector.Config()),
        )

    def test_03_default_config_round_trips_without_warnings(self):
        """Leave max_contacts_per_pair unset so rebuilding a default config emits no deprecation warning."""
        config = CollisionDetector.Config()
        self.assertIsNone(config.max_contacts_per_pair)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            CollisionDetector.Config(**config.__dict__)


class TestGeometryCollisionDetector(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

        self.build_func = basics.build_boxes_nunchaku
        self.expected_contacts = 9  # NOTE: specialized to build_boxes_nunchaku
        msg.debug(f"build_func: {self.build_func.__name__}")
        msg.debug(f"expected_contacts: {self.expected_contacts}")

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_primitive_pipeline(self):
        """
        Test the collision detector with the primitive pipeline
        on multiple worlds containing boxes_nunchaku model.
        """
        # Create and set up a model builder
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        data = model.data()

        # Create a collision detector with primitive pipeline
        config = CollisionDetector.Config(
            pipeline="primitive",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.world_count
        msg.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

        # Define expected contacts dictionary
        expected = {
            "model_active_contacts": sum(expected_world_contacts),
            "world_active_contacts": np.array(expected_world_contacts, dtype=np.int32),
        }

        # Check results
        check_contacts(
            detector.contacts,
            expected,
            case="boxes_nunchaku",
            header="primitive pipeline",
        )

    def test_02_unified_pipeline(self):
        """
        Test the collision detector with the unified pipeline
        on multiple worlds containing boxes_nunchaku model.
        """
        # Create and set up a model builder
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        data = model.data()

        # Create a collision detector with unified pipeline
        config = CollisionDetector.Config(
            pipeline="unified",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.world_count
        msg.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

        # Define expected contacts dictionary
        expected = {
            "model_active_contacts": sum(expected_world_contacts),
            "world_active_contacts": np.array(expected_world_contacts, dtype=np.int32),
        }

        # Check results
        check_contacts(
            detector.contacts,
            expected,
            case="boxes_nunchaku",
            header="unified pipeline",
        )

    def test_03_pair_based_world_budgets(self):
        """Verify per-world budgets follow geometry rather than an equal model split."""
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))

        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit")
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(detector.world_max_contacts, model.geoms.world_minimum_contacts)
        self.assertEqual(detector.model_max_contacts, sum(model.geoms.world_minimum_contacts))

    def test_04_max_contacts_caps_model_total(self):
        """Verify ``max_contacts`` caps rather than floors the geometry-based estimate."""
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        uncapped_total = sum(model.geoms.world_minimum_contacts)
        self.assertGreater(uncapped_total, 15)

        config = CollisionDetector.Config(
            pipeline="primitive",
            broadphase="explicit",
            max_contacts=15,
        )
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(detector.model_max_contacts, 15)
        self.assertEqual(sum(detector.world_max_contacts), 15)

    def test_05_heterogeneous_world_budgets(self):
        """Verify worlds with different geometry receive different contact budgets."""
        builder = ModelBuilder()
        basics.build_boxes_nunchaku(builder=builder)
        builder.begin_world(label="empty_world")
        builder.end_world()
        model = ModelKamino.from_newton(builder.finalize(self.default_device))

        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit")
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(len(detector.world_max_contacts), 2)
        self.assertGreater(detector.world_max_contacts[0], 0)
        self.assertEqual(detector.world_max_contacts[1], 0)
        self.assertEqual(detector.world_max_contacts, model.geoms.world_minimum_contacts)
        self.assertEqual(detector.contacts.world_max_contacts_host, detector.world_max_contacts)


class TestCollisionDetectorContactCapacity(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_02_resolve_contact_capacity_uses_pair_metadata(self):
        """Verify the default resolved budget matches pair-based metadata."""
        builder = basics.build_boxes_nunchaku()
        model = ModelKamino.from_newton(builder.finalize(self.default_device))

        detector = CollisionDetector(model=model, config=CollisionDetector.Config())
        self.assertEqual(detector.world_max_contacts, list(model.geoms.world_minimum_contacts))
        self.assertEqual(detector.model_max_contacts, model.geoms.model_minimum_contacts)
        self.assertEqual(
            list(ContactCapacity.resolve_from(model).world_max_contacts), list(model.geoms.world_minimum_contacts)
        )

    def test_03_solver_supplied_capacity_is_honored_verbatim(self):
        """Verify the detector allocates a caller-supplied ContactCapacity verbatim."""
        builder = ModelBuilder()
        builder.replicate(builder=basics.build_boxes_nunchaku(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        overridden = ContactCapacity(world_max_contacts=(4, 8, 15))

        detector = CollisionDetector(
            model=model,
            config=CollisionDetector.Config(pipeline="primitive", broadphase="explicit"),
            capacity=overridden,
        )
        self.assertEqual(detector.world_max_contacts, [4, 8, 15])
        self.assertEqual(detector.model_max_contacts, 27)
        self.assertEqual(detector.contacts.world_max_contacts_host, [4, 8, 15])


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
