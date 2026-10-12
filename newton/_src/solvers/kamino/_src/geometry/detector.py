# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides a unified interface for performing Collision Detection in Kamino.

Usage example:

    # Create a model builder
    builder = ModelBuilder()
    # ... add bodies and collision geometries to the builder ...

    # Finalize the model
    model = ModelKamino.from_newton(builder.finalize(device="cuda:0"))

    # Create a collision detector with desired config
    config = CollisionDetector.Config(
        pipeline="unified",
        broadphase="explicit",
        bvtype="aabb",
    )

    # Create the collision detector
    detector = CollisionDetector(model=model, config=config)
"""

from __future__ import annotations

from enum import IntEnum

import warp as wp

from .....core.types import override
from ...config import CollisionDetectorConfig
from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..geometry.capacity import ContactCapacity
from ..geometry.contacts import ContactsKamino
from ..geometry.primitive import CollisionPipelinePrimitive
from ..geometry.unified import CollisionPipelineUnifiedKamino

###
# Module interface
###

__all__ = [
    "BroadPhaseType",
    "CollisionDetector",
    "CollisionPipelineType",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class CollisionPipelineType(IntEnum):
    """Defines the collision detection pipelines supported in Kamino."""

    PRIMITIVE = 0
    """
    Use the "fast" collision detection pipeline specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.
    """

    UNIFIED = 1
    """
    Use Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a
    unified GJK/MPR-based narrow-phase. This pipeline is more general and
    supports arbitrary collision geometries, including meshes and SDFs.
    """

    @classmethod
    def from_string(cls, s: str) -> CollisionPipelineType:
        """Converts a string to a CollisionPipelineType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid CollisionPipelineType: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the collision detector mode."""
        return f"CollisionDetectorMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the collision detector mode."""
        return self.__str__()


class BroadPhaseType(IntEnum):
    """Defines the broad-phase collision detection modes supported in Kamino."""

    NXN = 0
    """
    Use an `NXN` broad-phase that considers all possible pairs of collision shapes as candidates.

    This mode is simple but can be inefficient for models with many collision shapes.
    """

    SAP = 1
    """
    Use a Sweep and Prune (SAP) broad-phase that sorts collision shapes along a chosen axis
    and only considers overlapping shapes as candidates for narrow-phase collision detection.

    This mode is more efficient than `NXN` for models with many collision
    shapes, especially when they are sparsely distributed in space.
    """

    EXPLICIT = 2
    """
    Use an explicit broad-phase that relies on pre-computed candidate pairs
    of collision shapes (see ``GeometriesModel.collidable_pairs``).

    This mode can be the most efficient when the candidate pairs are
    well-chosen, but it requires additional setup during model building.
    """

    @classmethod
    def from_string(cls, s: str) -> BroadPhaseType:
        """Converts a string to a BroadPhaseType enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid BroadPhaseType: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the broad-phase type."""
        return f"BroadPhaseType.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the broad-phase type."""
        return self.__str__()


###
# Interfaces
###


class CollisionDetector:
    """
    Provides a Collision Detection (CD) front-end for Kamino.

    This class is responsible for performing collision detection as well
    as managing the collision containers and their memory allocations.

    Supports two collision pipeline types:

    - `PRIMITIVE`: A fast collision pipeline with specialized for geometric
    primitives using an "explicit" broad-phase on pre-computed collision
    shape pairs and a narrow-phase using Newton's primitive colliders.

    - `UNIFIED`: Newton's unified collision-detection pipeline using a configurable
    broad-phase that supports `NXN`, `SAP`, or `EXPLICIT` modes, and a unified
    GJK/MPR-based narrow-phase. This pipeline is more general and supports arbitrary
    collision geometries, including meshes and SDFs.
    """

    Config = CollisionDetectorConfig
    """
    The configuration dataclass for the CollisionDetector, which includes parameters
    for selecting the collision pipeline type, broad-phase mode, bounding volume type,
    contact generation parameters, and other settings related to collision detection.

    See :class:`CollisionDetectorConfig` for the full
    list of configuration options and their descriptions.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        config: CollisionDetector.Config | None = None,
        capacity: ContactCapacity | None = None,
    ):
        """
        Initialize the CollisionDetector.

        Args:
            model: The model container holding the time-invariant data of the system being simulated.
                If provided, the detector will be finalized using the provided model and config.
                If `None`, the detector will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.
            config: Config for the CollisionDetector.
                If `None`, uses default config.
            capacity: The contact capacity to allocate, e.g. resolved by the solver so that the
                detector and the Newton contact buffer agree. If `None`, it is resolved with
                :meth:`ContactCapacity.resolve_from` from the model and config.
        """
        # Declare the device cache
        self._device: wp.DeviceLike = None

        # Cache a reference to the target model
        self._model: ModelKamino | None = model

        # Cache the collision detector config
        self._config: CollisionDetector.Config | None = config

        # Declare the contacts container
        self._contacts: ContactsKamino | None = None

        # Declare the collision detection pipelines
        self._pipeline_type: CollisionPipelineType | None = None
        self._unified_pipeline: CollisionPipelineUnifiedKamino | None = None
        self._primitive_pipeline: CollisionPipelinePrimitive | None = None

        # Declare and initialize the caches of contacts allocation sizes
        self._capacity: ContactCapacity | None = None
        self._model_max_contacts: int = 0

        # Finalize the collision detector if a model is provided
        if model is not None:
            self.finalize(model=model, config=config, capacity=capacity)

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """Returns the device on which the CollisionDetector data is allocated and executes."""
        return self._device

    @property
    def model(self) -> ModelKamino | None:
        """Returns the model associated with the CollisionDetector."""
        return self._model

    @property
    def config(self) -> CollisionDetector.Config | None:
        """Returns the config used to configure the CollisionDetector."""
        return self._config

    @property
    def contacts(self) -> ContactsKamino | None:
        """Returns the ContactsKamino container managed by the CollisionDetector."""
        return self._contacts

    @property
    def model_max_contacts(self) -> int:
        """Returns the total maximum number of contacts allocated for the model across all worlds."""
        return self._model_max_contacts

    @property
    def world_max_contacts(self) -> list[int]:
        """Returns the maximum number of contacts allocated for each world."""
        if self._capacity is None:
            return [0]
        return list(self._capacity.world_max_contacts)

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino | None = None,
        config: CollisionDetector.Config | None = None,
        capacity: ContactCapacity | None = None,
    ):
        """
        Allocates CollisionDetector data on the target device.

        Args:
            model: The model container holding the time-invariant data of the system being simulated.
                If provided, the detector will be finalized using the provided model and config.
                If `None`, the detector will be created empty without allocating data, and
                can be finalized later by providing a model to the `finalize` method.
            config: Config for the CollisionDetector.
                If `None`, uses default config.
            capacity: The contact capacity to allocate. If `None`, it is resolved with
                :meth:`ContactCapacity.resolve_from` from the model and config.
        """
        # Override the model if specified explicitly
        if model is not None:
            self._model = model

        # Check that the model is valid
        if self._model is None:
            raise ValueError("Cannot finalize CollisionDetector: model is `None`")
        elif not isinstance(self._model, ModelKamino):
            raise TypeError(f"Cannot finalize CollisionDetector: expected ModelKamino, got {type(self._model)}")

        # Use the model's device
        self._device = self._model.device

        # Override the config if specified, ensuring that they are valid
        if config is not None:
            if not isinstance(config, CollisionDetector.Config):
                raise TypeError(
                    f"Cannot finalize CollisionDetector: expected CollisionDetector.Config, got {type(config)}"
                )
            self._config = config
        # If no config is provided, use the defaults
        if self._config is None:
            self._config = CollisionDetector.Config()

        # Configure the collision detection pipeline type based on the config
        self._pipeline_type = CollisionPipelineType.from_string(self._config.pipeline)

        # Resolve the contact capacity unless the caller already resolved it
        if capacity is None:
            capacity = ContactCapacity.resolve_from(self._model, config=self._config)
        elif not isinstance(capacity, ContactCapacity):
            raise TypeError(f"Cannot finalize CollisionDetector: expected ContactCapacity, got {type(capacity)}")
        if capacity.num_worlds != self._model.size.num_worlds:
            raise ValueError(
                f"Cannot finalize CollisionDetector: capacity covers {capacity.num_worlds} worlds, "
                f"but the model has {self._model.size.num_worlds}"
            )
        self._capacity = capacity
        self._model_max_contacts = capacity.model_max_contacts

        # Proceed with allocations only if the model admits contacts, which
        # occurs when collision geometries defined in the builder and model
        # can form at least one collidable pair. Otherwise, set the contacts
        # container and pipeline to `None`.
        if self._model_max_contacts > 0:
            # Create the contacts interface which will allocate all contacts data arrays
            # NOTE: If internal allocations happen, then they will contain
            # the contacts generated by the collision detection pipelines
            self._contacts = ContactsKamino(capacity, device=self._device)

            # Initialize the configured collision detection pipeline
            match self._pipeline_type:
                case CollisionPipelineType.PRIMITIVE:
                    self._primitive_pipeline = CollisionPipelinePrimitive(
                        model=self._model,
                        bvtype=self._config.bvtype,
                        default_gap=self._config.default_gap,
                    )
                case CollisionPipelineType.UNIFIED:
                    self._unified_pipeline = CollisionPipelineUnifiedKamino(
                        model=self._model,
                        broadphase=self._config.broadphase,
                        default_gap=self._config.default_gap,
                        max_triangle_pairs=self._config.max_triangle_pairs,
                    )
                case _:
                    raise ValueError(f"Unsupported CollisionPipelineType: {self._pipeline_type}")
        else:
            self._contacts = None
            self._primitive_pipeline = None
            self._unified_pipeline = None

    def collide(self, data: DataKamino, contacts: ContactsKamino | None = None):
        """
        Executes collision detection given a model and its associated data.

        This operation will use the `primitive` or `unified` pipeline depending on
        the configuration set during the initialization of the CollisionDetector.

        Args:
            data: The solver data container holding solver-specific internal geome/shape data.
                Body poses are sourced from ``data.bodies.q_i`` so that detection follows the
                configuration the integrator is currently working at. Under a mid-point scheme
                such as :class:`IntegratorMoreauJean` this is the mid-step pose, whereas under
                :class:`IntegratorEuler` it still equals the pose at the start of the time-step.
            contacts: An optional ContactsKamino container to store the generated contacts.
                If `None`, uses the internal ContactsKamino container managed by the CollisionDetector.
        """
        # If no contacts can be generated, skip collision detection
        if contacts is not None:
            _contacts = contacts
        else:
            _contacts = self._contacts

        # Skip this operation if no contacts data has been allocated
        if _contacts is None or _contacts.model_max_contacts_host <= 0:
            return

        # Ensure that a collision detection pipeline has been created
        if self._primitive_pipeline is None and self._unified_pipeline is None:
            raise RuntimeError("Cannot perform collision detection: a collision pipeline has not been created")

        # Ensure that the data is valid
        if data is None:
            raise ValueError("Cannot perform collision detection: data is None")
        if not isinstance(data, DataKamino):
            raise TypeError(f"Cannot perform collision detection: expected DataKamino, got {type(data)}")

        # Execute the configured collision detection pipeline
        match self._pipeline_type:
            case CollisionPipelineType.PRIMITIVE:
                self._primitive_pipeline.collide(data, _contacts)
            case CollisionPipelineType.UNIFIED:
                self._unified_pipeline.collide(data, _contacts)
            case _:
                raise ValueError(f"Unsupported CollisionPipelineType: {self._pipeline_type}")
