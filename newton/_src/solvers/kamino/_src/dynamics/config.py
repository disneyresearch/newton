# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides types for holding configurations for constrained forward dynamics problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..core.model import Model as ModelKamino
from ..core.data import DataKamino
from ..core.size import SizeKamino
from ..geometry.contacts import ContactsKamino
from ..kinematics.constraints import get_max_constraints_per_world
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "ConstrainedDynamicsCfg",
    "ConstrainedDynamicsConfig",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class ConstrainedDynamicsCfg:
    """
    A Warp struct to hold on-device constrained dynamics configurations.
    """

    alpha: wp.float32
    """Baumgarte stabilization parameter for bilateral joint constraints."""
    beta: wp.float32
    """Baumgarte stabilization parameter for unilateral joint limit constraints."""
    gamma: wp.float32
    """Baumgarte stabilization parameter for unilateral contact constraints."""
    delta: wp.float32
    """Contact penetration margin used for unilateral contact constraints"""
    preconditioning: wp.bool
    """Flag to enable preconditioning of the dual problem."""


@dataclass
class ConstrainedDynamicsConfig:
    """
    A data container to hold host-side constrained dynamics configurations.
    """

    alpha: float = 0.01
    """
    Global default Baumgarte stabilization parameter for bilateral joint constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    beta: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral joint-limit constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    gamma: float = 0.01
    """
    Global default Baumgarte stabilization parameter for unilateral contact constraints.\n
    Must be in range `[0, 1.0]`.\n
    Defaults to `0.01`.
    """

    delta: float = 1.0e-6
    """
    Contact penetration margin used for unilateral contact constraints.\n
    Must be non-negative.\n
    Defaults to `1.0e-6`.
    """

    preconditioning: bool = True
    """
    Set to `True` to enable preconditioning of the dual problem.\n
    Defaults to `True`.
    """

    def __post_init__(self) -> None:
        """
        Performs validation checks on the configuration values after initialization.
        """
        self.check_values()

    def check_values(self) -> None:
        """
        Validates configuration values.
        """
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError(f"Invalid alpha: {self.alpha}. Must be in range [0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(f"Invalid beta: {self.beta}. Must be in range [0, 1.0].")
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(f"Invalid gamma: {self.gamma}. Must be in range [0, 1.0].")
        if self.delta < 0.0:
            raise ValueError(f"Invalid delta: {self.delta}. Must be non-negative.")

    def to_config(self) -> ConstrainedDynamicsCfg:
        """
        Converts the host-side configurations to a ConstrainedDynamicsCfg struct for on-device use.
        """
        config = ConstrainedDynamicsCfg()
        config.alpha = wp.float32(self.alpha)
        config.beta = wp.float32(self.beta)
        config.gamma = wp.float32(self.gamma)
        config.delta = wp.float32(self.delta)
        config.preconditioning = wp.bool(self.preconditioning)
        return config


@dataclass
class ConstrainedDynamicsInfo:
    """
    A container to hold the the dual forward dynamics problem data over multiple worlds.
    """

    ###
    # Meta-Data
    ###

    model_size: SizeKamino | None = None
    """A local reference to the model size meta-data container."""

    world_max_total_cts: list[int] | None = None
    """The maximum total number of constraints possible in each world."""

    sum_of_max_problem_dims: int = 0
    """The sum of the maximum problem dimensions (i.e. constraints) across all worlds."""

    max_of_max_problem_dims: int = 0
    """The largest maximum problem dimensions (i.e. constraints) across all worlds."""

    sum_of_max_total_cts: int = 0
    """The sum of the maximum total constraints across all worlds."""

    max_of_max_total_cts: int = 0
    """The largest maximum total constraints across all worlds."""

    sum_of_max_limits: int = 0
    """The sum of the maximum limits across all worlds."""

    max_of_max_limits: int = 0
    """The largest maximum limits across all worlds."""

    sum_of_max_contacts: int = 0
    """The sum of the maximum number of contacts across all worlds."""

    max_of_max_contacts: int = 0
    """The largest maximum number of contacts across all worlds."""

    ###
    # Configurations
    ###

    configs: wp.array | None = None
    """
    Problem configuration parameters for each world.\n
    Shape of `(num_worlds,)` and type :class:`ConstrainedDynamicsCfg`.
    """

    ###
    # Constraints Info
    ###

    njc: wp.array | None = None
    """
    The number of active joint constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    nl: wp.array | None = None
    """
    The number of active limit constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    nc: wp.array | None = None
    """
    The number of active contact constraints in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    lio: wp.array | None = None
    """
    The limit index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    cio: wp.array | None = None
    """
    The contact index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    uio: wp.array | None = None
    """
    The unilateral index offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    lcgo: wp.array | None = None
    """
    The limit constraint group offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    ccgo: wp.array | None = None
    """
    The contact constraint group offset of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.\n
    """

    constraints_start: wp.array | None = None
    """
    The vector index offset of each constraint vector block of each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    constraints_count: wp.array | None = None
    """
    The number of constraints active in each world.\n
    Shape of `(num_worlds,)` and type :class:`int32`.
    """

    ###
    # Properties
    ###

    @property
    def num_worlds(self) -> int:
        """
        Returns the number of worlds captured by this info container.
        """
        if self.model_size is None:
            raise ValueError("ModelSize reference has not been set.")
        return self.model_size.num_worlds

    ###
    # Utilities
    ###

    @staticmethod
    def validate_configs(
        settings: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None, num_worlds: int
    ) -> list[ConstrainedDynamicsConfig]:
        """
        Validates constrained dynamics configurations and returns a list of configurations for each world.

        If a single `ConstrainedDynamicsConfig` object is provided, it will be replicated for all worlds.
        If a list of settings is provided, it will ensure that the number of settings matches the number of worlds.
        """
        if settings is None:
            # If no settings are provided, use default settings
            return [ConstrainedDynamicsConfig()] * num_worlds
        elif isinstance(settings, ConstrainedDynamicsConfig):
            # If a single settings object is provided, replicate it for all worlds
            return [settings] * num_worlds
        elif isinstance(settings, list):
            # Ensure the settings are of the correct type and length
            if len(settings) != num_worlds:
                raise ValueError(f"Expected {num_worlds} settings, got {len(settings)}")
            for s in settings:
                if not isinstance(s, ConstrainedDynamicsConfig):
                    raise TypeError(f"Expected ConstrainedDynamicsConfig, got {type(s)}")
            return settings
        else:
            raise TypeError(
                f"Expected List[ConstrainedDynamicsConfig] or ConstrainedDynamicsConfig, got {type(settings)}"
            )

    @classmethod
    def from_containers(
        cls,
        sum_of_max_problem_dims: int,
        max_of_max_problem_dims: int,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        configs: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None = None,
        device: wp.DeviceLike = None,
    ) -> ConstrainedDynamicsInfo:
        # Ensure the simulation containers are valid
        assert_containers_are_valid(model=model, data=data, limits=limits, contacts=contacts)

        # If a device is not provided use the model's device
        if device is None:
            _device = model.device
        else:
            _device = device

        # Validate configurations
        configs = cls.validate_configs(configs, model.info.num_worlds)

        # Extract required maximum number of constraints possible in each world
        world_max_total_cts = get_max_constraints_per_world(model, limits, contacts)

        # Allocate and return the constrained dynamics info
        # container constructed from the given model and data
        with wp.ScopedDevice(_device):
            return ConstrainedDynamicsInfo(
                # Store meta-data about the problem dimensions
                model_size=model.size,
                world_max_total_cts=world_max_total_cts,
                sum_of_max_problem_dims=int(sum_of_max_problem_dims),
                max_of_max_problem_dims=int(max_of_max_problem_dims),
                sum_of_max_total_cts=int(sum(world_max_total_cts)),
                max_of_max_total_cts=int(max(world_max_total_cts)),
                sum_of_max_limits=int(limits.model_max_limits_host) if limits is not None else 0,
                max_of_max_limits=int(max(limits.world_max_limits_host)) if limits is not None else 0,
                sum_of_max_contacts=int(contacts.model_max_contacts_host) if contacts is not None else 0,
                max_of_max_contacts=int(max(contacts.world_max_contacts_host)) if contacts is not None else 0,
                # Store the constrained dynamics problem configurations
                configs=wp.array([c.to_config() for c in configs], dtype=ConstrainedDynamicsCfg),
                # Capture references to the mode and data info arrays
                njc=model.info.num_joint_cts,
                nl=data.info.num_limits,
                nc=data.info.num_contacts,
                lio=model.info.limits_offset,
                cio=model.info.contacts_offset,
                uio=model.info.unilaterals_offset,
                lcgo=data.info.limit_cts_group_offset,
                ccgo=data.info.contact_cts_group_offset,
                constraints_start=model.info.total_cts_offset,
                constraints_count=data.info.num_total_cts,
            )


###
# Utilities
###


def assert_containers_are_valid(
    model: ModelKamino,
    data: DataKamino | None = None,
    limits: LimitsKamino | None = None,
    contacts: ContactsKamino | None = None,
    model_required: bool = True,
    data_required: bool = False,
    limits_required: bool = False,
    contacts_required: bool = False,
) -> None:
    """
    Asserts that the provided model and data containers are valid.
    """
    if model_required and model is None:
        raise ValueError("A model of type `ModelKamino` must be provided.")
    elif model is not None and not isinstance(model, ModelKamino):
        raise ValueError("Invalid model provided. Must be an instance of `ModelKamino`.")

    if data_required and data is None:
        raise ValueError("A data container of type `DataKamino` must be provided.")
    elif data is not None:
        if not isinstance(data, DataKamino):
            raise ValueError("Invalid data container provided. Must be an instance of `DataKamino`.")

    if limits_required and limits is None:
        raise ValueError("A limits container of type `LimitsKamino` must be provided.")
    elif limits is not None:
        if not isinstance(limits, LimitsKamino):
            raise ValueError("Invalid limits container provided. Must be an instance of `LimitsKamino`.")

    if contacts_required and contacts is None:
        raise ValueError("A contacts container of type `ContactsKamino` must be provided.")
    elif contacts is not None:
        if not isinstance(contacts, ContactsKamino):
            raise ValueError("Invalid contacts container provided. Must be an instance of `ContactsKamino`.")
