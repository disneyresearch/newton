# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides configuration types shared by all formulations of the constrained forward dynamics problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import warp as wp

from .....core.types import override
from ...config import (
    ConfigBase,
    ConstrainedDynamicsConfig,
    ConstraintComplianceConfig,
    ConstraintStabilizationConfig,
)
from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..geometry.contacts import ContactsKamino
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "ConstrainedDynamicsConfigStruct",
    "ConstrainedDynamicsProblemConfig",
    "ProblemFormulation",
    "assert_containers_are_valid",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class ProblemFormulation(IntEnum):
    """
    Enumerates the formulations of the constrained forward dynamics problem.

    All formulations describe the same nonlinear complementarity problem in the post-event
    generalized velocities ``u+`` and constraint reactions ``lambda``, but differ in the
    space in which the inner linear system of a splitting method (e.g. PADMM) is solved.
    """

    DUAL = 0
    """Constraint-space formulation using the (compliant) Delassus matrix ``D + R``."""

    PRIMAL = 1
    """Generalized-velocity formulation using the constraint-augmented mass matrix ``M + J^T W J``."""

    KKT = 2
    """Saddle-point formulation using the quasi-definite KKT matrix ``[M, J^T; J, -R_eff]``."""

    def __str__(self):
        return f"{self.name.lower()}"


@wp.struct
class ConstrainedDynamicsConfigStruct:
    """
    A Warp struct to hold on-device configuration parameters of a constrained dynamics problem.
    """

    alpha: wp.float32
    """Baumgarte stabilization parameter for bilateral joint constraints."""
    beta: wp.float32
    """Baumgarte stabilization parameter for unilateral joint limit constraints."""
    gamma: wp.float32
    """Baumgarte stabilization parameter for unilateral contact constraints."""
    delta: wp.float32
    """Contact penetration margin used for unilateral contact constraints"""
    compliance_joints: wp.float32
    """Compliance of the kinematic (i.e. bilateral) joint constraints."""
    compliance_limits: wp.float32
    """Compliance of the unilateral joint-limit constraints."""
    compliance_contacts_normal: wp.float32
    """Compliance along the normal direction of unilateral contact constraints."""
    compliance_contacts_tangential: wp.float32
    """Compliance along the tangential directions of unilateral contact constraints."""
    preconditioning: wp.bool
    """Flag to enable diagonal preconditioning of the problem."""


@dataclass
class ConstrainedDynamicsProblemConfig(ConfigBase):
    """
    Per-world configurations of a constrained forward dynamics problem.
    """

    constraints: ConstraintStabilizationConfig = field(default_factory=ConstraintStabilizationConfig)
    """Constraint stabilization global defaults/override configurations."""

    dynamics: ConstrainedDynamicsConfig = field(default_factory=ConstrainedDynamicsConfig)
    """Constrained dynamics problem construction configurations."""

    compliance: ConstraintComplianceConfig = field(default_factory=ConstraintComplianceConfig)
    """Constraint compliance global defaults/override configurations."""

    @property
    def has_compliance(self) -> bool:
        """
        Returns `True` if any of the configurable constraint compliances is non-zero.
        """
        return (
            self.compliance.joints > 0.0
            or self.compliance.limits > 0.0
            or self.compliance.contacts_normal > 0.0
            or self.compliance.contacts_tangential > 0.0
        )

    def to_struct(self) -> ConstrainedDynamicsConfigStruct:
        """
        Converts the config to a :class:`ConstrainedDynamicsConfigStruct`.
        """
        config_struct = ConstrainedDynamicsConfigStruct()
        config_struct.alpha = wp.float32(self.constraints.alpha)
        config_struct.beta = wp.float32(self.constraints.beta)
        config_struct.gamma = wp.float32(self.constraints.gamma)
        config_struct.delta = wp.float32(self.constraints.delta)
        config_struct.compliance_joints = wp.float32(self.compliance.joints)
        config_struct.compliance_limits = wp.float32(self.compliance.limits)
        config_struct.compliance_contacts_normal = wp.float32(self.compliance.contacts_normal)
        config_struct.compliance_contacts_tangential = wp.float32(self.compliance.contacts_tangential)
        config_struct.preconditioning = wp.bool(self.dynamics.preconditioning)
        return config_struct

    @override
    def validate(self) -> None:
        """
        Validates the current values held by the :class:`ConstrainedDynamicsProblemConfig` instance.
        """
        self.constraints.validate()
        self.dynamics.validate()
        self.compliance.validate()

    @override
    def __post_init__(self):
        """Post-initialization to validate config."""
        self.validate()

    @staticmethod
    def check_configs(
        config: list[ConstrainedDynamicsProblemConfig] | ConstrainedDynamicsProblemConfig | None, num_worlds: int
    ) -> list[ConstrainedDynamicsProblemConfig]:
        """
        Checks and expands the given config(s) into a list with one config per world.

        If a single config is provided, it is replicated for all worlds. If a list of
        configs is provided, the number of configs must match the number of worlds.
        """
        if config is None:
            return [ConstrainedDynamicsProblemConfig()] * num_worlds
        elif isinstance(config, ConstrainedDynamicsProblemConfig):
            return [config] * num_worlds
        elif isinstance(config, list):
            if len(config) != num_worlds:
                raise ValueError(f"Expected {num_worlds} configs, got {len(config)}")
            for c in config:
                if not isinstance(c, ConstrainedDynamicsProblemConfig):
                    raise TypeError(f"Expected ConstrainedDynamicsProblemConfig, got {type(c)}")
            return config
        else:
            raise TypeError(
                "Expected List[ConstrainedDynamicsProblemConfig] or ConstrainedDynamicsProblemConfig, "
                f"got {type(config)}"
            )


###
# Utilities
###


def assert_containers_are_valid(
    model: ModelKamino | None,
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
    elif data is not None and not isinstance(data, DataKamino):
        raise ValueError("Invalid data container provided. Must be an instance of `DataKamino`.")

    if limits_required and limits is None:
        raise ValueError("A limits container of type `LimitsKamino` must be provided.")
    elif limits is not None and not isinstance(limits, LimitsKamino):
        raise ValueError("Invalid limits container provided. Must be an instance of `LimitsKamino`.")

    if contacts_required and contacts is None:
        raise ValueError("A contacts container of type `ContactsKamino` must be provided.")
    elif contacts is not None and not isinstance(contacts, ContactsKamino):
        raise ValueError("Invalid contacts container provided. Must be an instance of `ContactsKamino`.")
