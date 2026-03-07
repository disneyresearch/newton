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
TODO
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import warp as wp

from ..core.model import ModelKamino
from ..core.data import DataKamino
from ..geometry.contacts import ContactsKamino
from ..kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from ..kinematics.limits import LimitsKamino
from .config import (
    ConstrainedDynamicsCfg,
    ConstrainedDynamicsConfig,
    ConstrainedDynamicsInfo,
    assert_containers_are_valid,
)

###
# Module interface
###

__all__ = [
    "ConstrainedDynamicsCfg",
    "ConstrainedDynamicsConfig",
    "ConstrainedDynamicsProblem",
    "ConstrainedDynamicsProblemData",
    "ConstrainedSystem",
    "SystemJacobiansType",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


SystemJacobiansType = DenseSystemJacobians | SparseSystemJacobians
"""A utility type alias for the supported Jacobians containers."""


@dataclass
class ConstrainedSystem:
    """
    A convenience container for holding references to the core system
    containers that define a constrained dynamics problem, i.e. model,
    data, limits, contacts and Jacobians containers.
    """

    ###
    # Attributes
    ###

    model: ModelKamino | None = None
    data: DataKamino | None = None
    limits: LimitsKamino | None = None
    contacts: ContactsKamino | None = None
    jacobians: SystemJacobiansType | None = None

    ###
    # Operations
    ###

    @property
    def is_sparse(self) -> bool:
        """
        Returns whether the system is using sparse Jacobians.
        """
        if self.jacobians is None:
            raise ValueError("Cannot determine sparsity of system without Jacobians.")
        elif isinstance(self.jacobians, DenseSystemJacobians):
            return False
        elif isinstance(self.jacobians, SparseSystemJacobians):
            return True
        else:
            raise TypeError(f"Unsupported Jacobians type: {type(self.jacobians)}")


@dataclass
class ConstrainedDynamicsProblemData:
    """
    Defines a base class for containers holding the data of a constrained dynamics problem.

    This base container defines the common data arrays that are
    typically required by many CFD problem formulations, such as:
    - `v_star`: The constraint-space reference velocity vectors.
    - `mu`: An array containing the per-contact friction coefficients.
    - `P`: The problem preconditioner values.

    Note:
        Implementation-specific constrained dynamics problem classes should extend this base class
        to define the specific data arrays required by their formulation of the CFD problem. The
        final contents of this container will vary based on the formulation of the CFD problem.
    """

    ###
    # Attributes
    ###

    v_star: wp.array | None = None
    """
    Stack of constraint-space reference velocity vectors.\n
    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    mu: wp.array | None = None
    """
    Stack of per-contact friction coefficient vectors.\n
    Shape of `(sum_of_max_contacts,)` and type :class:`float32`.
    """

    sigma: wp.array | None = None
    """
    Array of constraint-space diagonal regularizer values.\n
    Shape of `(sum_of_max_total_cts,)` and type :class:`float32`.
    """

    P: wp.array | None = None
    """
    Array of preconditioner values.\n
    Shape of `(sum_of_max_problem_dims,)` and type :class:`float32`.
    """

    ###
    # Operations
    ###

    def reset(self):
        """
        Resets the problem data to zero (or identity for the preconditioner).
        """
        self.v_star.zero_()
        self.mu.zero_()
        self.sigma.zero_()
        self.P.fill_(1.0)


###
# Interfaces
###


class ConstrainedDynamicsProblem(ABC):
    """
    Provides a base class to define a common interface for constrained forward dynamics (CFD) problems in Kamino.
    """

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SystemJacobiansType | None = None,
        configs: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None = None,
        device: wp.DeviceLike = None,
        **kwargs: dict[str, Any],
    ):
        """
        TODO
        """
        self._device: wp.DeviceLike = None
        """The device on which the CFD problem data is allocated on."""

        self._configs: list[ConstrainedDynamicsConfig] = []
        """Host-side cache of the list of per-world CFD problem configs."""

        self._info: ConstrainedDynamicsInfo | None = None
        """The info container holding dimensional information and meta-data about the CFD problem."""

        self._system: ConstrainedSystem | None = None
        """
        The constrained dynamical system container holding references
        to the model, data, limits, contacts and Jacobians containers.
        """

        self._data: ConstrainedDynamicsProblemData | None = None
        """The CFD problem data container holding the data arrays that define the CFD problem."""

        # Finalize the CFD problem data if at least the core
        # system containers are provided at initialization
        # NOTE: limits and contacts are optional because
        # not all CFD problem formulations require them
        if model is not None and data is not None and jacobians is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                configs=configs,
                device=device,
                **kwargs,
            )

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """
        Returns the device the CFD problem is allocated on.
        """
        return self._device

    @property
    def sparse(self) -> bool:
        """
        Returns whether the CFD problem is using sparse operators.
        """
        if self._system is None:
            raise ValueError("Cannot determine sparsity of problem without a finalized system.")
        return self._system.is_sparse

    @property
    def max_of_max_dims(self) -> tuple[int, int]:
        """
        Returns the maximum dimension of any per-world problem across all worlds.
        """
        return self._info.max_of_max_total_cts

    @property
    def sum_of_max_dims(self) -> int:
        """
        Returns the sum of maximum dimensions of any per-world problem across all worlds.
        """
        return self._info.sum_of_max_total_cts

    @property
    def configs(self) -> list[ConstrainedDynamicsConfig]:
        """
        Returns the list of per-world CFD problem configs.
        """
        return self._configs

    @configs.setter
    def configs(self, value: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig):
        """
        Sets the list of per-world CFD problem configs.
        If a single `ConstrainedDynamicsConfig` object is provided, it will be replicated for all worlds.
        """
        # Ensure the provided value is valid
        self._configs = ConstrainedDynamicsInfo.validate_configs(value, self._info.num_worlds)

        # If the CFD problem info container has already been
        # created, update its configs array with the new values
        if self._info is not None and self._info.configs is not None:
            self._info.configs.assign([c.to_config() for c in self._configs])

    @property
    def info(self) -> ConstrainedDynamicsInfo:
        """
        Returns the CFD problem info container holding dimensional information and meta-data about the CFD problem.
        """
        if self._info is None:
            raise ValueError("CFD problem info is not available. Ensure that `finalize()` has been called.")
        return self._info

    @property
    def system(self) -> ConstrainedSystem:
        """
        Returns the constrained dynamical system container holding references
        to the model, data, limits, contacts and Jacobians containers.
        """
        if self._system is None:
            raise ValueError("Constrained system is not available. Ensure that `finalize()` has been called.")
        return self._system

    @property
    def preconditioner(self) -> wp.array | None:
        """
        Returns the preconditioner array of the CFD problem if it has been built, otherwise None.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()
        return self._data.P

    @preconditioner.setter
    def preconditioner(self, P: wp.array):
        """
        Sets the preconditioner for the CFD problem.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # Assign the provided preconditioner
        # values to the problem data container
        self._data.P = P

    @property
    def regularizer(self) -> wp.array | None:
        """
        Returns the regularizer array of the CFD problem if it has been built, otherwise None.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()
        return self._data.sigma

    @regularizer.setter
    def regularizer(self, sigma: wp.array):
        """
        Sets the regularizer array for the CFD problem.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()
        self._data.sigma = sigma

    ###
    # Operations
    ###

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino,
        jacobians: SystemJacobiansType,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        configs: list[ConstrainedDynamicsConfig] | ConstrainedDynamicsConfig | None = None,
        device: wp.DeviceLike = None,
        **kwargs: dict[str, Any],
    ):
        """
        TODO
        """
        # Ensure the simulation containers are valid
        assert_containers_are_valid(model=model, data=data, limits=limits, contacts=contacts)

        # Construct the constrained dynamical system container
        # by capturing references to the provided system containers.
        self._system = ConstrainedSystem(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            jacobians=jacobians,
        )

        # Compute total problem dimensions
        sum_of_max_problem_dims, max_of_max_problem_dims = self._get_sum_max_problem_dims_impl()

        # First create the constrained dynamics info container
        self._info = ConstrainedDynamicsInfo.from_containers(
            sum_of_max_problem_dims=sum_of_max_problem_dims,
            max_of_max_problem_dims=max_of_max_problem_dims,
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
            configs=configs,
            device=device,
        )

        # TODO: How can we have a single all-worlds mask that is provided by the SolverKamino class?
        # Allocate a default world mask that includes all worlds in the CFD problem
        with wp.ScopedDevice(self._device):
            self._all_worlds_mask = wp.ones((self.info.num_worlds,), dtype=wp.int32)

        # Call the implementation-specific finalization method to
        # allocate problem data and perform any additional setup
        # required by the specific CFD problem formulation.
        self._data = self._finalize_impl(**kwargs)

        # Ensure that the implementation-specific finalization method returns a valid problem data container
        if self._data is None:
            raise ValueError(
                "The implementation-specific `_finalize_impl` method must "
                "return a valid `ConstrainedDynamicsProblemData` container."
            )
        elif not issubclass(type(self._data), ConstrainedDynamicsProblemData):
            raise TypeError(
                f"The implementation-specific `_finalize_impl` method must return a container "
                f"derived from `ConstrainedDynamicsProblemData`, but got `{type(self._data)}`."
            )

    def reset(self):
        """
        Resets the CFD problem data to initial/sentinel values.

        Note:
            The exact reset behavior is implementation-specific, but typically
            involves setting the problem data to zero or other sentinel values.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # Call the implementation-specific reset method to
        # reset the problem data to initial/sentinel values.
        self._reset_impl()

    def build(self, reset_to_zero: bool = True):
        """
        Builds the constrained dynamics problem given the current values contained
        in the cached system containers, i.e. model, data, limits and contacts data.

        Args:
            reset_to_zero (bool):
                Whether to reset the problem data before building.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # Initialize problem data
        if reset_to_zero:
            self.reset()

        # Call the implementation-specific build method to populate the problem data
        self._build_impl(reset_to_zero=reset_to_zero)

        # Optionally build and apply the problem preconditioner
        if any(s.preconditioning for s in self._configs):
            self._build_preconditioner_impl()
            self._apply_preconditioner_impl()

    ###
    # Linear-System Operations
    ###

    def precompute(self, world_mask: wp.array | None = None):
        """
        Performs the necessary pre-computation steps for the CFD problem in preparation for solving, such as:
        - Computing problem preconditioners
        - Updating problem regularization parameters
        - Optionally performing any necessary matrix factorization steps, if applicable.

        Args:
            world_mask (wp.array, optional):
                An array containing masking values indicating which
                worlds to perform the pre-computation steps for.\n
                Shape of `(num_worlds,)` and type :class:`int32`.

        Raises:
            ValueError: If the problem matrix is not allocated or if a linear
            solver is not available for the required pre-computation steps.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Perform the implementation-specific pre-computation steps for the CFD problem
        self._precompute_impl(world_mask=_world_mask)

    def solve(self, v: wp.array, x: wp.array, world_mask: wp.array | None = None):
        """
        Solves the inner linear system of the forward dynamics to render the unconstrained solution.

        Args:
            v (wp.array):
                The input array holding the right-hand side vectors of each world's linear system.
            x (wp.array):
                The output array to hold the solution vectors of each world's linear system.
            world_mask (wp.array, optional):
                An array containing masking values indicating which
                worlds to solve the linear system for.\n
                Shape of `(num_worlds,)` and type :class:`int32`.

        Raises:
            ValueError: If the problem matrix is not allocated or if a linear solver is not available.
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Solve the linear system using the implementation-specific solve method
        # to compute x = A^{-1} v for the worlds specified in the world mask
        self._solve_impl(v=v, x=x, world_mask=_world_mask)

    ###
    # BLAS Operations
    ###

    def matvec(self, x: wp.array, y: wp.array, world_mask: wp.array | None = None):
        """
        TODO
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Call the implementation-specific matvec method to compute y = A x
        self._matvec_impl(x=x, y=y, world_mask=_world_mask)

    def matvec_transpose(self, y: wp.array, x: wp.array, world_mask: wp.array | None = None):
        """
        TODO
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Call the implementation-specific matvec transpose method to compute x = A^T y
        self._matvec_transpose_impl(y=y, x=x, world_mask=_world_mask)

    def gemv(self, x: wp.array, y: wp.array, world_mask: wp.array | None = None, alpha: float = 1.0, beta: float = 0.0):
        """
        TODO
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Call the implementation-specific gemv method to compute y = alpha A x + beta y
        self._gemv_impl(x=x, y=y, world_mask=_world_mask, alpha=alpha, beta=beta)

    def gemv_transpose(
        self,
        y: wp.array,
        x: wp.array,
        world_mask: wp.array | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        """
        TODO
        """
        # Ensure that `finalize()` was called
        self._assert_is_finalized()

        # If no world mask is provided, use the default all-worlds mask
        # to solve the linear system for all worlds in the CFD problem
        if world_mask is None:
            _world_mask = self._all_worlds_mask
        else:
            _world_mask = world_mask

        # Call the implementation-specific gemv transpose method to compute x = alpha A^T y + beta x
        self._gemv_transpose_impl(y=y, x=x, world_mask=_world_mask, alpha=alpha, beta=beta)

    ###
    # Internals
    ###

    def _assert_is_finalized(self):
        """
        Asserts that the CFD problem has been finalized and that
        the core system containers and info container are available.

        Raises:
            ValueError: If the CFD problem is not finalized or if the required containers are not available.
        """
        if self._system is None:
            raise ValueError("CFD problem is not finalized: system container is not available.")
        if self._info is None:
            raise ValueError("CFD problem is not finalized: info container is not available.")
        if self._data is None:
            raise ValueError("CFD problem is not finalized: problem data container is not available.")

    ###
    # Implementation API
    ###

    @abstractmethod
    def _get_sum_max_problem_dims_impl(self) -> tuple[int, int]:
        """Returns the formulation-specific sum of maximum problem dimensions across all worlds."""
        raise NotImplementedError("Implementation-specific `_get_sum_max_problem_dims_impl` method is not defined.")

    @abstractmethod
    def _finalize_impl(self, **kwargs: dict[str, Any]) -> ConstrainedDynamicsProblemData:
        raise NotImplementedError("Implementation-specific `_finalize_impl` method is not defined.")

    @abstractmethod
    def _reset_impl(self):
        # NOTE: The default implementation is a no-op since
        # not all CFD problem formulations may require resetting.
        pass

    @abstractmethod
    def _build_impl(self, reset_to_zero: bool = False):
        raise NotImplementedError("Implementation-specific `_build_impl` method is not defined.")

    @abstractmethod
    def _build_preconditioner_impl(self):
        # NOTE: The default implementation is a no-op since not
        # all CFD problem formulations may require preconditioning.
        pass

    @abstractmethod
    def _apply_preconditioner_impl(self):
        # NOTE: The default implementation is a no-op since not
        # all CFD problem formulations may require preconditioning.
        pass

    @abstractmethod
    def _precompute_impl(self, world_mask: wp.array) -> None:
        raise NotImplementedError("The `_precompute_impl` operation is not implemented.")

    @abstractmethod
    def _solve_impl(self, v: wp.array, x: wp.array, world_mask: wp.array) -> None:
        raise NotImplementedError("The `_solve_impl` operation is not implemented.")

    @abstractmethod
    def _matvec_impl(self, x: wp.array, y: wp.array, world_mask: wp.array):
        raise NotImplementedError("The `_matvec_impl` operation is not implemented.")

    @abstractmethod
    def _matvec_transpose_impl(self, y: wp.array, x: wp.array, world_mask: wp.array):
        raise NotImplementedError("The `_matvec_transpose_impl` operation is not implemented.")

    @abstractmethod
    def _gemv_impl(self, x: wp.array, y: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0):
        raise NotImplementedError("The `_gemv_impl` operation is not implemented.")

    @abstractmethod
    def _gemv_transpose_impl(
        self, y: wp.array, x: wp.array, world_mask: wp.array, alpha: float = 1.0, beta: float = 0.0
    ):
        raise NotImplementedError("The `_gemv_transpose_impl` operation is not implemented.")
