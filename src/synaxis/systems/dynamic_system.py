import abc
from collections.abc import Mapping
from typing import Any, TypeVar

from synaxis.core import StateDerivative, equation

from .static_system import StaticSystem

SignalType = TypeVar("SignalType")


class DynamicSystem(StaticSystem, abc.ABC):
    # TODO: Implicit state derivative registration is disabled for now, because it couldn't be discovered statically by
    # the IDE typing engine. Instead, we can rely on the user to explicitly define state derivatives as needed.
    @equation(name="compute_dynamics", otype=StateDerivative)
    def _compute_dynamics(self) -> StateDerivative[SignalType] | Mapping[StateDerivative[SignalType], SignalType]:
        return self.compute_dynamics()

    @abc.abstractmethod
    def compute_dynamics(self) -> Any:
        pass
