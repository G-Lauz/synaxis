import abc
from collections.abc import Mapping
from typing import Any, TypeVar

from synaxis.core import Output, System, equation

SignalType = TypeVar("SignalType")


class StaticSystem(System, abc.ABC):
    @equation(name="compute_outputs")
    def _compute_outputs(self) -> Output[SignalType] | Mapping[Output[SignalType], SignalType]:
        return self.compute_outputs()

    @abc.abstractmethod
    def compute_outputs(self) -> Any:
        pass
