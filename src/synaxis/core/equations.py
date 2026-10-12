from __future__ import annotations

import types
import uuid
from collections.abc import Callable
from typing import Any, Generic, TypeVar

R = TypeVar("R")


class Equation(Generic[R]):
    _uuid: uuid.UUID
    _function: Callable[[Any], R]
    _bound: Callable[[], R] | None

    def __init__(self, function: Callable[[Any], R]) -> None:
        if not callable(function):
            raise TypeError(f"an equation must be a callable takin the system instance, got {type(function).__name__}")

        self._uuid = uuid.uuid4()
        self._function = function
        self._bound = None

    @property
    def id(self) -> uuid.UUID:
        return self._uuid

    @property
    def name(self) -> str:
        return getattr(self._function, "__name__", repr(self._function))

    @property
    def unbound_function(self) -> Callable[[Any], R]:
        return self._function

    @property
    def is_bound(self) -> bool:
        return self._bound is not None

    @property
    def function(self) -> Callable[[], R]:
        if self._bound is None:
            raise RuntimeError(f"equation {self.name!r} is not bound to a system instance")
        return self._bound

    def bind(self, instance: object) -> None:
        if self._bound is not None:
            if getattr(self._bound, "__self__", None) is instance:
                return  # already bound to the same instance

            raise RuntimeError(f"equation {self.name!r} is already bound to another system instance")

        # Bind the equation to the given system instance
        self._bound = types.MethodType(self._function, instance)
