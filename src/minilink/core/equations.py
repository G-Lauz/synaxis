from __future__ import annotations

from typing import Any, Callable, Concatenate, Generic, ParamSpec, Protocol, TypeVar, cast

from .components import ComponentDescriptor, ComponentKind

OwnerType = TypeVar("OwnerType")
P = ParamSpec("P")  # parameters
R = TypeVar("R")  # return type
R_co = TypeVar("R_co", covariant=True)


class EquationProtocol(Protocol[P, R_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


class Equation(ComponentDescriptor, Generic[OwnerType, P, R]):
    kind = ComponentKind.EQUATION

    def __init__(self, func: Callable[Concatenate[OwnerType, P], R]) -> None:
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        owner_id = None if self.owner_id is None else self.owner_id.hex[:5]
        return f'Equation(name="{self.name}", id="{self._uuid.hex[:5]}", owner={self.owner_cls}("id={owner_id}"))'

    def clone(self):
        equation_type = type(self)
        return equation_type(self.func)

    def __get__(self, instance: object | None, owner: type | None = None) -> Any:
        if instance is None:
            return self

        # Materialize the per-instance descriptor object via ComponentDescriptor so identity
        # discovery cab still find owned Equation instances.
        equation_obj = super().__get__(instance, owner)
        owner_instance = cast(OwnerType, instance)

        def bound(*args: P.args, **kwargs: P.kwargs) -> R:
            return equation_obj.func(owner_instance, *args, **kwargs)

        return cast(EquationProtocol[P, R], bound)


def equation(func: Callable[Concatenate[OwnerType, P], R]) -> Equation[OwnerType, P, R]:
    return Equation(func)
