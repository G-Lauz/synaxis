from __future__ import annotations

import abc
import dataclasses
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

from .compiler import compile
from .context import is_signal_in_context, read_signal
from .declaration import Declaration
from .equations import Equation, equation
from .runtime import CompiledSystem
from .signals import Signal, SignalType, StateDerivative, _SourceSignal

if TYPE_CHECKING:
    from .signals import Output


@dataclasses.dataclass(frozen=True)
class Connection:
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, init=False)
    src: Signal
    tgt: Signal


class System(Declaration, abc.ABC):
    _connections: set[Connection]

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        object.__setattr__(instance, "name", cls.__name__)
        return instance

    def __init__(
        self,
        *,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._connections = set()

        self.name = name if name is not None else self.name

    def __getattribute__(self, name: str, /) -> Any:
        value = super().__getattribute__(name)
        if isinstance(value, _SourceSignal) and is_signal_in_context():
            return read_signal(value)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Declaration):
            value.bind_name(name)  # when a declaration is assigned to an attribute with `self.attr = Declaration()`
        super().__setattr__(name, value)

    def clone(self) -> Self:
        system_type = type(self)
        return system_type(name=self.name)

    def connect(self, source: Signal, target: Signal) -> None:
        """Connect two signals together."""
        self._connections.add(Connection(src=source, tgt=target))

    def compile(self, *, allow_algebraic_loop: bool = False) -> CompiledSystem:
        return compile(self, allow_algebraic_loop=allow_algebraic_loop)

    def components(self):
        subsystems: dict[str, System] = {}
        signals: dict[str, Signal] = {}
        equations: dict[str, Equation] = {}

        resolved_names = set()

        # collect class-level declarations from the MRO (Method Resolution Order)
        system_type = type(self)
        for cls in reversed(system_type.__mro__):
            for name, descriptor in vars(cls).items():
                # Any subclass descriptor overrides the inherited members
                subsystems.pop(name, None)
                signals.pop(name, None)
                equations.pop(name, None)

                resolved_names.discard(name)  # remove the name from resolved_names to allow overriding

                # get the instance-level component through the descriptor's __get__ method
                component_instance = getattr(self, name, None)

                if isinstance(component_instance, System):
                    subsystems[name] = component_instance
                    resolved_names.add(name)
                elif isinstance(component_instance, Signal):
                    signals[name] = component_instance
                    resolved_names.add(name)
                elif isinstance(component_instance, Equation):
                    equations[name] = component_instance
                    resolved_names.add(name)

        # collect instance-level declarations
        for name, component in vars(self).items():
            # instance declarations whose name appear in resolved_names where already handle through getattr()
            if name in resolved_names:
                continue

            if isinstance(component, System):
                subsystems[name] = component
            elif isinstance(component, Signal):
                signals[name] = component
            elif isinstance(component, Equation):
                equations[name] = component

        return subsystems.values(), signals.values(), equations.values()

    def pretty_print(self, *, indent: int = 0):
        subsystems, signals, equations = self.components()

        signal_names = [f"{signal.name}(id={signal.id.hex[:5]})" for signal in signals]
        equation_names = [f"{equation.name}(id={equation.id.hex[:5]})" for equation in equations]
        connection_names = [
            f"{conn.src.name}(id={conn.src.id.hex[:5]})->{conn.tgt.name}(id={conn.tgt.id.hex[:5]})"
            for conn in self._connections
        ]

        print(f"{'  ' * indent}System: {self.name}(id={self.id.hex[:5]})")
        print(f"{'  ' * indent}-----------------------------------------------------------------")
        print(f"{'  ' * indent}Signals: {signal_names}")
        print(f"{'  ' * indent}Equations: {equation_names}")
        print(f"{'  ' * indent}Blocks:")
        print(f"{'  ' * indent}Connections: {connection_names}")

        for subsystem in subsystems:
            subsystem.pretty_print(indent=indent + 1)

        print(f"{'  ' * indent}-----------------------------------------------------------------\n")
