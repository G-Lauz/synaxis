from __future__ import annotations

import abc
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .components import ComponentDescriptor
from .equations import Equation, equation
from .signals import Signal


class System(ComponentDescriptor, abc.ABC):
    _uuid: uuid.UUID

    _blocks: Optional[List[System]] = None
    _signals: Optional[List[Signal]] = None
    _connections: Optional[List[Tuple[Signal, Signal]]] = None
    _owned_components: Dict[str, ComponentDescriptor]

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._uuid = uuid.uuid4()
        instance.name = cls.__name__
        return instance

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # scan for decorated equation method
        for base in cls.__mro__[1:]:
            for name, template in vars(base).items():
                if not isinstance(template, Equation):
                    continue

                override = cls.__dict__.get(name)
                if override is None or isinstance(override, Equation):
                    continue

                wrapped = Equation(override)
                wrapped.__set_name__(cls, name)
                setattr(cls, name, wrapped)

    def __init__(self, name: Optional[str] = None):
        """
        Caution: __init__ could not be called by the user, hence everything here should be optional
        """
        super().__init__()
        self._owned_components = {}

        self._user_defined_name = name
        self.name = name if name is not None else self.name

    def __repr__(self) -> str:
        return f'System(name="{self.name}", id="{self._uuid.hex[:5]}")'

    def __setattr__(self, name: str, value: object) -> None:
        # call __set_name__ at instantiation to enable descriptor
        # bind owner if attribute has been defined within the init constructor
        if isinstance(value, ComponentDescriptor):
            usage_name = value._user_defined_name if value._user_defined_name is not None else name
            value.bind_owner(name=usage_name, owner=type(self), owner_id=self._uuid)

        super().__setattr__(name, value)

        if isinstance(value, ComponentDescriptor):
            self._register_owned(name, value)

    def _register_owned(self, name: str, value: ComponentDescriptor) -> None:
        if not hasattr(self, "_owned_components"):
            return

        self._owned_components[name] = value

    def clone(self):
        system_type = type(self)
        return system_type(name=self.name)

    def connect(self, source: Signal, target: Signal) -> None:
        if self._connections is None:
            self._connections = []

        self._connections.append((source, target))

    def compile(self):
        # 1. get global identity (a mapping of [id, obj] for all obj of interest including equation)
        # 2. get system child ownership (optional?)
        # 3. get signal (input, output, state, param)
        # 4. get connections
        #   - these become mathematical constraint such as: target - source = 0
        #   - they could later be simplified (see step 10)
        # 5. get equations (compute_output, compute_dynamics)
        #   - this mean we should have a way to know what signals are used within the function
        #   - may required tracing for intelligent behavior like per terms algebraic loop detection
        #     e.g. in the LTI system if D = 0 in Cx + Du then there's no algebraic loop
        #   - may be a map of readings (inputs, states, params) to outputs?
        # 6. build the equation graph as a bipartite graph: reading -> equation -> output where
        #    equation is just an edge. Then, add connections (step 4) that map the outputs and inputs
        #    of the equation graph
        #   - bipartite graph means that readings or outputs set aren't allowed to connect with
        #     another node within its set.
        # 7. topological sort and algebraic loop detections
        #
        # (Optional):
        # 8. build a global implicit model
        # 9. use the implicit model for explicit ODE reduction
        # 10. connection pruning: e.g. controller.u -> plant.u could be prune to a simple edge since
        #     plant.u = controller.u
        # 11. different possible runtime execution according to graph configuration:
        #   - static computation (by default), solve algebraic loop, general implicit model
        #
        # (Note):
        # equations and connections become the mathematical constraints of the composition of open
        # dynamical system. They later could be used to formulate the general implicit model:
        # F(xdot, x, z, r, p) = [xdot - Ax_plant - Bu_plant, ..., up - uc] = 0 see
        # differential-algebraic equations (DAE) for more details.
        pass

    def get_objects_identity(self):
        blocks = self._get_owned_members(System)
        signals = self._get_owned_members(Signal)
        equations = self._get_owned_members(Equation)

        blocks = {getattr(block, "_uuid"): block for block in blocks}
        signals = {getattr(signal, "_uuid"): signal for signal in signals}
        equations = {getattr(equation, "_uuid"): equation for equation in equations}

        return blocks, signals, equations


    def _get_owned_members(self, kind: type):
        # Ensure class-level descriptors are materialized and registered on this instance.
        system_type = type(self)
        for cls in reversed(system_type.__mro__):
            for name, template in vars(cls).items():
                if isinstance(template, kind):
                    value = getattr(self, name)
                    if isinstance(value, kind):
                        self._register_owned(name, value)

        members = []
        seen = set()

        # Discover via explicit registry (independent from descriptor return values).
        for name, value in self._owned_components.items():
            if isinstance(value, kind):
                members.append(value)
                seen.add(name)

        # Backward-compatible fallback for values present in __dict__ but not registered.
        for name, value in self.__dict__.items():
            if name in seen:
                continue

            if isinstance(value, kind):
                self._register_owned(name, value)
                members.append(value)

        return members

    def pretty_print_identity(self, indent: int = 0):
        blocks, signals, equations = self.get_objects_identity()

        signals_names = [signal.name for signal in signals.values()]
        equations_names = [equation.name for equation in equations.values()]

        print(f"{'  ' * indent}System: {self.name}")
        print(f"{'  ' * indent}-----------------------------------------------------------------")
        print(f"{'  ' * indent}Signals: {signals_names}")
        print(f"{'  ' * indent}Equations: {equations_names}")

        print(f"{'  ' * indent}Blocks:")

        for blck in blocks.values():
            blck.pretty_print_identity(indent + 1)

        print(f"{'  ' * indent}-----------------------------------------------------------------\n")


class StaticSystem(System):
    @equation
    def _compute_outputs(self):
        return self.compute_outputs()

    @abc.abstractmethod
    def compute_outputs(self) -> Any:
        pass


class DynamicSystem(System):
    @equation
    def _compute_outputs(self):
        return self.compute_outputs()

    @abc.abstractmethod
    def compute_outputs(self) -> Any:
        pass

    @equation
    def _compute_dynamics(self):
        return self.compute_dynamics()

    @abc.abstractmethod
    def compute_dynamics(self) -> Any:
        pass
