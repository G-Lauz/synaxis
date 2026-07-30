from __future__ import annotations

import abc
import dataclasses
import uuid
from collections.abc import Mapping
from typing import Any, Callable, TypeVar

from .components import ComponentDescriptor, ComponentKind
from .computation_graph import BipartiteComputationGraph, OperationKind, OperationNode, SignalNode
from .equations import Equation, equation
from .signals import Output, Signal, SignalKind, StateDerivative, _evaluate_signals, _trace_signals

ComponentT = TypeVar("ComponentT", bound=ComponentDescriptor)


@dataclasses.dataclass(frozen=True)
class Connection:
    src: Signal
    tgt: Signal
    _uuid: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, init=False)


class System(ComponentDescriptor, abc.ABC):
    kind = ComponentKind.SYSTEM
    _blocks: dict[uuid.UUID, System]
    _signals: dict[uuid.UUID, Signal]
    _equations: dict[uuid.UUID, Equation]
    _connections: list[Connection]

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        object.__setattr__(instance, "_blocks", {})
        object.__setattr__(instance, "_signals", {})
        object.__setattr__(instance, "_equations", {})
        object.__setattr__(instance, "_connections", [])
        object.__setattr__(instance, "name", cls.__name__)
        instance._materialize_class_components()
        return instance

    def __init__(self, name: str | None = None):
        """
        Caution: __init__ could not be called by the user, hence everything here should be optional
        """
        super().__init__()

        self._user_defined_name = name
        self.name = name if name is not None else self.name

    def __repr__(self) -> str:
        owner_id = None if self.owner_id is None else self.owner_id.hex[:5]
        owner = f'{self.owner_cls}("id={owner_id}")' if self.owner_cls is not None else "None"
        return f'System(name="{self.name}", id="{self._uuid.hex[:5]}, owner={owner}")'

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, ComponentDescriptor):
            self._bind_owned_component(name, value)

        super().__setattr__(name, value)

        if isinstance(value, ComponentDescriptor):
            self._register_owned(value)
        else:
            self._unregister_owned(name)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        self._unregister_owned(name)

    def _bind_owned_component(self, name: str, value: ComponentDescriptor) -> None:
        usage_name = value._user_defined_name if value._user_defined_name is not None else name
        value.bind_owner(name=usage_name, owner=type(self), owner_id=self._uuid)

    def _register_owned(self, value: ComponentDescriptor) -> None:
        component_id = value._uuid

        if isinstance(value, System):
            self._blocks[component_id] = value
        elif isinstance(value, Signal):
            self._signals[component_id] = value
        elif isinstance(value, Equation):
            self._equations[component_id] = value
        else:
            raise TypeError(f"unknown component type {type(value)} for {value.name}")

    def _unregister_owned(self, name: str) -> None:
        component = getattr(self, name, None)
        if component is None or not isinstance(component, ComponentDescriptor):
            return

        component_id = component._uuid
        if isinstance(component, System):
            self._blocks.pop(component_id, None)
        elif isinstance(component, Signal):
            self._signals.pop(component_id, None)
        elif isinstance(component, Equation):
            self._equations.pop(component_id, None)
        else:
            raise TypeError(f"unknown component type {type(component)} for {component.name}")

    def _materialize_class_components(self) -> None:
        system_type = type(self)
        for cls in reversed(system_type.__mro__):
            for name, template in vars(cls).items():
                if not isinstance(template, ComponentDescriptor):
                    continue

                if getattr(system_type, name) is template:
                    template._materialize(self)

    def clone(self):
        system_type = type(self)
        return system_type(name=self.name)

    def connect(self, source: Signal, target: Signal) -> None:
        self._connections.append(Connection(src=source, tgt=target))

    def compile(self, *, allow_algebraic_loops: bool = False) -> CompiledSystem:
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

        # 1, 2, 3, 4, 5-ish
        (
            descendants,
            classified_descendants,
            path_by_id,
            raw_connections,
        ) = self._parse_descendants()

        # 5. get equations hyperedges
        equation_edges: list[tuple[uuid.UUID, tuple[Signal, ...], tuple[Signal, ...]]] = []
        for eq in classified_descendants.get(ComponentKind.EQUATION, {}).values():
            # TODO: documentation: equations are expected to be pure functions
            # interdiction:
            # - no mutation of any signal held by self or any other system
            # - no reading of mutable higher scope variable
            # - no control flow based on signal values (except if provide by the framework)

            owner_obj = descendants.get(eq.owner_id, None)
            if owner_obj is None:
                raise ValueError(f"equation {self.describe_path(path_by_id[eq._uuid])} (id={eq._uuid}) has no owner")

            # trace the equation to find inputs
            try:
                with _trace_signals() as traced_signals:
                    result = eq.func(owner_obj)
                    # nested return valus are possible
                    # this make sure to record all the signals returned by the equation
                    nested_results = self._result_signals(result)
                    for signal in nested_results:
                        signal.get_value()
            except Exception as exception:
                raise RuntimeError(
                    f"failed to trace equation {self.describe_path(path_by_id[eq._uuid])} (id={eq._uuid})"
                ) from exception

            equation_inputs: tuple[Signal, ...] = tuple(signal for signal in traced_signals)

            # TODO (NEED IMPROVEMENT): all this constructs make the whole system couple to a state space model. This may limit the generality of the framework, preventing someone to implement its own equation and specify where it should be in the system computation graph.

            # TODO (NEED IMPROVEMENT): we currently assume that the equation outputs are the signals of the owner component. This may not be true in general, and we should find a way to specify the outputs of an equation.

            # parse output and state signals from the component to define equation outputs
            # although the state equations doesn't return a state but rather the state derivative, we still consider the state as an output since it shares the same metadata
            if eq.name == "_compute_outputs":
                equation_outputs = tuple(output for output in owner_obj._signals.values() if isinstance(output, Output))
            elif eq.name == "_compute_dynamics":
                equation_outputs = tuple(
                    state for state in owner_obj._signals.values() if isinstance(state, StateDerivative)
                )
            else:
                raise ValueError(
                    f"unsuported equation at {self.describe_path(path_by_id[eq._uuid])} (id={eq._uuid[5:]}); "
                    f"only `_compute_outputs` and `_compute_dynamics` are supported"
                )

            equation_edges.append((eq._uuid, equation_inputs, equation_outputs))

        # 6. build the computation graph
        # signals and equations are nodes of the graph, connections are edges.
        # directed graph: inputs -> equation -> outputs
        graph = BipartiteComputationGraph()

        signal_nodes: dict[uuid.UUID, SignalNode] = {
            signal_id: SignalNode(
                id=signal_id,
                path=self.describe_path(path_by_id[signal_id]),
                kind=signal.signal_kind,
            )
            for signal_id, signal in classified_descendants.get(ComponentKind.SIGNAL, {}).items()
        }

        for signal in signal_nodes.values():
            graph.add_node(signal)

        for eq_id, inputs, outputs in equation_edges:
            eq = descendants[eq_id]
            owner_obj = descendants[eq.owner_id]
            operation_node = OperationNode(
                id=eq_id,
                path=self.describe_path(path_by_id[eq_id]),
                kind=OperationKind.EQUATION,
                fn=make_equation_fn(eq, owner_obj, inputs, outputs),
            )
            graph.add_node(operation_node)

            for signal in inputs:
                graph.add_edge(source=signal_nodes[signal._uuid], target=operation_node)

            for signal in outputs:
                graph.add_edge(source=operation_node, target=signal_nodes[signal._uuid])

        for connection in raw_connections:
            operation = OperationNode(
                id=connection._uuid,
                path=f"{self.describe_path(path_by_id[connection.src._uuid])} -> {self.describe_path(path_by_id[connection.tgt._uuid])}",
                kind=OperationKind.CONNECTION,
                fn=make_connection_fn(source_id=connection.src._uuid, target_id=connection.tgt._uuid),
            )
            graph.add_edge(
                source=signal_nodes[connection.src._uuid],
                target=operation,
            )
            graph.add_edge(
                source=operation,
                target=signal_nodes[connection.tgt._uuid],
            )

        # 7. Check for algebraic loops (cycles) in the graph
        # For now we raise an error, but implicit/residual solver could be used in future instead
        if graph.has_algebraic_loop() and not allow_algebraic_loops:
            remaining_nodes = set(graph.successors) - set(graph.topological_order())

            # remove connection nodes from the remaining nodes to focus on the actual components involved in the loop
            remaining_nodes = [
                node
                for node in remaining_nodes
                if isinstance(node, SignalNode)
                or (isinstance(node, OperationNode) and node.kind != OperationKind.CONNECTION)
            ]

            remaining_paths = [node.path for node in remaining_nodes]
            raise ValueError(
                f"algebraic loop detected in the system. The following components are part of the loop: {remaining_paths}"
            )

        # get initial values for unconnected signals (inputs, states, params) to be used at runtime
        initial_values = {
            node.id: descendants[node.id].get_value()
            for node in graph.get_unconnected_nodes()
            if isinstance(node, SignalNode)
        }

        return CompiledSystem(graph, initial_values)

    def _result_signals(self, value):
        if isinstance(value, Signal):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._result_signals(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._result_signals(item)

    def describe_path(self, path: tuple[str, ...]) -> str:
        return ".".join(path) if path else "root"

    def _parse_descendants(self) -> Any:
        visited: set[uuid.UUID] = set()
        descendants: dict[uuid.UUID, ComponentDescriptor] = {}
        classified_descendants: dict[ComponentKind, dict[uuid.UUID, ComponentDescriptor]] = {}
        path_by_id: dict[uuid.UUID, tuple[str, ...]] = {}
        raw_connections: list[Connection] = []

        def _visit(system: System, path: tuple[str, ...] = ()) -> None:
            if system._uuid in visited:
                raise ValueError(f"circular ownership detected for component {system.name} (id={system._uuid})")

            visited.add(system._uuid)

            descendants[system._uuid] = system
            path_by_id[system._uuid] = path
            classified_descendants.setdefault(ComponentKind.SYSTEM, {})[system._uuid] = system
            if system._connections is not None:
                raw_connections.extend(system._connections)

            for signal in system._signals.values():
                descendants[signal._uuid] = signal
                path_by_id[signal._uuid] = path + (signal.name,)
                classified_descendants.setdefault(ComponentKind.SIGNAL, {})[signal._uuid] = signal

            for eq in system._equations.values():
                descendants[eq._uuid] = eq
                path_by_id[eq._uuid] = path + (eq.name,)
                classified_descendants.setdefault(ComponentKind.EQUATION, {})[eq._uuid] = eq

            for block in system._blocks.values():
                block_path = path + (block.name,)
                _visit(block, block_path)

        _visit(self)

        return (
            descendants,
            classified_descendants,
            path_by_id,
            raw_connections,
        )

    def pretty_print_identity(self, indent: int = 0):
        signals_names = [signal.name for signal in self._signals.values()]
        equations_names = [equation.name for equation in self._equations.values()]
        connections_str = [
            f"{connexion.src.owner_cls}({connexion.src.name}) -> {connexion.tgt.owner_cls}({connexion.tgt.name})"
            for connexion in self._connections
        ]

        print(f"{'  ' * indent}System: {self.name}")
        print(f"{'  ' * indent}-----------------------------------------------------------------")
        print(f"{'  ' * indent}Signals: {signals_names}")
        print(f"{'  ' * indent}Equations: {equations_names}")
        print(f"{'  ' * indent}Blocks:")
        print(f"{'  ' * indent}Connections: {connections_str}")

        for blck in self._blocks.values():
            blck.pretty_print_identity(indent + 1)

        print(f"{'  ' * indent}-----------------------------------------------------------------\n")


class CompiledSystem:
    def __init__(self, graph: BipartiteComputationGraph, initial_values: Mapping[uuid.UUID, Any]):
        self.graph = graph

        self._id_nodes_map: dict[uuid.UUID, SignalNode] = {
            node.id: node for node in self.graph.successors if isinstance(node, SignalNode)
        }

        self._signal_ids_by_kind: dict[SignalKind, tuple[uuid.UUID, ...]] = {
            kind: tuple(node.id for node in self._id_nodes_map.values() if node.kind == kind) for kind in SignalKind
        }

        self._initial_values = dict(initial_values)

    def __setitem__(self, signal: Signal, value: Any) -> None:
        if signal._uuid not in self._initial_values:
            available_paths = [self._id_nodes_map[id].path for id in self._initial_values]
            raise KeyError(
                f"signal {signal.name} (id={signal._uuid.hex[:5]}) is not part of the compiled system or is not a"
                f" source signal. Available source signals: {available_paths}"
            )

        self._initial_values[signal._uuid] = value

    # TODO: proper type hinting
    def initial_values(self) -> Mapping[uuid.UUID, Any]:
        """Return a copy of the values required to evaluate the system"""
        return self._initial_values.copy()

    def evaluate(self, source_values: Mapping[uuid.UUID, Any]) -> dict[uuid.UUID, Any]:
        """Evaluate one graph pass and return values for every signal."""
        source_ids = set(source_values)
        expected_ids = set(self._initial_values)
        missing_ids = expected_ids - source_ids
        unexpected_ids = source_ids - expected_ids
        if missing_ids or unexpected_ids:
            missing_paths = [self._id_nodes_map[id].path for id in missing_ids]
            unexpected_paths = [self._id_nodes_map[id].path for id in unexpected_ids]
            raise ValueError(
                f"source values do not match the compiled system; "
                f"missing signals: {missing_paths}, unexpected signals: {unexpected_paths}"
            )

        values = dict(source_values)

        for node in self.graph.topological_order():
            if not isinstance(node, OperationNode):
                continue

            updates = node.fn(values)
            values.update(updates)

        return values


class StaticSystem(System):
    @equation
    def _compute_outputs(self):
        return self.compute_outputs()

    @abc.abstractmethod
    def compute_outputs(self) -> Any:
        pass


class DynamicSystem(System):
    # TODO: Implicit state derivative registration is disabled for now, because it couldn't be discovered statically by
    # the IDE typing engine. Instead, we can rely on the user to explicitly define state derivatives as needed.
    # def _register_owned(self, value: ComponentDescriptor) -> None:
    #     super()._register_owned(value)

    #     if not isinstance(value, State):
    #         return

    #     derivative = StateDerivative.from_state(value)

    #     # bind the derivative to the system
    #     self._bind_owned_component(f"d{value.name}", derivative)
    #     super()._register_owned(derivative)

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


def make_equation_fn(
    equation: Equation, owner_obj: System, inputs: tuple[Signal, ...], outputs: tuple[Signal, ...]
) -> Callable[[Mapping[uuid.UUID, Any]], dict[uuid.UUID, Any]]:
    def resolve_signals(value) -> Any:
        if isinstance(value, Signal):
            return value.get_value()
        if isinstance(value, list):
            return [resolve_signals(item) for item in value]
        if isinstance(value, tuple):
            return tuple(resolve_signals(item) for item in value)
        if isinstance(value, Mapping):
            return {key: resolve_signals(item) for key, item in value.items()}
        return value

    # warm start for error handling and bounding strategy at compilation time
    dummy_values = {signal._uuid: signal.get_value() for signal in inputs}
    with _evaluate_signals(dummy_values):
        dummy_result = resolve_signals(equation.func(owner_obj))

    # define appropriate function bounding and error handling
    bound_result: Callable[[Any], dict[uuid.UUID, Any]] | None = None

    is_signal_mapping = isinstance(dummy_result, Mapping) and any(isinstance(key, Signal) for key in dummy_result)

    if not outputs and dummy_result is None:
        # no outputs and no result
        # TODO: does this mean that the equation is useless? or should it be considered as constraints?
        # if so, also adjust the function signature `outputs: tuple[Signal, ...]` -> `outputs: tuple[Signal]`
        bound_result = lambda result: {}

    if len(outputs) == 1 and not is_signal_mapping:
        # single output doesn't need to be a mapping
        bound_result = lambda result: {outputs[0]._uuid: result}

    if bound_result == None and not isinstance(dummy_result, Mapping):
        raise TypeError(
            f"equation {equation.name}({equation._uuid.hex[:5]}) has {len(outputs)} outputs and must return a mapping keyed by output signals"
        )

    if bound_result == None and dummy_result is not None:
        invalid_keys = [key for key in dummy_result if not isinstance(key, Output)]
        if invalid_keys:
            raise TypeError(
                f"equation {equation.name}({equation._uuid.hex[:5]}) must use output signals as mapping keys, but got invalid keys: {invalid_keys}"
            )

        expected_by_ids = {signal._uuid: signal for signal in outputs}
        actual_by_ids = {signal._uuid: signal for signal in dummy_result}
        expected_ids = set(expected_by_ids)
        actual_ids = set(actual_by_ids)
        missing_outputs = expected_ids - actual_ids
        unexpected_outputs = actual_ids - expected_ids
        if missing_outputs or unexpected_outputs:
            missing = [expected_by_ids[id].name for id in missing_outputs]
            unexpected = [actual_by_ids[id].name for id in unexpected_outputs]
            raise ValueError(
                f"equation {equation.name}({equation._uuid.hex[:5]}) must return a mapping with keys corresponding to its outputs, but got missing keys: {missing} and unexpected keys: {unexpected}"
            )

    if bound_result is None:
        # multiple outputs must be a mapping keyed by output signals
        bound_result = lambda result: {signal._uuid: value for signal, value in result.items()}

    def fn(values_by_signal_id: Mapping[uuid.UUID, Any]) -> dict[uuid.UUID, Any]:
        with _evaluate_signals(values_by_signal_id):
            result = resolve_signals(equation.func(owner_obj))
        return bound_result(result)

    return fn


def make_connection_fn(
    source_id: uuid.UUID, target_id: uuid.UUID
) -> Callable[[Mapping[uuid.UUID, Any]], dict[uuid.UUID, Any]]:
    def fn(values_by_signal_id: Mapping[uuid.UUID, Any]) -> dict[uuid.UUID, Any]:
        return {target_id: values_by_signal_id[source_id]}

    return fn
