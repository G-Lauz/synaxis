from __future__ import annotations

import abc
import dataclasses
import uuid
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
        equation_edges: list[tuple[uuid.UUID, list[uuid.UUID], list[uuid.UUID]]] = []
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

            equation_inputs: list[uuid.UUID] = [signal._uuid for signal in traced_signals]

            # TODO (NEED IMPROVEMENT): all this constructs make the whole system couple to a state space model. This may limit the generality of the framework, preventing someone to implement its own equation and specify where it should be in the system computation graph.

            # TODO (NEED IMPROVEMENT): we currently assume that the equation outputs are the signals of the owner component. This may not be true in general, and we should find a way to specify the outputs of an equation.

            # parse output and state signals from the component to define equation outputs
            # although the state equations doesn't return a state but rather the state derivative, we still consider the state as an output since it shares the same metadata
            equation_outputs: list[uuid.UUID] = []

            if eq.name == "_compute_outputs":
                equation_outputs = [id for id, output in owner_obj._signals.items() if isinstance(output, Output)]
            elif eq.name == "_compute_dynamics":
                equation_outputs = [
                    id for id, state in owner_obj._signals.items() if isinstance(state, StateDerivative)
                ]
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
                value=signal.get_value(),
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
                fn=make_equation_fn(eq, owner_obj, tuple(inputs), len(outputs)),
            )
            graph.add_node(operation_node)

            for port, signal_id in enumerate(inputs):
                graph.add_edge(
                    source=signal_nodes[signal_id],
                    target=operation_node,
                    port=port,
                )

            for port, signal_id in enumerate(outputs):
                graph.add_edge(
                    source=operation_node,
                    target=signal_nodes[signal_id],
                    port=port,
                )

        for connection in raw_connections:
            operation = OperationNode(
                id=connection._uuid,
                path=f"{self.describe_path(path_by_id[connection.src._uuid])} -> {self.describe_path(path_by_id[connection.tgt._uuid])}",
                kind=OperationKind.CONNECTION,
                fn=lambda x: (x[0],),  # identity function
            )
            graph.add_edge(
                source=signal_nodes[connection.src._uuid],
                target=operation,
                port=0,
            )
            graph.add_edge(
                source=operation,
                target=signal_nodes[connection.tgt._uuid],
                port=0,
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

        return CompiledSystem(graph)

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
    def __init__(self, graph: BipartiteComputationGraph):
        self.graph = graph

        self._id_nodes_map: dict[uuid.UUID, SignalNode] = {
            node.id: node for node in self.graph.successors if isinstance(node, SignalNode)
        }

        input_nodes = self.graph.get_unconnected_nodes()
        self._initial_values: dict[uuid.UUID, Any] = {
            node.id: node.value for node in input_nodes if isinstance(node, SignalNode)
        }

    # def __getitem__(self, key: uuid.UUID) -> ComponentDescriptor:

    def __setitem__(self, signal: Signal, value: Any) -> None:
        node = self._id_nodes_map.get(signal._uuid, None)

        if node is None:
            raise KeyError(f"signal {signal.name} (id={signal._uuid.hex[:5]}) not found in compiled system")

        # only allow setting values for unconnected signals (i.e., those without predecessors in the graph)
        if self.graph.predecessors[node]:
            raise ValueError(
                f"signal {signal.name} (id={signal._uuid.hex[:5]}) is connected in the graph and cannot be set directly"
            )

        self._initial_values[signal._uuid] = value

    # TODO: proper type hinting
    def initial_values(self) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        states = tuple(
            value for id, value in self._initial_values.items() if self._id_nodes_map[id].kind == SignalKind.STATE
        )
        inputs = tuple(
            value for id, value in self._initial_values.items() if self._id_nodes_map[id].kind == SignalKind.INPUT
        )
        params = tuple(
            value for id, value in self._initial_values.items() if self._id_nodes_map[id].kind == SignalKind.PARAM
        )
        return states, inputs, params

    def evaluate(self, states, inputs, params):
        """Evaluate one graph pass.

        Runtime tuples map to unconnected signals in graph registration order.
        Outputs and state derivatives are returned in that same order.
        """
        signal_nodes = tuple(node for node in self.graph.successors if isinstance(node, SignalNode))
        values = {}

        runtime_values = (
            (SignalKind.STATE, states, "states"),
            (SignalKind.INPUT, inputs, "inputs"),
            (SignalKind.PARAM, params, "params"),
        )
        for kind, supplied_values, name in runtime_values:
            nodes = tuple(node for node in signal_nodes if node.kind == kind and not self.graph.predecessors[node])
            if not isinstance(supplied_values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(supplied_values) != len(nodes):
                raise ValueError(f"expected {len(nodes)} {name}, got {len(supplied_values)}")
            values.update(zip(nodes, supplied_values))

        for node in self.graph.topological_order():
            if not isinstance(node, OperationNode):
                continue

            arguments = [None] * len(self.graph.predecessors[node])
            for predecessor, edge in self.graph.predecessors[node].items():
                arguments[edge.port] = values[predecessor]

            results = node.fn(tuple(arguments))
            if not isinstance(results, tuple):
                raise TypeError(f"operation {node.path} must return a tuple")
            if len(results) != len(self.graph.successors[node]):
                raise ValueError(
                    f"operation {node.path} returned {len(results)} values; expected {len(self.graph.successors[node])}"
                )

            for successor, edge in self.graph.successors[node].items():
                values[successor] = results[edge.port]

        outputs = tuple(values[node] for node in signal_nodes if node.kind == SignalKind.OUTPUT)
        derivatives = tuple(values[node] for node in signal_nodes if node.kind == SignalKind.STATE_DERIVATIVE)
        return outputs, derivatives


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
    equation: Equation, owner_obj: System, input_ids: tuple[uuid.UUID, ...], output_count: int
) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def resolve_signals(value):
        if isinstance(value, Signal):
            return value.get_value()
        if isinstance(value, list):
            return [resolve_signals(item) for item in value]
        if isinstance(value, tuple):
            return tuple(resolve_signals(item) for item in value)
        if isinstance(value, dict):
            return {key: resolve_signals(item) for key, item in value.items()}
        return value

    def fn(values):
        values_by_signal_id = dict(zip(input_ids, values))

        with _evaluate_signals(values_by_signal_id):
            result = resolve_signals(equation.func(owner_obj))

        if output_count == 0:
            return ()
        if output_count == 1:
            return (result,)
        if isinstance(result, dict):
            return tuple(result.values())
        return tuple(result)

    return fn
