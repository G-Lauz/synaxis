from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .context import evaluate, sample_cache, trace
from .equations import Equation
from .graph import ComputationGraph, ConnectionNode, EquationNode, SignalNode
from .runtime import CompiledSystem
from .signals import (
    Input,
    Internal,
    Noise,
    Param,
    Signal,
    State,
    StateDerivative,
    Time,
    _ComputedSignal,
)

if TYPE_CHECKING:
    from .systems import Connection, System


_SignalEntry = tuple[Signal[Any], tuple[str, ...]]
_ConnectionEntry = tuple["Connection", tuple[str, ...]]


def compile(system: System, *, allow_algebraic_loop: bool) -> CompiledSystem:
    """Compile the system into a graph representation."""
    # flatten the system composition
    signals, connections = _flatten(system)
    _check_definitions(signals, connections)

    path_by_id = {signal.id: _format_path(path) for signal, path in signals}

    # graph creation
    graph = ComputationGraph()

    # add signals to the graph
    signal_node_pool: dict[uuid.UUID, SignalNode] = {}
    signal_pool: dict[uuid.UUID, Signal] = {}
    for signal, path in signals:
        signal_node = SignalNode(id=signal.id, path=path_by_id[signal.id], stype=type(signal))
        graph.add_node(signal_node)

        signal_node_pool[signal.id] = signal_node
        signal_pool[signal.id] = signal

    # add equations to the graph
    # TODO: documentation: equations are expected to be pure functions; interdiction:
    # - no mutation of any signal held by self or any other system
    # - no reading of mutable higher scope variable
    # - no control flow based on signal values (except if provide by the framework)
    with sample_cache():  # shape of computed signals propagate while tracing
        for signal, path in signals:
            if isinstance(signal, _ComputedSignal) and signal.equation is not None:
                equation_node = _compile_equation(signal.equation, signal, path + ("equation",))
                graph.add_node(equation_node)

                for input_id in equation_node.inputs:
                    if input_id not in signal_node_pool:
                        raise ValueError(
                            f"equation {equation_node.path} reads a signal (id={input_id.hex[:5]}) that is not part of "
                            f"system {system.name}"
                        )
                    graph.add_edge(signal_node_pool[input_id], equation_node)  # equation input

                # only one output is expected for a computed signal
                graph.add_edge(equation_node, signal_node_pool[signal.id])  # equation output

    # add connections to the graph
    for connection, _ in connections:
        connection_node = ConnectionNode(
            id=connection.id,
            path=f"{path_by_id[connection.src.id]}->{path_by_id[connection.tgt.id]}",
            source=connection.src.id,
            target=connection.tgt.id,
        )
        graph.add_node(connection_node)

        graph.add_edge(signal_node_pool[connection.src.id], connection_node)
        graph.add_edge(connection_node, signal_node_pool[connection.tgt.id])

    # check for algebraic loops (cycles) in the graph
    if graph.has_algebraic_loop() and not allow_algebraic_loop:
        remaining_nodes = set(graph.successors) - set(graph.topological_order)

        # remove connection nodes from the remaining nodes to focus on component nodes (signals and equations)
        remaining_nodes = [node.path for node in remaining_nodes if not isinstance(node, ConnectionNode)]
        raise ValueError(f"algebraic loop detected in the system {system.name}: {remaining_nodes}")

    # aggregate default source values from source signals (no predecessor)
    source_values = {
        node.id: signal_pool[node.id].get_value()
        for node in graph.get_unconnected_nodes()
        if isinstance(node, SignalNode)
    }

    # get state/state derivative pairs
    # this allow a solver to know how to update the state of the system at each time step
    state_pairs = tuple(
        (signal.state.id, signal.id)
        for signal, _ in signals
        if isinstance(signal, StateDerivative) and signal.state is not None
    )

    return CompiledSystem(graph=graph, sources=source_values, state_pairs=state_pairs)


def _flatten(system: System, *, path: tuple[str, ...] = ()) -> tuple[list[_SignalEntry], list[_ConnectionEntry]]:
    subsystems, system_signals = system.components()

    signals: list[_SignalEntry] = [(signal, path + (signal.name,)) for signal in system_signals]
    connections: list[_ConnectionEntry] = [(connection, path) for connection in system._connections]

    for subsystem in subsystems:
        sub_signals, sub_connections = _flatten(subsystem, path=path + (subsystem.name,))

        signals.extend(sub_signals)
        connections.extend(sub_connections)

    return signals, connections


def _check_definitions(signals, connections) -> None:
    path_by_id = {signal.id: _format_path(path) for signal, path in signals}

    errors: list[str] = []
    incoming: dict[uuid.UUID, list[str]] = {signal.id: [] for signal, _ in signals}

    # check connections
    for connection, owner_path in connections:
        src, tgt = connection.src, connection.tgt

        label = f"{path_by_id[src.id]} -> {path_by_id[tgt.id]}"

        foreigners = [signal.name for signal in (src, tgt) if signal.id not in path_by_id]

        if foreigners:
            errors.append(
                f"connection {label} declared in {_format_path(owner_path) or '<root>'} "
                f"references foreign signals outside the system: {', '.join(foreigners)}"
            )

            continue

        incoming[tgt.id].append(label)

        if isinstance(src, Internal):
            errors.append(
                f"connection {label}: internal signal {path_by_id[src.id]} cannot be used as a source in a connection"
            )

        if isinstance(tgt, Internal):
            errors.append(
                f"connection {label}: internal signal {path_by_id[tgt.id]} cannot be used as a target in a connection"
            )
        elif isinstance(tgt, (Param, Noise, Time, State)):
            errors.append(
                f"connection {label}: signal {path_by_id[tgt.id]} of type {type(tgt).__name__} cannot be a connection "
                "target (only Input, Output and StateDerivative can)"
            )

    # check signals
    derivative_by_state: dict[uuid.UUID, list[str]] = {signal.id: [] for signal, _ in signals}

    for signal, path in signals:
        label, kind = _format_path(path), type(signal).__name__

        if isinstance(signal, _ComputedSignal):
            has_equation = signal.equation is not None
            n_definitions = int(has_equation) + len(incoming[signal.id])

            if isinstance(signal, Internal) and not has_equation:
                errors.append(
                    f"internal signal {label} must carry an equation; define it using `@Internal` as a decorator or "
                    "`Internal(fn)` as a constructor"
                )
            elif n_definitions == 0:
                errors.append(
                    f"signal {label} of type {kind} has no definition: give it an equation "
                    "(@Output / @Internal / @<state>.derivative) or connect a source signal to it"
                )
            elif n_definitions > 1:
                errors.append(
                    f"signal {label} of type {kind} has multiple definitions ({n_definitions}): exactly one equation "
                    "or connection is allowed per signal"
                )

        elif isinstance(signal, Input) and len(incoming[signal.id]) > 1:
            errors.append(
                f"input signal {label} has {len(incoming[signal.id])} incoming connections "
                f"[{', '.join(incoming[signal.id])};]; at most one is allowed"
            )

        if isinstance(signal, StateDerivative):
            state = signal.state

            if state is None:
                errors.append(
                    f"state derivative signal {label} is not bound to a state; declare it with `@<state>.derivative` "
                    "or `StateDerivative(of=state)`"
                )

            elif state.id not in path_by_id:
                errors.append(
                    f"state derivative signal {label} is bound to a state {state.name} that is not part of the system"
                )

            else:
                derivative_by_state[state.id].append(label)

    for signal, path in signals:
        if isinstance(signal, State):
            label, derivatives = _format_path(path), derivative_by_state[signal.id]

            if not derivatives:
                errors.append(
                    f"state signal {label} has no state derivative; declare it with `@{signal.name}.derivative`"
                )

            elif len(derivatives) > 1:
                errors.append(
                    f"state signal {label} has {len(derivatives)} state derivatives definitions: "
                    f"[{', '.join(derivatives)}]; exactly one is allowed"
                )

    if errors:
        raise ValueError("invalid system definition:\n  - " + "\n  - ".join(errors))


def _compile_equation(
    equation: Equation,
    signal: _ComputedSignal[Any],
    path: tuple[str, ...],
) -> EquationNode:
    label = _format_path(path)

    try:
        with trace() as inputs:
            sample_result = equation.function()
    except Exception as exception:
        raise RuntimeError(f"failed to trace equation {equation.name} at path {label}") from exception

    def execute(values: Mapping[uuid.UUID, Any]) -> dict[uuid.UUID, Any]:
        # TODO: validation of the result at runtime could be usefull for signal-dependent Python control flow detection
        with evaluate(values):
            return {signal.id: equation.function()}

    return EquationNode(
        id=equation.id,
        path=label,
        inputs={signal.id for signal in inputs},
        outputs={signal.id},  # only one output is expected for a computed signal
        equation=execute,
    )  # TODO: maybe we should create an independent function that trace and retrieve inputs (signle responsibility)


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)
