from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from .context import evaluate, trace
from .equations import Equation
from .graph import ComputationGraph, ConnectionNode, EquationNode, SignalNode
from .runtime import CompiledSystem
from .signals import Signal, State, StateDerivative, _OutputSignal

if TYPE_CHECKING:
    from .systems import System


def compile(system: System, *, allow_algebraic_loop: bool) -> CompiledSystem:
    """Compile the system into a graph representation."""
    graph = ComputationGraph()

    # flatten the system composition
    systems, signals, equations, connections = _flatten(system)

    # add signals to the graph
    signal_node_pool: dict[uuid.UUID, SignalNode] = {}
    signal_pool: dict[uuid.UUID, Signal] = {}
    for signal, path in signals:
        signal_node = SignalNode(id=signal.id, path=_format_path(path), stype=type(signal))
        graph.add_node(signal_node)

        signal_node_pool[signal.id] = signal_node
        signal_pool[signal.id] = signal

    # add equations to the graph
    # collect (equation, inputs, outputs) for later state pairs matching
    # TODO: clean up, this collection is ugly
    equation_definition: list[tuple[Equation, tuple[Signal[Any], ...], tuple[Signal[Any], ...]]] = []
    for equation, candidate_outputs, path in equations:
        # TODO: documentation: equations are expected to be pure functions; interdiction:
        # - no mutation of any signal held by self or any other system
        # - no reading of mutable higher scope variable
        # - no control flow based on signal values (except if provide by the framework)
        equation_node, expected_inputs = _compile_equation(equation, candidate_outputs, path)
        graph.add_node(equation_node)

        for signal_id in equation_node.inputs:
            graph.add_edge(signal_node_pool[signal_id], equation_node)

        for signal_id in equation_node.outputs:
            graph.add_edge(equation_node, signal_node_pool[signal_id])

        equation_definition.append((equation, expected_inputs, candidate_outputs))

    # add connections to the graph
    for connection, path in connections:
        connection_node = ConnectionNode(
            id=connection.id,
            path=_format_path(path),
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
        remaining_nodes = {node for node in remaining_nodes if not isinstance(node, ConnectionNode)}

        paths = [node.path for node in remaining_nodes]
        raise ValueError(f"algebraic loop detected in the system {system.name}: {paths}")

    # aggregate default source values from source signals (no predecessor)
    source_values = {
        node.id: signal_pool[node.id].get_value()
        for node in graph.get_unconnected_nodes()
        if isinstance(node, SignalNode)
    }

    # infer state/state derivative pairs
    # this allow a solver to know how to update the state of the system at each time step
    state_pairs = _infer_state_pairs(equation_definition)

    return CompiledSystem(graph=graph, sources=source_values, state_pairs=state_pairs)


def _flatten(system: System, *, path: tuple[str, ...] = ()):
    subsystems, signals, sys_equations = system.components()

    subsystems = {(subsystem, path + (subsystem.name,)) for subsystem in subsystems}
    signals = {(signal, path + (signal.name,)) for signal in signals}

    equations = set()
    for equation in sys_equations:
        # It is possible that a system has several output sources of the same type, but they aren't all connected to the same equation. Hence this `candidate_outputs` isn't necessarily the true output of the equation. In case of multiple outputs of the same type, the equation should return a mapping of output signals to values.
        candidate_outputs = tuple(signal for signal, _ in signals if isinstance(signal, equation.output_type))
        equations.add((equation, candidate_outputs, path + (equation.name,)))

    connections = set()
    for connection in system._connections:
        connection_path = path + (f"{connection.src.name}->{connection.tgt.name}",)
        connections.add((connection, connection_path))

    for subsystem, subsystem_path in subsystems.copy():
        sub_subsystems, sub_signals, sub_equations, sub_connections = _flatten(subsystem, path=subsystem_path)

        subsystems.update(sub_subsystems)
        signals.update(sub_signals)
        equations.update(sub_equations)
        connections.update(sub_connections)

    return subsystems, signals, equations, connections


def _compile_equation(
    equation: Equation,
    candidate_outputs: tuple[_OutputSignal, ...],
    path: tuple[str, ...],
):
    try:
        with trace() as inputs:
            sample_result = equation.function()
        _validate_result(sample_result, candidate_outputs, path)

    except Exception as exception:
        raise RuntimeError(f"failed to trace equation {equation.name} at path {_format_path(path)}") from exception

    expected_inputs = tuple(inputs)

    bind_result = _make_binding(sample_result, candidate_outputs)

    def execute(values: Mapping[uuid.UUID, Any]) -> dict[uuid.UUID, Any]:
        # TODO: validation of the result at runtime could be usefull for signal-dependent Python control flow detection
        with evaluate(values):
            result = equation.function()
        return bind_result(result)

    return (
        EquationNode(
            id=equation.id,
            path=_format_path(path),
            inputs={signal.id for signal in expected_inputs},
            outputs={signal.id for signal in candidate_outputs},
            equation=execute,
        ),
        expected_inputs,
    )  # TODO: maybe we should create an independent function that trace and retrieve inputs


def _validate_result(result: Any, candidate_outputs: tuple[_OutputSignal, ...], path: tuple[str, ...]):
    is_output_mapping = isinstance(result, Mapping) and any(isinstance(key, _OutputSignal) for key in result)
    have_multiple_outputs = len(candidate_outputs) > 1

    if not have_multiple_outputs:
        return  # TODO: better handling of single output?

    if have_multiple_outputs and not is_output_mapping:
        raise TypeError(
            f"system {_format_path(path[:-1])} has multiple outputs of the same type defined, therefore the equation "
            f"{_format_path(path)} must return a mapping of output signals to values, "
            f"but it returned {result!r} instead."
        )

    invalid_keys = [key for key in result if not isinstance(key, _OutputSignal)]
    if have_multiple_outputs and invalid_keys:
        raise TypeError(
            f"equation {_format_path(path)} must return a mapping of output signals (`_OutputSignal`) to values, but "
            f"it returned a mapping with invalid keys: {invalid_keys}"
        )

    have_at_least_one_output = any(key in candidate_outputs for key in result)
    if have_multiple_outputs and not have_at_least_one_output:
        expected = ", ".join(signal.name for signal in candidate_outputs)
        raise ValueError(
            f"equation {_format_path(path)} must return a mapping of at least one output signal to value corresponding "
            "to an output signal of the system if multiple output of the same type are defined in the system: "
            f"expected one of [{expected}]"
        )


def _make_binding(
    result: Any,
    candidate_outputs: tuple[_OutputSignal, ...],
) -> Callable[[Any], dict[uuid.UUID, Any]]:
    is_output_mapping = isinstance(result, Mapping) and any(isinstance(key, _OutputSignal) for key in result)

    if not candidate_outputs and result is None:
        # no outputs and no result
        # TODO: does this mean that the equation is useless? or should it be considered as constraints?
        # if so, also adjust the function signature `outputs: tuple[Signal, ...]` -> `outputs: tuple[Signal]`
        return lambda result: {}

    if len(candidate_outputs) == 1 and not is_output_mapping:
        # single output doesn't need to be a mapping
        return lambda result: {candidate_outputs[0].id: result}

    # multiple outputs must be a mapping keyed by output signals
    return lambda result: {signal.id: value for signal, value in result.items()}


def _infer_state_pairs(
    equation_definitions: list[tuple[Equation, tuple[Signal[Any], ...], tuple[Signal[Any], ...]]],
) -> tuple[tuple[uuid.UUID, uuid.UUID], ...]:
    state_pairs: list[tuple[uuid.UUID, uuid.UUID]] = []

    for definition in equation_definitions:
        equation, expected_inputs, candidate_outputs = definition

        if equation.output_type is not StateDerivative:
            continue

        states = _get_signals_of_type(expected_inputs, State)

        if len(states) != len(candidate_outputs):
            raise ValueError(
                f"expected {len(states)} state derivatives, got {len(candidate_outputs)}. "
                "Check that the system has the same number of state and state derivative signals."
            )

        if len(states) == 1:
            pair = (states[0].id, candidate_outputs[0].id)
            state_pairs.append(pair)
            continue

        derivative_by_state: dict[uuid.UUID, StateDerivative[Any]] = {}

        # because of the continue if output_type is not state derivative we could assume we are handling state
        # derivative
        # TODO: this is ugly and most probably a side effect of using tuple
        candidate_outputs = cast(tuple[StateDerivative[Any], ...], candidate_outputs)

        for derivative in candidate_outputs:
            derivative: StateDerivative[Any]

            state = derivative.state

            if state is None:
                raise NotImplementedError(
                    "inference of state/state derivative pairs is not implemented for systems with multiple"
                    "state derivatives. Please specify the state for each state derivative signal with"
                    "`StateDerivative(of=state)`."
                )

            state_match_any_candidate = any(state is candidate for candidate in states)
            if not state_match_any_candidate:
                raise ValueError(
                    f"state derivative {derivative.name} has no matching state "
                    "signal in the system. Please specify the state for each state derivative signal with "
                    "`StateDerivative(of=state)`."
                )

            derivative_by_state[state.id] = derivative

            if len(derivative_by_state) != len(states):
                raise ValueError(
                    f"expected {len(states)} state derivatives, got {len(derivative_by_state)}. "
                    "Check that the system has the same number of state and state derivative signals."
                )

            pairs = tuple((state.id, derivative_by_state[state.id].id) for state in states)
            state_pairs.extend(pairs)

    return tuple(state_pairs)


def _get_signals_of_type(signals: tuple[Signal[Any], ...], stype: type[Signal[Any]]) -> tuple[Signal[Any], ...]:
    return tuple(signal for signal in signals if isinstance(signal, stype))


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)
