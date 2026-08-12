import functools
import uuid
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar

from .graph import ComputationGraph, ConstraintNode
from .signals import Signal

P = ParamSpec("P")
R = TypeVar("R")


class CompiledSystem:
    def __init__(
        self,
        graph: ComputationGraph,
        sources: Mapping[uuid.UUID, Any],
        state_pairs: tuple[tuple[uuid.UUID, uuid.UUID], ...],
    ):
        self.graph = graph
        self._sources = dict(sources)
        self._state_pairs = state_pairs

    def __setitem__(self, signal: Signal[Any], value: Any) -> None:
        """
        Allow to set s source signal value like a dictionary, e.g. `compiled_system[system.signal] = value`
        """
        if signal.id not in self._sources:
            signal_nodes = {node.id: node for node in self.graph.signal_nodes}
            available_signals = [signal_nodes[sid].path for sid in self._sources]
            raise KeyError(
                f"signal {signal.name} (id={signal.id.hex[:5]}) is not a configurable source; "
                f"available sources: {available_signals}."
            )

        self._sources[signal.id] = value

    @property
    def state_pairs(self) -> tuple[tuple[uuid.UUID, uuid.UUID], ...]:
        """
        Return the state pairs (state, derivative) in the system.
        """
        return self._state_pairs

    def initial_values(self) -> dict[uuid.UUID, Any]:
        """Return a new source mapping while preserving backend value references."""
        return self._sources.copy()

    def signal_ids(self, stype: type[Signal[Any]]) -> tuple[uuid.UUID, ...]:
        return tuple(node.id for node in self.graph.signal_nodes if node.stype is stype)

    def vary(self, func: Callable[P, R]) -> Callable[P, R]:
        """Restore whole-value source assignments after each invocation of ``func``."""

        @functools.wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            values = self._sources.copy()
            try:
                return func(*args, **kwargs)
            finally:
                self._sources = values

        return wrapped

    def evaluate(self, sources: Mapping[uuid.UUID, Any]) -> Mapping[uuid.UUID, Any]:
        """
        Evaluate one graph pass and return values for all signals in the system.
        """
        expected = set(self._sources)
        supplied = set(sources)
        if expected != supplied:
            missing = expected - supplied
            unexpected = supplied - expected

            signal_nodes = {node.id: node for node in self.graph.signal_nodes}
            missing = [signal_nodes[sid].path for sid in missing]
            unexpected = [signal_nodes[sid].path for sid in unexpected]

            raise ValueError(
                f"supplied sources do not match expected sources; missing: {missing}, unexpected: {unexpected}"
            )

        values = dict(sources)
        for node in self.graph.topological_order:
            if not isinstance(node, ConstraintNode):
                continue

            new_signals = node.fn(values)
            values.update(new_signals)
        return values
