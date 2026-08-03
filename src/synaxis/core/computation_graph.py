import dataclasses
import enum
import uuid
from collections.abc import Mapping
from typing import Any, Callable

from .signals import SignalKind


class OperationKind(enum.Enum):
    EQUATION = enum.auto()
    CONNECTION = enum.auto()


@dataclasses.dataclass(frozen=True)
class Node:
    id: uuid.UUID  # the id of the frontend defined signal (to later index results)
    path: str  # human readable identifier for the signal


@dataclasses.dataclass(frozen=True)
class SignalNode(Node):
    kind: SignalKind


@dataclasses.dataclass(frozen=True)
class OperationNode(Node):
    # representing an operation as an identity operation on signals allow mathematical constraints such as:
    # equation:     output - fn(inputs) = 0
    # connection:   traget - source = 0
    # this allows to use a residual or implicit evaluation which allow handling algebraic loops, physical constraints,
    # differential algebraic equations, implicit time-stepping methods for stiff systems, simultaneous solving coupled
    # sybsystems variables, etc.
    # e.g. for
    # y = a + b*u
    # u = -k*y
    # form an algebraic loop which cannot be evaluated in topological order. Instead, we can treat (y, u) as unknowns
    # and solve the system of equations:
    # F(y, u) = (
    #    y - (a + b*u),
    #    u + k*y
    # )
    # searching for F(y, u) = 0
    kind: OperationKind
    fn: Callable[[Mapping[uuid.UUID, Any]], dict[uuid.UUID, Any]]


class BipartiteComputationGraph:
    """
    A dual adjacency list representation of a bipartite graph
    """

    def __init__(self) -> None:
        self.successors: dict[Node, tuple[Node, ...]] = {}
        self.predecessors: dict[Node, tuple[Node, ...]] = {}

        # cache for topological sorting
        self._topological_order = None

    def add_node(self, node: Node):
        # node already exist
        if node in self.successors:
            return
        # TODO: raise warning if node already exist

        self.successors[node] = ()
        self.predecessors[node] = ()

        # adding a node change topological order, so we invalidate the cache
        self._topological_order = None

    def add_edge(self, source: Node, target: Node):
        is_valid_edge = (
            isinstance(source, SignalNode)
            and isinstance(target, OperationNode)
            or isinstance(source, OperationNode)
            and isinstance(target, SignalNode)
        )

        if not is_valid_edge:
            raise ValueError(
                f"Invalid edge from {source} to {target}. "
                "Edges must connect a SignalNode to an OperationNode or vice versa."
            )

        self.add_node(source)
        self.add_node(target)

        # edge already exist
        if target in self.successors[source]:
            return
        # TODO: raise warning if edge already exist

        self.successors[source] += (target,)
        self.predecessors[target] += (source,)

        # adding an edge change topological order, so we invalidate the cache
        self._topological_order = None

    def topological_order(self):
        """Kahn's algorithm for topological sorting"""
        if self._topological_order is not None:
            return self._topological_order

        # copy the input degrees to avoid modifying the original graph
        input_degrees = {node: len(predecessors) for node, predecessors in self.predecessors.items()}

        topological_order: list[Node] = []
        queue = self.get_unconnected_nodes()

        while queue:
            node = queue.pop(0)
            topological_order.append(node)

            for neighbor in self.successors[node]:
                input_degrees[neighbor] -= 1
                if input_degrees[neighbor] == 0:
                    queue.append(neighbor)

        self._topological_order = topological_order
        return topological_order

    def has_algebraic_loop(self) -> bool:
        """Check if the graph has an algebraic loop (i.e., a cycle)."""
        return len(self.topological_order()) != len(self.successors)

    def get_unconnected_nodes(self) -> list[Node]:
        """
        Return a list of nodes without predecessors (without incomming edges).
        """
        return [node for node in self.successors if not self.predecessors[node]]
