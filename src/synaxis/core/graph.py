from __future__ import annotations

import abc
import dataclasses
import uuid
from collections.abc import Callable, Mapping
from typing import Any

SignalValues = Mapping[uuid.UUID, Any]


@dataclasses.dataclass(frozen=True)
class Node(abc.ABC):
    id: uuid.UUID  # unique identifier shared with the declaration that generated this node
    path: str  # human readable path


@dataclasses.dataclass(frozen=True, eq=False)
class ConstraintNode(Node, abc.ABC):
    """
    A constraint represent an identity operation on signals such as:
        - equation: outputs - f(inputs) = 0
        - connection: target - source = 0
    This allow residual/implicit evaluation which is useful for optimization and solving when handling algebraic loops, physical constraints, differential algebraic equations (DAE), implicit time-stepping for stiff systems, etc.

    As example, consider the following closed-loop system:
        y = a + b*u
        u = -k*y
    which form an algebraic loop that cannot be solved explicitly using a topological order evaluation. Instead, we can
    treat (y, u) as unknowns and solve the following system of equations:
        F(y, u) = (
            y - a - b*u,
            u + k*y
        )
    and search for F(y, u) = 0.
    """

    # TODO: we should later output both the output mapping and the residual, at the moment only the explicit output mapping is returned.
    @abc.abstractmethod
    def fn(self, inputs: SignalValues) -> dict[uuid.UUID, Any]:
        pass


@dataclasses.dataclass(frozen=True, eq=False)
class ConnectionNode(ConstraintNode):
    source: uuid.UUID
    target: uuid.UUID

    def fn(self, inputs: SignalValues) -> dict[uuid.UUID, Any]:
        return {self.target: inputs[self.source]}


@dataclasses.dataclass(frozen=True, eq=False)
class EquationNode(ConstraintNode):
    inputs: set[uuid.UUID]
    outputs: set[uuid.UUID]
    equation: Callable[[SignalValues], dict[uuid.UUID, Any]] = dataclasses.field(repr=False)

    def __post_init__(self):
        """
        Validate the equation with a warm start using dummy values.
        """
        # TODO

    def fn(self, inputs: SignalValues) -> dict[uuid.UUID, Any]:
        return self.equation(inputs)


@dataclasses.dataclass(frozen=True)
class SignalNode(Node):
    stype: type


class ComputationGraph:
    """
    An adjacency map representation of a bipartite computation graph.
    It is a directed graph where nodes are either signals or constraints (equations and connections), and edges
    represent the flow of data between signals and constraints.
    """

    def __init__(self) -> None:
        self.successors: dict[Node, tuple[Node, ...]] = {}

        # cache for the topological sorting
        self._topological_order: tuple[Node, ...] | None = None

    @property
    def predecessors(self) -> dict[Node, tuple[Node, ...]]:
        """
        Compute the predecessors of each node in the graph.
        This is the inverse of the successors mapping.
        """
        predecessors: dict[Node, list[Node]] = {node: [] for node in self.successors}
        for source, targets in self.successors.items():
            for target in targets:
                predecessors[target].append(source)
        return {node: tuple(sources) for node, sources in predecessors.items()}

    @property
    def signal_nodes(self) -> tuple[SignalNode, ...]:
        """
        Return all signal nodes in the graph.
        """
        return tuple(node for node in self.successors if isinstance(node, SignalNode))

    @property
    def topological_order(self) -> tuple[Node, ...]:
        """
        Return a topological order of the nodes in the graph using Kahn's algorithm.
        """
        # if the topological order is already computed, return it
        if self._topological_order is not None:
            return self._topological_order

        indegrees = {node: len(predecessors) for node, predecessors in self.predecessors.items()}

        topological_oder: list[Node] = []
        queue: list[Node] = list(self.get_unconnected_nodes())

        while queue:
            node = queue.pop(0)
            topological_oder.append(node)

            for neighbor in self.successors[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)

        self._topological_order = tuple(topological_oder)
        return self._topological_order

    def add_node(self, node: Node) -> None:
        if node in self.successors:
            return  # TODO: raise an error/warning if the node already exists

        self.successors[node] = ()  # Initialize with no successors

        # Invalidate the cached topological order since the graph has changed
        self._topological_order = None

    def add_edge(self, source: Node, target: Node) -> None:
        break_bipartite = isinstance(source, SignalNode) == isinstance(target, SignalNode)
        if break_bipartite:
            raise ValueError(f"invalid edge from {source} to {target}: edges must connect a signal and a constraint")

        self.add_node(source)
        self.add_node(target)

        if target in self.successors[source]:
            return  # TODO: raise an error/warning if the edge already exists

        self.successors[source] += (target,)

        # Invalidate the cached topological order since the graph has changed
        self._topological_order = None

    def get_unconnected_nodes(self) -> tuple[Node, ...]:
        """
        Return a list of nodes without predecessors (without incomming edges).
        """
        return tuple([node for node in self.successors if not self.predecessors[node]])

    def has_algebraic_loop(self) -> bool:
        """
        Check if the graph has an algebraic loop (i.e., a cycle).
        """
        return len(self.topological_order) != len(self.successors)
