from .core.computation_graph import (
    BipartiteComputationGraph,
    Node,
    OperationKind,
    OperationNode,
    SignalNode,
)
from .core.signals import SignalKind


def to_dot(graph: BipartiteComputationGraph, explicit_connection: bool = True) -> str:
    """Return a Graphviz DOT representation of a computation graph."""
    connection_nodes = {
        node for node in graph.successors if isinstance(node, OperationNode) and node.kind == OperationKind.CONNECTION
    }
    node_ids: dict[Node, str] = {
        node: f"node_{index}"
        for index, node in enumerate(graph.successors)
        if explicit_connection or node not in connection_nodes
    }
    lines = [
        "digraph computation_graph {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica"];',
    ]

    for node, node_id in node_ids.items():
        label = node.path
        shape = "box"
        color = "#ffffff"
        style = "filled"

        if isinstance(node, SignalNode):
            label = f"({node.kind.name.lower()})\n{node.path}"
            shape = "ellipse"
            style = "filled"
            color = "#ffffff"
            if node.kind == SignalKind.INPUT:
                color = "#d9ead3"
            elif node.kind == SignalKind.OUTPUT:
                color = "#cfe2f3"
            elif node.kind == SignalKind.STATE:
                color = "#fff2cc"
            elif node.kind == SignalKind.STATE_DERIVATIVE:
                shape = "box"
                style = style + ", rounded"
            elif node.kind == SignalKind.PARAM:
                color = "#e6e6e6"

        lines.append(
            f"  {node_id} [label={_quote(label)}, shape={shape}, style={_quote(style)}, fillcolor={_quote(color)}];"
        )

    for source, successors in graph.successors.items():
        for target in successors:
            if not explicit_connection and (source in connection_nodes or target in connection_nodes):
                continue
            lines.append(f"  {node_ids[source]} -> {node_ids[target]};")

    if not explicit_connection:
        for connection in connection_nodes:
            for source in graph.predecessors[connection]:
                for target in graph.successors[connection]:
                    lines.append(f"  {node_ids[source]} -> {node_ids[target]} [style=dotted];")

    lines.append("}")
    return "\n".join(lines)


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'
