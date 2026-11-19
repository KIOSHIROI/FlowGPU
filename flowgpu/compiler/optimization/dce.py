"""Dead-code elimination for pure SSA producers.

The dataflow graph is SSA, so a pure node whose result reaches no ``output``
and no live consumer is dead.  Structural inputs are always retained (they are
the program's ABI), and stateful cache operators are retained even when their
SSA result is unused because their runtime write still happens.
"""
from __future__ import annotations

from flowgpu.compiler.core.ir import INPUT, OUTPUT, DataflowGraph
from flowgpu.compiler.ir.effects import Purity, effect_of
from flowgpu.compiler.optimization.rewrite import Decision


def eliminate_dead_code(graph: DataflowGraph) -> tuple[DataflowGraph, tuple[Decision, ...]]:
    effects = [effect_of(node) for node in graph.nodes]

    # Start from the observable boundary and walk backwards.
    live_values: set[str] = set()
    live_indices: set[int] = set()
    for index, (node, effect) in enumerate(zip(graph.nodes, effects)):
        if node.op == OUTPUT:
            live_indices.add(index)
            live_values.update(node.inputs)
        elif effect.purity is Purity.STATEFUL:
            # A stateful write is observable even when its result is unused.
            live_indices.add(index)
            live_values.update(node.inputs)

    for index in range(len(graph.nodes) - 1, -1, -1):
        node, effect = graph.nodes[index], effects[index]
        if node.op in (INPUT, OUTPUT):
            if node.op == INPUT:
                live_indices.add(index)
            continue
        if index in live_indices or (node.output is not None and node.output in live_values):
            live_indices.add(index)
            live_values.update(node.inputs)

    # A multi-result operator is one computation represented by one node per
    # result.  Its lowering expects the siblings to stay together, and the
    # lowerer already records a fully dead group as ``skipped`` rather than
    # emitting it.  Keep all part nodes here: removing individual parts would
    # either break the sibling lowering or erase the skipped-group audit that
    # the P0 review fixed.  Single-result pure nodes are the DCE target.
    for index, node in enumerate(graph.nodes):
        if node.part is not None:
            live_indices.add(index)
            live_values.update(node.inputs)

    removed = [node for index, node in enumerate(graph.nodes)
               if index not in live_indices]
    if not removed:
        return graph, ()
    removed_outputs = {node.output for node in removed if node.output is not None}
    nodes = [node for index, node in enumerate(graph.nodes) if index in live_indices]

    kept_values = {
        value for node in nodes if node.output is not None
        for value in (node.output,)
    }
    # Every retained node may read an input or a value produced above; the
    # latter is already in kept_values.  User-facing aliases survive only when
    # their target survived too.
    for node in nodes:
        kept_values.update(node.inputs)
    values = {name: tensor for name, tensor in graph.values.items()
              if name in kept_values}
    aliases = {
        alias: target for alias, target in graph.aliases.items()
        if alias in values and target in values
    }
    decisions = tuple(
        Decision("dce", f"remove dead {node.op} node {node.output}")
        for node in removed if node.output is not None
    )
    return DataflowGraph(graph.pipeline, values, nodes, aliases), decisions


__all__ = ["eliminate_dead_code"]
