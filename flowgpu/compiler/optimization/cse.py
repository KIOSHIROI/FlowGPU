"""Graph-level common-subexpression elimination for pure dataflow nodes.

This is deliberately separate from ``backend.cse``: that module hash-conses
expressions *inside one emitted kernel*, after fusion has already decided the
partition.  This pass runs before fusion and removes duplicate SSA producers
that would otherwise become duplicate work or duplicate buffers.

Only ``Purity.PURE`` nodes may be merged.  Input/output nodes are structural,
and stateful cache operators are excluded because merging them would duplicate
or erase a runtime write.  The structural key includes the result type and the
static attributes, so two same-named operators over the same operands are not
merged if their shapes/dtypes or attributes differ.
"""
from __future__ import annotations

from flowgpu.compiler.core.ir import INPUT, OUTPUT, DataflowGraph, Node
from flowgpu.compiler.ir.effects import Purity, effect_of
from flowgpu.compiler.optimization.rewrite import Decision, rewrite_inputs

#: Floating-point `add`/`mul` are mathematically commutative but not a safe
#: default for hash-consing: NaN payload/sign and future non-IEEE backends make
#: the emitted bytes observable.  CSE therefore requires the exact operand
#: order.  A future algebraic pass may canonicalise integer-only commutative
#: nodes after an equivalence check.


def _type_key(graph: DataflowGraph, value: str) -> tuple:
    tensor = graph.values[value]
    return (tensor.dtype, tuple(str(d) for d in tensor.shape))


def _key(graph: DataflowGraph, node: Node) -> tuple:
    return (node.op, tuple(node.inputs), node.part, repr(node.attributes),
            _type_key(graph, node.output))


def common_subexpression_elimination(graph: DataflowGraph) -> tuple[DataflowGraph, tuple[Decision, ...]]:
    """Merge structurally identical pure nodes, preserving one producer."""
    seen: dict[tuple, str] = {}
    replacements: dict[str, str] = {}
    decisions: list[Decision] = []
    kept: list[Node] = []
    for node in graph.nodes:
        effect = effect_of(node)
        if (node.op not in (INPUT, OUTPUT) and node.part is None
                and effect.purity is Purity.PURE):
            key = _key(graph, node)
            previous = seen.get(key)
            if previous is not None:
                replacements[node.output] = previous
                decisions.append(Decision(
                    "cse",
                    f"node {node.output} duplicates {previous} ({node.op}); "
                    "consumers point at the first pure producer",
                ))
                continue
            seen[key] = node.output
        kept.append(node)
    if not replacements:
        return graph, ()
    # Resolve every use before constructing the graph: DataflowGraph.validate
    # rejects a node that reads a noncanonical alias, so the nodes and the
    # alias table must be built together.
    def resolve(value: str) -> str:
        seen = set()
        while value in replacements:
            if value in seen:
                raise ValueError(f"CSE replacement cycle at {value!r}")
            seen.add(value)
            value = replacements[value]
        return value

    nodes = [
        type(node)(node.op, [resolve(value) for value in node.inputs],
                   node.output, node.part, node.attributes)
        for node in kept
    ]
    aliases = dict(graph.aliases)
    aliases.update({value: resolve(target) for value, target in replacements.items()})
    return DataflowGraph(graph.pipeline, dict(graph.values), nodes, aliases), tuple(decisions)


__all__ = ["common_subexpression_elimination"]
