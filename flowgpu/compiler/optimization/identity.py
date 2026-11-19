"""Safe identity and view-canonicalisation rewrites.

The dialect has no tensor literal node, so classical constant folding has a
small surface here.  What *is* safe and useful is eliminating identities that
arise from layout lowering and type wiring:

* ``reshape(x, shape(x))`` -> ``x``;
* ``reshape(reshape(x, s), shape(x))`` -> ``x``;
* a transpose/move-first-last pair whose permutations compose to identity;
* ``cast_to(x, dtype(x))`` -> ``x``.

Each rule is guarded by the graph's actual value types.  A replacement is only
made when the source and target have the same shape and dtype.  The pass is
therefore a canonicalisation, not a speculative algebraic rewrite requiring an
SMT proof.
"""
from __future__ import annotations

from flowgpu.compiler.core.ir import DataflowGraph, Node
from flowgpu.compiler.ir.effects import Purity, effect_of
from flowgpu.compiler.optimization.rewrite import Decision, rewrite_inputs


def _exported(graph: DataflowGraph, value: str) -> bool:
    """An exported value is an observable storage boundary in this dialect."""
    return any(node.op == "output" and value in node.inputs for node in graph.nodes)


def _same_type(graph: DataflowGraph, one: str, other: str) -> bool:
    left, right = graph.values[one], graph.values[other]
    return (left.dtype == right.dtype
            and tuple(str(d) for d in left.shape) == tuple(str(d) for d in right.shape))


def _inverse(permutation: tuple) -> tuple:
    return tuple(permutation.index(index) for index in range(len(permutation)))


def _replace(graph: DataflowGraph, rule: str, replacements: dict[str, str],
             reasons: list[str]) -> tuple[DataflowGraph, tuple[Decision, ...]]:
    for value, target in replacements.items():
        if value not in graph.values or target not in graph.values:
            raise ValueError(f"{rule} replacement crosses the value table")
        if not _same_type(graph, value, target):
            raise ValueError(f"{rule} replacement changes the value type")
    removed_values = set(replacements)

    def resolve(value: str) -> str:
        seen = set()
        while value in replacements:
            if value in seen:
                raise ValueError(f"identity replacement cycle at {value!r}")
            seen.add(value)
            value = replacements[value]
        return value

    nodes = [node for node in graph.nodes if node.output not in removed_values]
    nodes = [
        type(node)(node.op, [resolve(value) for value in node.inputs],
                   node.output, node.part, node.attributes)
        for node in nodes
    ]
    aliases = dict(graph.aliases)
    aliases.update({value: resolve(target) for value, target in replacements.items()})
    rewritten = DataflowGraph(graph.pipeline, dict(graph.values), nodes, aliases)
    decisions = tuple(Decision(rule, reason) for reason in reasons)
    return rewritten, decisions


def fold_identities(graph: DataflowGraph) -> tuple[DataflowGraph, tuple[Decision, ...]]:
    """Apply guarded identity/view rules to a fixed point."""
    current = graph
    all_decisions: list[Decision] = []
    for _ in range(4):
        effects = {id(node): effect_of(node) for node in current.nodes}
        replacements: dict[str, str] = {}
        reasons: list[str] = []
        for node in current.nodes:
            if node.op in ("input", "output"):
                continue
            effect = effects[id(node)]
            if effect.purity is not Purity.PURE or node.output is None:
                continue
            if _exported(current, node.output):
                # Keep an exported identity copy: tests and the buffer contract
                # treat it as an observable storage boundary, not a free view.
                continue
            attrs = node.attributes
            if node.op == "reshape":
                target = tuple(str(d) for d in getattr(attrs, "target_shape", ()))
                source = current.values[node.inputs[0]]
                if target == tuple(str(d) for d in source.shape):
                    replacements[node.output] = node.inputs[0]
                    reasons.append(f"reshape {node.output} is the identity")
                    continue
            if node.op == "cast_to":
                if _same_type(current, node.inputs[0], node.output):
                    replacements[node.output] = node.inputs[0]
                    reasons.append(f"cast_to {node.output} preserves dtype/shape")
                    continue
            if node.op in ("transpose", "move_first_last"):
                producer = current.producer_of(node.inputs[0])
                if (producer is not None and producer.op in ("transpose", "move_first_last")
                        and attrs is not None and producer.output is not None):
                    outer = tuple(getattr(attrs, "permutation", ()))
                    inner = tuple(getattr(producer.attributes, "permutation", ()))
                    if outer and inner and outer == _inverse(inner):
                        replacements[node.output] = producer.inputs[0]
                        reasons.append(
                            f"{node.op} {node.output} cancels {producer.op} {producer.output}")
        if not replacements:
            break
        current, decisions = _replace(current, "identity", replacements, reasons)
        all_decisions.extend(decisions)
    return current, tuple(all_decisions)


__all__ = ["fold_identities"]
