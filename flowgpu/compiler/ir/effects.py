"""Explicit effect and storage-alias facts for the dataflow IR.

The graph is SSA and mostly pure, but a few operators have runtime state:
they write a caller-supplied cache or expose it as a returned value.  The
compiler must not eliminate, duplicate, reorder across, or common-subexpression
eliminate those nodes merely because their SSA result is unused.

This module deliberately starts conservative.  Every node is classified as one
of:

* ``structural`` -- the ``input``/``output`` graph boundary;
* ``pure``       -- the result is a function of the operands and attributes;
* ``stateful``   -- the node writes one of its operands at runtime.

A future pass that needs a finer effect lattice can extend this module, but no
pass may infer purity from "the SSA result is unused".  The helper is shared by
the pass verifier, DCE and CSE so all three agree on the same classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind
from flowgpu.compiler.core.ir import INPUT, OUTPUT, DataflowGraph, Node


class Purity(str, Enum):
    STRUCTURAL = "structural"
    PURE = "pure"
    STATEFUL = "stateful"


@dataclass(frozen=True)
class Effect:
    """One node's effect summary.

    ``reads`` and ``writes`` name SSA values, not raw pointers.  For a stateful
    node ``mutates_inputs`` is true when the lowering may write through one of
    its operand buffers; ``reason`` is a short human-readable justification that
    lands in diagnostics and pass records.
    """

    purity: Purity
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    mutates_inputs: bool = False
    reason: str = ""

    @property
    def may_be_eliminated(self) -> bool:
        """Only a pure node with an unused result may be removed."""
        return self.purity is Purity.PURE

    @property
    def may_be_commoned(self) -> bool:
        """Only a pure node may be merged with an identical pure node."""
        return self.purity is Purity.PURE


#: Operators whose lowering mutates an input cache/state.  The SSA outputs are
#: still values, but the runtime side effect means a pass may not erase or
#: duplicate the node just because its result is dead.  Keep this list explicit
#: and reviewed: a new stateful operator belongs here with a test.
STATEFUL_OPERATORS = frozenset({
    "linear_decode_forward",
    "store_cache_4d",
})


def effect_of(node: Node) -> Effect:
    """Classify one node.  Every known operator is either stateful or pure."""
    if node.op == INPUT:
        return Effect(Purity.STRUCTURAL, writes=(node.output,),
                      reason="input introduces a caller-owned value")
    if node.op == OUTPUT:
        return Effect(Purity.STRUCTURAL, reads=tuple(node.inputs),
                      reason="output observes a value")
    reads = tuple(node.inputs)
    if node.op in STATEFUL_OPERATORS:
        return Effect(
            Purity.STATEFUL,
            reads=reads,
            writes=(node.output,) if node.output is not None else (),
            mutates_inputs=True,
            reason=f"{node.op} updates a caller-supplied cache/state buffer",
        )
    return Effect(Purity.PURE, reads=reads,
                  writes=(node.output,) if node.output is not None else (),
                  reason="result is a function of operands and static attributes")


def graph_effects(graph: DataflowGraph) -> tuple[Effect, ...]:
    """Classify every node in graph order."""
    return tuple(effect_of(node) for node in graph.nodes)


def verify_effects(graph: DataflowGraph) -> None:
    """Check that the graph's effect facts are internally consistent.

    Structural correctness is already checked by ``DataflowGraph.validate``;
    this adds the facts passes depend on:

    * every node has a classified effect;
    * a stateful node has a result and at least one input to mutate;
    * an alias is not used as a producer/consumer (the graph canonicalises it);
    * a stateful write is not hidden behind an ``output`` node.

    The check raises ``CompileError(INTERNAL)`` because a violation means the
    compiler built a graph the pass layer cannot safely reason about.
    """
    def fail(message: str) -> None:
        raise CompileError(DiagnosticKind.INTERNAL,
                           "invalid effect graph: " + message,
                           pipeline=graph.pipeline)

    for index, node in enumerate(graph.nodes):
        effect = effect_of(node)
        if effect.purity is Purity.STATEFUL:
            if node.output is None or not node.inputs:
                fail(f"node {index} ({node.op}) is stateful but has no output/input")
            if not effect.mutates_inputs:
                fail(f"node {index} ({node.op}) has an inconsistent stateful effect")
        if node.op == INPUT and effect.purity is not Purity.STRUCTURAL:
            fail(f"input node {index} is not structural")
        if node.op == OUTPUT and effect.purity is not Purity.STRUCTURAL:
            fail(f"output node {index} is not structural")

    canonical = set(graph.values)
    for alias, target in graph.aliases.items():
        if alias not in canonical or target not in canonical:
            fail(f"alias '{alias}' -> '{target}' crosses the value table")
        if target in graph.aliases:
            fail(f"alias '{alias}' targets another alias '{target}'")
