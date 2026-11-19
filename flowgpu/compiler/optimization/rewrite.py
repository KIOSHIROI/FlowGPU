"""A small, verifiable IR pass framework.

The compiler historically called optimization functions directly from
``compile.py``.  This module gives those functions one contract:

* a pass consumes and returns a ``DataflowGraph``;
* the graph is verified before and after every pass;
* a pass records human-readable decisions so a report can explain what changed;
* a manager runs passes to a small fixpoint (bounded, so a buggy rule cannot
  loop forever).

The framework intentionally does not provide a general e-graph or an
untrusted rewrite DSL.  Soundness comes from each rule's explicit guard; this
module only makes composition and verification uniform.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from flowgpu.compiler.core.ir import DataflowGraph
from flowgpu.compiler.ir.verification import verify_graph


@dataclass(frozen=True)
class Decision:
    """One human-readable transformation decision."""
    rule: str
    reason: str


class PassFunction(Protocol):
    def __call__(self, graph: DataflowGraph) -> tuple[DataflowGraph, tuple[Decision, ...]]:
        ...


@dataclass(frozen=True)
class Pass:
    name: str
    function: PassFunction


@dataclass(frozen=True)
class PassRecord:
    name: str
    nodes_before: int
    nodes_after: int
    decisions: tuple[Decision, ...] = ()

    @property
    def changed(self) -> bool:
        return self.nodes_before != self.nodes_after or bool(self.decisions)


@dataclass(frozen=True)
class PassResult:
    graph: DataflowGraph
    records: tuple[PassRecord, ...] = ()

    @property
    def changed(self) -> bool:
        return any(record.changed for record in self.records)


class PassManager:
    """Run passes in order, to a bounded fixpoint, verifying every step."""

    def __init__(self, passes: tuple[Pass, ...], *, rounds: int = 3) -> None:
        if type(rounds) is not int or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        self.passes = tuple(passes)
        self.rounds = rounds

    def run(self, graph: DataflowGraph) -> PassResult:
        verify_graph(graph)
        records: list[PassRecord] = []
        current = graph
        for _ in range(self.rounds):
            changed = False
            for pass_ in self.passes:
                before = len(current.nodes)
                verify_graph(current)
                current, decisions = pass_.function(current)
                verify_graph(current)
                record = PassRecord(pass_.name, before, len(current.nodes),
                                    tuple(decisions))
                records.append(record)
                changed = changed or record.changed
            if not changed:
                break
        return PassResult(current, tuple(records))


def copy_graph(graph: DataflowGraph) -> DataflowGraph:
    """A mutable copy with the same SSA/value identity."""
    return DataflowGraph(
        pipeline=graph.pipeline,
        values=dict(graph.values),
        nodes=[
            type(node)(node.op, list(node.inputs), node.output, node.part,
                       node.attributes)
            for node in graph.nodes
        ],
        aliases=dict(graph.aliases),
    )


def rewrite_inputs(graph: DataflowGraph, replacements: dict[str, str]) -> DataflowGraph:
    """Replace operand names using a canonical resolution map."""
    replacements = dict(replacements)
    def resolve(name: str) -> str:
        seen = set()
        while name in replacements:
            if name in seen:
                raise ValueError(f"replacement cycle at {name!r}")
            seen.add(name)
            name = replacements[name]
        return name

    aliases = dict(graph.aliases)
    for source, target in replacements.items():
        aliases[source] = resolve(target)
    nodes = []
    for node in graph.nodes:
        nodes.append(type(node)(node.op, [resolve(value) for value in node.inputs],
                                node.output, node.part, node.attributes))
    return DataflowGraph(graph.pipeline, dict(graph.values), nodes, aliases)


__all__ = [
    "Decision", "Pass", "PassManager", "PassRecord", "PassResult",
    "copy_graph", "rewrite_inputs",
]
