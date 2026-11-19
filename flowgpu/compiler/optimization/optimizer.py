"""The default graph-optimisation pipeline for the compiler core.

The passes here are intentionally conservative and independently verifiable:

1. identity folding removes only same-shape/same-dtype view/type identities;
2. CSE merges only pure, structurally identical nodes;
3. DCE removes only pure nodes with no observable result.

Stateful operators are never removed, duplicated or commoned.  The pipeline is
run through ``PassManager`` so every step is verified and recorded.
"""
from __future__ import annotations

from flowgpu.compiler.core.ir import DataflowGraph
from flowgpu.compiler.optimization.cse import common_subexpression_elimination
from flowgpu.compiler.optimization.dce import eliminate_dead_code
from flowgpu.compiler.optimization.identity import fold_identities
from flowgpu.compiler.optimization.rewrite import Pass, PassManager, PassResult

DEFAULT_PASSES = (
    Pass("identity", fold_identities),
    Pass("cse", common_subexpression_elimination),
    Pass("dce", eliminate_dead_code),
)

#: Identity/view folding is on by default for non-exported values. Exported
#: identity copies are guarded out in the pass because they are observable
#: storage boundaries in this dialect.
IDENTITY_PASS = Pass("identity", fold_identities)


def optimize_graph(graph: DataflowGraph, *, enabled: bool = True,
                   identity_folding: bool = True,
                   rounds: int = 3) -> PassResult:
    """Run the conservative graph pipeline and return its records."""
    if not enabled:
        return PassResult(graph, ())
    passes = DEFAULT_PASSES
    if not identity_folding:
        passes = tuple(pass_ for pass_ in DEFAULT_PASSES if pass_.name != "identity")
    return PassManager(passes, rounds=rounds).run(graph)


__all__ = ["DEFAULT_PASSES", "optimize_graph"]
