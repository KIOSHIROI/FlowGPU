"""Verification entry points shared by passes and the compile pipeline."""
from __future__ import annotations

from flowgpu.compiler.core.ir import DataflowGraph
from flowgpu.compiler.ir.effects import verify_effects


def verify_graph(graph: DataflowGraph) -> None:
    """Run the structural and effect invariants a pass may rely on."""
    graph.validate()
    verify_effects(graph)


__all__ = ["verify_graph"]
