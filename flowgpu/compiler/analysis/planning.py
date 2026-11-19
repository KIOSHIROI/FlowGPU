"""A single, explainable cost summary of the chosen plan.

The compiler already has a traffic model, a liveness plan and a fusion
partition.  This module does not invent a second latency model; it collects the
structural quantities those decisions expose so a caller can compare plans and
see why a choice was made:

* kernel launches (fusion groups);
* values eliminated by fusion;
* buffers that need storage and compiler-allocated slots;
* layout nodes that remain materialised;
* modeled traffic ``A`` from :mod:`flowgpu.compiler.analysis.traffic`.

No single scalar is claimed to be latency: launch count and traffic dominate
different regimes, and the project's own measurements show that neither replaces
device timing.
"""
from __future__ import annotations

from dataclasses import dataclass

from flowgpu.compiler.analysis.traffic import plan_traffic
from flowgpu.compiler.backend.backends import LAYOUT_OPS
from flowgpu.compiler.core.ir import DataflowGraph
from flowgpu.compiler.optimization.fusion import FusionReport


@dataclass(frozen=True)
class PlanSummary:
    kernels: int
    launches: int
    fused_values: int
    materialized_buffers: int
    allocated_buffers: int
    slots: int
    layout_nodes: int
    traffic_a: float

    def as_dict(self) -> dict:
        return {
            "kernels": self.kernels,
            "launches": self.launches,
            "fused_values": self.fused_values,
            "materialized_buffers": self.materialized_buffers,
            "allocated_buffers": self.allocated_buffers,
            "slots": self.slots,
            "layout_nodes": self.layout_nodes,
            "traffic_a": round(self.traffic_a, 6),
        }


def summarize_plan(graph: DataflowGraph, fusion: FusionReport, memory) -> PlanSummary:
    """Collect structural plan costs without claiming a latency prediction."""
    traffic = plan_traffic(graph, fusion, memory)
    layout_nodes = sum(
        1 for node in graph.nodes if node.op in LAYOUT_OPS
    )
    return PlanSummary(
        kernels=len(fusion.groups),
        launches=len(fusion.groups),
        fused_values=len(fusion.fused_values),
        materialized_buffers=len(memory.materialized),
        allocated_buffers=len(memory.allocated),
        slots=len(memory.slots),
        layout_nodes=layout_nodes,
        traffic_a=float(traffic.a),
    )


__all__ = ["PlanSummary", "summarize_plan"]
