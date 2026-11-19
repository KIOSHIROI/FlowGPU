#!/usr/bin/env python3
"""Structural ablation report for the public compiler.

This is a CPU-only, deterministic companion to the device ablation protocol.  It
does NOT measure latency and must not be reported as one.  It answers a narrower
question: when a mechanism is disabled, which kernels, buffers and modeled
traffic change?

Usage:
    python tools/optimization_ablation.py examples/batch8/dcp_lse_combine.fgpu
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flowgpu.compiler.compile import compile_source  # noqa: E402

VARIANTS = {
    "default": {},
    "no_graph_pipeline": {"optimize": False},
    "no_identity_folding": {"identity_folding": False},
    "rule_fusion": {"fusion_policy": "rule"},
    "no_index_reduction_fusion": {"index_fusion": False, "reduction_fusion": False},
    "layout_views_off": {"layout_views": False},
}


def run(source: str) -> dict:
    report = {}
    for name, options in VARIANTS.items():
        try:
            pipeline = compile_source(source, **options).pipelines[0]
        except Exception as error:  # a refused variant is data, not a crash
            report[name] = {"error": type(error).__name__ + ": " + str(error)}
            continue
        report[name] = {
            "plan": pipeline.planning.as_dict() if pipeline.planning is not None else None,
            "optimization": [
                {"name": record.name, "nodes_before": record.nodes_before,
                 "nodes_after": record.nodes_after,
                 "decisions": len(record.decisions)}
                for record in pipeline.optimization
            ],
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    source = pathlib.Path(args.path).read_text()
    report = run(source)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for name, entry in report.items():
            if "error" in entry:
                print(name, "ERROR", entry["error"])
                continue
            plan = entry["plan"]
            print(f"{name}: kernels={plan['kernels']} slots={plan['slots']} "
                  f"A={plan['traffic_a']} fused={plan['fused_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
