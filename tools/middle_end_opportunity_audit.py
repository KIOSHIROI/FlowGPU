"""Middle-end opportunity audit: where does IR optimisation change kernel/launch counts?

Compiles a small representative corpus under the default configuration and
several ablations (no index fusion, no reduction fusion, no semantic view
folding, rule-only fusion), then reports the differences in:

* kernel/launch count;
* fusion groups and fused values;
* materialised buffers / allocated slots;
* layout operators and refused kernels.

The output is the evidence for choosing the next middle-end project: fusion,
layout/materialisation, reduction decomposition or buffer planning.  This is a
CPU audit (no device execution); it measures *structural* opportunity, which is
the right first filter before spending GPU time.

    python tools/middle_end_opportunity_audit.py --out reports/middle_end/audit.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgpu.compiler import compile_source  # noqa: E402

PROGRAMS = {
    "pointwise_chain": "pipeline p { input x:fp32[1024] y=relu(x) z=add(y,x) out=mul(z,x) output out }",
    "pointwise_split": "pipeline p { input x:fp32[32,64] a,b=split(x) out=add(a,b) output out }",
    "layout_reshape_relu": "pipeline p { input x:fp32[32,64] y=reshape(x,[64,32]) out=relu(y) output out }",
    "layout_concat": "pipeline p { input a:fp32[32,64] input b:fp32[32,64] c=concat(a,b) out=relu(c) output out }",
    "reduction_softmax": "pipeline p { input x:fp32[8,128] m=max_axis(x,1) mb=broadcast_like(m,x) e=exp(sub(x,mb)) s=sum_axis(e,1) sb=broadcast_like(s,e) out=mul(sb,e) output out }",
    "scan_prefix": "pipeline p { input x:fp32[8,128] t=sum_axis(x,1) c=cumsum_axis(x,1) out=add(c,broadcast_like(t,x)) output out }",
}

ABLATIONS = {
    "default": {},
    "no_index_fusion": {"index_fusion": False},
    "no_reduction_fusion": {"reduction_fusion": False},
    "no_view_folding": {"semantic_view_folding": False},
    "rule_fusion": {"fusion_policy": "rule"},
    "split_views": {"split_views": True},
    "layout_views": {"layout_views": True},
}

LAYOUT_OPS = {"reshape", "split", "concat", "interleave", "transpose", "move_first_last"}


def _metrics(result) -> dict:
    pipeline = result.pipelines[0]
    fusion = pipeline.fusion
    memory = pipeline.memory
    triton = pipeline.triton
    op_counts = {}
    for node in pipeline.dataflow.nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    groups = tuple(fusion.groups)
    layout_ops = {op: count for op, count in op_counts.items() if op in LAYOUT_OPS}
    layout_groups = [group for group in groups
                     if any(node.op in LAYOUT_OPS for node in group.members)]
    return {
        "groups": len(groups),
        "fused_groups": len(fusion.fusions),
        "fused_values": len(fusion.fused_values),
        "kernels": len(triton.kernels) if triton else 0,
        "launches": len(triton.kernels) if triton else 0,
        "kernel_kinds": sorted({kernel.kind for kernel in triton.kernels}) if triton else [],
        "kernel_ops": [list(kernel.ops) for kernel in triton.kernels] if triton else [],
        "refusals": len(triton.refusals) if triton else 0,
        "skipped": len(triton.skipped) if triton else 0,
        "materialized_buffers": len(memory.materialized),
        "allocated_buffers": len(memory.allocated),
        "slots": len(memory.slots),
        "buffer_sizes": [buffer.size for buffer in memory.buffers],
        "layout_ops": layout_ops,
        "layout_groups": len(layout_groups),
        "op_counts": op_counts,
        "triton_bytes": len(triton.host) + sum(len(k.source) for k in triton.kernels) if triton else 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/middle_end/audit.json")
    parser.add_argument("--only", nargs="*", help="subset of program names")
    args = parser.parse_args(argv)
    rows = []
    started = time.perf_counter()
    for name, source in PROGRAMS.items():
        if args.only and name not in args.only:
            continue
        variants = {}
        for ablation, options in ABLATIONS.items():
            try:
                result = compile_source(source, **options)
                variants[ablation] = _metrics(result)
            except Exception as error:  # noqa: BLE001 - report, never hide
                variants[ablation] = {"error": f"{type(error).__name__}: {error}"}
        default = variants.get("default", {})
        deltas = {}
        for ablation, metrics in variants.items():
            if ablation == "default" or "error" in metrics or "error" in default:
                continue
            deltas[ablation] = {
                "kernel_delta": metrics["kernels"] - default["kernels"],
                "materialized_delta": metrics["materialized_buffers"] - default["materialized_buffers"],
                "slot_delta": metrics["slots"] - default["slots"],
                "fused_group_delta": metrics["fused_groups"] - default["fused_groups"],
                "refusal_delta": metrics["refusals"] - default["refusals"],
            }
        rows.append({"program": name, "default": default, "ablations": variants,
                     "deltas": deltas})
    totals = {
        "programs": len(rows),
        "kernels_default": sum(row["default"].get("kernels", 0) for row in rows),
        "kernels_rule_fusion": sum(row["ablations"].get("rule_fusion", {}).get("kernels", 0)
                                   for row in rows),
        "materialized_default": sum(row["default"].get("materialized_buffers", 0) for row in rows),
        "layout_groups_default": sum(row["default"].get("layout_groups", 0) for row in rows),
        "refusals_default": sum(row["default"].get("refusals", 0) for row in rows),
    }
    output = {
        "corpus": {name: len(source.splitlines()) for name, source in PROGRAMS.items()},
        "ablations": list(ABLATIONS),
        "rows": rows,
        "totals": totals,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation": (
            "Structural opportunity audit only.  A positive kernel delta on an "
            "ablation means the default optimisation is already saving kernels; "
            "a zero delta means that decision does not matter for this corpus.  "
            "Layout/materialised-buffer deltas identify where a layout/"
            "materialisation synthesis pass could act; reduction-fusion deltas "
            "identify decomposition opportunities.  Device timing is the next "
            "filter, not this audit."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    md = ["# Middle-end opportunity audit", "",
          f"{len(rows)} programs, {len(ABLATIONS)} configurations; "
          f"{output['elapsed_seconds']} s.", "",
          "| program | kernels | groups | fused groups | materialised | slots | layout groups | refusals |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        d = row["default"]
        md.append(f"| {row['program']} | {d.get('kernels')} | {d.get('groups')} | "
                  f"{d.get('fused_groups')} | {d.get('materialized_buffers')} | "
                  f"{d.get('slots')} | {d.get('layout_groups')} | {d.get('refusals')} |")
    md += ["", "## Ablation deltas (non-default minus default)", "",
           "| program | ablation | Δkernels | Δmaterialised | Δslots | Δfused groups |",
           "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        for ablation, delta in row["deltas"].items():
            md.append(f"| {row['program']} | {ablation} | {delta['kernel_delta']} | "
                      f"{delta['materialized_delta']} | {delta['slot_delta']} | "
                      f"{delta['fused_group_delta']} |")
    md += ["", "## Totals", "", "```json", json.dumps(totals, indent=2), "```", "",
           output["interpretation"], ""]
    (out.parent / "audit.md").write_text("\n".join(md))
    print(json.dumps(output, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
