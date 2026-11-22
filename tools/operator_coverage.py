#!/usr/bin/env python3
"""Public operator-coverage audit against a representative LLM checklist.

This is a static inventory, not a correctness or performance benchmark.  It
answers two questions a compiler user asks first: can I express this operation,
and which lowering kind implements it?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowgpu.compiler.backend.backends import KERNELS  # noqa: E402
from flowgpu.compiler.core.operators import OPERATORS  # noqa: E402

CHECKLIST = [
    ("pointwise", "add", "add", "elementwise"),
    ("pointwise", "mul", "mul", "elementwise"),
    ("pointwise", "sub", "sub", "elementwise"),
    ("pointwise", "div", "div", "elementwise"),
    ("pointwise", "exp", "exp", "elementwise"),
    ("pointwise", "log", "log", "elementwise"),
    ("pointwise", "sqrt", "sqrt", "elementwise"),
    ("pointwise", "rsqrt", "rsqrt", "elementwise"),
    ("pointwise", "sigmoid", "sigmoid", "elementwise"),
    ("pointwise", "silu", "silu", "elementwise"),
    ("pointwise", "gelu", "gelu", "elementwise"),
    ("pointwise", "tanh", "tanh", "elementwise"),
    ("pointwise", "abs", "abs", "elementwise"),
    ("pointwise", "neg", "neg", "elementwise"),
    ("pointwise", "pow", "pow", "elementwise"),
    ("pointwise", "where", "where", "select"),
    ("reduction", "sum", "sum_last", "last-axis reduction"),
    ("reduction", "max", "max_last", "last-axis reduction"),
    ("reduction", "mean", "mean_last", "last-axis reduction"),
    ("reduction", "argmax", "argmax_last", "last-axis reduction"),
    ("reduction", "softmax", "softmax", "rowwise"),
    ("norm", "rmsnorm", "rms_norm", "rowwise"),
    ("norm", "layernorm", "layer_norm", "rowwise"),
    ("norm", "batch_norm", "batch_norm", "rowwise"),
    ("linear algebra", "matmul", "matmul", "tiled GEMM"),
    ("linear algebra", "conv2d", "conv2d", "convolution"),
    ("linear algebra", "embedding lookup", "gather_rows", "indexed gather"),
    ("attention", "scaled dot-product attention", "attention", "attention"),
    ("attention", "causal attention", "causal_attention", "attention"),
    ("position", "rope", "rope", "positional"),
    ("layout", "reshape", "reshape", "layout"),
    ("layout", "transpose", "transpose", "layout"),
    ("layout", "concat", "concat", "layout"),
    ("layout", "split", "split", "layout"),
    ("layout", "pad+concat", "pad_cat", "layout"),
    ("gather", "row gather", "gather_rows", "indexed gather"),
    ("gather", "last-axis gather", "gather_last", "indexed gather"),
    ("gather", "scatter by index", "scatter_by_index", "indexed scatter"),
    ("sort/search", "top-k", "topk_last", "topk"),
    ("sort/search", "sort", "sort_last", "sort"),
    ("scan", "cumsum", "cumsum", "scan"),
    ("scan", "segmented cumsum", "segmented_cumsum", "scan"),
    ("state", "cache store", "store_cache_4d", "stateful"),
    ("state", "linear decode", "linear_decode_forward", "stateful"),
    ("state", "recurrence", "gated_delta_recurrence", "stateful"),
    ("shape", "dynamic symbolic shapes", "dynamic_shape", "runtime dispatch"),
]

KIND = {op: kind for kind, ops in KERNELS.items() for op in ops}
MANUAL = {
    "dynamic_shape": (
        "partial",
        "symbolic runtime-extent lowering exists (SymbolicShapeRunner); literal-only operators still need per-shape specialization",
    ),
}


def report() -> dict:
    entries = []
    for category, canonical, flowgpu, note in CHECKLIST:
        manual = MANUAL.get(flowgpu)
        present = flowgpu in OPERATORS
        entries.append({
            "category": category,
            "canonical": canonical,
            "flowgpu": flowgpu,
            "status": manual[0] if manual else ("present" if present else "missing"),
            "kernel_kind": KIND.get(flowgpu) if present else None,
            "note": manual[1] if manual else note,
        })
    present = sum(entry["status"] == "present" for entry in entries)
    partial = sum(entry["status"] == "partial" for entry in entries)
    return {
        "schema": "flowgpu.operator_coverage/1",
        "summary": {"checklist": len(entries), "present": present,
                    "partial": partial,
                    "missing": len(entries) - present - partial,
                    "registered_operators": len(OPERATORS),
                    "registered_kernel_kinds": len(KERNELS)},
        "entries": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = report()
    if args.check:
        assert data["schema"] == "flowgpu.operator_coverage/1"
        assert len(data["entries"]) == data["summary"]["checklist"]
        assert (data["summary"]["present"] + data["summary"]["partial"]
                + data["summary"]["missing"]) == data["summary"]["checklist"]
        return 0
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    if args.markdown:
        print("| category | operation | FlowGPU | status | kind |")
        print("|---|---|---|---|---|")
        for entry in data["entries"]:
            print(f"| {entry['category']} | {entry['canonical']} | `{entry['flowgpu']}` | "
                  f"{entry['status']} | {entry['kernel_kind'] or '-'} |")
        return 0
    summary = data["summary"]
    print(f"checklist: {summary['present']}/{summary['checklist']} present, "
          f"{summary['missing']} missing; registered operators: {summary['registered_operators']}")
    for entry in data["entries"]:
        if entry["status"] == "missing":
            print(f"  missing: {entry['category']}: {entry['canonical']} ({entry['flowgpu']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
