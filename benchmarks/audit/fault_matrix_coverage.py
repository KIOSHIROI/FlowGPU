"""Audit coverage for the 13-case Phase-2 fault matrix."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report

ROOT = pathlib.Path(__file__).resolve().parents[2]

CHECKS = (
    ("shape_mismatch", "mlp", "guard", "input x dtype or shape"),
    ("dtype_mismatch", "mlp", "guard", "input x dtype or shape"),
    ("noncontiguous_layout", "mlp", "guard", "must be contiguous"),
    ("weight_dtype_mismatch", "mlp", "guard", "w1 dtype or shape"),
    ("cache_dtype_mismatch", "decode", "guard", "k_cache dtype or shape"),
    ("cache_loc_oob", "decode", "guard", "loc index is outside the cache range"),
    ("missing_mask", "decode", "risk", "mask/cache consistency"),
    ("valid_zero_mask", "decode", "risk", "mask/cache consistency"),
    ("skipped_cache_loc", "decode", "risk", "cache-position continuity"),
    ("duplicate_cache_loc", "decode", "risk", "cache-position continuity"),
    ("mask_too_short", "decode", "risk", "mask/cache consistency"),
    ("corrupt_cache_value", "decode", "risk", "cache values are caller-owned"),
    ("wrong_q_scale", "decode", "risk", "q_scale value is caller-owned"),
)


def main() -> int:
    sources = {
        "mlp": ROOT / "examples/mlp.fgpu",
        "decode": ROOT / "examples/llama3_decode_block_masked32.fgpu",
    }
    reports = {name: build_audit_report(path.read_text()) for name, path in sources.items()}
    rows = []
    for fault, source, kind, needle in CHECKS:
        report = reports[source]
        haystack = "\n".join(report["host_guards"] if kind == "guard" else report["semantic_risks"])
        rows.append({"fault": fault, "source": source, "kind": kind,
                     "visible": needle in haystack, "needle": needle})
    summary = {"faults": len(rows),
               "visible": sum(row["visible"] for row in rows),
               "direct": sum(row["visible"] and row["kind"] == "guard" for row in rows),
               "risk": sum(row["visible"] and row["kind"] == "risk" for row in rows)}
    print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
