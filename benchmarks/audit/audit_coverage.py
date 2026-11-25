"""Check what the audit sheet exposes for the six user-study faults."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    sources = {
        "mlp": ROOT / "examples/mlp.fgpu",
        "decode": ROOT / "examples/llama3_decode_block_masked32.fgpu",
        "moe": ROOT / "examples/batch10/moe_router_topk.fgpu",
    }
    reports = {name: build_audit_report(path.read_text()) for name, path in sources.items()}
    checks = [
        ("shape", "mlp", "input x dtype or shape", True),
        ("dtype", "mlp", "input x dtype or shape", True),
        ("loc_range", "decode", "loc index is outside the cache range", True),
        ("mask_missing", "decode", "attention", False),
        ("cache_skip", "decode", "attention", False),
        ("moe_tie", "moe", "topk", False),
    ]
    rows = []
    for task, source, needle, guard_expected in checks:
        report = reports[source]
        haystack = "\n".join(report["host_guards"])
        if not guard_expected:
            haystack = "\n".join(
                f"{kernel['kind']} {' '.join(kernel['ops'])}"
                for kernel in report["kernels"]) + "\n" + "\n".join(
                f"{kernel['kind']} {' '.join(kernel['ops'])}"
                for kernel in report["reference_kernels"])
        visible = needle in haystack
        rows.append({"task": task, "source": source, "evidence": needle,
                     "guard_expected": guard_expected, "visible_in_audit": visible})
    guard_rows = [row for row in rows if row["guard_expected"]]
    semantic_rows = [row for row in rows if not row["guard_expected"]]
    summary = {"tasks": len(rows),
               "visible": sum(1 for row in rows if row["visible_in_audit"]),
               "guard_tasks": len(guard_rows),
               "guard_visible": sum(1 for row in guard_rows if row["visible_in_audit"]),
               "semantic_tasks": len(semantic_rows),
               "semantic_visible": sum(1 for row in semantic_rows if row["visible_in_audit"])}
    print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
