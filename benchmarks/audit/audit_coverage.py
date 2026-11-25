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
        ("mask_missing", "decode", "mask/cache consistency", False),
        ("cache_skip", "decode", "cache-position continuity", False),
        ("moe_tie", "moe", "topk tie contract", False),
    ]
    rows = []
    for task, source, needle, guard_expected in checks:
        report = reports[source]
        haystack = "\n".join(report["host_guards"])
        kind = "guard" if guard_expected else "risk"
        if not guard_expected:
            haystack = "\n".join(report["semantic_risks"])
        visible = needle in haystack
        rows.append({"task": task, "source": source, "evidence": needle,
                     "evidence_kind": kind, "visible_in_audit": visible})
    direct = [row for row in rows if row["evidence_kind"] == "guard"]
    flagged = [row for row in rows if row["evidence_kind"] == "risk"]
    summary = {"tasks": len(rows),
               "visible": sum(1 for row in rows if row["visible_in_audit"]),
               "direct_visible": sum(1 for row in direct if row["visible_in_audit"]),
               "risk_flagged": sum(1 for row in flagged if row["visible_in_audit"])}
    print(json.dumps({"rows": rows, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
