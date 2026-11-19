"""Derive an operator × backend coverage matrix from the compiler's own tables.

A coverage row answers one question per (model operator, backend profile):
*``generated``* when the frontend can express it and the profile admits the
kernel kind, *``extern``* when the model entry names a ledger-recorded extern
kernel, and *``refused``*/*``missing``* with the exact reason otherwise.  It is
deliberately conservative: a profile that refuses a kind is reported as
refused, even if the CANN/FlagOS tooling could widen it, because widening is a
separate measured decision.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

from flowgpu.compiler.backend.backends import KERNELS  # noqa: E402
from flowgpu.compiler.backend.platform_configs import PLATFORMS  # noqa: E402
from flowgpu.compiler.backend import reference as _reference  # noqa: E402
from flowgpu.compiler.backend import triton_backend as _triton  # noqa: E402
from flowgpu.compiler.backend.triton_backend import EMITTERS  # noqa: E402
from flowgpu.compiler.frontend.operator import (  # noqa: E402
    AXIS_OPERATORS, COMPOSED_OPERATORS, OPERATORS)

STATUSES = ("generated", "extern", "refused", "missing")


def operator_kinds() -> dict:
    kinds = {}
    for kind, operators in KERNELS.items():
        for operator in operators:
            kinds[operator] = kind
    for syntactic, expansion in AXIS_OPERATORS.items():
        kinds[syntactic] = kinds.get(expansion, "unknown")
    for operator in COMPOSED_OPERATORS:
        kinds[operator] = "composed"
    for operator in (set(_triton.POINTWISE_EXPRESSION)
                     | set(_reference.POINTWISE_EXPRESSION)):
        kinds.setdefault(operator, "pointwise")
    for operator in OPERATORS:
        kinds.setdefault(operator, "unknown")
    return kinds


KINDS = operator_kinds()


def load_extern_ledger() -> dict:
    path = pathlib.Path(__file__).resolve().parent / "data" / "extern_ledger.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {entry["module"]: entry for entry in data.get("entries", [])}


def load_model(path) -> dict:
    data = json.loads(pathlib.Path(path).read_text())
    assert data.get("schema") == "flowgpu.migration.model/1", data.get("schema")
    names = [entry["name"] for entry in data["operators"]]
    assert len(names) == len(set(names)), "duplicate operator names"
    return data


def cover_operator(entry: dict, platform: str, extern: dict) -> dict:
    config = PLATFORMS[platform]
    if entry.get("frontend"):
        operator = entry["frontend"]
        kind = KINDS.get(operator)
        if operator not in OPERATORS:
            return {"status": "missing", "kind": kind,
                    "reason": f"frontend operator {operator!r} is not registered"}
        if kind is None or kind == "unknown":
            return {"status": "missing", "kind": kind,
                    "reason": f"no kernel kind is recorded for {operator!r}"}
        if kind not in config.allowed_kernel_kinds:
            return {"status": "refused", "kind": kind,
                    "reason": (f"profile {platform!r} allows only "
                               f"{', '.join(config.allowed_kernel_kinds)}; "
                               f"{operator!r} needs {kind!r}")}
        if kind not in EMITTERS:
            return {"status": "missing", "kind": kind,
                    "reason": f"no Triton emitter for kind {kind!r}"}
        return {"status": "generated", "kind": kind,
                "reason": f"{operator!r} lowers as {kind!r} on {platform!r}"}
    module = entry.get("extern")
    if module:
        record = extern.get(module)
        if record is None:
            return {"status": "missing", "kind": "extern",
                    "reason": f"extern module {module!r} is not in the ledger"}
        return {"status": "extern", "kind": record.get("lane", "extern"),
                "reason": f"extern ledger: {record.get('status', 'recorded')}"}
    if entry.get("external"):
        return {"status": "missing", "kind": "external",
                "reason": f"external operator {entry['external']!r} is not mapped "
                          "to a FlowGPU operator yet"}
    return {"status": "missing", "kind": None,
            "reason": "entry declares neither a frontend operator nor an extern module"}


def coverage(model_path, platforms) -> dict:
    model = load_model(model_path)
    extern = load_extern_ledger()
    rows = []
    for platform in platforms:
        if platform not in PLATFORMS:
            raise SystemExit(f"unknown platform profile {platform!r}; "
                             f"choose from {', '.join(PLATFORMS)}")
        for entry in model["operators"]:
            record = cover_operator(entry, platform, extern)
            rows.append({"operator": entry["name"], "platform": platform, **record})
    summary = {platform: {status: 0 for status in STATUSES} for platform in platforms}
    for row in rows:
        summary[row["platform"]][row["status"]] += 1
    return {"schema": "flowgpu.migration.coverage/1",
            "model": model["model"], "platforms": list(platforms),
            "rows": rows, "summary": summary}


def render_coverage(result: dict) -> str:
    platforms = result["platforms"]
    lines = [f"# Coverage: {result['model']}", "",
             "| operator | " + " | ".join(platforms) + " |",
             "|---|" + "---|" * len(platforms)]
    by_operator = {}
    for row in result["rows"]:
        by_operator.setdefault(row["operator"], {})[row["platform"]] = row
    for operator, cells in by_operator.items():
        marks = []
        for platform in platforms:
            row = cells[platform]
            mark = {"generated": "✅ generated", "extern": "🔌 extern",
                    "refused": "⛔ refused", "missing": "❓ missing"}[row["status"]]
            marks.append(f"{mark}")
        lines.append("| `%s` | %s |" % (operator, " | ".join(marks)))
    lines += ["", "## Summary", "", "| platform | generated | extern | refused | missing |",
              "|---|---:|---:|---:|---:|"]
    for platform in platforms:
        s = result["summary"][platform]
        lines.append("| %s | %d | %d | %d | %d |" % (
            platform, s["generated"], s["extern"], s["refused"], s["missing"]))
    reasons = [row for row in result["rows"] if row["status"] in {"refused", "missing"}]
    if reasons:
        lines += ["", "## Why the gaps exist", "", "| operator | platform | status | reason |",
                  "|---|---|---|---|"]
        for row in reasons:
            lines.append("| `%s` | %s | %s | %s |" % (
                row["operator"], row["platform"], row["status"], row["reason"]))
    lines.append("")
    return "\n".join(lines)
