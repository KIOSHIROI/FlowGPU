"""Measure the engineering-side auditability cost on small CPU-shape cases."""
from __future__ import annotations

import json
import pathlib
import statistics
import time

from flowgpu.compiler.audit import audit_from_result, render_markdown
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

ROOT = pathlib.Path(__file__).resolve().parents[2]


def timed(fn, repeats=5):
    values = []
    for _ in range(repeats):
        started = time.perf_counter(); fn(); values.append((time.perf_counter() - started) * 1e3)
    return statistics.median(values)


def main() -> int:
    cases = [
        ("mlp", ROOT / "examples/mlp.fgpu", {"B": 4, "D": 64, "H": 128}),
        ("residual_norm", ROOT / "examples/residual_norm.fgpu", {"N": 32, "D": 64}),
    ]
    payload = {"cases": []}
    for name, path, bindings in cases:
        source = path.read_text()
        compiled = {}
        def compile_once():
            compiled["result"] = compile_source(source, shape_bindings=bindings)
        compile_ms = timed(compile_once)
        program = compiled["result"].pipelines[0]
        report = audit_from_result(compiled["result"])
        audit_ms = timed(lambda: audit_from_result(compiled["result"]))
        markdown = render_markdown(report)
        ref_ms = timed(lambda: reference.run(program.reference), repeats=3)
        payload["cases"].append({
            "name": name,
            "compile_ms": compile_ms,
            "audit_report_ms": audit_ms,
            "reference_run_ms": ref_ms,
            "audit_over_compile": audit_ms / compile_ms,
            "reference_over_compile": ref_ms / compile_ms,
            "report_bytes": len(markdown.encode()),
            "kernels": len(report["kernels"]),
            "guards": len(report["host_guards"]),
        })
        print(json.dumps(payload["cases"][-1]))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
