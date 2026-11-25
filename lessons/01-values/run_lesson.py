"""Compile Lesson 1, show the kernel, and run the CPU reference."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

ROOT = pathlib.Path(__file__).resolve().parent
BINDINGS = {"N": 4, "D": 8}


def main() -> int:
    source = (ROOT / "lesson.fgpu").read_text()
    report = build_audit_report(source, shape_bindings=BINDINGS)
    values = reference.run(
        compile_source(source, shape_bindings=BINDINGS).pipelines[0].reference,
        sizes=BINDINGS)
    out = values["out"]
    print("# Lesson 1: add_scale")
    print(f"kernels: {report['kernel_count']}")
    for kernel in report["kernels"]:
        print(f"- {kernel['name']}: {' -> '.join(kernel['ops'])}")
    print(f"guards: {len(report['host_guards'])}")
    print(f"reference out[0]: {out[:BINDINGS['D']]}")
    print(f"reference out[1]: {out[BINDINGS['D']:2 * BINDINGS['D']]}")
    print("\nquiz:")
    quiz = json.loads((ROOT / "quiz.json").read_text())
    for item in quiz["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
