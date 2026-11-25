"""Compile Lesson 3, show the fused reduction, and run the CPU reference."""
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
    program = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    out = reference.run(program.reference, sizes=BINDINGS)["out"]
    print("# Lesson 3: row_scale")
    print(f"kernels: {report['kernel_count']}")
    for kernel in report["kernels"]:
        print(f"- {kernel['name']}: {kernel['kind']} {' -> '.join(kernel['ops'])}")
    print(f"reference out[0]: {out[:8]}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
