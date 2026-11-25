"""Compile the same SwiGLU two ways and compare the CPU reference."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

ROOT = pathlib.Path(__file__).resolve().parent
BINDINGS = {"B": 2, "D": 8, "H": 16}


def main() -> int:
    source = (ROOT / "lesson.fgpu").read_text()
    fused = build_audit_report(source, shape_bindings=BINDINGS)
    materialized = build_audit_report(source, shape_bindings=BINDINGS,
                                      materialize_all=True)
    fused_program = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    material_program = compile_source(source, shape_bindings=BINDINGS,
                                      materialize_all=True).pipelines[0]
    one = reference.run(fused_program.reference, sizes=BINDINGS)["out"]
    other = reference.run(material_program.reference, sizes=BINDINGS)["out"]
    print("# Lesson 6: same dataflow, two compile decisions")
    print(f"fused kernels: {fused['kernel_count']}")
    for kernel in fused["kernels"]:
        print(f"- {' -> '.join(kernel['ops'])}")
    print(f"materialized kernels: {materialized['kernel_count']}")
    for kernel in materialized["kernels"]:
        print(f"- {kernel['kind']}: {' -> '.join(kernel['ops'])}")
    print(f"reference identical: {one == other}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
