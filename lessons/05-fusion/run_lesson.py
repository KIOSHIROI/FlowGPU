"""Compile Lesson 5, show fusion groups, and run the CPU reference."""
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
    report = build_audit_report(source, shape_bindings=BINDINGS)
    program = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    out = reference.run(program.reference, sizes=BINDINGS)["out"]
    print("# Lesson 5: fused_mlp")
    print(f"kernels: {report['kernel_count']}")
    for kernel in report["kernels"]:
        print(f"- {kernel['name']}: {' -> '.join(kernel['ops'])}")
    print("fusion:")
    for group in report["fusion_groups"]:
        print(f"- {group['anchor']}: {' -> '.join(group['members'])} "
              f"fused={group['fused_values']}")
    print(f"out[0,0] = {out[0]}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
