"""Compile Lesson 4, show the matmul tile, and run the CPU reference."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform

ROOT = pathlib.Path(__file__).resolve().parent
BINDINGS = {"M": 4, "K": 8, "N": 16}


def main() -> int:
    source = (ROOT / "lesson.fgpu").read_text()
    report = build_audit_report(source, shape_bindings=BINDINGS)
    program = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    out = reference.run(program.reference, sizes=BINDINGS)["out"]
    print("# Lesson 4: matmul_intro")
    policy = resolve_platform()
    print(f"kernels: {report['kernel_count']}")
    print(f"tile: BLOCK_M={policy.matmul_block_m} "
          f"BLOCK_N={policy.matmul_block_n} BLOCK_K={policy.matmul_block_k}")
    for kernel in report["kernels"]:
        print(f"- {kernel['name']}: {kernel['kind']} {kernel['ops']}")
    print(f"out[0,0] = {out[0]}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
