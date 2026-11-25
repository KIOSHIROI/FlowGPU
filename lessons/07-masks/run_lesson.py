"""Compile Lesson 7 and print the generated load/store masks."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report
from flowgpu.compiler.backend import reference
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform

ROOT = pathlib.Path(__file__).resolve().parent
BINDINGS = {"M": 3, "K": 7, "N": 5}


def main() -> int:
    source = (ROOT / "lesson.fgpu").read_text()
    report = build_audit_report(source, shape_bindings=BINDINGS)
    program = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    policy = resolve_platform()
    out = reference.run(program.reference, sizes=BINDINGS)["out"]
    print("# Lesson 7: masked_matmul")
    print(f"kernels: {report['kernel_count']}")
    print(f"tile: BLOCK_M={policy.matmul_block_m} "
          f"BLOCK_N={policy.matmul_block_n} BLOCK_K={policy.matmul_block_k}")
    print("grid: (ceil(3/BM), ceil(5/BN)) = (1, 1)")
    print("masks:")
    for line in module([program.triton]).splitlines():
        stripped = line.strip()
        if stripped.startswith("masked_matmul_k0[") or "tl.load" in stripped or "tl.store" in stripped:
            print(f"- {stripped}")
    print(f"out[0,0] = {out[0]}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
