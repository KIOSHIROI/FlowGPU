"""Compare the generated implementation with the handwritten packet."""
from __future__ import annotations

import json
import pathlib

from flowgpu.compiler.audit import build_audit_report
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

ROOT = pathlib.Path(__file__).resolve().parent
PACKET = ROOT.parents[1] / "review" / "swiglu_split"


def main() -> int:
    source = (PACKET / "source.fgpu").read_text()
    report = build_audit_report(source)
    program = compile_source(source).pipelines[0]
    out = reference.run(program.reference)["out"]
    handwritten = (PACKET / "handwritten.py").read_text().splitlines()
    addresses = [line.strip() for line in handwritten
                 if "* WIDTH" in line or "split_copy_kernel" in line
                 or "def swiglu_kernel" in line]
    print("# Lesson 8: generated vs handwritten")
    print("generated kernels:")
    for kernel in report["kernels"]:
        print(f"- {kernel['kind']}: {' -> '.join(kernel['ops'])}")
    print("handwritten structure:")
    for line in addresses:
        print(f"- {line}")
    print(f"reference out[0]: {out[:8]}")
    print("\nquiz:")
    for item in json.loads((ROOT / "quiz.json").read_text())["questions"]:
        print(f"- {item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
