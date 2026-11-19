#!/usr/bin/env python3
"""Compile a pipeline from Python and inspect the generated products.

Run from the repository root after installing the package:

    python examples/integration/compile_pipeline.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flowgpu import compile_source  # noqa: E402

SOURCE = """
pipeline example_add {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
"""


def main() -> int:
    result = compile_source(SOURCE)
    pipeline = result.pipelines[0]
    triton = pipeline.triton
    print(f"pipeline: {pipeline.pipeline.name}")
    print(f"kernels: {len(triton.kernels)}")
    for kernel in triton.kernels:
        print(f"  {kernel.name} [{kernel.kind}]")
    print(f"host wrapper lines: {len(triton.host.splitlines())}")
    print(f"reference kernels: {len(pipeline.reference.kernels)}")
    assert triton.complete
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
