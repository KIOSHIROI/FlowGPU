#!/usr/bin/env python3
"""Compile and inspect the quickstart program without a GPU.

Run from the repository root:

    python examples/quickstart/demo.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flowgpu.compiler import compile_source
source = (ROOT / "examples" / "quickstart" / "add.fgpu").read_text()
compiled = compile_source(source)
pipeline = compiled.pipelines[0].triton

print(f"pipeline: {pipeline.pipeline}")
print(f"kernels:  {len(pipeline.kernels)}")
for kernel in pipeline.kernels:
    print(f"  {kernel.name} [{kernel.kind}] ops={', '.join(kernel.ops)}")
print("\nGenerated host wrapper (first 12 lines):")
for line in pipeline.host.splitlines()[:12]:
    print("  " + line)
print("\nNext: python -m flowgpu examples/quickstart/add.fgpu --run")
