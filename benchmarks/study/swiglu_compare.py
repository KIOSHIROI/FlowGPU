"""Controlled comparison: same SwiGLU source, fused vs materialized compile."""
from __future__ import annotations

import json
import pathlib
import random
import statistics

import torch

from flowgpu.compiler.audit import audit_from_result
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
H, I = 4096, 14336


def gpu_time(fn, iterations=20):
    fn(); torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> int:
    source = (ROOT / "examples/llama3_mlp_swiglu.fgpu").read_text()
    payload = {"environment": "H100 GPU0", "cases": []}
    for tokens in (1, 2048):
        bindings = {"T": tokens, "H": H, "I": I}
        fused_audit = audit_from_result(compile_source(source, shape_bindings=bindings))
        material_audit = audit_from_result(compile_source(source, shape_bindings=bindings,
                                                           materialize_all=True))
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        wg = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wu = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wd = torch.randn(I, H, device="cuda", dtype=torch.bfloat16)
        fused_run = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                     cache_dir=str(ROOT / ".swiglu-compare"),
                                     shape_bindings=bindings)
        material_run = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                        cache_dir=str(ROOT / ".swiglu-compare"),
                                        shape_bindings=bindings, materialize_all=True)
        eager = lambda: (torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd
        expected = eager(); torch.cuda.synchronize()
        f = fused_run(x, wg, wu, wd)["y"]
        m = material_run(x, wg, wu, wd)["y"]
        errors = {
            "fused": ((f.float() - expected.float()).abs().max()
                      / expected.float().abs().max()).item(),
            "materialized": ((m.float() - expected.float()).abs().max()
                             / expected.float().abs().max()).item(),
        }
        paths = {"fused": lambda: fused_run(x, wg, wu, wd),
                 "materialized": lambda: material_run(x, wg, wu, wd)}
        samples = {key: [] for key in paths}
        rng = random.Random(0)
        for _ in range(3):
            order = list(paths); rng.shuffle(order)
            for key in order:
                samples[key].append(gpu_time(paths[key]))
        medians = {key: statistics.median(values) for key, values in samples.items()}
        payload["cases"].append({
            "tokens": tokens,
            "fused_kernels": [kernel["ops"] for kernel in fused_audit["kernels"]],
            "materialized_kernels": [kernel["ops"] for kernel in material_audit["kernels"]],
            "relative_error": errors,
            "fused_us": medians["fused"],
            "materialized_us": medians["materialized"],
            "materialized_over_fused": medians["materialized"] / medians["fused"],
            "samples": samples,
        })
        print(json.dumps(payload["cases"][-1]))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
