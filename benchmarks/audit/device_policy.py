"""Measure the device-policy contribution on one H100 with an idle GPU."""
from __future__ import annotations

import json
import pathlib
import random
import statistics
import time

import torch

from flowgpu.compiler.backend.platform_configs import resolve_device_platform, resolve_platform
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
H, I = 4096, 14336


def p50(fn, iterations=15):
    fn(); torch.cuda.synchronize(); values = []
    for _ in range(iterations):
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1e6)
    return statistics.median(values)


def main() -> int:
    source = (ROOT / "examples/llama3_mlp_swiglu.fgpu").read_text()
    safe = resolve_platform("cuda")
    tuned = resolve_device_platform()
    payload = {"environment": "dedicated H100 GPU0", "safe": {"n": safe.matmul_block_n, "m": safe.matmul_block_m},
               "tuned": {"n": tuned.matmul_block_n, "m": tuned.matmul_block_m}, "cases": []}
    for tokens in (1, 2048):
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        wg = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wu = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wd = torch.randn(I, H, device="cuda", dtype=torch.bfloat16)
        bindings = {"T": tokens, "H": H, "I": I}
        safe_run = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                    cache_dir=str(ROOT / ".policy-cache-safe"),
                                    shape_bindings=bindings, platform_config=safe)
        tuned_run = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                     cache_dir=str(ROOT / ".policy-cache-tuned"),
                                     shape_bindings=bindings, platform_config=tuned)
        eager = lambda: (torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd
        expected = eager(); torch.cuda.synchronize()
        safe_out = safe_run(x, wg, wu, wd)["y"]
        error = ((safe_out.float() - expected.float()).abs().max()
                 / expected.float().abs().max()).item()
        samples = {"safe": [], "tuned": [], "eager": []}
        paths = {"safe": lambda: safe_run(x, wg, wu, wd),
                 "tuned": lambda: tuned_run(x, wg, wu, wd),
                 "eager": eager}
        rng = random.Random(0)
        for _ in range(3):
            order = list(paths); rng.shuffle(order)
            for key in order:
                samples[key].append(p50(paths[key]))
        medians = {key: statistics.median(values) for key, values in samples.items()}
        payload["cases"].append({
            "tokens": tokens, "relative_error": error,
            **{f"{key}_us": medians[key] for key in medians},
            "safe_over_tuned": medians["safe"] / medians["tuned"],
            "tuned_over_eager": medians["tuned"] / medians["eager"],
            "safe_over_eager": medians["safe"] / medians["eager"],
            "samples": samples,
        })
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
