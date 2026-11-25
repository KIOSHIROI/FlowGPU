"""Measure the host cost of the generated input-contract guards."""
from __future__ import annotations

import json
import pathlib
import random
import statistics
import time
from dataclasses import replace

import torch

from flowgpu.compiler.backend.platform_configs import resolve_device_platform
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
H, I = 4096, 14336


def p50(fn, iterations=30):
    fn(); torch.cuda.synchronize(); values = []
    for _ in range(iterations):
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1e6)
    return statistics.median(values)


def main() -> int:
    source = (ROOT / "examples/llama3_mlp_swiglu.fgpu").read_text()
    base = resolve_device_platform()
    guarded_config = replace(base, runtime_validation=True)
    unguarded_config = replace(base, runtime_validation=False)
    payload = {"environment": "dedicated H100 GPU0", "cases": []}
    for tokens in (1, 2048):
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        wg = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wu = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wd = torch.randn(I, H, device="cuda", dtype=torch.bfloat16)
        bindings = {"T": tokens, "H": H, "I": I}
        guarded = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                   cache_dir=str(ROOT / ".guard-cost-cache"),
                                   shape_bindings=bindings, platform_config=guarded_config)
        unguarded = compile_pipeline(source, pipeline="llama3_mlp_swiglu",
                                     cache_dir=str(ROOT / ".guard-cost-cache"),
                                     shape_bindings=bindings, platform_config=unguarded_config)
        paths = {"guarded": lambda: guarded(x, wg, wu, wd),
                 "unguarded": lambda: unguarded(x, wg, wu, wd)}
        samples = {key: [] for key in paths}
        rng = random.Random(0)
        for _ in range(7):
            order = list(paths); rng.shuffle(order)
            for key in order:
                samples[key].append(p50(paths[key]))
        medians = {key: statistics.median(values) for key, values in samples.items()}
        payload["cases"].append({
            "tokens": tokens,
            "guarded_us": medians["guarded"],
            "unguarded_us": medians["unguarded"],
            "guarded_over_unguarded": medians["guarded"] / medians["unguarded"],
            "samples": samples,
        })
        print(json.dumps(payload["cases"][-1]))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
