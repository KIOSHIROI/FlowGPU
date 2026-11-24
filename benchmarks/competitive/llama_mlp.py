"""Measure the real Llama 3 8B SwiGLU MLP subgraph.

Compares FlowGPU's compiled subgraph and its zero-copy GraphRunner against
PyTorch eager bf16 on the same H100. Ratios, not absolute microseconds, are
the stable result on a shared device.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
H, I = 4096, 14336


def _p50(fn, iterations):
    import torch
    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,2048")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--device-tile", action="store_true",
                        help="use resolve_device_platform() for the GEMM tile")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.graph import GraphRunner
    from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
    from flowgpu.compiler.runtime.runtime import compile_pipeline
    from flowgpu.compiler.backend.platform_configs import resolve_device_platform

    options = {"platform_config": resolve_device_platform()} if args.device_tile else {}
    source = (ROOT / "examples/llama3_mlp_swiglu.fgpu").read_text()
    payload = {"environment": "H100 PCIe, shared", "dtype": "bf16",
               "hidden": H, "intermediate": I, "cases": []}
    for tokens in (int(value) for value in args.tokens.split(",")):
        x = torch.randn(tokens, H, device="cuda", dtype=torch.bfloat16)
        wg = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wu = torch.randn(H, I, device="cuda", dtype=torch.bfloat16)
        wd = torch.randn(I, H, device="cuda", dtype=torch.bfloat16)
        bindings = {"T": tokens, "H": H, "I": I}
        run = compile_pipeline(source, pipeline="llama3_mlp_swiglu", shape_bindings=bindings,
                               cache_dir=str(ROOT / ".competitive-cache/llama3-mlp"), **options)
        out = run(x, wg, wu, wd)["y"]
        eager = lambda: (torch.nn.functional.silu(x @ wg) * (x @ wu)) @ wd
        want = eager()
        torch.cuda.synchronize()
        relative = ((out - want).abs().max() / want.abs().max()).item()
        plan = prepare_execution_plan(
            source, pipeline="llama3_mlp_swiglu", args=(x, wg, wu, wd),
            cache_dir=str(ROOT / ".competitive-cache/llama3-mlp"),
            validator=lambda result: True, shape_bindings=bindings, **options)
        graph = GraphRunner(plan, (x, wg, wu, wd), copy_inputs=False)
        medians = {"flowgpu": [], "graph": [], "eager": []}
        for _ in range(args.repeats):
            medians["flowgpu"].append(_p50(lambda: run(x, wg, wu, wd), args.iterations))
            medians["graph"].append(_p50(graph.replay, args.iterations))
            medians["eager"].append(_p50(eager, args.iterations))
        fg = statistics.median(medians["flowgpu"])
        gp = statistics.median(medians["graph"])
        eg = statistics.median(medians["eager"])
        payload["cases"].append({
            "tokens": tokens, "relative_error": relative,
            "flowgpu_us": fg, "graph_us": gp, "eager_us": eg,
            "flowgpu_over_eager": fg / eg, "graph_over_eager": gp / eg,
        })
        print(f"T={tokens}: rel={relative:.3e} flowgpu={fg:.1f}us "
              f"graph={gp:.1f}us eager={eg:.1f}us "
              f"ratio={fg/eg:.2f}x graph={gp/eg:.2f}x")
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
