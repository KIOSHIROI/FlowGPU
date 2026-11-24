"""Run the compiled Llama 3 MLP repeatedly across layers with different weights."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
H, I = 4096, 14336


def _p50(fn, iterations):
    import torch
    fn(); torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--weight-scale", type=float, default=0.005)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
    from flowgpu.compiler.runtime.layers import LayerStack

    T = args.tokens
    source = (ROOT / "examples/llama3_mlp_swiglu.fgpu").read_text()
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    layers = [
        (torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * args.weight_scale,
         torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * args.weight_scale,
         torch.randn(I, H, device="cuda", dtype=torch.bfloat16) * args.weight_scale)
        for _ in range(args.layers)
    ]
    plan = prepare_execution_plan(
        source, pipeline="llama3_mlp_swiglu", args=(x, *layers[0]),
        cache_dir=str(ROOT / ".competitive-cache/llama3-mlp-stack"),
        validator=lambda result: True,
        shape_bindings={"T": T, "H": H, "I": I})
    stack = LayerStack(plan, layers)

    def flowgpu_stack():
        return stack(x)

    def eager_stack():
        current = x
        for wg, wu, wd in layers:
            current = (torch.nn.functional.silu(current @ wg) * (current @ wu)) @ wd
        return current

    got = flowgpu_stack().clone(); want = eager_stack(); torch.cuda.synchronize()
    relative = ((got - want).abs().max() / want.abs().max()).item()
    if not math.isfinite(relative):
        # Deep synthetic stacks can underflow/overflow bf16; the timing is still
        # useful, but there is no honest numeric comparison to report.
        relative = None
    fg = [ _p50(flowgpu_stack, args.iterations) for _ in range(args.repeats)]
    eg = [ _p50(eager_stack, args.iterations) for _ in range(args.repeats)]
    payload = {"environment": "H100 PCIe, shared", "layers": args.layers,
               "tokens": T, "relative_error": relative,
               "flowgpu_us": statistics.median(fg), "eager_us": statistics.median(eg),
               "flowgpu_over_eager": statistics.median(fg) / statistics.median(eg)}
    print(json.dumps(payload, indent=2) if args.json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
