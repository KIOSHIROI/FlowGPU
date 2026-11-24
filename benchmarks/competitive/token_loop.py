"""Measure a short token-generation loop through the compiled decode step."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.decode import DecodeSession
    if args.inplace:
        source = (ROOT / "examples/quickstart/decode_gqa_attention.fgpu").read_text()
        pipeline = "quickstart_decode_gqa_attention"
    else:
        source = (ROOT / "examples/quickstart/decode_gqa_step.fgpu").read_text()
        pipeline = "quickstart_decode_gqa_step"
    session = DecodeSession(source, pipeline,
                            str(ROOT / ".competitive-cache/token-loop"),
                            prepared=args.prepared, inplace=args.inplace)
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    def run_steps(count):
        for _ in range(count):
            k = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
            session.step(q, k, v)
    session.reset(); run_steps(args.warmup); session.reset()
    torch.cuda.synchronize()
    start = time.perf_counter(); run_steps(args.steps); torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = dict(steps=args.steps, prepared=args.prepared, inplace=args.inplace, total_ms=elapsed * 1000.0,
                  per_token_us=elapsed * 1e6 / args.steps,
                  tokens_per_s=args.steps / elapsed)
    print(json.dumps(result, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
