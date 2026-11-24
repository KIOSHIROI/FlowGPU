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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.decode import DecodeSession
    source = (ROOT / "examples/quickstart/decode_gqa_step.fgpu").read_text()
    session = DecodeSession(source, "quickstart_decode_gqa_step",
                            str(ROOT / ".competitive-cache/token-loop"))
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
    result = dict(steps=args.steps, total_ms=elapsed * 1000.0,
                  per_token_us=elapsed * 1e6 / args.steps,
                  tokens_per_s=args.steps / elapsed)
    print(json.dumps(result, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
