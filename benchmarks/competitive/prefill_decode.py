"""Prefill a prompt into the KV cache, then decode tokens from that cache."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.decode import DecodeSession
    from flowgpu.compiler.runtime.runtime import compile_pipeline
    prefill_source = (ROOT / "examples/quickstart/prefill_gqa_cache.fgpu").read_text()
    prefill = compile_pipeline(prefill_source, pipeline="quickstart_prefill_gqa_cache",
                               cache_dir=str(ROOT / ".competitive-cache/prefill-decode"))
    q = torch.randn(4, 16, 64, device="cuda", dtype=torch.bfloat16)
    k_new = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    v_new = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(2, 128, 64, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    loc = torch.tensor([0], device="cuda", dtype=torch.int32)
    # Warm the kernel and host paths before timing.
    prefill(q, k_new, v_new, torch.zeros_like(k_cache), torch.zeros_like(v_cache), loc)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = prefill(q, k_new, v_new, k_cache, v_cache, loc)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - start) * 1000.0
    decode_source = (ROOT / "examples/quickstart/decode_gqa_attention_nocopy.fgpu").read_text()
    session = DecodeSession(decode_source, "quickstart_decode_gqa_attention_nocopy",
                            str(ROOT / ".competitive-cache/prefill-decode"), inplace=True)
    session.seed(torch.zeros_like(k_cache), torch.zeros_like(v_cache), 0)
    qd = torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        session.step(qd, torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
                     torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    session.seed(result["k2"], result["v2"], 16)
    start = time.perf_counter()
    for _ in range(args.tokens):
        session.step(qd, torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
                     torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    decode = time.perf_counter() - start
    payload = dict(prefill_tokens=16, decode_tokens=args.tokens,
                   prefill_ms=prefill_ms,
                   per_token_us=decode * 1e6 / args.tokens,
                   tokens_per_s=args.tokens / decode)
    print(json.dumps(payload, indent=2) if args.json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
