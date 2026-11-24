"""Simulate continuous batching over 16 requests with a 4-slot GPU batch."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.batching import ContinuousBatchSession, RequestState
    source = (ROOT / "examples/quickstart/decode_batch_attention.fgpu").read_text()
    pipeline = "quickstart_decode_batch_attention"
    session = ContinuousBatchSession(
        source, pipeline,
        str(ROOT / ".competitive-cache/continuous-batch"), capacity=4, inplace=True)
    bank = {
        request: [
            (torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16),
             torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
             torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(args.tokens)]
        for request in range(args.requests)
    }

    def run_session(session):
        pending = [RequestState(request_id=index, max_tokens=args.tokens)
                   for index in range(args.requests)]
        while pending or session.active():
            while pending and session.active() < session.capacity:
                session.admit(pending.pop(0))
            if not session.active():
                break
            tokens = {}
            for slot in session.slots:
                if slot is None:
                    continue
                tokens[slot.request_id] = bank[slot.request_id][slot.position]
            session.step(tokens)

    run_session(session)
    torch.cuda.synchronize()
    session2 = ContinuousBatchSession(
        source, pipeline,
        str(ROOT / ".competitive-cache/continuous-batch"), capacity=4, inplace=True)
    start = time.perf_counter()
    run_session(session2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = session2.generated_tokens
    result = dict(requests=args.requests, tokens=tokens, total_ms=elapsed * 1000.0,
                  tokens_per_s=tokens / elapsed,
                  per_token_us=elapsed * 1e6 / tokens)
    print(json.dumps(result, indent=2) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
