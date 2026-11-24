"""Continuous batching with a prompt prefill when a request is admitted."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--prompt", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from flowgpu.compiler.runtime.batching import ContinuousBatchSession, RequestState
    source = (ROOT / "examples/quickstart/decode_batch_attention.fgpu").read_text()
    pipeline = "quickstart_decode_batch_attention"
    bank = {}
    prompts = {}
    for request in range(args.requests):
        prompts[request] = (torch.randn(2, args.prompt, 64, device="cuda", dtype=torch.bfloat16),
                            torch.randn(2, args.prompt, 64, device="cuda", dtype=torch.bfloat16))
        bank[request] = [
            (torch.randn(4, 1, 64, device="cuda", dtype=torch.bfloat16),
             torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
             torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(args.tokens)]

    def run(session):
        pending = [RequestState(request_id=index, max_tokens=args.prompt + args.tokens,
                                prompt_length=args.prompt,
                                prompt_k=prompts[index][0], prompt_v=prompts[index][1])
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
                tokens[slot.request_id] = bank[slot.request_id][slot.position - args.prompt]
            session.step(tokens)

    session = ContinuousBatchSession(source, pipeline,
                                     str(ROOT / ".competitive-cache/continuous-prefill"),
                                     capacity=4, inplace=True)
    run(session)
    torch.cuda.synchronize()
    session2 = ContinuousBatchSession(source, pipeline,
                                      str(ROOT / ".competitive-cache/continuous-prefill"),
                                      capacity=4, inplace=True)
    start = time.perf_counter()
    run(session2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    decode_tokens = session2.generated_tokens
    payload = dict(requests=args.requests, prompt_tokens=args.prompt * args.requests,
                   decode_tokens=decode_tokens, total_ms=elapsed * 1000.0,
                   per_decode_us=elapsed * 1e6 / decode_tokens,
                   decode_tokens_per_s=decode_tokens / elapsed)
    print(json.dumps(payload, indent=2) if args.json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
