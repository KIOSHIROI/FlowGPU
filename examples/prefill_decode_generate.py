"""Closed loop: prompt -> prefill -> decode -> LM head -> sample -> next token."""
from __future__ import annotations

import pathlib

import torch

from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline
from flowgpu.compiler.runtime.sampling import greedy

ROOT = pathlib.Path(__file__).resolve().parents[1]
D, SLOTS, VOCAB, PROMPT = 64, 32, 128, 8


def main() -> None:
    prefill = compile_pipeline((ROOT / "examples/llama3_prefill_block_cached.fgpu").read_text(),
                               pipeline="llama3_prefill_block_cached",
                               cache_dir=str(ROOT / ".prefill-decode-generate-cache"))
    decode = compile_pipeline((ROOT / "examples/llama3_decode_block_masked32.fgpu").read_text(),
                              pipeline="llama3_decode_block_masked32",
                              cache_dir=str(ROOT / ".prefill-decode-generate-cache"))
    head = compile_pipeline((ROOT / "examples/lm_head.fgpu").read_text(),
                            pipeline="lm_head",
                            cache_dir=str(ROOT / ".prefill-decode-generate-cache"))
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    shared = [bf(D), bf(D, D), bf(D, D), bf(D, D), bf(D, D), bf(D),
              bf(D, 2 * D), bf(D, 2 * D), bf(2 * D, D)]
    prefill_weights = shared + [bf(PROMPT, D), bf(PROMPT, D)]
    decode_weights = shared + [bf(1, D), bf(1, D)]
    head_weight = bf(D, VOCAB)
    embedding = bf(VOCAB, D)

    cache = PagedKVCache(blocks=1, page_size=SLOTS, heads=1, depth=D)
    table = BlockTable(cache)
    table.add("req-1")
    for _ in range(PROMPT):
        table.append("req-1")

    prompt = list(range(PROMPT))
    cache_k = cache.k.reshape(1, SLOTS, D)
    cache_v = cache.v.reshape(1, SLOTS, D)
    out = prefill(embedding[prompt], *prefill_weights, cache_k, cache_v,
                  torch.tensor([0], device="cuda", dtype=torch.int32),
                  torch.tensor([PROMPT], device="cuda", dtype=torch.int32),
                  torch.tensor([0], device="cuda", dtype=torch.int32))
    cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
    cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
    hidden = out["out"][-1].reshape(1, D)
    token = int(greedy(head(hidden, head_weight)["logits"]).item())
    tokens = prompt + [token]
    q_scale = torch.full((1, D), 0.125, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        position = table.append("req-1")
        mask = torch.full((1, SLOTS), -1e9, device="cuda", dtype=torch.bfloat16)
        mask[0, :position + 1] = 0
        cache_k = cache.k.reshape(1, SLOTS, D)
        cache_v = cache.v.reshape(1, SLOTS, D)
        out = decode(embedding[token].reshape(1, D), *decode_weights, cache_k, cache_v,
                     torch.tensor([position], device="cuda", dtype=torch.int32), mask, q_scale)
        cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
        cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
        token = int(greedy(head(out["out"], head_weight)["logits"]).item())
        tokens.append(token)
    torch.cuda.synchronize()
    print("locations", table.locations("req-1"))
    print("tokens", tokens)
    print("in_vocab", all(0 <= value < VOCAB for value in tokens))


if __name__ == "__main__":
    main()
