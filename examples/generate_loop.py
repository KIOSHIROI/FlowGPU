"""End-to-end token loop: embedding -> cached decode block -> LM head -> sample."""
from __future__ import annotations

import pathlib

import torch

from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline
from flowgpu.compiler.runtime.sampling import greedy

ROOT = pathlib.Path(__file__).resolve().parents[1]
D, SLOTS, VOCAB = 64, 8, 128


def main() -> None:
    block = compile_pipeline((ROOT / "examples/llama3_decode_block_masked.fgpu").read_text(),
                             pipeline="llama3_decode_block_masked",
                             cache_dir=str(ROOT / ".generate-loop-cache"))
    head = compile_pipeline((ROOT / "examples/lm_head.fgpu").read_text(),
                            pipeline="lm_head",
                            cache_dir=str(ROOT / ".generate-loop-cache"))
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    block_weights = [bf(D), bf(D, D), bf(D, D), bf(D, D), bf(D, D), bf(D),
                     bf(D, 2 * D), bf(D, 2 * D), bf(2 * D, D), bf(1, D), bf(1, D)]
    head_weight = bf(D, VOCAB)
    embedding = bf(VOCAB, D)

    cache = PagedKVCache(blocks=1, page_size=SLOTS, heads=1, depth=D)
    table = BlockTable(cache)
    table.add("req-1")
    token = 3
    tokens = [token]
    q_scale = torch.full((1, D), 0.125, device="cuda", dtype=torch.bfloat16)
    for _ in range(SLOTS):
        loc = table.append("req-1")
        mask = torch.full((1, SLOTS), -1e9, device="cuda", dtype=torch.bfloat16)
        mask[0, :loc + 1] = 0
        x = embedding[token].reshape(1, D)
        k = cache.k.reshape(1, SLOTS, D)
        v = cache.v.reshape(1, SLOTS, D)
        out = block(x, *block_weights, k, v,
                    torch.tensor([loc], device="cuda", dtype=torch.int32), mask, q_scale)
        cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
        cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
        logits = head(out["out"], head_weight)["logits"]
        token = int(greedy(logits).item())
        tokens.append(token)
    torch.cuda.synchronize()
    print("locations", table.locations("req-1"))
    print("tokens", tokens)
    print("in_vocab", all(0 <= value < VOCAB for value in tokens))


if __name__ == "__main__":
    main()
