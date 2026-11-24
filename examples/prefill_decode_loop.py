"""Prefill 8 tokens, then decode from the same 32-slot paged cache."""
from __future__ import annotations

import pathlib

import torch

from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
D, SLOTS, PROMPT = 64, 32, 8


def main() -> None:
    prefill = compile_pipeline((ROOT / "examples/llama3_prefill_block_cached.fgpu").read_text(),
                               pipeline="llama3_prefill_block_cached",
                               cache_dir=str(ROOT / ".prefill-decode-cache"))
    decode = compile_pipeline((ROOT / "examples/llama3_decode_block_masked32.fgpu").read_text(),
                              pipeline="llama3_decode_block_masked32",
                              cache_dir=str(ROOT / ".prefill-decode-cache"))
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    shared = [bf(D), bf(D, D), bf(D, D), bf(D, D), bf(D, D), bf(D),
              bf(D, 2 * D), bf(D, 2 * D), bf(2 * D, D)]
    prefill_weights = shared + [bf(PROMPT, D), bf(PROMPT, D)]
    decode_weights = shared + [bf(1, D), bf(1, D)]
    q_scale = torch.full((1, D), 0.125, device="cuda", dtype=torch.bfloat16)

    cache = PagedKVCache(blocks=1, page_size=SLOTS, heads=1, depth=D)
    table = BlockTable(cache)
    table.add("req-1")
    for _ in range(PROMPT):
        table.append("req-1")              # prompt occupies slots 0..7

    k = cache.k.reshape(1, SLOTS, D)
    v = cache.v.reshape(1, SLOTS, D)
    loc = torch.tensor([0], device="cuda", dtype=torch.int32)
    valid = torch.tensor([PROMPT], device="cuda", dtype=torch.int32)
    prefix = torch.tensor([0], device="cuda", dtype=torch.int32)
    out = prefill(bf(PROMPT, D), *prefill_weights, k, v, loc, valid, prefix)
    cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
    cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
    x = out["out"][-1].reshape(1, D)

    for _ in range(4):
        position = table.append("req-1")
        mask = torch.full((1, SLOTS), -1e9, device="cuda", dtype=torch.bfloat16)
        mask[0, :position + 1] = 0
        k = cache.k.reshape(1, SLOTS, D)
        v = cache.v.reshape(1, SLOTS, D)
        out = decode(x, *decode_weights, k, v,
                     torch.tensor([position], device="cuda", dtype=torch.int32),
                     mask, q_scale)
        cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
        cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
        x = out["out"]
    torch.cuda.synchronize()
    print("locations", table.locations("req-1"))
    print("decode out", tuple(x.shape), "finite", bool(torch.isfinite(x).all().item()))
    print("slots 0..11 written", [bool(cache.k[index].abs().sum().item()) for index in range(12)])


if __name__ == "__main__":
    main()
