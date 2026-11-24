"""Drive the cached decode block for one request with BlockTable + PagedKVCache."""
from __future__ import annotations

import pathlib

import torch

from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
H, D = 8, 64


def main() -> None:
    source = (ROOT / "examples/llama3_decode_block.fgpu").read_text()
    run = compile_pipeline(source, pipeline="llama3_decode_block",
                           cache_dir=str(ROOT / ".decode-loop-cache"))
    cache = PagedKVCache(blocks=1, page_size=H, heads=1, depth=D)
    table = BlockTable(cache)
    table.add("req-1")

    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weights = [
        bf(D), bf(D, D), bf(D, D), bf(D, D), bf(D, D), bf(D),
        bf(D, 2 * D), bf(D, 2 * D), bf(2 * D, D), bf(1, D), bf(1, D),
    ]
    x = bf(1, D)
    for _ in range(H):
        loc = table.append("req-1")
        k = cache.k.reshape(1, H, D)
        v = cache.v.reshape(1, H, D)
        out = run(x, *weights, k, v, torch.tensor([loc], device="cuda", dtype=torch.int32))
        cache.k.copy_(out["k_out"].reshape(H, 1, D))
        cache.v.copy_(out["v_out"].reshape(H, 1, D))
        x = out["out"]
    torch.cuda.synchronize()
    print("locations", table.locations("req-1"))
    print("out", tuple(x.shape), "finite", bool(torch.isfinite(x).all().item()))
    print("cache slots written", [bool(cache.k[index].abs().sum().item()) for index in range(H)])


if __name__ == "__main__":
    main()
