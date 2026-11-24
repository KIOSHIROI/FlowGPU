"""Dynamic batching over the GQA decode block: two slots, per-request caches."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import torch

from flowgpu.compiler.runtime.batching import DynamicBatch
from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
B, QH, KVH, SLOTS, D = 2, 4, 2, 8, 64


@dataclass
class Request:
    x: torch.Tensor
    cache: PagedKVCache
    table: BlockTable


def main() -> None:
    source = (ROOT / "examples/llama3_decode_block_gqa.fgpu").read_text()
    run = compile_pipeline(source, pipeline="llama3_decode_block_gqa",
                           cache_dir=str(ROOT / ".decode-loop-gqa-cache"))
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weights = [bf(D), bf(D, QH * D), bf(D, KVH * D), bf(D, KVH * D),
               bf(QH * D, D), bf(D), bf(D, 2 * D), bf(D, 2 * D), bf(2 * D, D),
               bf(B, QH * D), bf(B, QH * D), bf(B, KVH * D), bf(B, KVH * D)]
    zero = lambda: torch.zeros(1, D, device="cuda", dtype=torch.bfloat16)
    batch = DynamicBatch(capacity=B)
    requests = {}

    def new_request(request_id):
        cache = PagedKVCache(blocks=1, page_size=SLOTS, heads=KVH, depth=D)
        table = BlockTable(cache)
        table.add(request_id)
        requests[request_id] = Request(bf(D), cache, table)
        batch.add(request_id, requests[request_id].x)

    def call(rows):
        k = torch.zeros(B, KVH, SLOTS, D, device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        locs = torch.zeros(B, device="cuda", dtype=torch.int32)
        for request_id, slot in batch.active.items():
            state = requests[request_id]
            locs[slot] = state.table.append(request_id)
            k[slot] = state.cache.k.permute(1, 0, 2)
            v[slot] = state.cache.v.permute(1, 0, 2)
        out = run(torch.stack(rows), *weights, k, v, locs)
        for request_id, slot in batch.active.items():
            state = requests[request_id]
            state.cache.k.copy_(out["k_out"][slot].permute(1, 0, 2))
            state.cache.v.copy_(out["v_out"][slot].permute(1, 0, 2))
        return [out["out"][slot] for slot in range(B)]

    new_request("r0")
    new_request("r1")
    for _ in range(3):
        outputs = batch.step(call, zero)
        for request_id, value in outputs.items():
            requests[request_id].x = value
            batch.update(request_id, value)
    print("after 3 steps", {rid: requests[rid].table.locations(rid) for rid in requests})

    batch.remove("r0")
    requests.pop("r0")
    new_request("r2")
    outputs = batch.step(call, zero)
    torch.cuda.synchronize()
    print("active", sorted(batch.active))
    print("locations", {rid: requests[rid].table.locations(rid) for rid in requests})
    print("finite", all(bool(torch.isfinite(value).all().item()) for value in outputs.values()))


if __name__ == "__main__":
    main()
