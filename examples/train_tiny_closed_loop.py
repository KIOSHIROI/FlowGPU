"""Train a tiny block, then compare PyTorch and FlowGPU token generation."""
from __future__ import annotations

import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowgpu.compiler.runtime.block_table import BlockTable
from flowgpu.compiler.runtime.paged_cache import PagedKVCache
from flowgpu.compiler.runtime.runtime import compile_pipeline
from flowgpu.compiler.runtime.sampling import greedy

ROOT = pathlib.Path(__file__).resolve().parents[1]
D, H, V, T, SLOTS = 64, 128, 128, 8, 32


def rope_tables(positions, dtype):
    half = D // 2
    frequency = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    angle = positions[:, None].float() * frequency[None, :]
    cosine = angle.cos().repeat(1, 2)
    sine = angle.sin().repeat(1, 2)
    return cosine.to(dtype), sine.to(dtype)


def apply_rope(x, cosine, sine):
    half = D // 2
    index = torch.arange(D)
    first = index < half
    partner = torch.where(first, index + half, index - half)
    rotated = torch.where(first, -x[..., partner], x[..., partner])
    return x * cosine + rotated * sine


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Parameter(torch.ones(D))
        self.wq = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.wk = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.wv = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.wo = nn.Parameter(torch.randn(D, D) / math.sqrt(D))
        self.norm2 = nn.Parameter(torch.ones(D))
        self.wg = nn.Parameter(torch.randn(D, H) / math.sqrt(D))
        self.wu = nn.Parameter(torch.randn(D, H) / math.sqrt(D))
        self.wd = nn.Parameter(torch.randn(H, D) / math.sqrt(H))

    def forward(self, x, cosine, sine):
        h = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.norm1
        q = apply_rope(h @ self.wq, cosine, sine)
        k = apply_rope(h @ self.wk, cosine, sine)
        v = h @ self.wv
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device))
        probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1)
        mid = x + (probs @ v) @ self.wo
        h = mid * torch.rsqrt(mid.square().mean(-1, keepdim=True) + 1e-6) * self.norm2
        return mid + (F.silu(h @ self.wg) * (h @ self.wu)) @ self.wd


def main() -> None:
    torch.manual_seed(0)
    embedding = nn.Parameter(torch.randn(V, D) * 0.02)
    block = TinyBlock()
    head = nn.Parameter(torch.randn(D, V) / math.sqrt(D))
    parameters = [embedding, *block.parameters(), head]
    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    pattern = torch.arange(T) % 8
    inputs = pattern.repeat(32, 1)
    targets = torch.roll(pattern, -1).repeat(32, 1)

    for _ in range(400):
        cosine, sine = rope_tables(torch.arange(T), torch.float32)
        logits = block(embedding[inputs], cosine, sine) @ head
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print("train loss", float(loss.detach()))

    # PyTorch greedy in bf16, matching the FlowGPU weight cast.
    embed_bf = embedding.detach().to(torch.bfloat16)
    block_bf = TinyBlock()
    block_bf.load_state_dict({name: value.detach() for name, value in block.state_dict().items()})
    block_bf = block_bf.to(torch.bfloat16)
    head_bf = head.detach().to(torch.bfloat16)
    tokens = pattern.tolist()
    with torch.no_grad():
        for _ in range(5):
            cosine, sine = rope_tables(torch.arange(len(tokens)), torch.bfloat16)
            logits = block_bf(embed_bf[torch.tensor(tokens)][None], cosine, sine)[0, -1] @ head_bf
            tokens.append(int(greedy(logits).item()))
    expected = tokens[T:]
    print("torch tokens", tokens)

    # FlowGPU: cached prefill then masked decode.
    prefill = compile_pipeline((ROOT / "examples/llama3_prefill_block_cached.fgpu").read_text(),
                               pipeline="llama3_prefill_block_cached",
                               cache_dir=str(ROOT / ".train-tiny-cache"))
    decode = compile_pipeline((ROOT / "examples/llama3_decode_block_masked32.fgpu").read_text(),
                              pipeline="llama3_decode_block_masked32",
                              cache_dir=str(ROOT / ".train-tiny-cache"))
    lm = compile_pipeline((ROOT / "examples/lm_head.fgpu").read_text(),
                          pipeline="lm_head", cache_dir=str(ROOT / ".train-tiny-cache"))
    bf = lambda value: value.detach().to("cuda", torch.bfloat16)
    shared = [bf(block_bf.norm1), bf(block_bf.wq), bf(block_bf.wk), bf(block_bf.wv), bf(block_bf.wo),
              bf(block_bf.norm2), bf(block_bf.wg), bf(block_bf.wu), bf(block_bf.wd)]
    cos8, sin8 = (bf(t) for t in rope_tables(torch.arange(T), torch.float32))
    cos1, sin1 = (bf(t) for t in rope_tables(torch.arange(T, T + 4), torch.float32))
    cache = PagedKVCache(blocks=1, page_size=SLOTS, heads=1, depth=D)
    table = BlockTable(cache); table.add("req")
    for _ in range(T):
        table.append("req")
    k = cache.k.reshape(1, SLOTS, D); v = cache.v.reshape(1, SLOTS, D)
    out = prefill(bf(embed_bf[torch.tensor(pattern.tolist())]), *shared, cos8, sin8,
                  k, v, torch.tensor([0], device="cuda", dtype=torch.int32),
                  torch.tensor([T], device="cuda", dtype=torch.int32),
                  torch.tensor([0], device="cuda", dtype=torch.int32))
    cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
    cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
    hidden = out["out"][-1].reshape(1, D)
    cos8_cpu, sin8_cpu = rope_tables(torch.arange(T), torch.bfloat16)
    with torch.no_grad():
        prompt_x = embed_bf[torch.tensor(pattern.tolist())][None]
        h = prompt_x * torch.rsqrt(prompt_x.square().mean(-1, keepdim=True) + 1e-6) * block_bf.norm1
        torch_q = apply_rope(h @ block_bf.wq, cos8_cpu, sin8_cpu)
        torch_k = apply_rope(h @ block_bf.wk, cos8_cpu, sin8_cpu)
        torch_v = h @ block_bf.wv
        torch_hidden = block_bf(prompt_x, cos8_cpu, sin8_cpu)[0, -1:]
    print("prefill hidden maxdiff", (hidden.float() - torch_hidden.float().to(hidden.device)).abs().max().item())
    print("cache k maxdiff", (out["k_out"][0, :T].float() - torch_k[0].float().to(hidden.device)).abs().max().item())
    print("cache v maxdiff", (out["v_out"][0, :T].float() - torch_v[0].float().to(hidden.device)).abs().max().item())
    token = int(greedy(lm(hidden, bf(head_bf))["logits"]).item())
    generated = [token]
    q_scale = torch.full((1, D), 0.125, device="cuda", dtype=torch.bfloat16)
    for step in range(4):
        position = table.append("req")
        mask = torch.full((1, SLOTS), -1e9, device="cuda", dtype=torch.bfloat16)
        mask[0, :position + 1] = 0
        out = decode(bf(embed_bf[token]).reshape(1, D), *shared, cos1[step:step+1], sin1[step:step+1],
                     cache.k.reshape(1, SLOTS, D), cache.v.reshape(1, SLOTS, D),
                     torch.tensor([position], device="cuda", dtype=torch.int32), mask, q_scale)
        cache.k.copy_(out["k_out"].reshape(SLOTS, 1, D))
        cache.v.copy_(out["v_out"].reshape(SLOTS, 1, D))
        token = int(greedy(lm(out["out"], bf(head_bf))["logits"]).item())
        generated.append(token)
    torch.cuda.synchronize()
    print("flowgpu tokens", generated)
    print("match", generated == expected)


if __name__ == "__main__":
    main()
