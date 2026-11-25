"""Phase-3 starter suite: three non-standard LLM subgraphs, eager + compile."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class Case:
    name: str
    source: pathlib.Path
    pipeline: str
    bindings: dict
    make_inputs: object
    eager: object
    output_names: tuple
    tolerance: float
    id_name: str | None = None


def _bf(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02


def _residual_inputs():
    return (_bf(2048, 4096), _bf(2048, 4096), _bf(4096))


def _residual_eager(x, residual, weight):
    y = x + residual
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * weight


def _moe_inputs():
    return (torch.randn(4, 16, device="cuda"), torch.randn(8, 16, device="cuda"),
            torch.tensor([[0.5]], device="cuda"))


def _moe_eager(x, router_weight, cap):
    logits = x @ router_weight.t()
    capped = torch.tanh(logits / cap) * cap
    probs = torch.softmax(capped, -1)
    values, ids = torch.topk(capped, 2, -1)
    return torch.gather(probs, -1, ids), ids


def _routed_inputs():
    return (torch.randn(2048, 14336, device="cuda", dtype=torch.float16) * 0.02,
            torch.randn(2048, 14336, device="cuda", dtype=torch.float16) * 0.02,
            torch.randn(2048, 1, device="cuda"), torch.ones(1, device="cuda", dtype=torch.float16))


def _routed_eager(gate, up, weight, like):
    p = F.silu(gate.float()) * up.float()
    return (p * weight.float()).to(like.dtype)


def _p50(fn, iterations):
    fn(); torch.cuda.synchronize(); samples = []
    for _ in range(iterations):
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples)


def _rope_tables(positions, width):
    half = width // 2
    frequency = torch.exp(-math.log(10000.0) * torch.arange(half, device="cuda") / half)
    angle = positions[:, None].float() * frequency[None, :]
    return angle.cos().repeat(1, 2), angle.sin().repeat(1, 2)


def _apply_rope(x, cosine, sine):
    width = x.shape[-1]
    index = torch.arange(width, device=x.device)
    first = index < width // 2
    partner = torch.where(first, index + width // 2, index - width // 2)
    rotated = torch.where(first, -x[..., partner], x[..., partner])
    return x * cosine + rotated * sine


def _gqa_inputs():
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    x = bf(2, 64)
    weights = [bf(64), bf(64, 256), bf(64, 128), bf(64, 128), bf(256, 64), bf(64),
               bf(64, 128), bf(64, 128), bf(128, 64)]
    qcos, qsin = _rope_tables(torch.tensor([0, 0], device="cuda"), 256)
    kcos, ksin = _rope_tables(torch.tensor([0, 0], device="cuda"), 128)
    cache_k = torch.zeros(2, 2, 8, 64, device="cuda", dtype=torch.bfloat16)
    cache_v = torch.zeros_like(cache_k)
    cache_k[0, :, :5] = bf(2, 5, 64); cache_v[0, :, :5] = bf(2, 5, 64)
    cache_k[1, :, :1] = bf(2, 1, 64); cache_v[1, :, :1] = bf(2, 1, 64)
    return (x, *weights, qcos.to(torch.bfloat16), qsin.to(torch.bfloat16),
            kcos.to(torch.bfloat16), ksin.to(torch.bfloat16), cache_k, cache_v,
            torch.tensor([1, 5], device="cuda", dtype=torch.int32),
            torch.tensor([2, 6], device="cuda", dtype=torch.int32))


def _gqa_eager(x, n1, wq, wk, wv, wo, n2, wg, wu, wd, qcos, qsin, kcos, ksin,
               k_cache, v_cache, loc, valid):
    xn = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * n1
    q = _apply_rope(xn @ wq, qcos, qsin).reshape(2, 4, 64)
    k = _apply_rope(xn @ wk, kcos, ksin).reshape(2, 2, 64)
    v = (xn @ wv).reshape(2, 2, 64)
    kc, vc = k_cache.clone(), v_cache.clone()
    for b in range(2):
        kc[b, :, loc[b]] = k[b]; vc[b, :, loc[b]] = v[b]
    k_rep = kc.repeat_interleave(2, dim=1)
    v_rep = vc.repeat_interleave(2, dim=1)
    scores = torch.einsum("bqd,bqsd->bqs", q.float(), k_rep.float()) / 8.0
    keep = torch.arange(8, device=x.device).view(1, 1, 8) < valid.view(2, 1, 1)
    scores = scores.masked_fill(~keep, float("-inf"))
    context = torch.einsum("bqs,bqsd->bqd", torch.softmax(scores, -1), v_rep.float()).to(torch.bfloat16)
    mid = x + context.reshape(2, 256) @ wo
    hn = mid * torch.rsqrt(mid.square().mean(-1, keepdim=True) + 1e-6) * n2
    return mid + (F.silu(hn @ wg) * (hn @ wu)) @ wd


def _gqa_large_inputs():
    bf = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02
    weights = [bf(128), bf(128, 4096), bf(128, 1024), bf(128, 1024), bf(4096, 128),
               bf(128), bf(128, 14336), bf(128, 14336), bf(14336, 128)]
    cache_k = torch.zeros(4, 8, 256, 128, device="cuda", dtype=torch.bfloat16)
    cache_v = torch.zeros_like(cache_k)
    locs = torch.tensor([8, 16, 24, 32], device="cuda", dtype=torch.int32)
    for b, loc in enumerate(locs.tolist()):
        cache_k[b, :, :loc] = bf(8, loc, 128)
        cache_v[b, :, :loc] = bf(8, loc, 128)
    valid = locs + 1
    return (bf(4, 128), *weights, cache_k, cache_v, locs, valid)


def _gqa_large_eager(x, n1, wq, wk, wv, wo, n2, wg, wu, wd,
                     k_cache, v_cache, loc, valid):
    h = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * n1
    q = (h @ wq).reshape(4, 32, 128)
    k = (h @ wk).reshape(4, 8, 128)
    v = (h @ wv).reshape(4, 8, 128)
    kc, vc = k_cache.clone(), v_cache.clone()
    for b in range(4):
        kc[b, :, loc[b]] = k[b]; vc[b, :, loc[b]] = v[b]
    k_rep = kc.repeat_interleave(4, dim=1)
    scores = torch.einsum("bqd,bqsd->bqs", q.float(), k_rep.float()) / math.sqrt(128)
    keep = torch.arange(256, device=x.device).view(1, 1, 256) < valid.view(4, 1, 1)
    scores = scores.masked_fill(~keep, float("-inf"))
    context = torch.einsum("bqs,bqsd->bqd", torch.softmax(scores, -1), vc.repeat_interleave(4, dim=1).float())
    mid = x + context.reshape(4, 4096).to(torch.bfloat16) @ wo
    hn = mid * torch.rsqrt(mid.square().mean(-1, keepdim=True) + 1e-6) * n2
    return mid + (F.silu(hn @ wg) * (hn @ wu)) @ wd


CASES = (
    Case("residual_norm", ROOT / "examples/residual_norm.fgpu", "residual_norm",
         {"N": 2048, "D": 4096}, _residual_inputs, _residual_eager, ("z",), 2e-2),
    Case("moe_router_topk", ROOT / "examples/batch10/moe_router_topk.fgpu", "moe_router_topk",
         {}, _moe_inputs, _moe_eager, ("weights",), 2e-2, "ids"),
    Case("routed_silu", ROOT / "examples/batch10/routed_silu.fgpu", "routed_silu",
         {"M": 2048, "N": 14336}, _routed_inputs, _routed_eager, ("out",), 2e-2),
    Case("gqa_decode_masked", ROOT / "examples/llama3_decode_block_gqa_masked.fgpu",
         "llama3_decode_block_gqa_masked", {}, _gqa_inputs, _gqa_eager, ("out",), 3e-2),
    Case("gqa_decode_large", ROOT / "examples/gqa_decode_large.fgpu", "gqa_decode_large",
         {}, _gqa_large_inputs, _gqa_large_eager, ("out",), 3e-2),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = {"environment": "H100 PCIe, shared", "cases": []}
    for case in CASES:
        inputs = tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in case.make_inputs())
        run = compile_pipeline(case.source.read_text(), pipeline=case.pipeline,
                               cache_dir=str(ROOT / ".suite-cache"), shape_bindings=case.bindings)
        result = run(*inputs)
        eager_result = case.eager(*inputs)
        generated = tuple(result[name] for name in case.output_names)
        expected = eager_result if isinstance(eager_result, tuple) else (eager_result,)
        if case.id_name is not None:
            expected = expected[:len(case.output_names)]
        errors = [float((a.float() - b.float()).abs().max()) for a, b in zip(generated, expected)]
        ids_equal = None
        if case.id_name is not None:
            ids_equal = bool(torch.equal(result[case.id_name].to(expected[0].dtype), eager_result[1]))
        flowgpu = statistics.median([_p50(lambda: run(*inputs), args.iterations)
                                     for _ in range(args.repeats)])
        eager = statistics.median([_p50(lambda: case.eager(*inputs), args.iterations)
                                   for _ in range(args.repeats)])
        compiled = None
        try:
            compiled_fn = torch.compile(case.eager)
            compiled_fn(*inputs); torch.cuda.synchronize()
            compiled = statistics.median([_p50(lambda: compiled_fn(*inputs), args.iterations)
                                          for _ in range(args.repeats)])
        except Exception as error:  # noqa: BLE001 - baseline may be unsupported
            compiled = None
            payload.setdefault("baseline_errors", {})[case.name] = str(error).splitlines()[0][:120]
        entry = {"name": case.name,
                 "max_abs_error": max(errors), "tolerance": case.tolerance,
                 "passed": max(errors) <= case.tolerance,
                 "id_name": case.id_name,
                 "ids_equal": ids_equal,
                 "flowgpu_us": flowgpu, "eager_us": eager,
                 "flowgpu_over_eager": flowgpu / eager,
                 "compiled_us": compiled,
                 "flowgpu_over_compiled": flowgpu / compiled if compiled else None}
        payload["cases"].append(entry)
        print(json.dumps(entry) if args.json else entry)
    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
