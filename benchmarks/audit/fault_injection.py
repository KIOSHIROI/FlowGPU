"""Phase-2 auditability experiment: which LLM-subgraph faults are caught?

Structural faults are injected at the compiled host boundary; semantic faults
(wrong mask, skipped cache position) are injected by changing the values while
keeping every declared shape/dtype valid. The script records whether FlowGPU
raises, or silently returns a different / non-finite result.
"""
from __future__ import annotations

import json
import pathlib

import torch

from flowgpu.compiler.runtime.runtime import compile_pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
D, H, B = 64, 128, 4


def _bf(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.02


def _outcome(fn, correct=None, key="y"):
    try:
        value = fn()
        if isinstance(value, dict):
            value = value[key]
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001 - the point is the stage
        return {"caught": "launch", "type": type(error).__name__,
                "message": str(error).splitlines()[0][:120]}
    if correct is None:
        return {"caught": None}
    if not torch.isfinite(value).all():
        return {"caught": None, "silent": "non_finite"}
    difference = (value.float() - correct.float()).abs().max().item()
    return {"caught": None, "silent": "wrong_value", "maxdiff": difference}


def main() -> int:
    source = (ROOT / "examples/mlp.fgpu").read_text()
    run = compile_pipeline(source, pipeline="mlp",
                           cache_dir=str(ROOT / ".research-fault-cache"),
                           shape_bindings={"B": B, "D": D, "H": H})
    x = _bf(B, D); w1 = _bf(D, H); w2 = _bf(H, D)
    correct = run(x, w1, w2)["y"]
    cases = {}

    cases["shape_mismatch"] = _outcome(lambda: run(_bf(B - 1, D), w1, w2))
    cases["dtype_mismatch"] = _outcome(
        lambda: run(torch.randn(B, D, device="cuda", dtype=torch.float32), w1, w2))
    noncontiguous = torch.randn(D, B, device="cuda", dtype=torch.bfloat16).t()
    cases["noncontiguous_layout"] = _outcome(lambda: run(noncontiguous, w1, w2))
    cases["weight_dtype_mismatch"] = _outcome(
        lambda: run(x, torch.randn(D, H, device="cuda", dtype=torch.float32), w2))

    block = (ROOT / "examples/llama3_decode_block_masked32.fgpu").read_text()
    decode = compile_pipeline(block, pipeline="llama3_decode_block_masked32",
                              cache_dir=str(ROOT / ".research-fault-cache"))
    weights = [_bf(D), _bf(D, D), _bf(D, D), _bf(D, D), _bf(D, D), _bf(D),
               _bf(D, 2 * D), _bf(D, 2 * D), _bf(2 * D, D), _bf(1, D), _bf(1, D)]
    cache_k = torch.zeros(1, 32, D, device="cuda", dtype=torch.bfloat16)
    cache_v = torch.zeros_like(cache_k)
    cache_k[0, :8] = _bf(8, D); cache_v[0, :8] = _bf(8, D)
    good_mask = torch.full((1, 32), -1e9, device="cuda", dtype=torch.bfloat16)
    good_mask[0, :9] = 0
    q_scale = torch.full((1, D), 0.125, device="cuda", dtype=torch.bfloat16)
    base = (x[0:1], *weights, cache_k, cache_v,
            torch.tensor([8], device="cuda", dtype=torch.int32), good_mask, q_scale)
    good = decode(*base)["out"]
    cases["cache_loc_oob"] = _outcome(lambda: decode(
        x[0:1], *weights, cache_k, cache_v,
        torch.tensor([99], device="cuda", dtype=torch.int32), good_mask, q_scale), good, key="out")
    wrong_mask = torch.zeros_like(good_mask)
    cases["missing_mask"] = _outcome(lambda: decode(
        x[0:1], *weights, cache_k, cache_v,
        torch.tensor([8], device="cuda", dtype=torch.int32), wrong_mask, q_scale), good, key="out")
    zero_mask = torch.full_like(good_mask, -1e9)
    cases["valid_zero_mask"] = _outcome(lambda: decode(
        x[0:1], *weights, cache_k, cache_v,
        torch.tensor([8], device="cuda", dtype=torch.int32), zero_mask, q_scale), good, key="out")
    cases["skipped_cache_loc"] = _outcome(lambda: decode(
        x[0:1], *weights, cache_k, cache_v,
        torch.tensor([9], device="cuda", dtype=torch.int32), good_mask, q_scale), good, key="out")
    cases["cache_dtype_mismatch"] = _outcome(lambda: decode(
        x[0:1], *weights, cache_k.float(), cache_v, 
        torch.tensor([8], device="cuda", dtype=torch.int32), good_mask, q_scale), key="out")

    summary = {"cases": cases,
               "counts": {"total": len(cases),
                          "caught": sum(1 for c in cases.values() if c["caught"]),
                          "silent_wrong": sum(1 for c in cases.values() if c.get("silent") == "wrong_value"),
                          "silent_non_finite": sum(1 for c in cases.values() if c.get("silent") == "non_finite")}}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
