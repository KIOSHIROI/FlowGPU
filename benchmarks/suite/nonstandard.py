"""Phase-3 starter suite: three non-standard LLM subgraphs, eager + compile."""
from __future__ import annotations

import argparse
import json
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


CASES = (
    Case("residual_norm", ROOT / "examples/residual_norm.fgpu", "residual_norm",
         {"N": 2048, "D": 4096}, _residual_inputs, _residual_eager, ("z",), 2e-2),
    Case("moe_router_topk", ROOT / "examples/batch10/moe_router_topk.fgpu", "moe_router_topk",
         {}, _moe_inputs, _moe_eager, ("weights",), 2e-2, "ids"),
    Case("routed_silu", ROOT / "examples/batch10/routed_silu.fgpu", "routed_silu",
         {"M": 2048, "N": 14336}, _routed_inputs, _routed_eager, ("out",), 2e-2),
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
