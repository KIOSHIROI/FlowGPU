#!/usr/bin/env python3
"""M2 device runner: same inputs, tolerance, Graph and full-call separation.

Requires a CUDA device, PyTorch and Triton.  The CPU-only ``--check`` validates
the case matrix without importing torch; CI therefore runs that form.

Backends:
* ``flowgpu``  -- FlowGPU-generated Triton host;
* ``eager``    -- dtype-faithful PyTorch expression;
* ``inductor`` -- ``torch.compile`` (TorchInductor, Triton backend).

A separate handwritten-Triton baseline can be added without changing the
correctness/timing protocol.  Graph and full-call are intentionally separate:
Graph excludes host guards/allocation/dispatch, full-call includes them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "int32": "int32"}


@dataclass(frozen=True)
class DeviceCase:
    name: str
    source: str
    pipeline: str
    bindings: dict
    inputs: tuple
    output: str
    atol: float
    rtol: float


def load_cases(path: pathlib.Path) -> list[DeviceCase]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "flowgpu.competitive.cases/1":
        raise ValueError("unsupported competitive case schema")
    cases = []
    for case in payload["cases"]:
        cases.append(DeviceCase(
            name=case["name"], source=case["source"], pipeline=case["pipeline"],
            bindings=dict(case.get("bindings", {})),
            inputs=tuple((name, tuple(shape), dtype)
                         for name, shape, dtype in case["inputs"]),
            output=case["output"], atol=float(case.get("atol", 1e-4)),
            rtol=float(case.get("rtol", 1e-4)),
        ))
    return cases


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - device-only path
        raise SystemExit("M2 device runner needs PyTorch: " + str(error)) from error
    if not torch.cuda.is_available():
        raise SystemExit("M2 device runner needs CUDA; use --check on a CPU-only host")
    return torch


def make_inputs(case: DeviceCase, *, seed: int = 0):
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = []
    for _, shape, dtype in case.inputs:
        torch_dtype = getattr(torch, DTYPES[dtype])
        if dtype == "int32":
            tensor = torch.randint(-8, 9, shape, generator=generator, dtype=torch.int32)
        else:
            tensor = torch.randn(shape, generator=generator).to(torch_dtype)
        tensors.append(tensor.cuda())
    return tuple(tensors)


def _as_dict(outputs, output_name: str):
    if isinstance(outputs, dict):
        return outputs
    if isinstance(outputs, (tuple, list)) and len(outputs) == 1:
        return {output_name: outputs[0]}
    return {output_name: outputs}


def _correctness(case: DeviceCase, actual, expected, torch):
    if actual is None:
        return {"passed": False, "reason": "no output"}
    values = _as_dict(actual, case.output)
    if case.output not in values:
        return {"passed": False, "reason": f"output {case.output!r} missing"}
    got = values[case.output].double()
    want = expected[case.output].double()
    if got.shape != want.shape:
        return {"passed": False, "reason": f"shape {tuple(got.shape)} != {tuple(want.shape)}"}
    diff = (got - want).abs()
    if case.name == "mapping_pointwise":
        exact = bool(torch.equal(got, want))
        return {"passed": exact, "max_abs": float(diff.max()), "exact": exact}
    max_abs = float(diff.max()) if diff.numel() else 0.0
    tolerance = case.atol + case.rtol * float(want.abs().max()) if want.numel() else case.atol
    return {"passed": max_abs <= tolerance, "max_abs": max_abs,
            "tolerance": tolerance}


def _event_ms(torch, fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _graph_ms(torch, fn, *, warmup: int, iters: int) -> float | None:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters
    except Exception:
        return None


def _flowgpu_runner(case: DeviceCase, args, *, cache_dir: pathlib.Path):
    from flowgpu.compiler.compile import compile_source
    from flowgpu.compiler.runtime.runtime import compile_pipeline
    source = (ROOT / case.source).read_text()
    started = time.perf_counter()
    compiled = compile_source(source, shape_bindings=case.bindings)
    by_name = {item.pipeline.name: item for item in compiled.pipelines}
    if case.pipeline not in by_name:
        raise ValueError(f"no pipeline {case.pipeline!r}")
    run = compile_pipeline(source, pipeline=case.pipeline, cache_dir=str(cache_dir),
                           shape_bindings=case.bindings)
    compile_ms = (time.perf_counter() - started) * 1000.0
    def call():
        return _as_dict(run(*args), case.output)
    return call, compile_ms


def _eager_runner(case: DeviceCase, args):
    from benchmarks.competitive.torch_oracles import eager
    return lambda: eager(case.name, args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(pathlib.Path(__file__).with_name("cases.json")))
    parser.add_argument("--check", action="store_true", help="validate the matrix without torch/CUDA")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backends", default="flowgpu,eager,inductor")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(ROOT / ".competitive-cache"))
    args = parser.parse_args(argv)
    cases = load_cases(pathlib.Path(args.cases))
    if args.list:
        for case in cases:
            print(case.name, case.source, case.pipeline)
        return 0
    if args.check:
        for case in cases:
            if not (ROOT / case.source).exists():
                raise SystemExit("missing source: " + case.source)
            if case.output not in {"out", "q", "z"}:
                raise SystemExit("unexpected output name: " + case.output)
        print(f"competitive device matrix ok: {len(cases)} cases")
        return 0

    if args.device != "cuda":
        raise SystemExit("M2 runner currently supports --device cuda only")
    torch = _torch()
    from benchmarks.competitive.torch_oracles import eager, oracle
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    cache_dir = pathlib.Path(args.cache_dir)
    report = {
        "schema": "flowgpu.competitive.device/1",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "protocol": {"warmup": args.warmup, "iterations": args.iterations,
                     "graph": "CUDA Graph replay; excludes host guards/allocation/dispatch",
                     "full_call": "guarded host call including allocation and dispatch",
                     "oracle": "float64 torch oracle; per-case atol/rtol"},
        "cases": [],
    }
    for case in cases:
        tensors = make_inputs(case, seed=args.seed)
        expected = oracle(case.name, tensors)
        entry = {"name": case.name, "backend": {}}
        runners = {}
        if "flowgpu" in backends:
            try:
                runners["flowgpu"] = _flowgpu_runner(case, tensors, cache_dir=cache_dir)
            except Exception as error:
                entry["flowgpu_error"] = type(error).__name__ + ": " + str(error)
        if "eager" in backends:
            prepared = _eager_runner(case)
            runners["eager"] = (prepared(tensors), 0.0)
        if "inductor" in backends:
            eager_fn = _eager_runner(case)
            def fn(*inner):
                return eager_fn(inner)(*inner) if False else eager(case.name, inner)
            started = __import__("time").perf_counter()
            compiled = torch.compile(lambda *a: eager(case.name, a))
            compiled(*tensors)
            compile_ms = (time.perf_counter() - started) * 1000.0
            runners["inductor"] = (lambda: compiled(*tensors), compile_ms)
        for name, (call, compile_ms) in runners.items():
            actual = call()
            correctness = _correctness(case, actual, expected, torch)
            full = _event_ms(torch, call, warmup=args.warmup, iters=args.iterations)
            graph = _graph_ms(torch, call, warmup=args.warmup, iters=args.iterations)
            entry["backend"][name] = {
                "correctness": correctness,
                "graph_us": graph,
                "full_us": full,
                "compile_ms": compile_ms,
            }
        report["cases"].append(entry)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for entry in report["cases"]:
            print(entry["name"])
            for name, data in entry["backend"].items():
                print(f"  {name:9s} correct={data['correctness']['passed']} "
                      f"graph={data['graph_us']}us full={data['full_us']:.2f}us "
                      f"compile={data['compile_ms']:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
