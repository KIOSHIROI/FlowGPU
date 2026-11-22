#!/usr/bin/env python3
"""Compare one symbolic compile with per-shape specialization on a CUDA host.

For each program: compile once without shape_bindings, run every shape; then
compile one module per shape with shape_bindings. Report compile time, full-call
latency and correctness for both. This is a capability probe, not a leaderboard.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROGRAMS = [
    {"name": "vectoradd", "source": "examples/gpu_mode/vectoradd.fgpu",
     "pipeline": "vectoradd", "shapes": [[128, 128], [64, 64], [17, 19]],
     "inputs": ["a", "b"], "dtype": "fp16", "oracle": "add"},
    {"name": "matmul", "source": "examples/gpu_mode/matmul.fgpu",
     "pipeline": "matmul", "shapes": [[64, 128, 64], [32, 64, 32]],
     "inputs": ["a", "b"], "dtype": "fp16", "oracle": "matmul"},
    {"name": "softmax", "source": "examples/batch8/wide_row_softmax.fgpu",
     "pipeline": "wide_plain", "shapes": [[8, 1024], [4, 512]],
     "inputs": ["x"], "dtype": "fp32", "oracle": "softmax"},
]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit("symbolic probe needs PyTorch") from error
    if not torch.cuda.is_available():
        raise SystemExit("symbolic probe needs CUDA; use --check on CPU")
    return torch


def _inputs(torch, spec, shape):
    dtype = getattr(torch, {"fp16": "float16", "fp32": "float32"}[spec["dtype"]])
    if spec["oracle"] == "matmul":
        m, k, n = shape
        return (torch.randn(m, k, device="cuda", dtype=dtype),
                torch.randn(k, n, device="cuda", dtype=dtype))
    return tuple(torch.randn(*shape, device="cuda", dtype=dtype)
                 for _ in spec["inputs"])


def _torch_call(torch, spec, args, shape):
    if spec["oracle"] == "add":
        return {"out": args[0] + args[1]}
    if spec["oracle"] == "matmul":
        return {"out": args[0] @ args[1]}
    if spec["oracle"] == "softmax":
        return {"q": torch.softmax(args[0], dim=-1)}
    raise ValueError(spec["oracle"])


def _event_ms(torch, fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def run_symbolic(torch, spec):
    from flowgpu.compiler.runtime.symbolic_shape import SymbolicShapeRunner
    source = (ROOT / spec["source"]).read_text()
    import time
    started = time.perf_counter()
    runner = SymbolicShapeRunner(source, spec["pipeline"],
                                 str(ROOT / ".symbolic-cache"))
    compile_ms = (time.perf_counter() - started) * 1000.0
    rows = []
    for dimensions in spec["shapes"]:
        shape = tuple(dimensions)
        args = _inputs(torch, spec, shape)
        expected = _torch_call(torch, spec, args, shape)
        actual = runner(*args)
        key = "q" if spec["oracle"] == "softmax" else "out"
        error = float((actual[key].double() - expected[key].double()).abs().max())
        full = _event_ms(torch, lambda: runner(*args))
        rows.append({"shape": dimensions, "max_abs": error, "full_us": full})
    return {"compile_ms": compile_ms, "shapes": rows}


def run_per_shape(torch, spec):
    from flowgpu.compiler.runtime.runtime import compile_pipeline
    source = (ROOT / spec["source"]).read_text()
    rows = []
    for dimensions in spec["shapes"]:
        shape = tuple(dimensions)
        bindings = ({name: value for name, value in
                     zip(("M", "K", "N"), dimensions)} if spec["oracle"] == "matmul"
                    else {"M": dimensions[0], "N": dimensions[1]} if spec["oracle"] == "add"
                    else {"B": dimensions[0], "V": dimensions[1]})
        import time
        started = time.perf_counter()
        run = compile_pipeline(source, pipeline=spec["pipeline"],
                               cache_dir=str(ROOT / ".symbolic-cache"),
                               shape_bindings=bindings)
        compile_ms = (time.perf_counter() - started) * 1000.0
        args = _inputs(torch, spec, shape)
        expected = _torch_call(torch, spec, args, shape)
        actual = run(*args)
        key = "q" if spec["oracle"] == "softmax" else "out"
        error = float((actual[key].double() - expected[key].double()).abs().max())
        full = _event_ms(torch, lambda: run(*args))
        rows.append({"shape": dimensions, "max_abs": error, "compile_ms": compile_ms,
                     "full_us": full})
    return {"shapes": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        from flowgpu.compiler.compile import compile_source
        for spec in PROGRAMS:
            compiled = compile_source((ROOT / spec["source"]).read_text())
            by_name = {item.pipeline.name: item for item in compiled.pipelines}
            if spec["pipeline"] not in by_name:
                raise SystemExit("missing pipeline " + spec["pipeline"])
        print(f"symbolic probe matrix ok: {len(PROGRAMS)} programs")
        return 0
    torch = _torch()
    report = {"schema": "flowgpu.competitive.symbolic/1",
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(0)},
              "programs": []}
    for spec in PROGRAMS:
        report["programs"].append({"name": spec["name"],
                                   "symbolic": run_symbolic(torch, spec),
                                   "per_shape": run_per_shape(torch, spec)})
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for program in report["programs"]:
            print(program["name"], "symbolic compile",
                  round(program["symbolic"]["compile_ms"], 1), "ms")
            for row in program["symbolic"]["shapes"]:
                print("  ", row["shape"], "max_abs", row["max_abs"],
                      "full_us", round(row["full_us"], 2))
            for row in program["per_shape"]["shapes"]:
                print("  per-shape", row["shape"], "compile_ms",
                      round(row["compile_ms"], 1), "full_us", round(row["full_us"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
