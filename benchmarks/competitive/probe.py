#!/usr/bin/env python3
"""A CPU-only, reproducible first pass over the competitive case matrix.

This probe does not claim a latency win.  It records what can be compared
without a device and without a competitor installed:

* whether each public case compiles at the declared shape;
* compile wall time (a local machine measurement, not a portable benchmark);
* kernels, slots, materialized buffers and modeled traffic A;
* graph-pass decisions.

The device-latency and competitor columns are deliberately absent until a
runner with the same inputs, tolerance and timing definition is added.  That
separation is the point: structural parity and performance parity are different
claims.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flowgpu.compiler.backend import reference  # noqa: E402
from flowgpu.compiler.compile import compile_source  # noqa: E402
from benchmarks.competitive.oracles import ORACLES, compare  # noqa: E402


def load_cases(path: pathlib.Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "flowgpu.competitive.cases/1":
        raise ValueError("unsupported competitive case schema")
    return payload["cases"]


def _oracle_check(pipeline, case: dict) -> dict:
    name = case.get("oracle")
    if name is None or name not in ORACLES:
        return {"status": "not_available"}
    program = pipeline.reference
    sizes = dict(case["bindings"])
    try:
        inputs = reference.inputs_for(program, sizes)
        namespace = {}
        exec(compile(reference.module([program]), "<oracle>", "exec"), namespace)
        host = namespace["run_" + program.pipeline]
        tensors = [entry[3] for entry in inputs]
        actual = host(*tensors, *[sizes[d] for d in program.dimensions])
        values = [entry[3] for entry in inputs]
        expected = ORACLES[name](values, sizes)
        passed = True
        errors = {}
        for output_name, output_value in actual.items():
            if output_name not in expected:
                continue
            ok, error = compare(output_name, expected, output_value,
                                     atol=float(case.get("oracle_atol", 1e-4)))
            passed = passed and ok
            errors[output_name] = error
        return {"status": "pass" if passed else "fail", "oracle": name,
                "max_abs_error": errors}
    except Exception as error:
        return {"status": "error", "oracle": name,
                "error": type(error).__name__ + ": " + str(error)}


def run_case(case: dict) -> dict:
    source = (ROOT / case["source"]).read_text()
    started = time.perf_counter()
    compiled = compile_source(source, shape_bindings=case["bindings"])
    compile_ms = (time.perf_counter() - started) * 1000.0
    by_name = {item.pipeline.name: item for item in compiled.pipelines}
    if case["pipeline"] not in by_name:
        raise ValueError(f"{case['name']}: no pipeline {case['pipeline']!r}; "
                         f"found {sorted(by_name)}")
    pipeline = by_name[case["pipeline"]]
    plan = pipeline.planning.as_dict() if pipeline.planning is not None else None
    return {
        "name": case["name"],
        "source": case["source"],
        "bindings": case["bindings"],
        "compile_ms": round(compile_ms, 3),
        "plan": plan,
        "optimization": [
            {"name": record.name, "nodes_before": record.nodes_before,
             "nodes_after": record.nodes_after, "decisions": len(record.decisions)}
            for record in pipeline.optimization
        ],
        "triton_refusals": len(pipeline.triton.refusals) if pipeline.triton is not None else None,
        "oracle": _oracle_check(pipeline, case) if case.get("oracle") else {"status": "not_available"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(pathlib.Path(__file__).with_name("cases.json")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="compile every case and fail on the first error")
    args = parser.parse_args(argv)
    cases = load_cases(pathlib.Path(args.cases))
    results = []
    failures = 0
    for case in cases:
        try:
            results.append(run_case(case))
        except Exception as error:
            failures += 1
            results.append({"name": case.get("name"), "error": type(error).__name__ + ": " + str(error)})
    if args.json:
        print(json.dumps({"schema": "flowgpu.competitive.probe/1", "results": results}, indent=2,
                         ensure_ascii=False))
    else:
        for result in results:
            if "error" in result:
                print(f"{result['name']}: ERROR {result['error']}")
                continue
            plan = result["plan"]
            print(f"{result['name']}: compile={result['compile_ms']}ms "
                  f"kernels={plan['kernels']} slots={plan['slots']} "
                  f"A={plan['traffic_a']} refusals={result['triton_refusals']}")
    if args.check:
        oracle_failures = [
            entry["name"] for entry in results
            if entry.get("oracle", {}).get("status") in {"fail", "error"}
        ]
        if failures or oracle_failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
