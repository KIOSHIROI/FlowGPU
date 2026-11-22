#!/usr/bin/env python3
"""Repeat the M2 device matrix over several seeds and report medians.

A single seed is a smoke measurement, not a performance claim. This wrapper
keeps the exact device_runner protocol (same inputs per seed, same tolerance,
Graph and full-call separated) and aggregates per-backend medians; a correctness
failure in any seed is retained, not averaged away.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks" / "competitive" / "device_runner.py"


def load_seed(seed: int, *, backends: str, warmup: int, iterations: int) -> dict:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--json", "--seed", str(seed),
         "--backends", backends, "--warmup", str(warmup),
         "--iterations", str(iterations)],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)


def merge(reports: list[dict]) -> dict:
    cases = {}
    for report in reports:
        for case in report["cases"]:
            entry = cases.setdefault(case["name"], {"name": case["name"], "backend": {}})
            for backend, value in case["backend"].items():
                target = entry["backend"].setdefault(backend, {"full": [], "graph": [],
                                                               "compile": [], "passed": True})
                target["full"].append(value["full_us"])
                target["graph"].append(value.get("graph_us"))
                target["compile"].append(value.get("compile_ms", 0.0))
                target["passed"] = target["passed"] and value["correctness"]["passed"]
    merged = []
    for entry in cases.values():
        output = {"name": entry["name"], "backend": {}}
        for backend, values in entry["backend"].items():
            graph = [value for value in values["graph"] if value is not None]
            output["backend"][backend] = {
                "passed": values["passed"],
                "full_us_median": statistics.median(values["full"]),
                "graph_us_median": statistics.median(graph) if graph else None,
                "compile_ms_median": statistics.median(values["compile"]),
                "seeds": len(values["full"]),
            }
        merged.append(output)
    return {
        "schema": "flowgpu.competitive.multiseed/1",
        "environment": reports[0]["environment"],
        "protocol": dict(reports[0]["protocol"], seeds=len(reports),
                         aggregation="median across seeds"),
        "cases": merged,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--backends", default="flowgpu,flowgpu_prepared,eager,inductor,triton")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")
    reports = [load_seed(seed, backends=args.backends, warmup=args.warmup,
                         iterations=args.iterations) for seed in seeds]
    data = merge(reports)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for case in data["cases"]:
            print(case["name"])
            for backend, value in case["backend"].items():
                print(f"  {backend:16s} passed={value['passed']} "
                      f"full={value['full_us_median']:.2f}us "
                      f"graph={value['graph_us_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
