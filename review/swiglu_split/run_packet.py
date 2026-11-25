"""Run the review packet: reference vs generated vs handwritten."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def p50(fn, iterations=20):
    fn(); torch.cuda.synchronize(); values = []
    for _ in range(iterations):
        started = time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1e6)
    return statistics.median(values)


def main() -> int:
    spec = json.loads((ROOT / "inputs.json").read_text())
    torch.manual_seed(spec["seed"])
    rows, width = spec["rows"], spec["width"]
    x = torch.randn(rows, width, device="cuda", dtype=torch.float32)
    alpha = torch.tensor([[0.5]], device="cuda")
    beta = torch.tensor([[0.1]], device="cuda")
    limit = torch.tensor([[1.5]], device="cuda")
    reference = load("reference.py", "reference").swiglu_oai_split
    generated = load("generated_module.py", "generated").run_swiglu_oai_split
    handwritten = load("handwritten.py", "handwritten").run_handwritten
    expected = reference(x, alpha, beta, limit)
    got_generated = generated(x, alpha, beta, limit)["out"]
    got_handwritten = handwritten(x, alpha, beta, limit)
    torch.cuda.synchronize()
    result = {
        "generated_max_abs": float((got_generated - expected).abs().max()),
        "handwritten_max_abs": float((got_handwritten - expected).abs().max()),
        "generated_us": p50(lambda: generated(x, alpha, beta, limit)),
        "handwritten_us": p50(lambda: handwritten(x, alpha, beta, limit)),
        "manifest": json.loads((ROOT / "manifest.json").read_text()),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["generated_max_abs"] <= 1e-6 and result["handwritten_max_abs"] <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
