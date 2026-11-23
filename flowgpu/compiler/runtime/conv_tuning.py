"""Shape-specific conv tile selection: compile a few candidates and measure.

The emitted conv tile extents are constexpr launch arguments, so the same
program can be compiled at several geometries. This utility measures those
variants on the caller's tensors and returns the fastest compiled run plus a
record of the candidates. It is a tuning helper, not a different lowering.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

DEFAULT_CANDIDATES = (
    # Small 1x1 shapes: the default emitter resolves BN/BK to the channel /
    # output extent (16 here), and a forced 32-wide tile is slower.
    (32, 16, 16),
    (32, 32, 32),
    (64, 32, 32),
    (64, 64, 32),
    (128, 64, 64),
    (128, 128, 64),
)


@dataclass(frozen=True)
class ConvTuningCandidate:
    block_m: int
    block_n: int
    block_k: int
    median_us: float
    passed: bool
    error: str = ""


@dataclass(frozen=True)
class ConvTuningResult:
    block_m: int
    block_n: int
    block_k: int
    median_us: float
    candidates: tuple
    compile_ms: float


def tune_conv2d(source, *, pipeline, args, cache_dir, validator,
                candidates=DEFAULT_CANDIDATES, iterations=20, warmup=5):
    """Compile and measure conv tile candidates; return ``(run, result)``."""
    import torch
    from dataclasses import replace as _replace
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    from flowgpu.compiler.runtime.runtime import compile_pipeline
    if not candidates:
        raise ValueError("at least one conv tile candidate is required")
    base = resolve_platform()
    measured = []
    best = None
    started = time.perf_counter()
    for block_m, block_n, block_k in candidates:
        config = _replace(base, conv_block_m=block_m, conv_block_n=block_n,
                          conv_block_k=block_k)
        try:
            run = compile_pipeline(source, pipeline=pipeline,
                                   cache_dir=f"{cache_dir}/conv-tile-{block_m}-{block_n}-{block_k}",
                                   platform_config=config)
            result = run(*args)
            torch.cuda.synchronize()
            if validator(result) is not True:
                raise ValueError("candidate output failed validation")
            for _ in range(warmup):
                run(*args)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                run(*args)
            end.record()
            torch.cuda.synchronize()
            median_us = start.elapsed_time(end) * 1000.0 / iterations
            candidate = ConvTuningCandidate(block_m, block_n, block_k, median_us, True)
            if best is None or median_us < best[1].median_us:
                best = (run, candidate)
        except Exception as error:  # a candidate may not compile on this shape
            candidate = ConvTuningCandidate(block_m, block_n, block_k, float("inf"),
                                            False, f"{type(error).__name__}: {error}")
        measured.append(candidate)
    if best is None:
        raise ValueError("no conv tile candidate compiled and validated")
    compile_ms = (time.perf_counter() - started) * 1000.0
    run, candidate = best
    return run, ConvTuningResult(candidate.block_m, candidate.block_n, candidate.block_k,
                                 candidate.median_us, tuple(measured), compile_ms)


__all__ = ["DEFAULT_CANDIDATES", "ConvTuningCandidate", "ConvTuningResult", "tune_conv2d"]
