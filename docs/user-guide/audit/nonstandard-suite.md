# Phase 3 starter suite: non-standard subgraphs

This is the first slice of the Phase-3 workload suite. It is deliberately
small and reproducible before expanding to GQA/MLA/MoE variants.

| case | FlowGPU / eager (median, range) | FlowGPU / compiled |
|---|---|---|
| `residual_norm` | 0.98x (0.96–0.98) | 1.01x (1.00–1.01) |
| `moe_router_topk` | 0.99x (0.99–1.00) | 1.00x (1.00–1.01) |
| `routed_silu` | **0.78x** (0.77–0.79) | 1.00x (1.00–1.02) |
| `gqa_decode_masked` | not interpreted* | not interpreted* |
| `gqa_decode_large` | not interpreted* | not interpreted* |

All five pass correctness (max abs error <= 3e-2). Ratios are medians of three
randomly interleaved rounds x 8 iterations on a shared H100; ranges are across
rounds. *The GQA cases are correctness/contract coverage: their eager baseline
has different host overhead / shape, so the ratio is not a compiler result.

## Measurement protocol

- Randomize the order of FlowGPU, eager and compiled **inside each round**;
- run 3 rounds, report median and min/max ratio;
- keep correctness separate from timing;
- do not rank compilers from one shared session: within-session variance is
  small, but **session-level load variance remains** (an earlier run had
  `routed_silu` compiled 3x faster; this run has FlowGPU faster).

The stable output of this phase is the correctness/contract result, not a
performance leaderboard.

## What the suite already exposed

The first MoE run had equal routing **weights** but different `ids`: the
capped logits contained exact ties (`+0.5`), and top-k tie order is not a
specified contract. That is not a numerical error, but the suite must record
`ids` separately and compare them only when the contract says so.

The GQA baseline itself exposed the second methodology issue: a per-batch/per-head
Python loop made eager look ~9x slower than FlowGPU. Replacing it with a
vectorized einsum baseline removed that artifact. **Baseline implementation is
part of the result.**

A second methodology lesson: the five-case run produced different ordering of
FlowGPU vs `torch.compile` than the previous run, on the same shared device.
That is a measurement problem, not a compiler result. The suite now labels
ratios "indicative" and keeps correctness as the stable signal.

This is the kind of issue a "fastest kernel" benchmark would miss:

> **A non-standard subgraph suite needs a semantics column, not only latency.**

## Method

- FlowGPU: `compile_pipeline` with literal/symbolic bindings, first call
  warmup, p50 per call;
- eager: the same math in PyTorch;
- compiled: `torch.compile(eager)` where it compiles;
- correctness: max absolute error against eager, with `ids` treated as a
  separate contract.

Raw data:
[`nonstandard-suite-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/nonstandard-suite-20261125.json).
