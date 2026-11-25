# Phase 3 starter suite: non-standard subgraphs

This is the first slice of the Phase-3 workload suite. It is deliberately
small and reproducible before expanding to GQA/MLA/MoE variants.

| case | shape | what is non-standard | FlowGPU / eager | FlowGPU / compiled |
|---|---|---|---|---|
| `residual_norm` | 2048x4096 | fused add + RMSNorm | **0.65x** | **0.67x** |
| `moe_router_topk` | 4x16, 8 experts | tanh cap + global softmax + top-k gather | 0.99x | 1.00x |
| `routed_silu` | 2048x14336 | fp16 SiLU*mul + fp32 per-row weight + cast | **0.78x** | **1.00x** |

All three pass the tolerance (max abs error <= 2e-2). Ratios are from one
shared-H100 session, 15 iterations, 2 repeats; absolute times moved with load.

## What the suite already exposed

The first MoE run had equal routing **weights** but different `ids`: the
capped logits contained exact ties (`+0.5`), and top-k tie order is not a
specified contract. That is not a numerical error, but the suite must record
`ids` separately and compare them only when the contract says so.

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
