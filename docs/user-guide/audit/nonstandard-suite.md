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

## Dedicated-GPU rerun

GPU0 was idle (0% utilization immediately before and after; GPU1 was busy), so
the same five cases were rerun with five randomized interleaved rounds x 20
iterations:

| case | FlowGPU / eager (range) | FlowGPU / compiled (range) |
|---|---|---|
| `residual_norm` | **0.40x** (0.39–0.42) | **0.87x** (0.86–0.89) |
| `moe_router_topk` | 1.19x (1.17–1.23) | **0.85x** (0.84–0.87) |
| `routed_silu` | **0.15x** (0.151–0.155) | 1.02x (1.01–1.03) |
| `gqa_decode_masked` | **0.34x** (0.34–0.36) | **0.32x** (0.32–0.33) |
| `gqa_decode_large` | **0.45x** (0.44–0.46) | **0.36x** (0.36–0.37) |

All pass correctness. This is the first table in the project that is stable
enough to discuss as a performance result, with three caveats:

- it is **one H100**, not a cross-device claim;
- it is five cases, not a general LLM workload;
- FlowGPU is **not universally faster**: MoE is 1.19x slower than eager, while
  it is 0.85x of `torch.compile`; `routed_silu` is at parity with compiled.

The shared-device numbers earlier in this page are kept as a negative result:
they changed ordering between sessions. Dedicated measurement is the only
basis for a ranking.

## Measurement protocol

- Randomize the order of FlowGPU, eager and compiled **inside each round**;
- run 5 rounds, report median and min/max ratio;
- keep correctness separate from timing;
- verify the device is idle before and after;
- do not merge shared-device runs with dedicated runs.
