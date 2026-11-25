# Phase 3 suite: non-standard subgraphs

Seven cases:

| case | coverage |
|---|---|
| `residual_norm` | fused add + RMSNorm |
| `moe_router_topk` | MoE routing, global softmax, tie-sensitive top-k |
| `routed_silu` | fp16 SiLU*mul + fp32 per-row scale |
| `gqa_decode_masked` | small masked GQA decode |
| `gqa_decode_large` | batch 4, 32q/8kv, depth 128, 256 slots |
| `awq_dequantize` | int4 AWQ dequant |
| `mla_latent_attention` | low-rank latent K/V (MLA-shaped) |

All seven pass correctness. `awq_dequantize` is correctness-only (tiny shape)
and matches `packed_layout.awq_dequantize` exactly (max abs error 0.0). The MLA
case is the latent-KV core, not the full DeepSeek decoupled-RoPE MLA.

## Performance is not yet stable

Three runs of the same suite — busy GPU1, idle GPU0, and idle GPU0 with a
500 ms clock warm-up — gave different ratios for the same code and weights:

| case | FlowGPU / eager across runs |
|---|---:|
| `residual_norm` | 0.39–0.98x |
| `moe_router_topk` | 0.73–1.19x |
| `routed_silu` | 0.15–0.78x |
| `gqa_decode_masked` | 0.10–0.36x |
| `gqa_decode_large` | 0.17–0.79x |
| `mla_latent_attention` | 0.34–0.65x |

The ordering and magnitude are **not** stable enough for a compiler ranking.
0% utilization in `nvidia-smi` is not a clock/isolation guarantee, and a
warm-up kernel did not remove the variance.

**Decision:** correctness/contract is the stable signal. Performance requires a
clock-locked, isolated run with longer repetitions; until then these ratios are
diagnostic only.

## MoE tie contract

FlowGPU's `topk_last` contract is explicit: **lower column wins on a tie**.
The first suite baseline used `torch.topk`, whose tie order is unspecified, so
equal routing weights produced different `ids`. Replacing the baseline with a
stable descending argsort gives `ids_equal=true` across runs.

This is the kind of issue a latency-only benchmark would miss: a non-standard
subgraph suite needs a semantics column, not only timing.

## Measurement protocol

- Randomize FlowGPU / eager / compiled order inside each round;
- run multiple rounds, report median and min/max;
- keep correctness separate from timing;
- record whether the device was idle before/after;
- do not merge shared-device runs with dedicated runs;
- treat cross-session variance as a first-class result until clocks are locked.

## Raw data

- [`nonstandard-suite-dedicated-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/nonstandard-suite-dedicated-20261125.json)
- [`nonstandard-suite-warm-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/nonstandard-suite-warm-20261125.json)
- earlier shared-device runs are kept in the same directory.
