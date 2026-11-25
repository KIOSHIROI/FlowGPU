# FlowGPU research report

## Thesis

> Can LLM inference subgraphs be compiled with **auditable evidence** while
> staying performance-competitive with vendor libraries?

FlowGPU is the artifact. The research contribution is not "another compiler";
it is an empirical answer about the cost and boundary of auditability.

## Artifacts

- `.fgpu` subgraph DSL with shape/dtype/precision contracts;
- native Triton kernels plus an independent CPU reference;
- generated host guards and prepared/graph/custom-op execution;
- trained-tiny-model parity harness;
- 13-case fault-injection audit;
- 6-case non-standard subgraph suite;
- dedicated-device measurement protocol.

## Phase 1: semantic parity

A tiny one-layer block (D=64, H=128, vocab=128) is trained on a fixed 8-token
pattern, then PyTorch and FlowGPU both generate greedy tokens through
embedding -> cached prefill -> masked decode -> LM head -> sampling.

| item | bf16 | fp32 control |
|---|---:|---:|
| generated tokens | `[0,0,0,0,7]` both | `[0,0,0,0,7]` both |
| token match | **true** | **true** |
| prefill hidden max diff | 0.09375 | 2.9e-6 |
| prefill K max diff | 0.0234 | 7.2e-7 |

**Meaning:** RoPE, RMSNorm, attention scale, cached prefix, causal mask, cache
layout, LM head and sampling agree with an independent implementation.
**Not meaning:** language quality or speed.

## Phase 2: what do guards catch?

Thirteen injected faults on an MLP and a masked decode host:

| class | count | caught at launch | silent | oracle |
|---|---:|---:|---:|---:|
| structural + cache loc | 6 | 6 | 0 | - |
| observable semantic | 5 | 0 | 5 | 5/5 |
| no observable effect | 2 | 0 | - | - |

Caught: shape, dtype, layout, weight/cache dtype, `loc` range.
Silent: missing mask, all-masked mask, skipped/duplicate cache position, short
mask. Oracle catches all five observable semantic faults.

**Finding:** guards cover declarations, not state invariants.
**Negative result:** two injections had no observable bf16 effect; they are not
counted as silent errors. Fault-injection studies must distinguish "not
caught" from "not effective".

## Phase 3: non-standard subgraph suite

Six cases: fused residual+RMSNorm, MoE routing, fp16 routed SiLU, small masked
GQA, serving-shaped GQA, and AWQ int4 dequant. All pass correctness.

Dedicated H100 GPU0 (0% util before/after), 5 randomized interleaved rounds:

| case | FlowGPU / eager | FlowGPU / compiled |
|---|---:|---:|
| residual_norm | 0.40x | 0.87x |
| moe_router_topk | 1.19x | 0.85x |
| routed_silu | 0.15x | 1.02x |
| gqa_decode_masked | 0.34x | 0.32x |
| gqa_decode_large | 0.45x | 0.36x |

**Meaning:** on these five cases, FlowGPU is competitive and often faster, but
**not universally**: MoE is slower than eager.
**Not meaning:** general LLM performance or a cuBLAS win.

### Measurement lessons

1. A looped GQA eager baseline made FlowGPU look 9x faster; a vectorized
   baseline removed the artifact.
2. On a shared GPU the ordering of FlowGPU and `torch.compile` flipped between
   sessions. Dedicated measurement is required for a ranking.
3. MoE top-k tie order is not a specified contract; weights and ids must be
   reported separately.

## Phase 4: device-policy contribution

Same subgraph, weights and device; only the tile policy changes:

| tokens | safe L4 policy / H100 policy | H100 policy / eager |
|---:|---:|---:|
| 1 | 1.027x | 1.009x |
| 2048 | **1.163x** | 1.063x |

The device-aware policy is worth 2.7% at decode and 16.3% at prefill on H100.
This quantifies the cost of one global policy; it does **not** prove L4
behavior.

## Threats to validity

- single H100, no L4 evidence;
- random weights in most performance runs;
- six-case suite, not a model zoo;
- no user study of "auditability" with engineers;
- no full-model serving measurement;
- fault matrix is hand-designed, not exhaustive;
- tiny trained model is a synthetic pattern.

## What the evidence supports

- FlowGPU can match an independent implementation token-for-token on a trained
  tiny model;
- it can catch structural faults before launch and exposes the remaining
  semantic boundary;
- on a dedicated H100, a small non-standard suite is competitive, with at least
  one loss (MoE vs eager);
- device policy materially changes prefill performance.

## What it does not support

- "FlowGPU beats cuBLAS" in general;
- "FlowGPU is production serving-ready";
- "auditable" as formal verification;
- cross-device performance claims.

## Next experiments

1. L4 / second-device portability with the same policy protocol;
2. MLA-shaped latent attention and a larger quant case;
3. end-to-end trained model on a real tokenizer, not a synthetic pattern;
4. user study: can engineers use the audit trail to localize a fault faster?
5. dedicated rerun with longer repetitions and clock lock.
