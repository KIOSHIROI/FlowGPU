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

## Phase 2c: the audit artifact

The audit sheet exposes, for one compiled pipeline:

- inputs (name/dtype/shape);
- kernels (kind, fused ops, tile);
- fusion groups;
- generated host guards;
- caller-owned semantic risks;
- independent reference kernels;
- refusals/skips.

Coverage of the 13-phase fault matrix:

| kind | count | visible |
|---|---:|---:|
| direct host guard | 6 | 6/6 |
| semantic risk flag | 7 | 7/7 |

The risk flag is not a diagnosis: it names state the host does not check
(mask/cache consistency, cache values, `q_scale`, tie order). This is the
artifact's honest boundary.

Cost:

- building the sheet from an already compiled result: **~0.2%** of compile
  time on small CPU shapes;
- CPU reference: ~0.8–1.4x compile time on those shapes;
- host guard runtime: ~1.8% at decode (Phase 2b).

A user-study protocol and scoring/session harness are ready, but **no
participants have been run**.

## Phase 3: non-standard subgraph suite

Seven cases: fused residual+RMSNorm, MoE routing, fp16 routed SiLU, small masked
GQA, serving-shaped GQA, AWQ int4 dequant, and MLA-shaped latent attention. All
pass correctness.

Performance ratios were not stable across sessions, even on idle GPU0 with a
clock warm-up: `routed_silu` ranged 0.15x–0.78x of eager, `moe_router_topk`
0.73x–1.19x. The stable result is correctness: all seven cases pass. A
clock-locked, isolated run is required before any performance ranking.

**Meaning:** correctness and the guard/oracle boundary are stable.
**Not meaning:** a compiler performance ranking; the ratios are diagnostic.

### Measurement lessons

1. A looped GQA eager baseline made FlowGPU look 9x faster; a vectorized
   baseline removed the artifact.
2. On a shared GPU the ordering of FlowGPU and `torch.compile` flipped between
   sessions. Dedicated measurement is required for a ranking.
3. MoE top-k tie order is not a specified contract; weights and ids must be
   reported separately.
4. CUDA-event timing reduces host round-trip noise but does not remove
   clock/occupancy variance: large GQA ratios ranged 0.05x–0.49x across
   rounds. Clock locking + isolation is still required.

## Phase 2b: what do the guards cost?

Same subgraph, weights and dedicated GPU0; only `runtime_validation` changes
(7 interleaved rounds x 30 iterations):

| tokens | guarded / unguarded |
|---:|---:|
| 1 | 1.018x (1.013–1.025) |
| 2048 | 1.003x (0.89–1.05) |

Host guards cost about 1.8% at decode and are inside noise at prefill. This is
the **host guard** tax only; reference/oracle tests and compile-time checks are
outside it.

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
- guard-cost result is one subgraph;
- random weights in most performance runs;
- seven-case suite, not a model zoo;
- no user study of "auditability" with engineers (protocol + harness ready);
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
- device policy materially changes prefill performance;
- the host-guard audit tax is about 1.8% at decode on the measured subgraph;
- the audit sheet covers all 13 injected faults as 6 direct guards + 7 risk
  flags, and costs ~0.2% of compile time incrementally.

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
5. dedicated rerun with longer repetitions and clock lock;
6. include reference/oracle test time in the total auditability cost.
