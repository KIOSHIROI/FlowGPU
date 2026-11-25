# Changelog

## 1.38.0

- Add Lesson 6 (lessons/06-materialized): same SwiGLU source compiles to 3
  fused or 4 materialized kernels; both CPU references agree. Dataflow card,
  runner, exercise and quiz.

## 1.37.0

- Add Lesson 5 (lessons/05-fusion): SwiGLU as 3 kernels; gate/hidden are
  materialized, up/act fuse into k1 epilogue. CPU runner, dataflow card,
  exercise and quiz.

## 1.36.0

- Add Lesson 4 (lessons/04-matmul): K as the reduction axis, output tile,
  CPU-only runner, dataflow card, exercise and quiz.

## 1.35.0

- Add Lesson 3 (lessons/03-reduction): sum_last + broadcast_like + mul in
  one primitive kernel; CPU-only runner, dataflow card, exercise and quiz.

## 1.34.0

- Add Lesson 2 (lessons/02-layout): split/interleave/concat as different
  address functions. One primitive kernel; no a/b buffers. Dataflow card,
  CPU-only runner, exercise and six-question quiz.

## 1.33.0

- Add Lesson 1 (lessons/01-values): values, shapes and explicit broadcast.
  One fused kernel; CPU-only compile + reference; dataflow card, exercise and
  five-question quiz.

## 1.32.0

- Add a dataflow learning card to the SwiGLU review packet: shape flow,
  address expressions, buffering, tail masks and four exercises. Entry is now
  dataflow -> reference -> implementations, not compiler internals.

## 1.31.0

- Add the SwiGLU split review packet: PyTorch reference, FlowGPU-generated
  Triton, handwritten two-kernel Triton, inputs and manifest with source map.
  GPU execution pending remote availability.

## 1.30.0

- Add flowgpu.compiler.source_map: kernel -> DSL statement lines, with the
  first explicit split gate/up correlation. Statement-level; line-level
  address mapping remains future work.

## 1.29.0

- Make the fused/materialized comparison CPU-first: the audit CLI accepts
  shape bindings and --materialize-all, so the kernel split can be inspected
  without a GPU.

## 1.28.0

- Add materialize_all=True: one SwiGLU source compiles to 3 fused kernels or
  4 kernels with the hidden tensor materialized. CPU reference matches; H100
  correctness matches; timing is pending an idle device.

## 1.27.0

- Add optional state invariants for cache position continuity, valid range and
  mask prefix; they reject the five observable semantic faults the generated
  guards miss. Caller-invoked, not an automatic compiler check.

## 1.26.0

- Add Phase 2c to the research report: audit-sheet coverage (13/13 as 6 direct
  + 7 risk flags), incremental cost (~0.2% of compile), reference cost and the
  unrun user-study status.

## 1.25.0

- Extend audit risk flags to cache values and q_scale; check all 13 Phase-2
  faults against the audit sheet: 6 direct guards + 7 risk flags = 13/13.

## 1.24.0

- Add StudySession: collect per-participant answers and emit both/root/stage
  accuracy, median time and per-task responses. Protocol remains unrun.

## 1.23.0

- Add audit_from_result(), avoiding recompilation when building the audit
  sheet; measure its incremental cost at ~0.2% of compile time on small
  CPU-shape cases and the CPU-reference cost separately.

## 1.22.0

- Add the user-study scoring harness: root-cause keyword and stage accuracy
  are scored separately; fixtures carry scoring keywords.

## 1.21.0

- Extend the audit sheet with input declarations and caller-owned semantic
  risks; all 6 user-study tasks now have evidence (3 direct guards, 3 risk
  flags), while mask/cache values remain undiagnosed.

## 1.20.0

- Add an audit-coverage check for the six user-study faults: 3/3 guard tasks
  and 1/3 semantic tasks are visible in the audit sheet; mask/cache state is
  not. Document why an oracle remains necessary.

## 1.19.0

- Add six user-study task fixtures (3 host-guard, 3 semantic) with reference
  root cause, stage and audit evidence; protocol remains unrun.

## 1.18.0

- Add flowgpu.compiler.audit: a compiled-artifact audit sheet with kernels,
  fusion groups, host guards, reference kernels and refusals.
- Add the pre-registered user-study protocol (not yet run).

## 1.17.0

- Add CUDA-event timing to the suite; absolute times improve but ratios still
  vary by up to an order of magnitude on large launch-bound cases.
- Record the negative result: no timing mode tested replaces clock locking and
  isolation; correctness remains the stable signal.

## 1.16.0

- Fix the MoE reference to a stable descending argsort, honoring the
  documented lower-column tie contract; ids now match across runs.
- Report that suite performance ratios are unstable across sessions even on
  idle GPU0 with a warm-up; correctness is the stable signal.

## 1.15.0

- Update the research report with the 7-case suite, MLA-shaped latent
  attention and the 1.8% host-guard cost; refresh threats/next experiments.

## 1.14.0

- Measure generated guard overhead on dedicated GPU0: 1.018x guarded /
  unguarded at decode; prefill within noise. Auditability tax is small on the
  launch-bound path.

## 1.13.0

- Add mla_latent_attention to the Phase-3 suite: low-rank latent K/V, masked
  attention, SwiGLU; correctness passes and it is 0.42x eager / 0.34x compiled
  on dedicated H100.

## 1.12.0

- Add a consolidated research report covering the thesis, Phase 1-4
  experiments, negative results, threats to validity and next experiments.

## 1.11.0

- Add awq_dequantize to the Phase-3 suite as quant pre/post coverage; it
  matches packed_layout.awq_dequantize exactly (max abs error 0.0).

## 1.10.0

- Add a controlled device-policy experiment on dedicated H100 GPU0: the
  L4-safe tile is 1.027x slower at decode and 1.163x slower at prefill than
  the H100 device policy, with everything else held fixed.

## 1.9.0

- Add a dedicated-GPU0 run of the Phase-3 suite: all five cases pass; FlowGPU
  is 0.40x/0.15x/0.34x/0.45x of eager on residual/routed/masked-GQA/large-GQA,
  but 1.19x on MoE, and 0.85x of compiled on MoE.

## 1.8.0

- Add an interleaved multi-round measurement protocol to the Phase-3 suite:
  randomized order per round, median and min/max ratios.
- Document that within-session variance is controlled but session-level load
  variance remains; GQA ratios are not interpreted as compiler results.

## 1.7.0

- Add a serving-shaped GQA decode case (batch 4, 32q/8kv, depth 128,
  256 slots) to the Phase-3 suite; correctness passes.
- Mark shared-device ratios as indicative: ordering changed between runs, so
  compiler ranking needs a dedicated-GPU rerun.

## 1.6.0

- Add the masked GQA decode block to the Phase-3 suite (correctness/contract
  coverage; tiny shape, not a performance claim).
- Replace the looped GQA eager baseline with a vectorized einsum baseline;
  document that baseline implementation is part of the measurement.

## 1.5.0

- Add the Phase-3 starter suite: residual_norm, moe_router_topk and
  routed_silu, with eager and torch.compile baselines.
- All pass the tolerance; the MoE run exposed that top-k tie order is not a
  specified contract and must be recorded separately from weights.

## 1.4.0

- Expand the fault matrix to 13 cases: 6 caught at launch, 5 observable
  semantic faults silent under guards but caught by an oracle; 2 injections
  had no observable effect and are excluded.

## 1.3.0

- Add a host range guard for cache_write_hsd loc; the fault matrix now catches
  6/9 cases, with 3 remaining silent state-semantic faults.
- Document the updated audit boundary: declaration + loc range are guarded;
  mask/prefix consistency still needs an oracle.

## 1.2.0

- Add a Phase-2 fault-injection experiment: 5/5 structural faults caught,
  4/4 semantic faults silent (cache loc, mask, skipped position).
- Document the auditability boundary: guards cover declarations, not state
  invariants; reference/oracle tests are required for the latter.

## 1.1.0

- Add a trained tiny-model parity experiment: PyTorch and FlowGPU generate
  the same 5 tokens; bf16 hidden diff is 0.094 and the fp32 control is 2.9e-6.
- This is semantic/correctness evidence, not language quality or speed.

## 1.0.0

- Add examples/prefill_decode_generate.py: prompt -> cached prefill -> LM head ->
  greedy sampling -> masked decode -> next token, on one paged cache.
- H100 run: prompt [0..7], decode locations [8..11], generated ids inside the
  vocabulary. This is the integration milestone, not a full serving engine or
  a trained model.

## 0.99.0

- Add examples/prefill_decode_loop.py: one 32-slot PagedKVCache written by
  cached prefill (slots 0..7) then masked decode (8..11).
- Add llama3_decode_block_masked32.fgpu for the shared 32-slot cache shape.

## 0.98.0

- Add cached_attention: causal attention over a cached prefix plus the new
  query block, with valid/prefix counts.
- Add llama3_prefill_block_cached.fgpu; invalid future slots do not contribute
  (bit-identical output in the H100 smoke test).

## 0.97.0

- Add window_attention: native masked attention with a per-batch valid-key
  count (4-D) or a scalar count (2-D/3-D).
- Add llama3_decode_block_gqa_masked.fgpu; invalid slots no longer contribute
  (bit-identical output in the H100 smoke test). The batched loop uses it.

## 0.96.0

- Add llama3_decode_block_masked.fgpu: explicit masked attention with host
  mask and q_scale. Invalid cache slots no longer contribute (bit-identical
  output in the smoke test).
- decode_loop / generate_loop use the masked block; the unmasked example now
  requires all cache slots to hold real tokens.

## 0.95.0

- Add examples/generate_loop.py: embedding -> cached decode block -> lm_head ->
  greedy sampling -> next token, with PagedKVCache + BlockTable.
- H100 run: 8 steps, locations 0..7, all sampled ids inside the vocabulary.

## 0.94.0

- Add examples/decode_loop_batched.py: DynamicBatch + per-request PagedKVCache
  and BlockTable over the batched GQA decode block; add/remove requests and
  finite outputs verified on H100.

## 0.93.0

- Add examples/decode_loop.py: BlockTable + PagedKVCache driving the cached
  decode block for 8 steps; all slots written and output finite on H100.

## 0.92.0

- Add DynamicBatch: fixed-capacity request slots, padded steps and active-only
  outputs over a compiled fixed-shape decode step.

## 0.91.0

- Add BlockTable: request -> blocks mapping, per-token append, flat locations
  and release over PagedKVCache.

## 0.90.0

- Add llama3_prefill_block.fgpu: 8-token causal prefill with contiguous KV
  range write and the full MLP/residual block (single-head, batch 1).
- H100 smoke test writes slots loc..loc+7 and leaves the rest zero.

## 0.89.0

- Add llama3_decode_block_gqa.fgpu: batch 2, 4 query heads, 2 KV heads,
  depth 64, 8 slots; cache_write_hsd_batch plus full MLP/residual block.
- H100 smoke test writes each batch's selected slot and leaves other slots zero.

## 0.88.0

- Add llama3_decode_block.fgpu: RMSNorm, QKV, RoPE, cache write, attention,
  output projection, residual, SwiGLU MLP and residual in one cached decode
  subgraph (single-head, batch 1).

## 0.87.0

- Add PagedKVCache: host block allocator with block reuse, zeroing, page-bound
  checks and flat locations for cache-aware pipelines.

## 0.86.0

- Add inference-only token selection helpers: greedy and temperature/top-k
  sampling over the last axis.

## 0.85.0

- Re-run the full 30-case M2 matrix after the device-aware GEMM policy:
  0 correctness failures across flowgpu, prepared, graph and eager.

## 0.84.0

- Add LayerStack(residual=True): host-level residual add outside the compiled
  MLP, with the two alternating buffer sets. 32-layer timing 0.99x eager in
  one session; no correctness claim because the synthetic bf16 stack goes
  non-finite.

## 0.83.0

- Add LayerStack: one prepared plan reused across layers with two alternating
  buffer sets and no per-layer copy. 4-layer real MLP: 0.995x eager.
- 32-layer synthetic timing is noisy and its scaled bf16 stack underflows; no
  32-layer correctness is claimed.

## 0.82.0

- Add a stacked-MLP benchmark: one prepared plan reused across 4 layer weight
  sets, 1.007x eager at T=1. Document the host buffer ping-pong requirement
  and why a static graph is not the reuse path when weights change.

## 0.81.0

- Add register_torch_op(): expose a compiled FlowGPU subgraph as a PyTorch
  custom op while the host owns weights and scheduling.
- Verified on the real Llama 3 MLP (relative error 1.7e-3).

## 0.80.0

- compile_pipeline() auto-resolves the device GEMM policy like prepared plans.
- Real Llama 3 MLP default path: decode 0.99x eager, prefill 0.98x; prepared
  decode 0.99x, prefill 0.97x in one shared session.

## 0.79.0

- H100 small-M tile uses 32x64x128 / 8 warps / 4 stages: prepared real
  Llama 3 decode 0.99x eager, graph 0.99x, prepared prefill 0.96x.

## 0.78.0

- Small-M GEMM tile uses 8 warps / 4 stages: real Llama 3 decode drops from
  1.05x to 1.01x eager; prepared prefill measures 0.95x in one shared session.

## 0.77.0

- Clamp the device 128x256 tile to literal M/N powers of two; 64-column GEMMs
  no longer launch mostly-masked blocks. Prepared real MLP: prefill 0.98x
  eager in one session, decode 1.05x; shared-device observation.

## 0.76.0

- Prepared execution plans auto-select resolve_device_platform(): H100 uses
  128x256, L4 keeps 128x128. In one session prefill prepared/eager was 0.94x
  (decode 1.04x); shared-device observation, not a universal claim.

## 0.75.0

- Add resolve_device_platform()/large_n_matmul_policy(): H100-class devices
  opt into 128x256x64 for GEMM; L4-class devices keep the safe 128x128 tile.
- Real Llama 3 MLP with the opt-in tile: 1.01x eager on both decode and
  prefill (parity, not a win). Add --device-tile to the benchmark.

## 0.74.0

- Add the real Llama 3 8B SwiGLU MLP example and benchmark. Three kernels;
  latest 3-repeat ratio is 1.02x eager (parity, not a win); graph is slower on
  this compute-bound subgraph.

## 0.73.0

- Add CUDA Graph replay to DecodeSession: 20.83 us/token, 48,002.3 tokens/s
  vs 38.76 us/token for the compiled in-place loop.
- Fix the token-loop timing protocol: tokens are precomputed before timing; the
  old 72.11 us/token included torch.randn allocation.

## 0.72.0

- Add a small-M wide-N matmul tile: M=64 MLP kernels drop from 11.81 us to
  5.45 us on H100, and a graph path measures 11.9 us per call vs eager 31.3.
- Add the MLP case to the M2 matrix (30 cases).
- Add prefill-aware continuous batching (0.71.0 entry below).

## 0.71.0

- Add prefill-aware continuous-batch admission: 16 requests with 16-token
  prompts schedule at 43.26 us/decode token, 23,116.7 tokens/s.

## 0.70.0

- Add cache_write_hsd_range and a prefill pipeline: graph 6.42 us vs Inductor
  73.28 us; prefill+decode path: 0.564 ms for 16 prompt tokens and 83.71
  us/token decode.

## 0.69.0

- Add ContinuousBatchSession with dynamic admission/retirement and an in-place
  cache path: 30,997.9 tokens/s on 16 requests with a 4-slot batch.

## 0.68.0

- Add rank-3 attention and a no-copy GQA decode: graph 7.09 us vs 9.04 us
  with the reshape copy.

## 0.67.0

- Add cache_write_hsd_batch and a 4-request batched GQA decode: graph 7.61 us
  vs eager 316.11 us / Inductor 1213.24 us.

## 0.66.0

- Add an in-place decode session path: 72.11 us/token and 13,867.4 tokens/s
  vs 105.53 us/token for the functional cache loop.

## 0.65.0

- Add DecodeSession and a 32-step token-generation benchmark: 105.53 us per
  token, 9,475.8 tokens/s on a shared H100; functional cache copy is the next
  optimization target.

## 0.64.0

- Add head-major cache_write_hsd and a 4-query/2-KV-head GQA decode with
  cache write: graph 9.04 us vs eager 190.72 / Inductor 234.33.

## 0.63.0

- Execution plans keep host layout helpers, so cache_write + reshape + attention
  supports prepared/graph: 8.68 us vs eager 119.61 us.

## 0.62.0

- Add cache_write and a decode-with-cache-write pipeline: 62.23 us vs eager
  121.56 / Inductor 202.49; prepared/graph blocked on a layout host call.

## 0.61.1

- Use the native attention kernel for fixed-cache decode; graph improves from
  9.88 to 3.49 us and prepared/graph correctness passes.

## 0.61.0

- Add a native Triton causal/GQA attention kernel; prepared/graph work and
  graph is 2.93 us vs eager 186.26 us / Inductor 77.34 us.

## 0.60.0

- Add a causal GQA attention case; correctness passes, but the library path
  loses to Inductor and is not prepared/graph compatible. Native attention is
  the next production gap.

## 0.59.0

- Add decode-step attention over a fixed KV cache: graph 9.88 us vs eager
  25.29 us / Inductor 59.71 us.

## 0.58.0

- Add the first end-to-end decoder layer to M2: graph 32.16 us vs eager
  475.55 us / Inductor 197.63 us.

## 0.57.1

- Align the competitive summary with the 21-case matrix and the measured
  FlagOS/FlagGems baseline; record TVM/TensorRT as external blockers.

## 0.57.0

- Add the FlagOS/FlagGems baseline backend and publish a 10-case comparison.
- Document TVM: the PyPI wheel is CPU-only, so no GPU-comparable baseline.

## 0.56.0

- Declaration-only shape detection: literal-only operators (sort/topk/conv)
  now work from symbolic declarations through per-shape rewriting.
- Verified symbolic sort_last and conv2d across two shapes on H100.

## 0.55.1

- Validate all 21 cases: every backend passes, median compile 11.3 ms.
- Small MoE is within noise of eager; large MoE graph clears both competitors.

## 0.55.0

- Add a larger MoE router case ([32,128], 64 experts, k=8); graph 27.98 us
  vs eager 61.58 us, showing the small-case gap is launch-bound.

## 0.54.2

- Refresh the summary for 20 cases and record the MoE router as an open
  eager/Inductor gap; PyPI remains externally blocked.

## 0.54.1

- Validate the full 20-case matrix: all backends correct, median compile
  15.6 ms. Document the MoE router as a remaining eager/Inductor gap.

## 0.54.0

- Add the QK RMSNorm+RoPE case to M2; graph 6.62 us vs eager 184.5 us.

## 0.53.1

- Refresh the one-page competitive summary for the 19-case matrix, tuning,
  dynamic fallback, and the remaining external baseline/PyPI gaps.

## 0.53.0

- Update compile-time and peak-memory reports to all 19 cases, including MoE
  and top-k; MoE shows the same 32 MiB first-use allocation effect.

## 0.52.1

- Lock ShapeRunner symbolic-to-literal fallback with constructor and runtime
  tests.

## 0.52.0

- ShapeRunner falls back from a symbolic module to literal per-shape rewriting
  when an operator has no symbolic lowering; verified on symbolic mean_last.

## 0.51.1

- Validate the full 19-case matrix in one run: every backend passes, median
  compile 15.0 ms; publish the grouped latency snapshot.

## 0.51.0

- Add the topk_last sampling case to M2; 4.70 us vs eager 18.44 us on H100.

## 0.50.0

- Add the multi-operator MoE router case to M2: matmul + tanh soft-cap +
  softmax + topk + gather; 48-53 us vs eager 151.6 us on H100.

## 0.49.0

- Add conv2d_ex(x, weight, bias, stride, padding), one parameterized
  implicit-GEMM convolution for arbitrary static stride/padding.

## 0.48.0

- Cache tune_conv2d choices by source/shape/platform; a warm call compiles only
  the winning geometry and reports cached=True.

## 0.47.1

- Add flowgpu_tuned to the M2 runner and publish tuned conv results: stride-2
  cases improve from 18-22 us to 3.68-3.74 us; small 1x1 is unchanged.

## 0.47.0

- Add tune_conv2d, a measured conv tile selector. On the 3x3 same case it
  finds a 3.91 us geometry where the default candidate is 19.33 us.

## 0.46.1

- Publish the full 17-case latency/correctness snapshot and mark shared-GPU
  outliers separately from prepared/graph values.

## 0.46.0

- Add conv2d_stride2_same (3x3, stride 2, padding 1); prepared/graph are
  3.56-3.78 us vs eager 39.92 us on H100.

## 0.45.2

- Publish peak-allocated memory for all 16 cases; first-use matmul/conv
  allocations are separated from prepared/static buffers.

## 0.45.1

- Publish the full 16-case matrix and compile-time distribution: 11.7-27.5 ms
  after a cold first compile, median 15.2 ms.

## 0.45.0

- Add conv2d_stride2 (stride-2, no padding) on the implicit-GEMM lowering;
  prepared/graph are 3.7 us vs eager 55.55 us on H100.

## 0.44.0

- Add conv2d_same: odd-K same-padded convolution on the implicit-GEMM
  lowering; 3.90 us vs eager 62.04 us on H100.

## 0.43.1

- Add a CPU Docker image, a CI build job, and container documentation.

## 0.43.0

- Support 4-D [O,C,KH,KW] weights and lower small KxK conv2d to an
  implicit-GEMM tl.dot kernel; 3x3 is 4.03 us vs eager 50.25 us on H100.

## 0.42.1

- Cap conv2d 1x1 tile extents (BM/BN 128, BK 64); a [4,16,16,32] x [64,32]
  case is within fp16 tolerance on H100.

## 0.42.0

- Lower conv2d 1x1 to one tiled tl.dot kernel (2.37 us vs 136 us for the
  four-kernel composition); prepared and graph paths now work.

## 0.41.0

- Extend the M2 matrix to 13 cases with batch_norm and 1x1 conv2d.
- batch_norm is fused and fast; conv2d_1x1 exposes a 136 us four-kernel gap.

## 0.40.1

- Keep a portable layer_norm expansion for symbolic or wider-than-4096 rows;
  literal in-register rows still use the native kernel.

## 0.40.0

- Support rank-0 scalar arguments in prepared execution plans and GraphRunner;
  layer_norm prepares and captures, all four FlowGPU paths pass on H100.

## 0.39.0

- Replace the layer_norm composition with one native Triton row kernel;
  exact for eps 1e-5/1e-3 on H100 and 2.42 us vs eager 36.75 us (shared GPU).

## 0.38.0

- Replace the mean_last composition with one native Triton row-mean kernel.

## 0.37.0

- Replace the quadratic topk sort composition with a native Triton `tl.sort`
  row kernel; sort_last is 2.23 us vs eager 13.87 us on H100.

## 0.36.2

- Publish a one-page competitive capability/latency summary with honest limits.

## 0.36.1

- Add the prepared+graph M2 backend and publish the 9-case comparison;
  copy+replay is 8.56–14.65 µs, faster than allocating prepared but slower than
  zero-copy graph.

## 0.36.0

- Add `prepare_graph_pipeline` combining prepared execution plans with graph
  capture; copy+replay measured 11.12/10.96 µs for add/residual on H100.

## 0.35.1

- Add `ShapeRunner`, which selects symbolic vs literal dynamic handling from
  the program declarations; verified on H100 for both paths.

## 0.35.0

- Add `LiteralShapeRunner`: rewrite named literal input declarations per shape
  and compile a static specialization; verified on H100 for 1024/2048/17.

## 0.34.2

- Add peak-allocated-memory reporting to the M2 device runner and publish the
  9-case result; prepared/static matmul avoids a 32 MiB first-use allocation.

## 0.34.1

- Publish 3-seed medians including the zero-copy graph path: 2.99–4.31 µs
  across all 9 cases.

## 0.34.0

- Add the `flowgpu_graph` M2 backend and publish zero-copy CUDA Graph replay
  across all 9 cases: 2.62–4.30 µs, below eager and explicit Triton.

## 0.33.1

- Measure symbolic vs per-shape compile cost and full-call latency on H100 GPU1;
  one module is not automatically cheaper for a small shape set.

## 0.33.0

- Add `SymbolicShapeRunner` and document one-module runtime-extent dispatch;
  verified on H100 for vectoradd, matmul and softmax across shapes.
- Coverage records dynamic shapes as partial; no checklist gaps remain.

## 0.32.0

- Add `sort_last` as a correctness-first composition (neg + topk(k=width) + neg);
  H100 GPU1 matches torch.sort exactly on literal rows.
- Coverage moves to 45/46; only symbolic dynamic shapes remain.

## 0.31.0

- Add a 1x1 conv2d (NHWC, flattened [O,C] weight) composed from reshape,
  transpose, matmul and bias; H100 GPU1 matches torch exactly.
- Coverage moves to 44/46 with 91 registered operators.

## 0.30.2

- Add a multi-seed M2 wrapper and publish 3-seed medians for all 9 cases;
  all seeds pass correctness.

## 0.30.1

- Share the row-count reduction between layer_norm's mean and variance;
  kernels drop 4 -> 3 and H100 GPU1 full-call is 60.9 us.

## 0.30.0

- Add rank-2 inference `batch_norm` with provided mean/var, affine weight/bias
  and scalar eps; H100 GPU1 matches torch exactly.
- Coverage moves to 43/46 with 90 registered operators.

## 0.29.2

- Extend the M2 matrix to nine cases with `neg`, `pow` and `mean_last`; all
  correctness checks pass and prepared `mean_last` is 12.89 µs on H100 GPU1.

## 0.29.1

- Add `GraphRunner(copy_inputs=False)` zero-copy capture for caller-owned static
  buffers; in-place updates need no per-call copy.

## 0.29.0

- Add `layer_norm` as a composed fp32-mean/variance operator with affine
  weight/bias and scalar eps; H100 GPU1 matches torch layer_norm exactly.
- Coverage moves to 42/46, with 89 registered operators.

## 0.28.0

- Add `mean_last` as a composed operator (sum_last / count of ones), with an
  H100 GPU1 correctness check and coverage at 41/46.

## 0.27.0

- Add `pow` as a value-layer binary operator with torch-compatible NaN
  semantics; H100 GPU1 matches torch.pow exactly, including NaN masks.
- Update the coverage audit to 40/46 present.

## 0.26.0

- Add `neg` as a value-layer unary operator with reference and Triton lowerings;
  H100 GPU1 correctness is exact.
- Update the coverage audit to 39/46 present.

## 0.25.2

- Add and publish the operator-coverage audit (38/46 present, 8 gaps).

## 0.25.1

- Batch GraphRunner input copies with `_foreach_copy_`, reducing copy + replay
  from 9.3–18.3 µs to 8.1–11.2 µs on H100 GPU1.

## 0.25.0

- Add `GraphRunner`: capture a prepared execution plan and replay fixed-shape
  static buffers with no guards, allocation or Triton dispatch.
- Measure graph replay on H100 GPU1 and document the fixed-shape deployment
  contract.

## 0.24.1

- Make the prepared execution-plan path first-class: select one named pipeline
  from a multi-pipeline source, document caller-owned buffers, and publish the
  H100 GPU1 comparative result and optimization order.
- Add the prepared FlowGPU backend to the M2 device runner.
- Add the CPU competitive probe, six-case matrix, independent CPU oracles and
  competitive roadmap; wire compile/oracle checks into CI.
- Add the M2 device runner with shared inputs/tolerance and separate CUDA-Graph
  and full-call metrics for FlowGPU vs eager vs TorchInductor vs hand-written
  Triton baselines.

## 0.24.0

- Add `DynamicShapeRunner`: derive symbolic bindings per call and cache static
  per-shape specializations with explicit rebuilds.
- Document the practical dynamic-shape boundary and M3 status.

## 0.23.0

- Add a declarative, contract-filtered tuning-space generator for measured
  cross-domain selection.
- Add `ShapeSpecializedRunner` for bounded per-shape prepared-runner caches.
- Document frozen autotune, per-shape specialization and prepared deployment.

## 0.22.0

- Add an explicit effect/alias model and graph verifier; stateful cache
  operators are never removed or commoned.
- Add a verified pass framework with graph-level CSE, DCE and guarded identity
  folding, plus `--no-optimize` / `--no-identity-folding` controls.
- Add a structural planning summary and `--emit planning`.
- Document the static-only dynamic-shape boundary and the optimization pipeline.

## Unreleased

- Prohibit private evidence-path tokens across the public tree and neutralize the
  public audit helper default output path.

## Unreleased

- Replace private evidence-path references in public technical notes with
  neutral placeholders.

## 0.21.7

- Add `flowgpu --doctor-format json` for machine-readable environment reports.

## 0.21.6

- Validate the release pipeline against the upgraded GitHub Actions and record
  the Dependabot resolution.

## Unreleased

- Apply the grouped Dependabot GitHub Actions upgrades to the workflow files;
  close the stale PR and leave no open pull requests.

## 0.21.5

- Remove internal evidence-path references from public code comments.
- Ship an explicit empty public extern ledger instead of reading a private path.

## 0.21.4

- Fix the release workflow by keeping the runtime SBOM in the weekly
  security-audit artifact instead of the release job.

## 0.21.3

- Attach the runtime CycloneDX SBOM to GitHub Release assets and checksums.

## 0.21.2

- Neutralize the last stakeholder wording in the public migration bundle output;
  extend the boundary check to catch it.

## Unreleased

- Add an explicit limitations page covering platform, API and evidence scope.

## Unreleased

- Extend the relative-link check to all public documentation.

## Unreleased

- Add an examples catalog and record the current roadmap status.

## Unreleased

- Document the tested platform matrix (Linux/macOS; Windows via WSL2).

## Unreleased


## 0.21.1

- Neutralize remaining customer/engagement positioning in the public migration
  tooling and bundled model provenance.
- Add internal-positioning phrases to the public-boundary check.

## 0.21.0

- Remove the internal commercial pricing module and `flowgpu-migration price`
  from the public package.
- Fix `flowgpu-migration bundle new` to emit explicit gaps for every
  operator/platform and update its reproduction command.
- Prohibit internal pricing text in the public-boundary check.

## 0.20.1

- Fix `flowgpu-migration demo` for the public tree: resolve repository paths
  correctly and bundle the demo model in the package.
- Add a public test for the migration demo.

## Unreleased

- Add a public-boundary check for personal paths, credentials and organizer files.

## Unreleased

- Remove stale version numbers from the security policy and getting-started docs.

## Unreleased

- Require release and index-publication tags to be ancestors of `main`.

## Unreleased

- Validate sdist contents alongside the wheel in the release metadata check.

## 0.20.0

- Add `flowgpu --doctor` environment diagnostics and document it.

## 0.19.2

- Add the user-facing security and release-verification page.
- Validate distributions with `twine check` in CI/release/publish workflows.
- Add a VS Code/Codespaces dev container and contributor tooling polish.

## Unreleased

- Validate wheel/sdist metadata with `twine check` in CI, release and
  index-publication workflows.
- Add a VS Code/Codespaces dev container.

## Unreleased

- Add a user-facing security and release-verification page.

## Unreleased

- Add macOS CPU CI coverage and a governance document.

## 0.19.1

- Add the PEP 561 `py.typed` marker and ship it in the wheel.
- Add `.editorconfig` and pre-commit configuration for contributors.

## Unreleased

- Harden workflow permissions and add job timeouts to prevent stuck CI.

## 0.19.0

- Add tested Python 3.14 support to the CI matrix and classifiers.
- Include the Python integration guide/example and release tooling since
  0.18.2.

## Unreleased

- Add a runnable Python-integration example and guide.

## 0.18.2

- Package the 0.18 migration guide, release guide and TestPyPI workflow.
- Point package metadata at the live documentation site.
- Add the wheel release-metadata check to CI.

## Unreleased

- Add upgrade guidance for the 0.18 namespace migration, a release guide,
  TestPyPI trusted publishing, and a wheel metadata invariant check.

## 0.18.1

- Fix the CI coverage source path after the 0.18.0 namespace migration.
- Keep the public suite and the release workflow green on the new layout.

## 0.18.0

- **Breaking:** move the implementation into `flowgpu.compiler` and
  `flowgpu.migration`; the former top-level `compiler` / `migration` packages
  are removed. Import from `flowgpu` and use `python -m flowgpu`.
- The wheel now contains one top-level package, `flowgpu`.

## Unreleased

- Publish the user guide on GitHub Pages and link it from the README.

## Unreleased

- Generate a CycloneDX runtime SBOM in the security-audit workflow.

## 0.17.0

- Add the `flowgpu` public import namespace (`from flowgpu import compile_source`)
  and `python -m flowgpu`, while keeping the legacy `compiler` imports. This is
  the first step of the pre-1.0 namespace transition.

## 0.16.6

- Generate GitHub build-provenance attestations for release wheel/sdist files.
- Document optional attestation verification.

## 0.16.5

- Keep every public example compilable (remove a staged `store_at` sample and
  add a compile-all-examples test).
- Attach `SHA256SUMS` to GitHub Releases.
- Document installing a published release wheel and verifying its checksum.

## 0.16.4

- Add a GitHub Release on tag pushes with generated notes and wheel/sdist assets.
- Add a weekly `pip-audit` runtime-dependency audit.
- Add an sdist install check to CI.
- Update dependency-triage documentation.

## 0.16.3

No runtime code changes. This release packages the documentation, security
policy, maintenance tooling and CI quality gates added after 0.16.2.

- Add a security policy with supported versions and disclosure expectations.
- Add a stability/versioning guide and CodeQL scanning.
- Add an automated relative-link check for README and the user guide.
- Enforce the configured coverage gate in CI and report coverage artifacts.

## 0.16.2

- Add maintainer, maintenance and release-process documentation.
- Add Dependabot and issue templates for ongoing maintenance.
- Add weekly scheduled CI, a maintenance invariant check and a PR checklist.
- Add a user guide, FAQ, troubleshooting, Python API and CLI docs, and a
  strict MkDocs documentation site build.
- Improve package metadata for discoverability and add `docs` extras.
- Add contributor/roadmap/architecture pages, issue contact links and a guarded
  manual PyPI publishing workflow.
- Add a one-command quickstart demo, a five-minute notebook and manual GitHub
  Pages deployment workflow.
- Finish the user-facing README and package metadata cleanup.

## 0.16.1

- Fix the release/CI wheel smoke check.
- Keep the public CPU test suite green.

## 0.16.0

First public snapshot of the FlowGPU compiler prototype.

- DSL frontend, typed dataflow IR, fusion, buffer reuse, Triton and reference backends.
- CPU test suite, examples, backend contracts and contribution guide.
- Competition/private evidence is intentionally not part of this release.
