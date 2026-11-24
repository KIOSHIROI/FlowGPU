# Competitive summary

A one-page status against the competitor matrix. All numbers are from the
public probes; no private fixture is used.

## Capability scorecard

| area | status | evidence |
|---|---|---|
| LLM operator checklist | **45 present / 1 partial / 0 missing** | [Operator coverage](operator-coverage.md) |
| dynamic shapes | symbolic one-module, literal rewrite, automatic fallback | [Prepared deployment](deployment.md) |
| fixed-shape deployment | prepared → copy+graph → zero-copy graph | [M2 H100 report](competitive-m2-h100.md) |
| tuning | measured conv tile selection with a cache | [M2 H100 report](competitive-m2-h100.md#conv-tile-tuning) |
| correctness | **26 cases × 6 backends**, one full-matrix run | [M2 H100 report](competitive-m2-h100.md#21-case-validation-run) |
| compile time | 21-case distribution, median 11.3 ms | [M2 H100 report](competitive-m2-h100.md#compile-time-19-case-matrix) |
| memory | 19-case peak-allocated metric | [M2 H100 report](competitive-m2-h100.md#peak-memory-all-19-cases) |
| FlagOS/FlagGems | 10-case baseline measured; matmul/add excluded by Triton-3.8 incompatibility | [FlagOS baseline](competitive-flagos.md) |
| TVM/Relax | **blocked**: PyPI wheel is CPU-only (`cuda` disabled) | [FlagOS baseline](competitive-flagos.md#tvm) |
| packaging | wheel/sdist, Python 3.11–3.14 CI, clean-venv installs, CPU Docker image | CI |
| PyPI | **blocked**: trusted publisher not configured; both indexes 404 | external action |

## Latency snapshot (H100 GPU1, full-call µs)

| path | stable range |
|---|---:|
| eager | 2.2–23.4 (up to 151 on MoE) |
| explicit Triton | 2.1–17.0 where a baseline exists |
| FlowGPU prepared | 2.2–4.0 |
| FlowGPU copy+graph | 4.4–7.6 |
| FlowGPU zero-copy graph | **2.2–4.3** |
| FlowGPU tuned | stride-2 conv **3.7** vs 18–22 default |
| MoE router (small) | 49.32 (graph) vs 56.19 eager |
| MoE router (large) | **29.18 (graph)** vs 57.14 eager / 87.89 Inductor |

- The kernel path is competitive with explicit Triton.
- After the allocated path, host overhead is removed by prepared/copy+graph;
  zero-copy replay is the fixed-metadata floor.
- Default-path outliers on a shared GPU are separated in the M2 report, not
  hidden.

## Honest limitations

- default calls still lose to fresh prepared/graph calls for some cases on a
  shared GPU; zero-copy graph is not a general dispatch replacement;
- `sort_last` and `topk_last` require literal extents; sort exposes values;
- conv supports NHWC and static stride/padding (via `conv2d_ex`), not arbitrary
  layouts/dilation;
- dynamic shapes are partial: literal-only bodies may need per-shape rewrite;
- the small MoE router is launch-bound and only near eager, not ahead;
- FlagGems has no GPU matmul/add baseline on this Triton stack;
- TVM/Relax has no GPU wheel; TensorRT-LLM is not available here;
- PyPI publication is blocked by an external trusted-publisher setting.

## Next

1. default host path only if it can be improved without weakening the
   exact-context contract;
2. general dynamic handling for literal-only bodies;
3. TVM CUDA build and FlagTree-class compiler when their environments are available;
4. adoption: PyPI trusted publisher.
