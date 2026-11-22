# Competitive summary

A one-page status against the competitor matrix. All numbers are from the
public probes; no private fixture is used.

## Capability scorecard

| area | status | evidence |
|---|---|---|
| LLM operator checklist | **45 present / 1 partial / 0 missing** | [Operator coverage](operator-coverage.md) |
| dynamic shapes | symbolic one-module, literal rewrite, per-shape fallback | [Prepared deployment](deployment.md) |
| fixed-shape deployment | allocated prepared → copy+graph → zero-copy graph | [M2 H100 report](competitive-m2-h100.md) |
| correctness | 9-case matrix × 6 backends, 3 seeds | [M2 H100 report](competitive-m2-h100.md) |
| memory | peak-allocated metric published | [M2 H100 report](competitive-m2-h100.md#peak-allocated-memory) |
| packaging | wheel/sdist, Python 3.11–3.14 CI, clean-venv installs | CI |
| PyPI | **blocked**: trusted publisher not configured; both indexes 404 | external action |

## Latency snapshot (H100 GPU1, full-call µs)

| path | simple cases | fused residual |
|---|---:|---:|
| eager | 4.8–9.6 | 62.5 |
| explicit Triton | 11.6–16.5 | 12.7 |
| FlowGPU prepared | 12.0–16.5 | 16.5 |
| FlowGPU copy+graph | 8.6–14.7 | 11.9 |
| FlowGPU zero-copy graph | **2.9–4.3** | **2.9** |

- The kernel path is competitive with explicit Triton.
- The remaining default-path gap is host guards/allocation/Triton dispatch, not
  math; CUDA Graph removes it for fixed metadata.
- Static zero-copy replay is under eager on every measured case.

## Honest limitations

- default/prepared calls still lose to eager on simple pointwise/softmax;
- `sort_last` requires literal extents and currently exposes values only;
- `conv2d` is 1x1, NHWC, flattened `[O,C]` weight only;
- dynamic shapes are partial: literal-only bodies may need per-shape rewrite;
- PyPI publication is blocked by an external trusted-publisher setting.

## Next

1. default host path (guards/launch) if it can be improved without weakening
   the exact-context contract;
2. general dynamic handling for literal-only bodies;
3. real sort/conv2d kernels only after a measured workload justifies them;
4. adoption: PyPI trusted publisher, issue/PR templates already present.
