# Audit cost beyond the guards

The guard-cost experiment measured the runtime tax. This one measures the
**engineering-side** cost on small CPU-shape cases:

- compile the pipeline once;
- build the audit sheet **from the compiled result** (no recompilation);
- execute the CPU reference once.

| case | compile | + audit sheet | CPU reference | audit / compile |
|---|---:|---:|---:|---:|
| `mlp` (4x64x128) | 3.89 ms | 0.008 ms | 5.32 ms | 0.002x |
| `residual_norm` (32x64) | 2.70 ms | 0.005 ms | 2.20 ms | 0.002x |

The audit sheet is nearly free once the pipeline is compiled: **0.2%** of
compile time. The independent CPU reference costs roughly 0.8–1.4x of compile
time on these small shapes.

## Limits

- small CPU shapes, not a real LLM;
- reference cost grows with shape, so the full-model reference is much more
  expensive;
- this measures construction time, not the human time to read the sheet.

Still, it supports the claim that **the artifact is cheap to produce**; the
expensive part of auditability is the test/reference strategy, not the report.

Raw data:
[`audit-cost-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/audit-cost-20261125.json).
