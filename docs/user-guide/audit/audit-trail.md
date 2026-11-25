# Audit trail

`flowgpu.compiler.audit` turns one compiled pipeline into a short,
machine-readable and human-readable sheet:

```bash
python -m flowgpu.compiler.audit examples/llama3_mlp_swiglu.fgpu
python -m flowgpu.compiler.audit examples/llama3_mlp_swiglu.fgpu --json
```

It is derived from the compiled artifacts, not from re-reading the source:

- kernels: name, kind, fused ops, tile, grid, parameters;
- fusion groups: anchor, members, fused values, epilogue;
- host guards: the generated `raise ValueError` lines;
- reference kernels: the independent CPU implementation;
- refusals/skips: what the backend declined to emit.

From Python:

```python
from flowgpu.compiler.audit import build_audit_report, render_markdown
report = build_audit_report(source, shape_bindings={"T": 1, "H": 4096, "I": 14336})
print(render_markdown(report))
```

## Why this is the research artifact

The fault-injection study showed the boundary: guards catch declarations, not
mask/cache state. The audit sheet makes both sides visible in one place:

- what was compiled and fused;
- what the host will reject;
- what independent reference exists;
- what the backend refused.

This is the artifact a user study should test: does an engineer localize a
subgraph fault faster with this sheet than with source + logs alone?

## Limits

- It reports the first pipeline only;
- it does not prove semantic correctness;
- it does not include runtime values or timings.

## Coverage of the six user-study faults

The audit sheet now includes input declarations and **caller-owned semantic
risks**:

| task | evidence in sheet |
|---|---|
| shape | direct (host guard) |
| dtype | direct (host guard) |
| cache loc range | direct (host guard) |
| mask missing | risk flagged (`mask/cache consistency`) |
| skipped cache position | risk flagged (`cache-position continuity`) |
| MoE tie order | risk flagged (`topk tie contract`) |

Summary: 6/6 tasks have evidence — **3 direct guards and 3 risk flags**.

A risk flag is not a diagnosis: the sheet says *the host does not check mask or
cache-position consistency*, not that a particular mask is wrong. The two
layers are complementary:

- direct evidence catches declaration/index faults;
- risk flags tell the reader where an oracle/runtime check is still needed.

Raw data:
[`audit-coverage-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/audit-coverage-20261125.json).
