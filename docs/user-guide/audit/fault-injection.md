# Phase 2: what do the guards actually catch?

This experiment injects nine faults into two compiled hosts: an MLP with bound
`B/D/H` and the masked 32-slot decode block. It separates **structural faults**
(wrong declaration) from **semantic faults** (valid tensors, wrong state).

## Result

| class | cases | caught at launch | silently wrong |
|---|---:|---:|---:|
| structural | 5 | 5 | 0 |
| cache index (fixed) | 1 | 1 | 0 |
| semantic | 3 | 0 | 3 |

Caught at launch:

- input shape mismatch;
- input dtype mismatch;
- non-contiguous input layout;
- weight dtype mismatch;
- cache dtype mismatch.

Silently wrong after the loc guard:

- mask missing (all zeros) at the decode step;
- mask all `-1e9` (no valid keys);
- skipped cache position (`loc=9` while slot 8 is empty).

`loc=99` was silently wrong in the first run and is now caught by a host
`cache_write_hsd` range check; the three remaining cases are state semantics,
not declaration errors.

## Interpretation

FlowGPU's host guards cover the **declaration boundary** well: shape, dtype,
layout, cache dtype and now the cache `loc` range are rejected before any
launch. They still do **not** cover **state invariants**: whether the mask
matches the written prefix, or whether a cache position was skipped.

This is the auditability boundary the project should state explicitly:

> **Guards are not an oracle.** A subgraph can be structurally valid and still
> compute the wrong result because the host passed inconsistent cache state.

The four silent cases are exactly the ones a reference-backed test or an
explicit invariant check must catch. This motivates:

1. ~~a host guard for `cache_write_hsd` locations~~ (added in 1.3.0);
2. a test/oracle path for mask and cache-position consistency;
3. reporting "caught vs silent" as a benchmark metric, not only latency.

Raw data:
[`fault-injection-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/fault-injection-20261125.json).
