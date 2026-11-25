# Phase 2: what do the guards actually catch?

This experiment injects nine faults into two compiled hosts: an MLP with bound
`B/D/H` and the masked 32-slot decode block. It separates **structural faults**
(wrong declaration) from **semantic faults** (valid tensors, wrong state).

## Result

| class | cases | caught at launch | silently wrong | oracle catches |
|---|---:|---:|---:|---:|
| structural + loc range | 6 | 6 | 0 | - |
| observable semantic | 5 | 0 | 5 | 5/5 |
| injection with no observable effect | 2 | 0 | - | - |

Caught at launch:

- input shape mismatch;
- input dtype mismatch;
- non-contiguous input layout;
- weight dtype mismatch;
- cache dtype mismatch;
- `loc` outside the 32-slot cache range.

Silently wrong under guards, detected by the oracle:

- mask missing (all zeros);
- mask all `-1e9` (no valid keys);
- skipped cache position;
- duplicate cache position;
- mask shorter than the written prefix.

Two injections — corrupting one cached key and setting `q_scale=0` — produced
the same bf16 output in this setup, so they are reported as "no observable
effect", not as silent errors. A stronger injection or a more sensitive
subgraph is needed to study those.

## Interpretation

FlowGPU's host guards cover the **declaration boundary** and the cache `loc`
range. They still do **not** cover **state invariants**: whether the mask
matches the written prefix, or whether a cache position was skipped. An
independent correct-output oracle detects all five observable semantic faults.

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
