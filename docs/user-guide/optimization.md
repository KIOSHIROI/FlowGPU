# Optimization pipeline

FlowGPU has two optimization layers: the graph-level pass pipeline and the
planning layer that turns the optimized graph into kernels and buffers.

## Graph passes

By default, every compile runs a small, verified pipeline after semantic
normalization:

| pass | what it removes | guard |
|---|---|---|
| identity | identity reshape, same-dtype cast, cancelling transpose/move-last pairs | only non-exported values; exported copies are storage boundaries |
| CSE | duplicate pure producers | only `pure` nodes; inputs/outputs and stateful cache writes are excluded |
| DCE | pure producers whose result reaches no output | only `pure` nodes; stateful writes and multi-result groups are retained |

The passes share an explicit effect model. A stateful operator (currently
`linear_decode_forward` and `store_cache_4d`) is never commoned, duplicated or
erased merely because its SSA result is unused. A pass verifies the graph before
and after every step, and the compile pipeline verifies it again before fusion.

The pipeline is on by default. For a conservative control:

```sh
flowgpu program.fgpu --no-optimize
flowgpu program.fgpu --no-identity-folding
```

In Python:

```python
from flowgpu import compile_source

result = compile_source(source, optimize=True, identity_folding=True)
pipeline = result.pipelines[0]
for record in pipeline.optimization:
    print(record.name, record.nodes_before, record.nodes_after)
```

`optimize=False` builds the graph without graph CSE/DCE/identity folding, while
still running the structural/effect verifier.

## Planning summary

Fusion, liveness and layout analysis additionally produce a structural cost
summary. It is deliberately *not* a latency prediction; launch count, modeled
traffic and buffer count matter in different regimes.

```sh
flowgpu program.fgpu --emit planning
```

The JSON contains:

- `kernels` / `launches`
- `fused_values`
- `materialized_buffers` and compiler `allocated_buffers`
- `slots`
- remaining `layout_nodes`
- modeled traffic `traffic_a`

The planning choices themselves are the established cost policies:
`fusion_policy="traffic"` for pointwise placement, layout-view cost checks on
platforms with a validated contract, and liveness slot reuse for disjoint
buffers. Changing a default still requires parent/candidate evidence under
[the acceptance protocol](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/COMPILER_ACCEPTANCE.md).

## What is not implemented

This pipeline is not a general optimizing compiler. It does not yet provide
whole-program constant folding, global DCE across side effects, loop
transformations, dynamic-shape guards or automatic tuning-space generation.
See [Limitations](limitations.md) for the current scope.
