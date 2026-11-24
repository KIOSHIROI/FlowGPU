# Dynamic batching

A compiled decode step has a fixed batch dimension. `DynamicBatch` keeps a
fixed-capacity set of active requests, pads inactive slots and returns only the
active outputs:

```python
from flowgpu.compiler.runtime.batching import DynamicBatch

batch = DynamicBatch(capacity=2)
batch.add("req-1", payload_one)
batch.add("req-2", payload_two)
outputs = batch.step(call=run_padded, empty=make_empty_payload)
batch.remove("req-1")
```

- `add` returns the slot and rejects duplicates/full batches;
- `update` replaces a request's payload;
- `step` preserves slot order, calls `run_padded(rows)`, and returns
  `{request_id: output}` for active requests only;
- `empty()` builds a padded payload for an inactive slot;
- the step must return exactly one output per batch slot.

Cache ownership stays outside: `PagedKVCache` allocates blocks and `BlockTable`
maps request ids to flat locations. `DynamicBatch` only decides which requests
share a fixed-shape invocation.

This is a host scheduling primitive, not a continuous-batching policy or a
priority/KV-eviction scheme.
