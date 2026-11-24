# PyTorch custom-op integration

A compiled FlowGPU subgraph can be exposed as a PyTorch custom op while the
host keeps ownership of the weights and the schedule:

```python
import torch
from flowgpu.compiler.runtime.runtime import compile_pipeline
from flowgpu.compiler.runtime.torch_op import register_torch_op

run = compile_pipeline(
    source, pipeline="llama3_mlp_swiglu",
    cache_dir=".flowgpu-cache",
    shape_bindings={"T": 1, "H": 4096, "I": 14336})
mlp = register_torch_op(
    "llama3_mlp", run, arity=4, output_shape=(1, 4096),
    output_dtype=torch.bfloat16, output_name="y")

y = mlp(x, w_gate, w_up, w_down)
```

The op is inference-only and takes tensor arguments in the compiled host's
order. Weights are passed by the caller; FlowGPU does not own parameters. A
Meta kernel is registered so FakeTensor / `torch.compile` can infer the output
shape without launching. The helper retains the `torch.library.Library` object:
letting it be collected removes the registration and the next call segfaults
on torch 2.14.

## Verified on the real MLP

On the H100, `llama3_mlp_op(x, w_gate, w_up, w_down)` returned `[1, 4096]`
with a relative error of 1.7e-3 against eager bf16.

This is the host-integration primitive, not a full model or serving engine:
the host still owns attention, KV cache, batching and sampling.
