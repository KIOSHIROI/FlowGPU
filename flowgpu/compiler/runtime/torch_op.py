"""Register a compiled FlowGPU subgraph as a PyTorch custom op.

The host keeps owning the weights and the invocation schedule; FlowGPU only
supplies the compiled ``run_*`` function. The op is inference-only and takes
tensor arguments in the same order as the compiled host.
"""
from __future__ import annotations


# The Library object owns the schema/impl registrations. Keeping only the
# OpOverloadPacket lets the Library be collected, which removes the operator
# and segfaults the next call (observed on torch 2.14).
_LIBRARIES = {}


def register_torch_op(name, run, *, arity, output_shape, output_dtype,
                      output_name=None, device_type="CUDA", namespace="flowgpu"):
    """Define ``torch.ops.<namespace>.<name>`` backed by a compiled host.

    ``run`` may return a tensor or the generated host's output dict; pass
    ``output_name`` in the latter case. A Meta implementation is registered so
    ``torch.compile`` / FakeTensor can infer the output without launching.
    """
    import torch

    library = torch.library.Library(namespace, "DEF")
    parameters = ", ".join(f"Tensor arg{index}" for index in range(arity))
    schema = f"{name}({parameters}) -> Tensor"
    library.define(schema)

    def implementation(*args):
        result = run(*args)
        if isinstance(result, dict):
            if output_name is None:
                raise ValueError("compiled host returned a dict; pass output_name")
            return result[output_name]
        return result

    def meta(*args):
        return torch.empty(tuple(output_shape), dtype=output_dtype, device="meta")

    library.impl(name, implementation, device_type)
    library.impl(name, meta, "Meta")
    _LIBRARIES[(namespace, name)] = library
    return getattr(getattr(torch.ops, namespace), name)


__all__ = ["register_torch_op"]
