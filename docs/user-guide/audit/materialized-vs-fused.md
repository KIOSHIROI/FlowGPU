# Materialized vs fused SwiGLU

One source, two compile modes:

```python
fused = compile_source(source, shape_bindings=...)
materialized = compile_source(source, shape_bindings=..., materialize_all=True)
```

`materialize_all=True` makes every non-pointwise anchor opaque, so the
`matmul + silu + mul` epilogue no longer fuses into the up-projection.

## Kernel shape

| mode | kernels |
|---|---|
| fused | `matmul`; `matmul -> silu -> mul`; `matmul` (3) |
| materialized | `matmul`; `matmul`; `pointwise: silu, mul`; `matmul` (4) |

The pointwise kernel is the materialized `hidden` tensor.

## Correctness

- CPU reference: both forms produce identical values on the `T=2,H=8,I=16`
  slice (test `test_materialize_all.py`);
- H100: both forms agree with eager to the same relative error
  (`3.4e-3` at T=1, `0.0` at T=2048).

## Performance

Not reported yet: GPU0 was 88% busy during the correctness run, so the timing
in the raw file is contaminated. The structural comparison is valid; the
latency comparison needs an idle device.

Raw data:
[`swiglu-materialize-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/swiglu-materialize-20261125.json).

## CPU-first inspection

The comparison does not need a GPU:

```bash
python -m flowgpu.compiler.audit examples/llama3_mlp_swiglu.fgpu \
  --shape T=2 --shape H=8 --shape I=16
python -m flowgpu.compiler.audit examples/llama3_mlp_swiglu.fgpu \
  --shape T=2 --shape H=8 --shape I=16 --materialize-all
```

The first prints 3 kernels and the fused epilogue; the second prints 4 kernels
and the standalone `pointwise: silu, mul` kernel. Shape bindings make the
CPU-only path concrete on a small slice.
