# Lesson 6: materialized vs fused

Entry point: [`lessons/06-materialized/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/06-materialized).

Same SwiGLU source, two compile modes:

```python
compile_source(source, shape_bindings=...)
compile_source(source, shape_bindings=..., materialize_all=True)
```

| mode | kernels |
|---|---|
| fused | `matmul`; `matmul -> silu -> mul`; `matmul` (3) |
| materialized | `matmul`; `matmul`; `pointwise: silu, mul`; `matmul` (4) |

The dataflow does not change. The only difference is whether `up` stays in
registers or is written to a buffer before the pointwise kernel.

Both CPU references are identical.

Run without a GPU:

```bash
python lessons/06-materialized/run_lesson.py
```

Teaching point: **dataflow determines dependencies; compile decisions determine
buffers and kernel boundaries.**
