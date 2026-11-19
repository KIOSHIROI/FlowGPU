# Examples catalog

All examples are runnable with the CLI. Paths are relative to the repository
root.

## Start here

| Example | What it shows |
|---|---|
| `examples/quickstart/add.fgpu` | pointwise add, no GPU needed |
| `examples/gpu_mode/vectoradd.fgpu` | symbolic-shape vector add |
| `examples/gpu_mode/matmul.fgpu` | a tiled matmul pipeline |
| `examples/residual_norm.fgpu` | residual + normalization pattern |
| `examples/mlp.fgpu` | a small multi-layer pipeline |
| `examples/errors/dtype_mismatch.fgpu` | a diagnostic example |
| `examples/errors/shape_mismatch.fgpu` | a diagnostic example |

```sh
flowgpu examples/quickstart/add.fgpu --run
flowgpu examples/gpu_mode/vectoradd.fgpu --emit triton
flowgpu examples/errors/dtype_mismatch.fgpu --emit summary
```

## Model and operator patterns

The `examples/batch8` and `examples/batch10` directories contain larger
attention, MoE, quantisation, scan, gather and layout programs used during
development. They are useful as real-world language samples.

The `examples/packed` programs demonstrate packed low-precision layouts
(AWQ, NVFP4 and MXFP8) in the DSL. `examples/migration` contains examples used
by the operator-coverage tooling.

## Errors

`examples/errors/` intentionally contains invalid programs. Use them to see
diagnostic quality:

```sh
flowgpu examples/errors/shape_mismatch.fgpu --emit summary
```

## Run the examples as tests

The public test suite compiles and executes many examples on the CPU:

```sh
python -m pytest -q
```

If you add an example, add a small test or run it through the reference backend
as part of your change.
