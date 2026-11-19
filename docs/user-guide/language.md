# Language overview

FlowGPU programs describe operator pipelines, not kernels. The canonical
contract is [`Design.md`](https://github.com/KIOSHIROI/FlowGPU/blob/main/Design.md);
the grammar is [`EBNF`](https://github.com/KIOSHIROI/FlowGPU/blob/main/EBNF).

## A program

```text
pipeline add {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
```

- `pipeline name { ... }` declares a pipeline.
- `input name: dtype[shape]` declares a typed tensor input.
- `name = expression` binds an SSA value.
- `output value` marks a pipeline result.

## Types

Common tensor dtypes are `fp16`, `bf16`, `fp32` and integer types. A shape is a
list of positive literal dimensions and optionally symbolic dimensions:

```text
input x: fp16[N, 128]
```

Symbolic programs are compiled once and receive shape/dtype checks in the host
wrapper. Use `--shape` to specialize a symbolic dimension for inspection.

## Expressions

Pointwise arithmetic and comparisons are available (`add`, `sub`, `mul`, `div`,
`maximum`, `abs`, ...). Reductions reduce the last axis (`sum_axis`,
`max_axis`, `min_axis`). Layout operators include `reshape`, `split`, `concat`,
`interleave` and `transpose`.

Scalars are values too: a scalar input is a 0-d fp32/int value, and scalar
parameters can be specialized at compile time. See the [examples](examples.md)
for real programs.

## What the language deliberately does not do

- No arbitrary control flow or loops. A pipeline is a dataflow graph.
- No implicit dtype coercion that would silently change precision.
- No task-specific special cases in the compiler.

If a program cannot be expressed or lowered exactly, the compiler reports a
diagnostic or refusal rather than approximating it.

## Inspecting a program

```sh
.venv/bin/python -m compiler.cli program.fgpu --emit dataflow
.venv/bin/python -m compiler.cli program.fgpu --emit fusion
.venv/bin/python -m compiler.cli program.fgpu --emit buffers
.venv/bin/python -m compiler.cli program.fgpu --emit triton
```

The `--emit` stages are useful when learning the compiler or debugging a
transformation.

## Further reading

- [Architecture](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/ARCHITECTURE.md)
- [Compiler acceptance criteria](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/COMPILER_ACCEPTANCE.md)
- [Support matrix](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/SUPPORT_MATRIX.md)
- [Backend contracts](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/BACKEND_CONTRACTS_20261019.md)
