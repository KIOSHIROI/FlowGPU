# Tutorial: your first pipeline

This tutorial writes a tiny FlowGPU program, compiles it, inspects the generated
Triton, and runs the portable reference backend. No GPU is required.

## 1. Create a program

Create `hello.fgpu`:

```text
pipeline hello {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
```

A program declares a pipeline, typed inputs, a sequence of assignments and one or
more outputs. Shape `1024` is literal; replace it with a symbol such as `N` to
compile a shape-generic program.

## 2. Inspect the typed pipeline

```sh
.venv/bin/python -m compiler.cli hello.fgpu --emit summary
```

You should see the pipeline name, its typed values and a final `OK`. The summary
is a quick way to catch shape or dtype mistakes before emission.

## 3. Inspect the generated Triton

```sh
.venv/bin/python -m compiler.cli hello.fgpu --emit triton
```

The output contains a complete module: a `@triton.jit` kernel plus a host wrapper
that allocates outputs, validates shapes/dtypes and launches the kernel. The
kernel is compiler output, not a template string.

## 4. Run the reference backend

```sh
.venv/bin/python -m compiler.cli hello.fgpu --run
```

The reference backend interprets the same typed pipeline in Python and prints
the result. It is the semantic oracle used by tests.

## 5. Try a reduction

```text
pipeline row_sum {
  input x: fp32[32, 128]
  total = sum_axis(x, 1)
  output total
}
```

```sh
.venv/bin/python -m compiler.cli row_sum.fgpu --emit summary
.venv/bin/python -m compiler.cli row_sum.fgpu --emit triton
```

`sum_axis` reduces the last axis; the generated kernel contains the reduction
and the host wrapper. The exact kernel shape depends on the selected platform
profile.

## 6. Read a diagnostic

Programs are checked before emission. A type error looks like:

```text
CompileError: type error at line 3, column 9: ...
```

The message includes a source position and a concrete explanation. If a
construct is not supported by the compiler/backend, the compiler emits a
refusal instead of silently approximating it.

## Next

- Browse the [examples catalog](examples.md).
- Read the [language overview](language.md).
- Integrate the compiler through the [Python API](api.md).
