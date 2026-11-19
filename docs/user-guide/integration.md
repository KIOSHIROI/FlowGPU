# Integrating FlowGPU into a Python project

The compiler is a normal Python API. A build script, test harness or code
generator can compile a pipeline and inspect the typed products without running
a GPU.

```python
from flowgpu import compile_source

SOURCE = """
pipeline add {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
"""

result = compile_source(SOURCE)
pipeline = result.pipelines[0]

print(pipeline.triton.kernels[0].source)  # Triton kernel
print(pipeline.triton.host)               # host wrapper
print(pipeline.reference.host)            # portable reference implementation
```

The runnable version is
[`examples/integration/compile_pipeline.py`](https://github.com/KIOSHIROI/FlowGPU/blob/main/examples/integration/compile_pipeline.py).

## Typical patterns

1. **Build-time code generation.** Compile once, inspect `pipeline.triton`, and
   store the generated module in your own build output. Generated source and
   host ABIs are not a stable serialized interface before 1.0; version and test
   the generated output with your application.
2. **Reference-backed tests.** Use the portable reference backend for CPU tests
   and as an oracle when a device is available. The CLI exposes this as
   `--run`; the API exposes `pipeline.reference`.
3. **Device validation.** On a CUDA machine, `flowgpu ... --run-gpu` executes the
   emitted Triton and compares it with the reference backend. It exits with
   code 2 instead of silently falling back when the device is unavailable.
4. **Tracing your own model.** Feed an operator graph you already have through
   the DSL (or a custom frontend) and keep the typed dataflow products for
   diagnostics and evidence.

## What the compiler does not do

- It does not import PyTorch or Triton while compiling; generated host code does
  that at runtime.
- It does not promise a stable generated-source ABI before 1.0.
- It does not execute arbitrary untrusted programs for you; treat device
  execution as running generated code.

If you are embedding FlowGPU in a larger project, pin the version and include a
small golden-output test around the generated module.
