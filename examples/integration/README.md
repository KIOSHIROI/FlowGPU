# Python integration example

`compile_pipeline.py` compiles a small program through the public `flowgpu` API
and prints the generated kernel, host-wrapper and reference-backend summary.

```sh
.venv/bin/python examples/integration/compile_pipeline.py
```

The example does not require a GPU. See the
[integration guide](../../docs/user-guide/integration.md) for the patterns
behind it.
