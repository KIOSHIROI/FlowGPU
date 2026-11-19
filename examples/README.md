# Examples

Start with the quickstart, then browse by use case.

| Directory | Contents |
|---|---|
| `quickstart/` | add/reduction examples, one-command demo and notebook |
| `integration/` | compile and inspect a pipeline from Python |
| `gpu_mode/` | vector add, matmul and prefix-sum programs |
| `packed/` | AWQ, NVFP4 and MXFP8 packed-layout programs |
| `batch8/`, `batch10/` | larger attention/MoE/quantisation/scan samples |
| `errors/` | intentionally invalid programs for diagnostics |

```sh
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run
.venv/bin/python examples/quickstart/demo.py
.venv/bin/python examples/integration/compile_pipeline.py
```

Every non-error `.fgpu` example is compiled by the public test suite; see
[docs/user-guide/examples.md](../docs/user-guide/examples.md).
