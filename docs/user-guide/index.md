# FlowGPU

FlowGPU is a compiler for a small dataflow DSL that targets **Triton** and a
portable **reference backend**.

```text
.fgpu source -> typed SSA dataflow -> fusion -> buffers -> Triton / reference
```

It is an **alpha compiler prototype**: useful for experimenting with operator
pipelines and code generation, not a production dependency. The reference
backend runs anywhere with Python; generated Triton needs a CUDA machine with
PyTorch and Triton.

## Start in two minutes

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --emit summary
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run
```

## Where to go next

- [Install FlowGPU](getting-started.md) — get a working checkout and verify it.
- [Tutorial](tutorial.md) — write, compile and run your first pipeline.
- [Language overview](language.md) — the DSL surface.
- [Python API](api.md) — integrate the compiler into a program.
- [CLI guide](cli.md) — inspect, run and debug from the terminal.
- [Examples](examples.md) — a catalog of runnable programs.
- [Backends](backends.md) — reference vs. Triton and platform profiles.
- [FAQ](faq.md) / [Troubleshooting](troubleshooting.md) — common questions.

The source is at <https://github.com/KIOSHIROI/FlowGPU>. It is MIT licensed.
