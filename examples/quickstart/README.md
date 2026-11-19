# Quickstart

Install once (from the `flowgpu/` directory):

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Then, with no GPU at all:

```sh
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit summary
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit triton
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run
```

* `--emit summary` shows the typed pipeline;
* `--emit triton` prints the generated Triton module (host + kernel);
* `--run` executes the **reference backend** (generated Python, CPU) and prints
  the outputs.

With a CUDA device and `torch` + `triton` installed, one more command runs the
generated Triton and compares it against the same reference:

```sh
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run-gpu
```

Expected shape of that command:

```
quickstart_add (sizes {}, CUDA)
  out  fp16[1024]  PASS  max abs diff 2.44e-04 (0 over tolerance)
OK  generated Triton matches the reference backend on CUDA
```

If torch/CUDA is missing the command exits 2 with a message pointing back at
`--run`; it never falls back silently.

After `pip install`, the same CLI is available as `flowgpu`:

```sh
flowgpu examples/quickstart/add.fgpu --run-gpu
```
