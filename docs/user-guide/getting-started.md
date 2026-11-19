# Getting started

## Requirements

- **Python 3.11, 3.12 or 3.13**.
- No GPU is needed to compile programs or to run the reference backend.
- To execute generated Triton you additionally need a CUDA-capable machine with
  PyTorch and Triton installed.

FlowGPU is not on PyPI yet. Install from source or directly from Git.

## Install from a checkout

```sh
git clone https://github.com/KIOSHIROI/FlowGPU.git
cd FlowGPU
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

On Windows PowerShell the interpreter path is `.venv\Scripts\python.exe`.

## Install from Git

```sh
python -m pip install "git+https://github.com/KIOSHIROI/FlowGPU.git"
```

This installs the runtime dependency (`lark`) and the `flowgpu` command.

## Verify

```sh
.venv/bin/python -m compiler.cli --version
# FlowGPU 0.16.1

.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit summary
```

The `summary` command should print a typed pipeline (`quickstart_add`) with one
`out` value of type `fp16[1024]` and `OK`.

If you installed the package, use the console script instead:

```sh
flowgpu examples/quickstart/add.fgpu --emit summary
flowgpu --version
```

## Run the reference backend

```sh
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run
```

This executes the generated Python reference implementation on the CPU and
prints the output values. It does not require `torch`.

## Run generated Triton (optional)

```sh
.venv/bin/python -m pip install torch triton
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run-gpu
```

If PyTorch, Triton or CUDA is unavailable, this command exits with code 2 and a
message pointing back to `--run`. It never silently falls back to a torch kernel.

## Next

- [Tutorial](tutorial.md)
- [CLI guide](cli.md)
- [Troubleshooting](troubleshooting.md)
