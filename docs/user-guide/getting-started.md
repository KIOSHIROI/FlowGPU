# Getting started

## Requirements

- **Python 3.11, 3.12, 3.13 or 3.14**.
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

## Install a published release wheel

Download the wheel from the latest
[GitHub Release](https://github.com/KIOSHIROI/FlowGPU/releases), then verify it
against the attached `SHA256SUMS` before installing:

```sh
sha256sum -c SHA256SUMS
python -m pip install flowgpu-<version>-py3-none-any.whl
```

The release page also contains the sdist for environments that prefer building
from source. Release artifacts are covered by GitHub build-provenance
attestations; if `gh` is available you can verify a downloaded file with:

```sh
gh attestation verify flowgpu-<version>-py3-none-any.whl --repo KIOSHIROI/FlowGPU
```

## Install from Git

```sh
python -m pip install "git+https://github.com/KIOSHIROI/FlowGPU.git"
```

This installs the runtime dependency (`lark`) and the `flowgpu` command.

## Verify

```sh
.venv/bin/python -m flowgpu --version
# prints the installed version, e.g. FlowGPU 0.x.y

.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --emit summary
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
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run
```

This executes the generated Python reference implementation on the CPU and
prints the output values. It does not require `torch`.

## Run generated Triton (optional)

```sh
.venv/bin/python -m pip install torch triton
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run-gpu
```

If PyTorch, Triton or CUDA is unavailable, this command exits with code 2 and a
message pointing back to `--run`. It never silently falls back to a torch kernel.

## Import from Python

```python
from flowgpu import compile_source
```

The old `from flowgpu.compiler import ...` path remains supported during pre-1.0.

## Next

- [Tutorial](tutorial.md)
- [CLI guide](cli.md)
- [Troubleshooting](troubleshooting.md)
