# FlowGPU

[![CI](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/ci.yml/badge.svg)](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

FlowGPU is a compiler for a small dataflow DSL that targets **Triton** and a
portable **reference backend**. Write an operator pipeline, get typed SSA
dataflow, conservative fusion, buffer reuse and complete emitted host + kernel
code.

```text
.fgpu source -> typed SSA dataflow -> fusion -> buffers -> Triton / reference
```

**Status:** alpha compiler prototype. It is useful for experimenting with
operator pipelines and code generation; it is not a production dependency yet.
The reference backend runs anywhere with Python. Running generated Triton
requires a CUDA machine with PyTorch and Triton.

## Why FlowGPU?

- **One program, two backends.** Inspect the same typed pipeline as emitted
  Triton or as executable reference Python.
- **Compiler output, not templates.** Host wrappers, launch arguments, shape and
  dtype checks, and kernels are generated together.
- **Designed for evidence.** Refusals are explicit, default emission is pinned
  by tests, and generated vs. hand-written code is kept separate.
- **CPU-first development.** The test suite executes emitted source through a CPU
  Triton double, so no GPU is needed to contribute.

## Install

FlowGPU is not on PyPI yet. Install from source or directly from Git:

```sh
# from a checkout
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# or directly from GitHub
python -m pip install "git+https://github.com/KIOSHIROI/FlowGPU.git"
```

Requires **Python 3.11, 3.12 or 3.13**.

## Quickstart (no GPU needed)

```sh
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit summary
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit triton
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run
```

`--emit summary` prints the typed pipeline, `--emit triton` prints the generated
Triton module, and `--run` executes the reference backend and prints outputs.
After installation the same commands are available as `flowgpu ...`.

On a CUDA machine with PyTorch and Triton installed:

```sh
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run-gpu
```

It compares generated Triton against the reference backend and exits non-zero if
the device cannot be used; it never silently falls back to torch.

## Python API

```python
from compiler import CompileError, compile_source

source = """
pipeline add {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
"""

compiled = compile_source(source)
pipeline = compiled.pipelines[0]
print(pipeline.triton.host)      # generated host wrapper
print(pipeline.triton.kernels[0].source)  # generated Triton kernel
print(pipeline.reference.host)   # CPU reference implementation
```

The public surface is `compile_source`, `compile_pipelines`, `CompileError` and
`CompileResult`. See the [Python API guide](docs/user-guide/api.md) for the objects and
common options.

## What works today

| Area | Status |
|---|---|
| Typed dataflow DSL, shape/dtype checking | ✅ supported |
| Pointwise, reductions, layout views, scans | ✅ supported with documented contracts |
| Triton source emission + host wrappers | ✅ CPU-tested; device path opt-in |
| Portable reference backend | ✅ runs on CPU |
| CUDA platform profiles | ✅ default |
| Ascend / KunlunXin profiles | ⚠️ experimental; some contracts fail closed |
| Full PyTorch operator coverage | ❌ not a goal yet |
| Production stability / ABI guarantees | ❌ not yet |

## Documentation

Start here:

- [Getting started](docs/user-guide/getting-started.md)
- [Tutorial: your first pipeline](docs/user-guide/tutorial.md)
- [Examples catalog](docs/user-guide/examples.md)
- [Language overview](docs/user-guide/language.md)
- [CLI guide](docs/user-guide/cli.md)
- [Python API](docs/user-guide/api.md)
- [Backends and platform support](docs/user-guide/backends.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [FAQ](docs/user-guide/faq.md) · [Troubleshooting](docs/user-guide/troubleshooting.md)

The language contract is [Design.md](Design.md); the grammar is [EBNF](EBNF).
The deeper design/evidence notes live in [`docs/`](docs/).

## Project layout

| Path | Purpose |
|---|---|
| `compiler/` | parser, typed IR, optimization, backends, CLI |
| `migration/` | operator coverage/evidence tooling |
| `examples/` | runnable `.fgpu` programs |
| `models/` | small model operator manifests |
| `test/` | public CPU test suite |
| `docs/` | user guides and technical documentation |

## Development

```sh
.venv/bin/python -m pip install -e ".[dev,docs]"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
mkdocs build --strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the evidence-first contribution
rules, [PUBLIC_TESTS.md](docs/PUBLIC_TESTS.md) for the public test scope, and
[MAINTAINERS.md](MAINTAINERS.md) for maintenance.

## Support

- Usage questions and bug reports: [GitHub Issues](https://github.com/KIOSHIROI/FlowGPU/issues)
- Support policy: [SUPPORT.md](SUPPORT.md)
- Security reports: see [SECURITY.md](SECURITY.md)
- Community expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

There is no support SLA; this is a volunteer-maintained prototype.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
