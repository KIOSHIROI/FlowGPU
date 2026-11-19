# FlowGPU

[![CI](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/ci.yml/badge.svg)](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/ci.yml)
[![CodeQL](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/codeql.yml/badge.svg)](https://github.com/KIOSHIROI/FlowGPU/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/KIOSHIROI/FlowGPU)](https://github.com/KIOSHIROI/FlowGPU/releases)
[![Docs](https://img.shields.io/badge/docs-live-blue)](https://kioshiroi.github.io/FlowGPU/)
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

Published wheel/sdist files, their `SHA256SUMS`, build-provenance attestations,
and release notes are on the [Releases page](https://github.com/KIOSHIROI/FlowGPU/releases).

Requires **Python 3.11, 3.12, 3.13 or 3.14**.

## Quickstart (no GPU needed)

```sh
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --emit summary
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --emit triton
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run
```

The one-command demo compiles the same program and prints its generated kernels:

```sh
.venv/bin/python examples/quickstart/demo.py
```

`--emit summary` prints the typed pipeline, `--emit triton` prints the generated
Triton module, and `--run` executes the reference backend and prints outputs.
After installation the same commands are available as `flowgpu ...`.

On a CUDA machine with PyTorch and Triton installed:

```sh
.venv/bin/python -m flowgpu examples/quickstart/add.fgpu --run-gpu
```

It compares generated Triton against the reference backend and exits non-zero if
the device cannot be used; it never silently falls back to torch.

## Python API

```python
from flowgpu import CompileError, compile_source

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
`CompileResult`, available from the `flowgpu` package. The former top-level
`compiler` import was removed in 0.18.0. See the [Python API guide](docs/user-guide/api.md) for the objects and
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

**Docs site:** <https://kioshiroi.github.io/FlowGPU/>

Start here:

- [Getting started](docs/user-guide/getting-started.md)
- [Tutorial: your first pipeline](docs/user-guide/tutorial.md)
- [Examples catalog](docs/user-guide/examples.md)
- [Language overview](docs/user-guide/language.md)
- [CLI guide](docs/user-guide/cli.md)
- [Python API](docs/user-guide/api.md)
- [Python integration](docs/user-guide/integration.md)
- [Backends and platform support](docs/user-guide/backends.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [FAQ](docs/user-guide/faq.md) · [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Contributing](docs/user-guide/contributing.md) · [Roadmap](docs/user-guide/roadmap.md)
- [Stability and versioning](docs/user-guide/stability.md)
- [Security and release verification](docs/user-guide/security.md)
- [Limitations](docs/user-guide/limitations.md)
- [Upgrading from pre-0.18](docs/user-guide/upgrading.md)

The language contract is [Design.md](Design.md); the grammar is [EBNF](EBNF).
The deeper design/evidence notes live in [`docs/`](docs/).

## Project layout

| Path | Purpose |
|---|---|
| `flowgpu/compiler/` | parser, typed IR, optimization, backends, CLI |
| `flowgpu/migration/` | operator coverage/evidence tooling |
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

## Project governance

FlowGPU is maintainer-led; see [GOVERNANCE.md](GOVERNANCE.md) and [MAINTAINERS.md](MAINTAINERS.md).

## Support

- Usage questions and bug reports: [GitHub Issues](https://github.com/KIOSHIROI/FlowGPU/issues)
- Support policy: [SUPPORT.md](SUPPORT.md)
- Security reports: see [SECURITY.md](SECURITY.md)
- Community expectations: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

There is no support SLA; this is a volunteer-maintained prototype.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
