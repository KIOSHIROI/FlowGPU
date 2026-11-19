# FlowGPU

FlowGPU is a compiler for a small dataflow DSL that targets Triton and a portable
reference backend. Write an operator pipeline, get typed SSA dataflow, fusion,
buffer reuse and emitted Triton.

**Status:** research/compiler prototype. Not a production dependency.

## Quickstart

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# print the pipeline / generated Triton / run the reference backend
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit summary
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --emit triton
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run

# on a CUDA machine with torch + triton
.venv/bin/python -m compiler.cli examples/quickstart/add.fgpu --run-gpu
```

After install, the same CLI is available as `flowgpu`.

## What is in the compiler

```
frontend -> typed SSA dataflow IR -> fusion -> liveness/buffers
         -> backend (Triton source or reference) -> runtime host code
```

The public Python surface is:

```python
from compiler import compile_source, CompileError, compile_pipelines
```

The language contract is [Design.md](Design.md) and the grammar is [EBNF](EBNF).

## Repository layout

| Path | Purpose |
|---|---|
| `compiler/` | parser, core IR, optimization, backends, CLI |
| `migration/` | operator coverage/evidence tooling |
| `examples/` | runnable `.fgpu` programs |
| `models/` | small model operator manifests |
| `test/` | public CPU test suite |
| `docs/` | compiler, backend and evidence-protocol documentation |

## Tests

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

The suite executes generated Triton source through a CPU Triton double, so no
GPU is required for CI. Device execution is opt-in (`--run-gpu`).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Public test suite](docs/PUBLIC_TESTS.md)
- [Roadmap](docs/ROADMAP.md)
- [Maintenance and release process](docs/MAINTENANCE.md)
- [Compiler acceptance criteria](docs/COMPILER_ACCEPTANCE.md)
- [Backend contracts](docs/BACKEND_CONTRACTS_20261019.md)

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
