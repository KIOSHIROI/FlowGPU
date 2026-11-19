# Stability and versioning

FlowGPU is pre-1.0. The public API is intentionally small, but some names and
result objects may change in minor releases until the surface is frozen.

## Versioning

Releases use `MAJOR.MINOR.PATCH`:

- **MAJOR** — reserved for a stable public API; no 1.0 yet.
- **MINOR** — new features, docs/CLI additions, and possibly breaking changes
  while pre-1.0. Breaking changes are called out in the
  [CHANGELOG](https://github.com/KIOSHIROI/FlowGPU/blob/main/CHANGELOG.md).
- **PATCH** — backwards-compatible fixes and documentation corrections.

Python support is **3.11–3.14**. Dropping a Python version is a minor change and
is documented in the changelog.

## What is stable

- The package name, console scripts (`flowgpu`, `flowgpu-migration`) and the
  documented install paths.
- The public imports `compile_source`, `compile_pipelines`, `CompileError` and
  `CompileResult`, subject to the pre-1.0 caveat above.
- The documented CLI commands shown in the user guide.

## What is not stable yet

- Internal modules below `flowgpu/compiler/` are implementation details; importing them
  directly is not a compatibility promise.
- Generated Triton source, host-wrapper ABI and JSON emitted by advanced
  `--emit` modes are not a stable serialized interface.
- Platform profiles marked experimental (Ascend/KunlunXin) are contract-driven
  experiments; unsupported constructs fail closed.

## Deprecation

When practical, a deprecation is announced in the CHANGELOG and docs for at
least one minor release before removal. Because the project is pre-1.0, a
security or correctness fix may require a faster change; that will be stated
explicitly.

## Reporting compatibility problems

Open an issue with `flowgpu --version`, the previous working version, a minimal
program and the exact failure. Compatibility bugs are treated as real bugs.

## Platforms

| Platform | CI status |
|---|---|
| Linux | tested on Python 3.11–3.14 |
| macOS | tested on Python 3.13 |
| Windows | not CI-verified; use WSL2 |

Device execution (`--run-gpu`) additionally requires a Linux CUDA environment
with a compatible PyTorch/Triton pair.
