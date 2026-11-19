# CLI guide

The CLI compiles, inspects and runs `.fgpu` programs.

```sh
flowgpu path/to/program.fgpu --emit summary
# or, from a checkout without installing the console script:
python -m flowgpu path/to/program.fgpu --emit summary
```

## Inspect

```sh
flowgpu program.fgpu --emit summary     # typed pipeline
flowgpu program.fgpu --emit dataflow    # SSA dataflow graph
flowgpu program.fgpu --emit fusion      # fusion groups
flowgpu program.fgpu --emit buffers     # buffer/storage plan
flowgpu program.fgpu --emit planning    # launches/traffic/buffer cost summary
flowgpu program.fgpu --emit triton      # generated Triton module
```

These stages are the fastest way to understand what the compiler did.

## Run

```sh
flowgpu program.fgpu --run              # portable reference backend, CPU
flowgpu program.fgpu --run-gpu          # generated Triton on CUDA
```

`--run` does not require `torch`. `--run-gpu` requires PyTorch, Triton and a
CUDA device; it compares generated Triton against the reference backend and
exits with code 2 when that environment is unavailable.

## Select a platform

```sh
flowgpu program.fgpu --platform cuda
flowgpu program.fgpu --platform cuda-tuned
flowgpu program.fgpu --platform ascend-wide
```

`cuda` is the default. Non-CUDA profiles are experimental; unsupported contracts
fail closed with a diagnostic instead of being silently approximated.

## Shape specialization

```sh
flowgpu program.fgpu --shape N=1024 --emit summary
```

Symbolic dimensions are normally checked at runtime by the generated host code.
`--shape` specializes them for inspection or a concrete generated module.

## Environment diagnostics

If installation or device execution is unclear, print a compact environment report:

```sh
flowgpu --doctor
```

It reports the FlowGPU, Python, lark, torch and Triton versions and whether
the reference backend is available; it never runs a GPU kernel. For issue
reports or automation use the machine-readable form:

```sh
flowgpu --doctor --doctor-format json
```

## Optimization controls

Graph identity folding, CSE and DCE run by default. Use `--no-optimize` to
compile the conservative graph, or `--no-identity-folding` to keep identity view
nodes while retaining CSE/DCE. `--emit planning` shows the structural plan cost.

## Help and errors

```sh
flowgpu --help
flowgpu --version
```

Compile errors include a source position. For common failures, see
[Troubleshooting](troubleshooting.md).

The `flowgpu-migration` console script provides operator-coverage and evidence
tooling. A GPU-free demo is available with:

```sh
flowgpu-migration demo
```

Run `flowgpu-migration --help` for all commands.
