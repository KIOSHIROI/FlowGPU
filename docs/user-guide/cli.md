# CLI guide

The CLI compiles, inspects and runs `.fgpu` programs.

```sh
flowgpu path/to/program.fgpu --emit summary
# or, from a checkout without installing the console script:
python -m compiler.cli path/to/program.fgpu --emit summary
```

## Inspect

```sh
flowgpu program.fgpu --emit summary     # typed pipeline
flowgpu program.fgpu --emit dataflow    # SSA dataflow graph
flowgpu program.fgpu --emit fusion      # fusion groups
flowgpu program.fgpu --emit buffers     # buffer/storage plan
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

## Help and errors

```sh
flowgpu --help
flowgpu --version
```

Compile errors include a source position. For common failures, see
[Troubleshooting](troubleshooting.md).

The `flowgpu-migration` console script provides operator-coverage and evidence
tooling; run `flowgpu-migration --help` for its commands.
