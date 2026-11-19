# Troubleshooting

## `flowgpu: command not found`

Use the module form from a checkout:

```sh
python -m flowgpu --version
```

or install the package in the active environment:

```sh
python -m pip install -e .
```

Then `flowgpu` should be on `PATH`.

## `ModuleNotFoundError: No module named 'compiler'`

You are probably running Python outside the repository or the virtual
environment. Activate the environment and run from the repository root, or
install the package with `pip install -e .`.

## `ModuleNotFoundError: No module named 'lark'`

Install the runtime dependency:

```sh
python -m pip install -e .
```

## `--run-gpu` says PyTorch/Triton/CUDA is unavailable

That is expected on a CPU-only machine; the command exits with code 2 rather
than falling back to torch. Use `--run` for the reference backend, or install
`torch` and `triton` on a CUDA machine.

## A compile error points at the wrong place

Include the full diagnostic and the minimal `.fgpu` file in a bug report.
Diagnostics carry source positions and categories; a reproducible minimal
program is much easier to fix than a large one.

## A program works on CUDA but fails on Ascend/KunlunXin

Non-CUDA profiles are experimental and contract-gated. Some layout/reduction
combinations are deliberately refused until their device contracts are
validated. Try `--platform cuda` for the portable CUDA path, or report the
specific refusal with the program and target.

## `layout_views=True` fails on a non-CUDA platform

That is intentional fail-closed behaviour: the target has no validated
layout-view contract. Use `layout_views=False`, or the platform default.

## The CLI prints too much output

Use `--emit summary` for a compact view. `--emit triton` is the full generated
module and is expected to be long.

## Shapes differ from a tutorial

Examples use literal shapes for clarity. Use symbolic dimensions in your own
program when you need one compiled program to serve several shapes; the
generated host wrapper checks the runtime shapes and dtypes.

## Still stuck?

Open a GitHub issue with:

- `flowgpu --version`
- OS and Python version
- the complete command
- the minimal `.fgpu` program
- the full traceback/diagnostic
