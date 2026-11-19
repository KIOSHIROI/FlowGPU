# FAQ

## Is FlowGPU production-ready?

No. It is an alpha compiler prototype. It is useful for research and
experimentation, but it does not yet promise API/ABI stability, full operator
coverage or production support.

## Do I need a GPU?

No, for compiling and for the reference backend. `--run` works on a CPU-only
machine. A GPU with PyTorch and Triton is needed only for `--run-gpu`.

## Why is FlowGPU not on PyPI?

The project is in an early alpha phase. Install from the repository or from Git:

```sh
python -m pip install "git+https://github.com/KIOSHIROI/FlowGPU.git"
```

A future release may publish wheels once the public API and release process are
stable.

## Which Python versions are supported?

CPython 3.11, 3.12 and 3.13. The CI matrix runs the public suite on all three.

## Why did the compiler refuse my program?

FlowGPU prefers an explicit diagnostic over a silent approximation. Common
reasons are an unsupported operator, a dtype/shape mismatch, a layout that a
backend cannot lower exactly, or a platform contract that has not been
validated. The error includes a source position and a category.

If you believe the refusal is wrong, open an issue with the minimal `.fgpu`
program and the exact command.

## Why does `--run-gpu` exit with code 2?

It requires PyTorch, Triton and a usable CUDA device. Without them it exits 2 and
points back to `--run`; it never falls back to torch kernels. Install the
optional runtime dependencies or use `--run`.

## Is the generated code traceable?

Yes. `--emit triton` prints the module; `--emit reference` prints the reference
implementation; and the Python API exposes `pipeline.triton` and
`pipeline.reference`. Generated and extern kernels are reported separately by
the evidence tooling.

## Can I use it in a commercial project?

The code is MIT licensed. The project is alpha, so evaluate stability yourself.
See [LICENSE](https://github.com/KIOSHIROI/FlowGPU/blob/main/LICENSE).

## How do I contribute?

See [CONTRIBUTING.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/CONTRIBUTING.md).
Bug fixes should include a regression test; default emitted bytes should stay
unchanged unless the change is explicitly about emission.

## How do I report a security issue?

Do not open a public issue. Follow
[SECURITY.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/SECURITY.md).
