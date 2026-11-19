# Limitations

FlowGPU is an alpha compiler prototype. The following are deliberate limits, not
hidden gaps.

## Product status

- Not production-ready; no support SLA.
- Not yet published on PyPI. Install from the repository or a GitHub Release.
- The API is pre-1.0 and may change in minor releases; see
  [Stability and versioning](stability.md).
- Generated Triton source and host ABIs are not a stable serialized interface.

## Language and compiler

- The DSL has no arbitrary control flow or loops; a pipeline is a dataflow graph.
- Operator coverage is the subset in the
  [support matrix](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/SUPPORT_MATRIX.md),
  not the full PyTorch operator set.
- Unsupported constructs are refused with a diagnostic rather than approximated.
- Intended for inference-operator pipelines, not training/autograd.

## Dynamic shapes

FlowGPU compiles static shapes. A symbolic dimension can be specialized with
`shape_bindings`, and `DynamicShapeRunner` caches one static specialization per
observed input metadata tuple. There is still no symbolic-shape kernel or
general runtime dynamic-dispatch path; a new shape is an explicit compile miss.

## Platforms

- Linux and macOS are CI-tested; native Windows is not, use WSL2.
- CUDA is the default platform. Ascend/KunlunXin profiles are experimental and
  contract-gated.
- `--run-gpu` requires PyTorch, Triton and a compatible CUDA device; the compiler
  and reference backend do not.
- Cross-chip performance claims are tied to their measurement evidence and are not
  general guarantees.

## Evidence and migration tooling

- The migration CLI works on model operator inventories; it does not trace a model
  automatically.
- Public releases do not include competition submissions, private evidence or
  organizer task packages. Reports in the docs are historical research notes, not
  redistributed data.
- Reproducing a historical benchmark may require a device and environment that
  are not part of this repository.

If a limitation blocks your use case, open an issue with a minimal reproducer;
the project prefers explicit refusals over silent approximations.
