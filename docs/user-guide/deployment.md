# Prepared deployment

FlowGPU separates **preparation** from the hot call. Preparation may compile,
measure and freeze configurations; the returned callable contains a fixed
launch and exact metadata guards, so it does no autotuner dispatch in the hot
path.

## Frozen autotune

`prepare_pipeline` is the CUDA deployment entry point:

```python
from flowgpu.compiler.runtime.preparation import prepare_pipeline

run, record = prepare_pipeline(
    source,
    pipeline="p",
    args=(x, y),
    cache_dir=".flowgpu-cache",
    validator=lambda result: check(result),
    autotune_fold=True,          # or autotune=True for GEMM
)
run(x, y)
```

What it does:

1. compiles/prepares the emitted module;
2. executes the autotuned kernels once and captures their winning configs;
3. calls `freeze_autotune` to rewrite the host/kernel source into an ordinary
   guarded launch;
4. validates the prepared kernel independently;
5. caches the prepared source and selections by compiler/source/metadata/device
   identity.

The prepared callable guards:

- shape and stride;
- dtype and device;
- pointer alignment modulo 128;

and raises `ValueError` when any of them changes. **The cache stores source and
configurations, never outputs or input contents.** Preparation must happen
outside CUDA Graph capture; an unchanged cache hit performs no tuning.

`prepare_pipeline` is intentionally explicit rather than a CLI flag: it needs
CUDA tensors and an independent validator.

## Per-shape specialization

FlowGPU does not dispatch symbolic shapes at runtime. A deployment with several
fixed shapes can build one prepared runner per input metadata tuple:

```python
from flowgpu.compiler.runtime.shape_specialization import ShapeSpecializedRunner

runner = ShapeSpecializedRunner(
    lambda x, y: prepare_pipeline(
        source, pipeline="p", args=(x, y),
        cache_dir=".flowgpu-cache",
        validator=lambda result: check(result),
        autotune_fold=True,
    )[0]
)

runner(x1, y1)   # builds/caches metadata for shape 1
runner(x1, y1)   # cache hit, no tuning
runner(x2, y2)   # new shape/stride/device tuple -> explicit rebuild
```

The key includes shape, stride, dtype, device and alignment. The cache is
bounded and evicts the least recently used tuple. This is **explicit per-shape
specialization**, not dynamic-shape compilation.

## Candidate spaces

`flowgpu.compiler.runtime.tuning_space` describes measured candidate families as
labels, options and the geometry they emit. `runtime.tile_selection` uses it for
cross-domain tiles and filters candidates by the target's contract (for
non-CUDA targets only the materialize arm is offered until an indexed contract
is validated).

## Limits

- Preparation requires CUDA tensors and an independent validator.
- The resulting guard is exact-context; a changed shape requires rebuilding.
- Tuning results are evidence for the recorded context, not a universal default.
- Autotune dispatch overhead is removed, but cold preparation cost remains and
  is reported separately by the preparation record.
