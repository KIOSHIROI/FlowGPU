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


## Prepared execution plan

For fixed-shape, allocation-sensitive deployment, `prepare_execution_plan` builds
a plan with caller-owned buffers:

```python
from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan

plan = prepare_execution_plan(
    source,
    pipeline="p",
    args=(x, y),
    cache_dir=".flowgpu-cache",
    validator=lambda result: check(result),
)
buffers = plan.allocate_buffers()       # allocate before CUDA Graph capture
for batch in batches:
    plan.run_into(batch.x, batch.y, buffers=buffers)
```

It prepares the same frozen host as `prepare_pipeline`, then rewrites the host
into `run_into_<pipeline>(..., buffers)` with:

- the same exact input guards;
- explicit buffer metadata/alignment and overlap checks;
- no per-call output allocation.

The caller owns buffer lifetime; use a distinct buffer set for overlapping
streams. A multi-pipeline source is accepted: the named pipeline is selected
before preparation. Preparation itself still requires CUDA tensors and an
independent validator, and must happen outside CUDA Graph capture.


## Fixed-shape CUDA Graph

For a fixed metadata context, capture a prepared plan once and replay only the
graph:

```python
from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
from flowgpu.compiler.runtime.graph import GraphRunner

plan = prepare_execution_plan(source, pipeline="p", args=(x, y),
                              cache_dir=".flowgpu-cache",
                              validator=lambda result: check(result))
runner = GraphRunner(plan, (x, y))

runner.inputs[0].copy_(new_x)   # caller updates the static buffers
runner.inputs[1].copy_(new_y)
out = runner.replay()           # no guards, allocation or Triton dispatch
```

`GraphRunner(plan, args)` allocates static inputs and captures the prepared plan.
`replay()` is the hot path; `runner(*args)` is a convenience that copies caller
tensors into the static inputs first and is therefore slower. For a caller that
already owns stable buffers, `GraphRunner(plan, args, copy_inputs=False)` captures
those buffers directly; update `runner.inputs` in place and call `replay()`, with
no per-call copy.

The captured outputs alias the plan buffers: copy them before the next replay.
Capture is fixed to shape, stride, dtype, device and pointer identity; a new
metadata context needs a new capture. Capture outside an existing CUDA Graph
capture.


## Symbolic shape dispatch

When every operator in the program supports runtime extents, compile once with
no `shape_bindings` and call the same module with different shapes:

```python
from flowgpu.compiler.runtime.symbolic_shape import SymbolicShapeRunner

runner = SymbolicShapeRunner(source, "vectoradd", ".flowgpu-cache")
runner(x_128, y_128)   # [128, 128]
runner(x_64, y_64)     # [64, 64]; same compiled module
```

Verified on H100 GPU1 for vectoradd (128x128, 64x64, 17x19), matmul
(64x128x64, 32x64x32) and softmax (8x1024, 4x512). Programs containing
literal-only operators still use `DynamicShapeRunner` or explicit
specialization; `SymbolicShapeRunner` does not silently fall back.


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

For a symbolic FlowGPU source, `DynamicShapeRunner` derives the bindings from
each call and wraps the same cache:

```python
from flowgpu.compiler.runtime.dynamic_shape import DynamicShapeRunner

runner = DynamicShapeRunner(source, "matmul", ".flowgpu-cache")
runner(a_64x128, b_128x32)   # compiles with M=64, K=128, N=32
runner(a_64x128, b_128x32)   # cache hit
runner(a_32x64,  b_64x16)    # compiles with M=32, K=64, N=16
```

Literal extents are validated; a symbolic name that appears twice must receive
the same value. A mismatch raises rather than inventing a second meaning.

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
