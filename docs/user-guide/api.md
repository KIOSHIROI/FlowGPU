# Python API

The public API is intentionally small. Import it from the `flowgpu` package; the legacy `compiler` package remains available while the pre-1.0 namespace transition is in progress.

```python
from flowgpu import CompileError, compile_source, compile_pipelines
```

The API is **pre-1.0**: names and result objects may change in minor releases
until the public surface is frozen. Pin a commit or version if you depend on it.

## Compile source

```python
from flowgpu import compile_source

source = """
pipeline add {
  input a: fp16[1024]
  input b: fp16[1024]
  out = add(a, b)
  output out
}
"""

result = compile_source(source)
pipeline = result.pipelines[0]

print(pipeline.pipeline.name)        # add
print(pipeline.triton.host)          # generated host wrapper
print(pipeline.triton.kernels[0].source)
print(pipeline.reference.host)       # reference implementation
```

`compile_source` returns a `CompileResult` with:

| Attribute | Meaning |
|---|---|
| `pipelines` | one `PipelineResult` per pipeline in the source |
| `source_path` | diagnostic source path, if supplied |

Each `PipelineResult` exposes the intermediate and backend products:

| Attribute | Meaning |
|---|---|
| `pipeline` | parsed pipeline declaration |
| `semantic` | typed semantic program |
| `dataflow` | typed SSA dataflow graph |
| `fusion` | fusion groups |
| `memory` | buffer/storage plan |
| `triton` | emitted Triton program (kernels, host source, refusals) |
| `reference` | emitted reference program |
| `platform_conflicts` | contract conflicts, if any |

## Common options

Only a small subset is needed for normal use. All are keyword-only after
`source`.

| Option | Purpose |
|---|---|
| `source_path=` | path used in diagnostics |
| `platform_config=` | target profile (see `compiler.backend.platform_configs`) |
| `shape_bindings=` / `shape_constraints=` | specialize or constrain symbolic dimensions |
| `index_fusion=` | enable/disable index-region fusion (default on) |
| `reduction_fusion=` | enable/disable reduction fusion (default on) |
| `layout_views=` | layout-view policy; `None` uses the platform default |
| `scalar_placement=` | `"off"` or `"sps"` scalar placement |
| `emit=` | set `False` to skip backend source emission |
| `cache_dir=` | content-addressed emission cache |

For the complete signature, see
[`compiler/compile.py`](https://github.com/KIOSHIROI/FlowGPU/blob/main/compiler/compile.py).
The CLI exposes many advanced strategy options; use `flowgpu --help` for the
list.

## Errors

Invalid programs raise `CompileError`:

```python
from flowgpu import CompileError, compile_source

try:
    compile_source("pipeline p { input x: fp16[4] out = add(x, x) output nope }")
except CompileError as error:
    print(error)
```

Diagnostics include a source position, a category and a human-readable message.
The compiler chooses to refuse a construct rather than silently approximating it.

## Compile already-parsed pipelines

`compile_pipelines(pipelines, **options)` is the lower-level entry used by custom
frontends. Most users should call `compile_source`.

For examples of frontends, see
[`compiler/frontends`](https://github.com/KIOSHIROI/FlowGPU/tree/main/compiler/frontends).
