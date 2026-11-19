# Upgrading from pre-0.18

**0.18.0 is a breaking pre-1.0 release.** It removes the generic top-level
`compiler` and `migration` packages so the wheel has one importable namespace:
`flowgpu`.

## Imports

Before:

```python
from compiler import compile_source
```

After:

```python
from flowgpu import compile_source
```

The same applies to `CompileError`, `CompileResult` and `compile_pipelines`.

## CLI

Before:

```sh
python -m compiler.cli program.fgpu --emit summary
```

After:

```sh
python -m flowgpu program.fgpu --emit summary
# or the installed console script:
flowgpu program.fgpu --emit summary
```

`flowgpu-migration` remains the console script for the migration tooling;
its module moved to `flowgpu.migration`.

## Source paths

| Before | After |
|---|---|
| `compiler/` | `flowgpu/compiler/` |
| `migration/` | `flowgpu/migration/` |

This is why the 0.18.0 wheel reports exactly one top-level package, `flowgpu`.
There is intentionally no `compiler` compatibility shim: keeping that generic
top-level name would preserve the namespace collision the migration removes.
Pin `flowgpu<0.18` if an intermediate dependency still uses the old imports.
