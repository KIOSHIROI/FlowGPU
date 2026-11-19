# Compiler Architecture

FlowGPU is organized as a layered flowgpu.compiler. The top-level `flowgpu.compiler.*` modules remain compatibility facades for existing scripts and tests; new implementation code should import from the layer packages directly.

```text
frontend -> ir -> analysis -> optimization -> backend -> runtime
                         ^              ^
                         |              |
                       policy/schedule --+
```

## Layers

### `frontend/`

Source grammar, lexer/parser, AST, symbol table, operator typing, shape constraints, diagnostics, and source positions.

### `ir/`

Semantic IR, SSA-like dataflow IR, and post-fusion structured lowering IR. These modules describe programs and lowering contracts; they do not choose device-specific tile parameters.

### `analysis/`

Liveness, traffic, precision, stability, portability, module complexity, and kernel dependency analysis.

### `optimization/`

Fusion and execution-plan transformations, including indexed/reduction/cross-domain/row-gather fusion and strategy cost models.

### `backend/`

Backend contracts, platform policy, Triton/reference/library lowering, primitive and reduction emitters, custom-op emission, and platform-specific lowering helpers.

### `runtime/`

Generated-module cache and runtime integration. Cache fingerprints include the canonical root grammar and every implementation-layer Python module.

## Entry points

`flowgpu/compiler/compile.py` remains the public orchestration entry point. `flowgpu/compiler/cli.py` remains the command-line entry point. `triton_backend.py` and `reference.py` are intentionally still single backend compilation units; splitting them is deferred until their shared helper boundaries are stable.

## Compatibility policy

Legacy paths such as `flowgpu.compiler.ast_node`, `flowgpu.compiler.fusion`, and `flowgpu.compiler.triton_backend` are thin facades that re-export the moved implementation. Internal implementation imports should use the layer path directly. `flowgpu/compiler/grammar.lark` is the canonical grammar; the frontend parser and lexer both resolve it from that path.

## - `flowgpu/compiler/extern.py` is the documented escape hatch for hand-written kernels:
  it wraps author Triton/host sources in the generated-module envelope
  (provenance, content hash, structural checks) without pretending the compiler
  emitted them.  Adoption rule: promote a construct into the language only when
  at least two independent tasks need it (docs/FRONTEND_STRATEGY_20261026.md).
`flowgpu/compiler/shared.py`

Pure functions of the graph and the program's names -- `Names`, `identifier`,
`product`, `group_operands/results`, `store_rotation_chain`, `is_pointwise`,
`row_key`, the primitive/value-layer op-name groups, `compiler_fingerprint` --
live in `flowgpu/compiler/shared.py`.  They were scattered across `backend/backends.py`,
`optimization/*` and `runtime/cache.py`, which is why `ir`, `analysis` and
`optimization` used to import layers above themselves just to reach them.  The
module imports only `frontend.ast_node`, so every layer may import it.

The 21.6 dtype/precision rules live in `frontend/dtype_rules.py`
(`analysis/precision.py` is a compatibility facade) because the frontend
enforces them at the call site and may not import `analysis`.

## Dependency rules

- Frontend and IR do not import backend modules.
- When two rules below point in opposite directions, `shared.py` is the place
  to put the helper, not a new upward import.
- Measurement harnesses under `optimization/` (`strategy*.py`, `gather_*.py`)
  are the one documented exception: they compile and time candidate modules, so
  they import `flowgpu.compiler.compile`, `backend.platform_configs`,
  `backend.triton_backend` and `runtime.runtime` at module level.  They are
  measurement tools that live in `optimization/` for historical reasons, not
  passes: the strict rule holds for `fusion.py`, `indexed_fusion.py`,
  `reduction_fusion.py` and `cross_domain_fusion.py`, which only consume
  `flowgpu.compiler.shared`, IR, analysis and policy.
- Analysis may consume IR and frontend types, but does not emit device source.
- Optimization consumes IR and analysis contracts.
- Backends consume the preceding layers and may add device-specific lowering.
- Runtime may invoke compilation and generated artifacts, but core IR does not depend on runtime.
- Experimental strategy modules belong under optimization until they become stable public passes.
