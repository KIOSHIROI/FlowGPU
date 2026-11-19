# Architecture

FlowGPU is a small compiler pipeline. Each stage has a narrow contract and a
CPU-side test surface.

```text
source .fgpu
   |
   v
lexer / parser (Lark) -> AST
   |
   v
type and shape checking -> typed SSA dataflow IR
   |
   v
fusion / index analysis -> liveness / buffer plan
   |
   v
backend:  Triton source emitter      reference Python interpreter
   |
   v
host wrapper (allocations, launches, scalar ABI)
```

The reference backend is the semantic oracle. The Triton backend must produce
the same values; tests execute emitted source through a CPU Triton double, so CI
does not need a GPU.

Design rules:

1. Refuse constructs that cannot be lowered exactly.
2. Default source emission is byte-identical across changes unless the change is
   explicitly an emission change.
3. Backend behaviour is a measured contract, not an inference from another chip.
4. Hand-written Triton is recorded as extern.

For the full technical description see
[ARCHITECTURE.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/ARCHITECTURE.md).
