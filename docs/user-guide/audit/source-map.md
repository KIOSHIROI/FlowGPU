# Kernel-to-source map

The first source-correlation artifact maps each generated kernel back to the
`.fgpu` statements it came from:

```bash
python -m flowgpu.compiler.source_map examples/batch10/swiglu_oai_split.fgpu
python -m flowgpu.compiler.source_map examples/batch10/swiglu_oai_split.fgpu --json
```

For the `split` example it prints:

```text
## swiglu_oai_split_k0
- ops: split, split, broadcast_like, minimum, ...
- DSL lines: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
  - L16: `gate, up = split(x)`
  ...
```

The two outputs of `split` map to the same line, which is the explicit
correlation the audit trail lacked before.

## Scope

This is **statement-level** correlation: kernel → DSL lines. It does not yet
map every generated `tl.load`/`tl.store` to an instruction, and it assumes the
fusion-group order matches the kernel order (checked for the examples, not a
language guarantee).

The next step is line-level correlation for layout indexing: logical
coordinate → generated address expression → tail mask.
