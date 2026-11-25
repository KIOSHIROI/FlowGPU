# Lesson 8: generated vs handwritten Triton

Entry point: [`lessons/08-handwritten/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/08-handwritten).

The lesson reuses the SwiGLU split review packet:

| piece | file |
|---|---|
| DSL | `review/swiglu_split/source.fgpu` |
| generated | `review/swiglu_split/generated_module.py` |
| handwritten | `review/swiglu_split/handwritten.py` |
| dataflow | `review/swiglu_split/dataflow.md` |

Compile shape:

```text
generated:   1 primitive kernel
             split -> split -> ... (split folded into index loads)
handwritten: 2 kernels
             split_copy -> swiglu
```

Addresses are the same in both:

```text
gate[r,c] = x[r*16 + c]
up[r,c]   = x[r*16 + 8 + c]
```

The difference is where they live:

- generated: one kernel, no `gate`/`up` buffers;
- handwritten: `split_copy` writes `gate`/`up`, then a pointwise kernel reads
  them.

Run without a GPU:

```bash
python lessons/08-handwritten/compare.py
```

It prints the generated kernel ops, the handwritten address lines, and the CPU
reference.

Teaching point: **the same semantics can be one fused kernel or two explicit
kernels; which is easier to audit is separate from which is faster.**
