# Lesson 8: generated vs handwritten Triton

This lesson reuses the review packet:

- [`review/swiglu_split/source.fgpu`](../review/swiglu_split/source.fgpu)
- [`review/swiglu_split/generated_module.py`](../review/swiglu_split/generated_module.py)
- [`review/swiglu_split/handwritten.py`](../review/swiglu_split/handwritten.py)
- [`review/swiglu_split/dataflow.md`](../review/swiglu_split/dataflow.md)

Run:

```bash
python lessons/08-handwritten/compare.py
```

No GPU needed: it compiles the DSL, prints the generated kernel shape and the
handwritten address lines, then runs the CPU reference.

The point: the same semantics can be one fused kernel or two explicit kernels.
Which one is easier to audit is a separate question from which one is faster.
