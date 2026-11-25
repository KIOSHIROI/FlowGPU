# Lesson 7: address and masks

Read `dataflow.md`, then run:

```bash
python lessons/07-masks/run_lesson.py
```

No GPU needed. The point: a tile larger than the matrix makes the tail masks
visible. Loads clamp with `other=0.0`; stores must be masked or they write out
of bounds.
