# Lesson 3: reduce the last axis

Read `dataflow.md`, then run:

```bash
python lessons/03-reduction/run_lesson.py
```

No GPU needed. The teaching point: a reduction produces a smaller tensor, and
broadcasting it back uses the row index, not the column index.
