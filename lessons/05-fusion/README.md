# Lesson 5: fusion

Read `dataflow.md`, then run:

```bash
python lessons/05-fusion/run_lesson.py
```

No GPU needed. The point: some intermediates stay in registers, others must be
materialized because another kernel reads them.
