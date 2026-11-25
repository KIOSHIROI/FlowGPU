# Lesson 6: materialized vs fused

Same source, two compile modes:

```python
compile_source(source, shape_bindings=...)
compile_source(source, shape_bindings=..., materialize_all=True)
```

Run:

```bash
python lessons/06-materialized/run_lesson.py
```

The dataflow does not change. Only the decision to keep `up` in registers or
write it to a buffer changes.
