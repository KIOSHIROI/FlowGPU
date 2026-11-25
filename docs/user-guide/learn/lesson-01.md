# Lesson 1: values, shapes, broadcast

Entry point: [`lessons/01-values/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/01-values).

```fgpu
pipeline add_scale {
    input x     : fp32[N,D]
    input scale : fp32[1,D]
    scale_full = broadcast_like(scale, x)
    y = mul(x, scale_full)
    out = add(y, x)
    output out
}
```

The lesson workbook:

- `dataflow.md`: shape and address flow;
- `reference.py`: PyTorch statement;
- `run_lesson.py`: CPU compile + reference output;
- `exercise.md`: predict before running;
- `quiz.json`: five answers.

Run without a GPU:

```bash
python lessons/01-values/run_lesson.py
```

Output:

```text
kernels: 1
- add_scale_k0: broadcast_like -> mul -> add
reference out[0]: ...
```

## Teaching intent

Lesson 1 introduces only three ideas:

1. a name is a value, not a mutable variable;
2. broadcast is explicit (`broadcast_like`), and it changes the address;
3. a single-use intermediate (`y`) does not need a buffer.

The exercise asks the learner to predict shape, buffer and kernel count before
looking at the output. That prediction is the handhold.
