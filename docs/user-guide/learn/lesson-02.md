# Lesson 2: layout operators

Entry point: [`lessons/02-layout/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/02-layout).

```fgpu
pipeline layout_mix {
    input x : fp32[N,8]
    a, b = split(x)
    interleaved = interleave(a, b)
    joined = concat(a, b)
    out = add(interleaved, joined)
    output out
}
```

The same `a`/`b` feed two different address functions:

```text
interleaved[r, 2c]     = a[r,c]
interleaved[r, 2c + 1] = b[r,c]

joined[r, c]     = a[r,c]
joined[r, H + c] = b[r,c]
```

Yet the compiler emits **one primitive kernel**:

```text
split, split, interleave, concat, add
```

No `a`/`b` buffers. The two consumers become two address computations inside
the same kernel.

Run:

```bash
python lessons/02-layout/run_lesson.py
```

The exercise asks the learner to predict even/odd interleave addresses and the
concat addresses before running. The teaching point:

> **layout is an address function; multiple uses do not force materialization.**
