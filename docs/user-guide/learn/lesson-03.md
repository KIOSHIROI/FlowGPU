# Lesson 3: reduce the last axis

Entry point: [`lessons/03-reduction/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/03-reduction).

```fgpu
pipeline row_scale {
    input x : fp32[N,D]
    total = sum_last(x)
    total_full = broadcast_like(total, x)
    out = mul(x, total_full)
    output out
}
```

Flow:

```text
x          [N,D]
total      [N]      -- sum over D
total_full [N,D]    -- broadcast total[r] across the row
out        [N,D]
```

Address:

```text
total[r] = sum_c x[r*D + c]
out[r,c] = x[r*D + c] * total[r]
```

The broadcast uses the **row** index, not the column index:

```text
total_full[r,c] = total[r]
```

The compiler emits one primitive kernel:

```text
sum_last -> broadcast_like -> mul
```

`total` stays in registers; it does not need a buffer.

Run without a GPU:

```bash
python lessons/03-reduction/run_lesson.py
```

Teaching point: **a reduction shrinks the shape, and broadcasting it back is a
row-level address function.**
