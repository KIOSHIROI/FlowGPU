# Lesson 7: address and masks

Entry point: [`lessons/07-masks/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/07-masks).

```fgpu
pipeline masked_matmul {
    input a : fp32[M,K]
    input b : fp32[K,N]
    out = matmul(a, b)
    output out
}
```

Lesson shape: `a [3,7]`, `b [7,5]`, tile `128/128/64`.

```text
grid = (ceil(3/128), ceil(5/128)) = (1, 1)
```

One program covers the output, but the tile is much larger than `3x5`, so the
generated code contains three masks:

```text
A load: (m < 3) & (k < 7)     other=0.0
B load: (k < 7) & (n < 5)     other=0.0
store:  (m < 3) & (n < 5)
```

- out-of-range **loads** use `other=0.0`, so they add zero to the accumulator;
- out-of-range **stores** must be masked, or they write out of bounds.

K loop: `cdiv(7, 64) = 1`.

Run without a GPU:

```bash
python lessons/07-masks/run_lesson.py
```

It prints the actual generated `tl.load` and `tl.store` lines.
