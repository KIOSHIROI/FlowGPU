# Lesson 5: fusion in SwiGLU

Entry point: [`lessons/05-fusion/`](https://github.com/KIOSHIROI/FlowGPU/tree/main/lessons/05-fusion).

```fgpu
pipeline fused_mlp {
    input x      : fp32[B,D]
    input w_gate : fp32[D,H]
    input w_up   : fp32[D,H]
    input w_down : fp32[H,D]
    gate = matmul(x, w_gate)
    up = matmul(x, w_up)
    act = silu(gate)
    hidden = mul(act, up)
    out = matmul(hidden, w_down)
    output out
}
```

Three kernels:

```text
k0: matmul
k1: matmul -> silu -> mul
k2: matmul
```

Which values are materialized:

| value | buffer? | why |
|---|---|---|
| `gate` | yes | k1's prologue reads it |
| `up` | no | k1's matmul, straight to epilogue |
| `act` | no | `silu(gate)` inside k1 |
| `hidden` | yes | k2 reads it |
| `out` | yes | final output |

The teaching point:

> `silu(gate) * up` becomes k1's epilogue.  
> `up` and `act` never get a buffer; `gate` and `hidden` do.

Run without a GPU:

```bash
python lessons/05-fusion/run_lesson.py
```
