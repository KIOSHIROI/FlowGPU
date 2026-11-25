# Lesson 6 数据流：同一份 SwiGLU，两种编译决策

## 同一份公式

```text
gate   = x @ w_gate
up     = x @ w_up
hidden = silu(gate) * up
out    = hidden @ w_down
```

## 默认：fused

```text
k0: matmul
k1: matmul -> silu -> mul
k2: matmul
```

- `gate` 物化（k1 读）；
- `up` / `act` 融合进 k1；
- `hidden` 物化（k2 读）。

## `materialize_all=True`

```text
k0: matmul
k1: matmul
k2: pointwise: silu, mul
k3: matmul
```

- `up` 不再进 epilogue，而是先写入缓冲；
- `silu * up` 变成独立 pointwise kernel；
- `hidden` 仍然物化。

## 数据流没有变

变的只是编译决策：

| 决策 | fused | materialized |
|---|---|---|
| up 是否缓冲 | 否 | 是 |
| `silu*mul` 位置 | k1 epilogue | 独立 kernel |
| kernel 数 | 3 | 4 |

数学结果必须完全相同。

## 练习

1. 两种模式各几个 kernel？
2. materialized 多出来的 kernel 算什么？
3. 哪种模式多一次 `hidden` 的写/读？
4. 为什么两种模式的结果必须一致？
5. 如果只看数据流图，能判断该用哪种模式吗？
