# Lesson 5 数据流：SwiGLU 与融合

## 公式

```text
gate   = x @ w_gate
up     = x @ w_up
act    = silu(gate)
hidden = act * up
out    = hidden @ w_down
```

## 形状流

```text
x       [B, D]
w_gate  [D, H]
w_up    [D, H]
w_down  [H, D]

gate    [B, H]
up      [B, H]
act     [B, H]
hidden  [B, H]
out     [B, D]
```

## 哪些中间值需要缓冲

编译结果是 3 个 kernel：

```text
k0: matmul
k1: matmul -> silu -> mul
k2: matmul
```

| 值 | 是否物化 | 原因 |
|---|---|---|
| `gate` | 是 | k0 的结果，k1 的 prologue 要读它 |
| `up` | 否 | k1 的 matmul，直接进 epilogue |
| `act` | 否 | k1 内的 `silu(gate)` |
| `hidden` | 是 | k1 的结果，k2 要读它 |
| `out` | 是 | 最终输出 |

关键点：

> `silu(gate) * up` 被放进 k1 的 epilogue；  
> `up` 和 `act` 不落地，`hidden` 落地。

## 为什么 gate/up 不合并成一个 matmul

`gate` 和 `up` 是两次独立的 `x @ W`，共享 `x` 但权重不同。

编译器选择：

- k0 算 gate 并物化；
- k1 算 up，同时读 gate、算 silu、乘 up。

没有把两次 matmul 合成一个 kernel，因为那需要同时持有两个累加器和两套权重 tile。

## 练习

1. `gate` / `hidden` / `out` 的形状？
2. 为什么 `gate` 必须物化？
3. 为什么 `up` / `act` 不物化？
4. k1 的 epilogue 包含哪些操作？
5. 如果 `silu(gate)` 还要被 k2 使用，融合会怎么变？
