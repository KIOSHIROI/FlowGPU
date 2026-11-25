# Lesson 8 数据流：生成版 vs 手写版

同一份 `swiglu_oai_split`：

```fgpu
gate, up = split(x)
...
```

## 生成版：1 个 primitive kernel

```text
split, split, broadcast_like, minimum, ...
```

`split` 不产生独立缓冲，而是变成两条取数地址：

```text
gate[r,c] = x[r * 16 + c]
up[r,c]   = x[r * 16 + 8 + c]
```

## 手写版：2 个 kernel

```text
split_copy_kernel: gate/up 先写入缓冲
swiglu_kernel:     读取 gate/up，算点式公式
```

地址仍然相同，但被拆成两个阶段：

```text
split_copy:  gate_buf[r,c] = x[r*16 + c]
             up_buf[r,c]   = x[r*16 + 8 + c]
swiglu:      out[r,c] = f(gate_buf[r,c], up_buf[r,c])
```

## 对照

| 问题 | 生成版 | 手写版 |
|---|---|---|
| split 的输出 | 索引加载 | 显式缓冲 |
| kernel 数 | 1 | 2 |
| 中间缓冲 | 无 | gate/up |
| 地址表达式 | 编译器生成 | 手写显式 |

## 练习

1. 生成版为什么不需要 gate/up 缓冲？
2. 手写版为什么需要？
3. 两份实现的 `gate` 地址是否相同？
4. 如果 split 改成奇偶交错，哪份实现更容易改错？
5. 哪份实现更适合作为“地址对照”的基准？
