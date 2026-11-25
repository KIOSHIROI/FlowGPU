# Lesson 2 数据流：`split` / `interleave` / `concat`

## 公式

```text
a, b       = split(x)          # 前后两半
interleaved = interleave(a,b)  # 交替 a,b
joined      = concat(a,b)      # 先 a 后 b
out         = interleaved + joined
```

## 形状流

```text
x           [N, 8]
a, b        [N, 4]
interleaved [N, 8]
joined      [N, 8]
out         [N, 8]
```

## 地址流

对行 `r`、列 `c`（`0 <= c < H`）：

```text
a[r,c] = x[r * 2H + c]
b[r,c] = x[r * 2H + H + c]

interleaved[r, 2c]     = a[r,c]
interleaved[r, 2c + 1] = b[r,c]

joined[r, c]     = a[r,c]
joined[r, H + c] = b[r,c]
```

`interleave` 和 `concat` 用的是同一对输入，但输出地址完全不同：

- `concat`：先放完 `a` 的 `H` 列，再放 `b`；
- `interleave`：按列交替。

## 中间缓冲

这个例子生成 **1 个 primitive kernel**：

```text
split, split, interleave, concat, add
```

`a` / `b` 虽然被两个消费者读取，但都留在同一个 primitive kernel 内，
没有单独的 `a` / `b` 缓冲。它们的两个用途直接变成两条地址计算：

```text
interleaved[r, 2c]     = a[r,c]
interleaved[r, 2c + 1] = b[r,c]

joined[r, c]     = a[r,c]
joined[r, H + c] = b[r,c]
```

这正是布局融合的关键：**同一个值可以被多个地址函数使用，而不必先落地。**

## 练习

1. `interleaved[r, 2c]` 来自 `a` 还是 `b`？
2. `joined[r, H+c]` 来自 `a` 还是 `b`？
3. `a` / `b` 被两个消费者读取，为什么仍然可以没有缓冲？
4. 如果交换 `interleave(a,b)` 的参数，哪些地址变？
