# Lesson 3 数据流：跨列归约

## 公式

```text
total[r]  = sum_c x[r,c]
out[r,c]  = x[r,c] * total[r]
```

## 形状流

```text
x          [N, D]
total      [N]      -- 对最后一维 D 求和
total_full [N, D]   -- 把 total 按行广播回来
out        [N, D]
```

## 地址流

```text
total[r]  = sum_{c=0}^{D-1} x[r * D + c]
out[r,c]  = x[r * D + c] * total[r]
```

注意广播的是 **行号 `r`**，不是列号 `c`：

```text
total_full[r,c] = total[r]
```

## 编译结果

这个例子生成 **1 个 primitive kernel**：

```text
sum_last -> broadcast_like -> mul
```

`total` 不单独占缓冲：归约结果留在寄存器里，随后按行广播给同一行的输出。

## 尾部

`D` 不一定是 2 的幂。生成代码需要：

```text
mask = column < D
```

归约时只对有效列求和，广播/乘法时也只写有效列。

## 练习

1. `total` 的形状是什么？
2. `total` 是按行还是按列求和？
3. `total_full[r,c]` 的地址里有 `c` 吗？
4. 为什么这个例子仍然只有 1 个 kernel？
5. 当 `D=5` 时，归约 mask 应该是什么？
