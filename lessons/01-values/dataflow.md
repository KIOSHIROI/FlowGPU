# Lesson 1 数据流：`add_scale`

## 公式

```text
out[r, c] = x[r, c] * scale[c] + x[r, c]
```

## 形状流

```text
x          [N, D]
scale      [1, D]
scale_full [N, D]  -- 显式广播
y          [N, D]  -- 中间值，不需要单独缓冲
out   [N, D]
```

## 地址流

```text
out[r, c] = x[r * D + c] * scale[c] + x[r * D + c]
```

`scale` 没有行偏移，只有列偏移 `c`；这是广播在地址上的表现。

## 编译决策

- `mul` 和 `add` 都是逐元素；
- `y` 只被 `add` 读一次；
- 所以 `y` 不分配缓冲，两个操作融进同一个 pointwise kernel。

## 练习

1. 把 `scale` 改成 `fp32[1,1]`，形状流哪里变？
2. 把 `scale` 改成 `fp32[N,D]`，地址流哪里变？
3. 如果把 `y` 写成 `output`，它会物化吗？
4. 这个 pipeline 应该生成几个 kernel？
