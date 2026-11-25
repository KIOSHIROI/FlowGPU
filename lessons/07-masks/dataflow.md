# Lesson 7 数据流：tile 大于矩阵时的 mask

## 公式

```text
out[i,j] = sum_k a[i,k] * b[k,j]
```

课堂 shape：

```text
a   [3, 7]
b   [7, 5]
out [3, 5]
```

tile：

```text
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
```

## 网格

```text
grid = (ceil(3 / 128), ceil(5 / 128)) = (1, 1)
```

一个 program 负责整个输出，但 tile 远大于 `3x5`，所以必须 mask。

## 三类 mask

```text
A load: (m < 3) & (k < 7)
B load: (k < 7) & (n < 5)
store:  (m < 3) & (n < 5)
```

- `m < 3`, `n < 5`：输出行/列尾部；
- `k < 7`：累加维尾部；
- 越界的 `A/B` 读用 `other=0.0`，保证不污染累加；
- 越界的 `out` 写必须被挡住，否则越界写。

## K 循环

```text
for k in range(0, cdiv(7, 64)):   # 1 次
```

`K=7 < BLOCK_K=64`，所以一个 tile 就覆盖了 K，但仍然需要 `k < 7` 的 mask。

## 练习

1. `M=3, N=5` 时 grid 是多少？
2. `M=130, N=5` 时 grid 是多少？
3. 哪些 load 需要 `k < 7`？
4. store 的 mask 少了 `m < 3` 会发生什么？
5. 为什么越界 load 用 `other=0.0` 而不是任意值？
