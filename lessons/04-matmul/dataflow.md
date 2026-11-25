# Lesson 4 数据流：matmul

## 公式

```text
out[i,j] = sum_k a[i,k] * b[k,j]
```

## 形状流

```text
a   [M, K]
b   [K, N]
out [M, N]
```

对齐规则：

- `a` 的最后一维 `K` 必须等于 `b` 的第一维 `K`；
- 这个共同的 `K` 被消掉，留下 `M` 和 `N`。

## 地址流

```text
a[i,k]   = a[i * K + k]
b[k,j]   = b[k * N + j]
out[i,j] = out[i * N + j]
```

三个不同的步长：

| 张量 | 行步长 | 列步长 |
|---|---:|---:|
| `a` | `K` | 1 |
| `b` | `N` | 1 |
| `out` | `N` | 1 |

## 累加维

`K` 不是输出维度，而是 **reduction 维**：

```text
for k in 0..K-1:
    acc += a[i,k] * b[k,j]
```

这就是为什么 matmul 比逐元素算子难：每个输出元素都要遍历 `K`。

## 编译结果

默认生成 `matmul` kernel。编译器会选 tile：

```text
BLOCK_M, BLOCK_N, BLOCK_K
```

`program_id(0)` 选择输出行块，`program_id(1)` 选择输出列块。

## 练习

1. `a [4,8]`、`b [8,16]`，`out` 的形状？
2. 哪个维度被消掉？
3. `a[i,k]` 和 `b[k,j]` 的列步长分别是多少？
4. 把 `K` 从 8 改成 16，哪个循环变长？
5. `N=6` 时，尾部 mask 应该加在哪个轴？
