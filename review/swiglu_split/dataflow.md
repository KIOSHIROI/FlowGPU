# 数据流卡片：SwiGLU OAI split

先看数据流，不要先看编译器。

## 1. 形状流

```text
x [8,16]
├─ split 前半列 0:8   -> gate [8,8]
└─ split 后半列 8:16  -> up   [8,8]

gate -> minimum(gate, limit)              -> upper   [8,8]
up   -> maximum(up, -limit) -> minimum(..., limit) -> bounded [8,8]

upper -> sigmoid(alpha * upper) -> gated [8,8]
bounded + beta                   -> shifted [8,8]
gated * shifted                  -> out [8,8]
```

## 2. 地址流

对行 `r`、列 `c`：

```text
gate[r,c] = x[r * 16 + c]
up[r,c]   = x[r * 16 + 8 + c]
out[r,c]  = out[r * 8 + c]
```

`split` 没有自己的输出缓冲；它就是这两条取数地址。

## 3. 中间缓冲

| 版本 | gate/up 是否有独立缓冲 | kernel 数 |
|---|---|---|
| FlowGPU 生成 | 否，split 折进索引加载 | 1 |
| 手写两 kernel | 是，split_copy 显式写入 | 2 |

## 4. 尾部/掩码

`HALF=8` 是 2 的幂，当前输入没有尾部。把 `width` 改成 12、`half=6` 后：

- Triton 的 `tl.arange(0, HALF)` 需要 `HALF` 为 2 的幂；
- 多出来的 lane 必须用 `column < HALF` 的 mask 挡住；
- `split_copy` 的第二个地址 `r*WIDTH + HALF + c` 也要同样 mask。

这正是“改一个宽度，检查地址和掩码”的练习点。

## 5. 练习

1. 把 `width` 从 16 改成 32：哪些常量变？哪些地址变？
2. 把 split 从“前后两半”改成“奇偶交错”：  
   `gate[r,c] = x[r*16 + 2*c]`，`up[r,c] = x[r*16 + 2*c + 1]`。  
   哪一份实现会先暴露错误？
3. 去掉 `minimum(gate, limit)`：数据流少哪个节点？生成版 kernel 数会变吗？
4. 当 `width` 不是 2 的幂时，尾部 mask 应该加在哪几个 load 上？

## 6. 这张卡片的作用

它把“公式 → 形状 → 地址 → 缓冲 → 掩码”串成一条可检查的线。  
读者不必先理解整个 FlowGPU，只需要沿着数据流回答：

> 这个值从哪里来？它有没有被物化？地址怎么算？尾部怎么挡？
