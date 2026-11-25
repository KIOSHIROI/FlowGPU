# 练习：同一语义，两种实现

先读 `review/swiglu_split/dataflow.md`，再看这里。

1. 生成版有几个 kernel？手写版有几个？
2. 生成版的 `gate` / `up` 在哪里？
3. 手写版的 `split_copy_kernel` 写了哪两个地址？
4. 两份实现的 `gate[r,c]` 地址是否相同？
5. 哪份实现更适合检查“奇偶交错”这种布局改动？

运行：

```bash
python lessons/08-handwritten/compare.py
```
