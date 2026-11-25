# 练习：先预测，再运行

1. `a [3,7]`、`b [7,5]`，输出形状？
2. 默认 tile 下的 grid 是多少？
3. 哪些 load 需要 `k < 7`？
4. store 的 mask 是什么？
5. `M=130` 时 grid 变成多少？

运行：

```bash
python lessons/07-masks/run_lesson.py
```

它会打印生成的 `tl.load` / `tl.store` mask。
