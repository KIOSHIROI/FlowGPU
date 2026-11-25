# 练习：先预测，再运行

1. `x [2,8]`、`H=16`，`gate` / `hidden` / `out` 的形状？
2. 哪两个中间值必须物化？
3. 哪两个中间值被融合掉？
4. k1 的 epilogue 有哪几个操作？
5. 为什么 gate 和 up 不合并成一个 matmul？

运行：

```bash
python lessons/05-fusion/run_lesson.py
```
