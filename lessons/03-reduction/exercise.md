# 练习：先预测，再运行

1. `x [4,5]` 时，`total` 的形状？
2. 归约的是哪一维？
3. `out[2,3]` 用的是哪个 `total`？
4. 为什么 `total` 不需要单独缓冲？
5. `D=5` 时哪些 lane 必须被 mask？

运行：

```bash
python lessons/03-reduction/run_lesson.py
```
