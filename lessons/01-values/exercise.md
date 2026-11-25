# 练习：先预测，再运行

先不要看生成代码。

1. `x [4,8]`, `scale [1,8]`，`out` 的形状是什么？
2. `scale` 的地址里有没有行偏移？
3. `y` 会不会单独占一块显存？
4. 这个 pipeline 生成几个 kernel？
5. 把 `scale` 改成 `[1,1]` 后，地址表达式变成什么？

运行：

```bash
python lessons/01-values/run_lesson.py
```

它会编译、打印 kernel 和 CPU reference 的第一行。
先写下你的答案，再对照。
