# 练习：布局就是地址

先预测，再看运行结果。

1. `x [4,8]`，`a` / `b` 的形状？
2. `interleaved[0,0]` 和 `interleaved[0,1]` 分别来自哪里？
3. `joined[0,0]` 和 `joined[0,4]` 分别来自哪里？
4. `a` / `b` 被两个消费者读取，为什么仍然不需要缓冲？
5. 交换 `interleave(a,b)` 的参数后，哪些输出位置变？

运行：

```bash
python lessons/02-layout/run_lesson.py
```
