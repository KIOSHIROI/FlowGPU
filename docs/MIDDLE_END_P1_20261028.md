# Middle-end P1：layout 扩展、成本模型、真实 corpus、scan 分解（2026-10-28）

在 P0（layout-view fusion）之后，1–4 四步的结果。

## 1. 扩展 layout-view fusion

`layout_views=True` 现在额外支持：

- `interleave`：2 个 kernel → **1**，H100 **38.51 → 20.24 µs（1.90×）**；
- `transpose`：已有 transpose emitter 本来就把 relu 融进同一个 kernel，
  不产生额外收益（1→1），不强行改；
- 后端 reference/device 都增加了 interleave 的 parity+mask 映射。

## 2. view vs materialise 成本模型

`flowgpu/compiler/optimization/layout_view_cost.py`：

- `view_cost(numel, layout_ops, consumer_ops)`：一次 launch + 每元素索引重映射
  + consumer 算术；
- `materialize_cost(...)`：两次 launch + 中间 buffer 读写 traffic + consumer；
- `should_fuse_layout_view(...)` 返回决策和 `view_us/materialize_us/saving_us`。

在 H100 实测的 4K–1M 元素范围，launch 项占主导，模型对**所有当前合法 view**
都选择融合；cost policy 目前是 advisory（供审计/报告使用），
`layout_views` 仍是 opt-in 的 `"always"` 行为。测试里用病态高索引成本证明模型
确实可以翻转为 materialise。

## 3. 真实 .fgpu corpus 验证

`tools/validate_layout_views_corpus.py` 扫描 51 个 example，10 个含 layout 算子：

| 指标 | 结果 |
|---|---:|
| 编译成功 | 10 |
| kernel 总数变化 | **−42** |
| materialised buffer 变化 | **−43** |
| 改善文件 | 3 |
| 不变文件 | 7 |
| 变差文件 | **0** |

关键文件：

| file | default kernels | layout_views kernels |
|---|---:|---:|
| `examples/packed/awq_dequant.fgpu` | 34 | **1** |
| `examples/packed/fp4_indexer.fgpu` | 10 | **3** |
| `examples/batch10/swiglu_oai_split.fgpu` | 3 | **1** |

参考后端数值对照（symbolic size=2）：8 个文件全部 `reference_match=true`；
`awq_dequant`/`fp4_indexer` 的参考执行失败是它们自身的 negative shift 问题
（两种配置都失败，与 layout-view 无关），已原样记录，不隐藏。

**这是静态/参考执行验证，不是 H100 真实任务执行**；大融合（34→1）还需要真实的
device parent/candidate 验证。

## 4. Reduction/Scan 分解

新增 `scan_epilogue=True`：

```
t = sum_axis(x, 1); c = cumsum_axis(x, 1)
out = add(c, broadcast_like(t, x))
```

被重写为：

```
c, t = cumsum_totals(x)
out = add(c, broadcast_like(t, x))
```

H100：scan prefix **3 → 1 kernel，42.39 → 16.14 µs（2.63×）**。

实现上，scan backend 现在允许 carried totals 进入 `add_broadcast` epilogue：
先做一次 cheap row-total prepass，再在主 scan 循环里广播 total；reference
backend 做同样的两遍循环。非 row-broadcast 的 totals 使用仍然拒绝。

## 验证与产物

- 新测试：`test_layout_views.py` 增加 interleave/transpose；
  `test_scan_epilogue.py`（2 条）；`test_layout_view_cost.py`（3 条）；
- H100 结果：`private evidence`；
- corpus 结果：`private evidence`；
- 全量 suite 见 `private evidence`。

## 边界

- `layout_views` 和 `scan_epilogue` 都是 opt-in，默认 emission 不变；
- 真实 corpus 是静态验证，awq/fp4 的 34→1/10→3 需要 device 正确性证据；
- cost model 尚未接入自动策略（当前 always-view 在实测范围都更优）；
- interleave/transpose 的 backend 契约目前以 CUDA indexed 路径为准。
