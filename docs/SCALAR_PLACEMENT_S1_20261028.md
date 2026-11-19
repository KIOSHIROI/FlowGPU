# SPS S1：IR-native scalar placement（2026-10-28）

S1 把标量放置从 `research/sps` 的独立模型变成编译器里的 **IR-native pass**，并
真正落地了 HOST 与 CONSTEXPR 两种 placement。输入是 frontend-neutral 的 core
AST + dataflow graph，所以 DSL、Python eDSL、Triton import 三条前端路径都会走
同一份分析。

## 1. 规划器

`compiler/optimization/scalar_placement.py`：

- 收集 core AST 中所有 rank-0 `InputDecl`（标量输入）；
- 对每个标量在 HOST / CONSTEXPR 之间选择：
  - 无编译期值 → HOST；
  - 有值且 `calls * host_saving > compile_cost` → CONSTEXPR；
- 输出 `PlacementPlan`，包含每个 decision 的 reason、literal、host/constexpr 集合；
- UNIFORM/TILE 记录为 S1 未实现，不伪装成可选。

当前成本模型：host formal 0.10 µs/call；constexpr 编译 10 ms；只在调用次数
足以摊销编译时选择 SPECIALIZE。

## 2. 编译 API

`compile_source` / `compile_pipelines` 新增：

```python
compile_pipelines(pipeline,
                  scalar_placement="sps",
                  scalar_specialization={"c0": 2.0},
                  scalar_calls=10_000_000)
```

- 默认 `scalar_placement="off"`，行为与之前完全一致；
- 特殊化名字必须是标量输入，否则 `ValueError`；
- `PipelineResult.scalar_placement_plan` 暴露决策，便于报告/测试。

## 3. 代码生成落地

- Triton kernel：constexpr 标量不再生成形参，直接绑定 literal；
- 生成的 host：constexpr 标量从签名、校验和 launch 参数中全部移除；
- reference backend 保持原样（独立可执行规范，不是性能产物）；
- HOST placement 完全沿用现有路径。

## 4. H100 验证

`tools/verify_scalar_specialization_h100.py`（scale kernel，N=1M，BLOCK=1024）：

| path | host signature | max abs diff | latency us |
|---|---|---|---:|
| original Triton | `scale_kernel` | 0 | 9.77 |
| S1 off (host formal) | `run_scale_kernel(x_ptr, c0)` | 0.0 | 18.45 |
| S1 constexpr | `run_scale_kernel(x_ptr)` | 0.0 | **16.24** |

- 数值：两种路径都与原始 kernel `max_abs_diff=0.0`；
- constexpr 比 host formal 快约 **12%**（launch-bound kernel，5 批 interleaved
  测量中位数；这是 sanity check，不是调优结论）；
- 结果：`results/triton_import/h100_scalar_specialization.{json,md}`。

## 5. 测试

- `test/test_scalar_placement.py` 7 条：host/constexpr 决策、host ABI 保留/移除、
  literal 内联、默认 off 不变、非法输入拒绝、Triton importer 常量特殊化；
- 全量 suite 见 `results/pytest.log`。

## 6. 边界与下一步

- S1 只实现 HOST/CONSTEXPR；UNIFORM 需要接 program-scalar IR，TILE 需要
  materialisation rewrite；
- reference backend 尚未特殊化（独立语义规范）；
- 多 pipeline 的 specialization 名字按 pipeline 过滤；
- 下一步 S2：DSL/eDSL/Triton 三前端同一 plan 的差分验证；S3：uniform placement；
  S4：fusion-aware 成本；S5：与 CCSA acceptance 结合。
