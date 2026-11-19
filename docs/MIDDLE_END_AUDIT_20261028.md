# 中间端机会审计（2026-10-28）

第一步不是直接做优化，而是回答：**中间端哪个决策最值得投入？** 本审计用
6 个代表性程序做结构性消融，并在 H100 上对同一 corpus 做端到端计时。

## 方法

- 工具：`tools/middle_end_opportunity_audit.py`（CPU 结构审计）+
  `tools/benchmark_middle_end_audit_h100.py`（H100 计时）；
- corpus：pointwise chain、split、reshape、concat、softmax reduction、prefix scan；
- ablations：`no_index_fusion`、`no_reduction_fusion`、`no_view_folding`、
  `rule_fusion`、`split_views`；
- 指标：kernel/launch 数、fusion groups、materialised buffers、slots、layout
  groups、refusals；
- H100 计时：每个 variant 都先与 default 输出做数值对照，再 interleaved
  5 批 × 100 次取中位数。

## 结构审计（default）

| program | kernels | groups | fused groups | materialised buffers | layout groups |
|---|---:|---:|---:|---:|---:|
| pointwise_chain | **1** | 1 | 1 | 2 | 0 |
| pointwise_split | **3** | 3 | 0 | 4 | 2 |
| layout_reshape_relu | **2** | 2 | 0 | 3 | 1 |
| layout_concat | **2** | 2 | 0 | 4 | 1 |
| reduction_softmax | **1** | 1 | 1 | 2 | 0 |
| scan_prefix | **3** | 3 | 1 | 4 | 0 |

消融的关键 delta：

- `reduction_softmax`：关掉 index fusion → **+6 kernels**；关掉 reduction
  fusion → **+3 kernels**；说明现有融合在 reduction/softmax 上非常有效；
- `scan_prefix`：关掉 index fusion → **+1 kernel / +1 materialised buffer / +1 slot**；
- `pointwise_split` / `reshape` / `concat`：现有 ablation 都不改变 kernel 数，
  说明 **layout/materialisation 决策还没有被现有旋钮覆盖**；
- `rule_fusion` 与 default 无差异：rule 和 traffic 在这个 corpus 上一致。

## H100 实测（interleaved，median µs）

| program | variant | kernels | latency µs |
|---|---|---:|---:|
| pointwise_chain | default | 1 | **16.61** |
| pointwise_split | default | **3** | **46.10** |
| layout_reshape_relu | default | 2 | 25.89 |
| layout_concat | default | 2 | 24.28 |
| reduction_softmax | default | 1 | **23.36** |
| reduction_softmax | no_index_fusion | 7 | 83.80 |
| reduction_softmax | no_reduction_fusion | 4 | 49.80 |
| scan_prefix | default | 3 | 42.78 |
| scan_prefix | no_index_fusion | 4 | 54.84 |

## 结论：P0 应该是 Layout / Materialization Synthesis

1. **布局/物化是最大的、尚未被现有旋钮覆盖的机会**：
   - `split + add` 仍要 **3 个 kernel / 46.1 µs**，而同样 pointwise 的
     chain 只要 **1 个 kernel / 16.6 µs**——约 **29.5 µs/次**的结构性开销；
   - `reshape+relu`、`concat+relu` 各 **2 个 kernel / 24–26 µs**；
   - `split_views` ablation 不改变任何指标，说明这个开关没有覆盖这些常见布局链。
2. **融合本身已经很有效，但只在它适用的形态上**：
   - softmax 1 kernel vs 关掉 index fusion 的 7 个（3.6×）；
   - reduction fusion 也贡献 2.1×；
   - 下一步不是"重新发明 fusion"，而是把 layout 决策纳入同一个 cost/legal
     框架，并为 scan/reduction 找到更细的分解点。
3. **scan 仍有 3 个 kernel**：index fusion 只省 1 个，说明 scan 的
   materialisation/分解还有空间，列为 P1。

## 对下一步的建议

- **P0：Layout / Materialization Synthesis**
  - 范围先定：`split/concat/reshape` 与 pointwise consumer 的 view fusion；
    materialize vs view 的成本模型；backend contract 合法性；
  - 目标：把 `split+add` 从 3 kernels 降到 1–2，reshape/concat 从 2 降到 1；
  - 用现有 H100 audit 作为 parent/candidate 基线。
- **P1：Reduction / Scan decomposition**：scan 3 kernels → 更少阶段，需
  backend-specific 证据；
- **P2：fusion cost**：把 rule/traffic 融合升级为带 backend 成本的分区；
- 标量 placement（S1–S4）保留为支撑案例，不再扩展。

## 局限

- corpus 只有 6 个合成程序，不是完整任务集；
- H100 计时包含 host 开销，launch-bound 程序对 host 变化敏感；
- 结构审计是 CPU 静态分析，device 时间只覆盖已能编译的变体；
- `split_views` 无变化可能是实现覆盖不足，正好是 P0 的入口。
