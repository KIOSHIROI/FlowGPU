# SPS S2–S4：三前端、program-uniform、fusion-aware（2026-10-28）

在 S1（IR-native HOST/CONSTEXPR pass）之后，S2–S4 把 SPS 扩到多前端、
program-uniform placement 和 fusion-aware 成本。

## S2：DSL / eDSL / Triton 同一 plan

`test/test_sps_frontends.py`（3 条）用同一个语义程序分别构造三条前端路径：

| 前端 | 程序 | 标量名 |
|---|---|---|
| DSL | `pipeline p { input x:fp32[1024] input c:fp32 y=mul(x,c) ... }` | `c` |
| Python eDSL | `program("p")` + `x * c` | `c` |
| Triton importer | `x * 2.0` 的 `@triton.jit` kernel | `c0` |

结果：

- 高调用次数（10M）三个前端都得到 `constexpr` decision，host 签名都只剩一个
  tensor 参数；DSL/eDSL 的 literal 是 `2.0`，importer 的显式常量 `c0` 也是 `2.0`；
- 1 次调用三个前端都得到 `host` decision；
- 非 input/output 的 dataflow op 序列三者相同（`["mul"]`）；
- 说明 placement plan 是**前端无关**的，不依赖 DSL 文本。

## S3：program-uniform placement 进入同一 plan

`flowgpu/compiler/optimization/scalar_placement.py` 现在识别 dataflow 中
`op == "program_scalar"` 的 rank-0 producer，并给出 `Placement.UNIFORM`
decision；`PlacementPlan.uniform_scalars` 可直接读取。

`test/test_sps_uniform_fusion.py` 注册 staged program-scalar rule 后，用 eDSL 构造：

```python
table = p.input("table", "int32[S]")
b = p.call("program_scalar", table)
p.output(p.add(x, b))
```

编译后 plan 中有 UNIFORM decision；生成的 kernel `ops == ("program_scalar",
"add")`，且只有一条 kernel 顶层、无 `tl.arange`/`mask` 的 uniform load。
这证明 S3 把已有的 program-scalar lowering 接入了 SPS 的统一 placement 语言。

边界：S3 **不**把 host scalar 自动改写成 uniform——那需要调用方提供 device table
作为源，并改变输入 ABI，属于后续工作；S3 解决的是"已经由 program-scalar IR 表达的
值，在 plan 中不再隐形"。

## S4：fusion-aware 成本

S1/S2 的成本模型假设一个 scalar 的 host 成本与 kernel 数无关。S4 改为：

- host scalar 每次调用还要作为 launch argument 传给**每一个**读取它的 fusion
  group；
- `host_arg_per_group_us = 0.05`；
- `plan_placements(..., fusion=...)` 计算每个 scalar 被多少个 group 读取
  （`_groups_using`），并把它计入 constexpr 的节省。

`test/test_sps_uniform_fusion.py` 的 threshold 测试：

- 60k 次调用、1 个 group：`0.1 * 60000 = 6000 us < 10000 us` → HOST；
- 60k 次调用、2 个 group：`(0.1 + 2*0.05) * 60000 = 12000 us > 10000 us`
  → CONSTEXPR；
- reason 里出现 `2 group(s)`；真实 compile 的 plan 也会带 group 数。

这说明"一个 scalar 被多少 kernel 使用"会改变 specialization 的盈亏平衡点，
S4 把这一层纳入 SPS 决策。

## 下一步

- **S5**：SPS + CCSA acceptance（候选 placement 的 anytime-valid 验证）；
- S3 的"host → uniform"改写（需要 device-table 源与 ABI 变更）；
- S4 的成本参数用 H100/FlagOS 多 kernel 数据标定；
- 把 S2 的三前端差分结果写入论文/报告的系统章节。
