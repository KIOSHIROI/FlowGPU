# P0：Layout / Materialization Synthesis（2026-10-28）

中间端机会审计指出：`split/concat/reshape + pointwise` 仍有 2–3 个 kernel，
现有 `split_views`/`view_folding` 旋钮覆盖不到。P0 实现了一个 opt-in 的
layout-view fusion：

```
compile_source(source, layout_views=True)
```

它把 layout anchor（`split`/`concat`/`reshape`/cast）和它的 pointwise consumer
合并成一个 **indexed region**，由现有 indexed emitter 直接按输出下标重映射读取，
不再物化中间 layout buffer。

## 实现

- `flowgpu.compiler.optimization.indexed_fusion.fuse_indexed(..., layout_views=...)`：
  - opt-in 允许 `split`/`concat` 进入 indexed region；
  - 当 region 没有 primitive/reduction anchor 时，允许以**最后一个 pointwise
    sink** 作为 anchor（此前直接 `continue`，所以 layout+pointwise 永远不融合）；
- `flowgpu/compiler/backend/triton_backend.py` 与 `reference.py`：
  - `group.indexed` 的 kernel 统一走 primitive/indexed dispatch，不再按 anchor
    op 误判为 `pointwise`；
- `flowgpu/compiler/backend/indexed_backend.py`：
  - 新增 `concat` 的 view 映射（左右 operand 各自带 mask，避免越界 load）；
  - `split` 的 view 映射沿用已有分支；
  - reference indexed emitter 同样支持 concat/split，保持参考后端可执行；
- `flowgpu/compiler/compile.py`：新增 `layout_views` 选项（默认 **False**，保持现有
  emission 不变）。

## H100 结果（interleaved，median µs，与 default 输出逐元素对照）

| program | default kernels | default µs | layout_views kernels | layout_views µs | speedup |
|---|---:|---:|---:|---:|---:|
| split + add | 3 | 37.78 | **1** | **16.86** | **2.24×** |
| reshape + relu | 2 | 29.14 | **1** | **18.79** | **1.55×** |
| concat + relu | 2 | 26.47 | **1** | **20.38** | **1.30×** |

所有 `layout_views` 输出与 default 输出通过 `torch.testing.assert_close`
（rtol/atol 1e-4），数值正确性已由 H100 脚本检查。

结果：`results/middle_end/h100_layout_views.{json,md}`；审计更新：
`results/middle_end/audit.json`。

## 测试

`test/test_layout_views.py`（5 条）：

- default 仍保持 3/2/2 kernel（不改变现有行为）；
- split+add → 1 个 indexed kernel（`ops=("split","split","add")`）；
- reshape+relu → 1 个 kernel；
- concat+relu → 1 个 kernel，且左右 operand 使用分区 mask；
- 无关程序在 `layout_views=True` 下行为不变。

## 边界与下一步

- P0 目前 **opt-in**；要成为默认需要更大 corpus（含真实任务/多后端）验证；
- 覆盖 `split/concat/reshape/cast/move`，未覆盖 `interleave`；
- concat 的右 operand 地址在掩码下才会被 load，虽然安全，但可以进一步简化；
- 后端契约尚未泛化到 Ascend/昆仑芯，先以 CUDA indexed 路径为准；
- 下一步：把 layout-view fusion 纳入成本模型（view vs materialize），扩展
  interleave/transpose，并用真实任务做 parent/candidate 对照后考虑默认开启。
