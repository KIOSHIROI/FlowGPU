# 跨后端 layout-view 契约（2026-10-28）

CUDA 默认开启 layout-view fusion 之后，跨后端的边界必须写进 PlatformConfig，
而不是靠平台名字符串判断。

## 契约字段

`PlatformConfig.layout_views: bool = False`

- `cuda` / `cuda-tuned`：**True**（3 个真实 FlagOS 任务 H100 device 验证通过）；
- `ascend` / `ascend-910b` / `ascend-wide`：**False**；
- `kunlunxin` / `kunlun-r300`：**False**。

字段进入 PlatformConfig 的 bool 校验，并在 `resolve_platform` 的构造/替换中保留。

## 编译语义

- `compile_source(..., layout_views=None)`：使用平台的 `layout_views` 字段；
  - CUDA：融合；
  - Ascend/昆仑：物化（旧 emission）。
- 显式 `layout_views=True` / `"cost"` 在 `layout_views=False` 的平台上**fail-closed**：
  ```
  CompileError(OPERATOR): platform 'kunlunxin' has no validated layout-view
  fusion contract; compile with layout_views=False ...
  ```
  不再发未验证的 indexed region，也不把失败留给下游 refusal；
- `layout_views=False` 在所有平台都可用，是旧 emission 的逃生舱。

## 测试

`test_layout_views_contract.py`：

- 只有 CUDA 声明 `layout_views=True`；
- 平台默认按字段而非名字生效；
- 非 CUDA 显式开启被 `CompileError` 拒绝；
- 非 CUDA 显式关闭仍走物化路径。

## 为什么 Ascend/昆仑暂不开启

`docs/LAYOUT_VIEWS_DEFAULT_EVAL_20261028.md` 的评估：

- Ascend：10/10 layout 文件编译报错（indexed/layout 契约未验证）；
- kunlunxin：可编译但全部有 refusals（`primitive` kernel kind 不在 allowed list，
  且索引/地址形状契约未验证）。

要开启它们，需要：

1. 把 indexed primitive 纳入该平台的 allowed_kernel_kinds；
2. 在真实 Ascend/昆仑设备上做和 CUDA 相同的 default-vs-view + reference 对照；
3. 把实测结果写回 PlatformConfig，而不是仅改默认。

在此之前，fail-closed 是正确行为。
