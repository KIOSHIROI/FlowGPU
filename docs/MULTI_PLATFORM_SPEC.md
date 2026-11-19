# 多芯片生成配置与验证边界

2026-10-01 审计修订。平台表是代码生成配置，所有预设的 `tuning_status` 均为 `untuned`。型号别名不代表在该型号上完成了编译、正确性或性能验证。

| 平台 | Torch 设备类型 | 逐元素默认映射 | 默认放行的模板 | 证据范围 |
|---|---|---|---|---|
| cuda | cuda | direct_grid | pointwise、rowwise、matmul、transpose、rope | 部分 A800 工作负载实测，不是全算子生产认证 |
| ascend / ascend-910b | npu | grid_stride_loop | pointwise | 本地生成代码与 AST 检查，未在昇腾执行 |
| kunlunxin / kunlun-r300 | cuda | direct_grid | pointwise | 本地生成代码与 AST 检查，未在昆仑芯执行 |

昇腾的循环配置必须由调用者提供实际 Vector Core 数。型号名称不能推导固定核数。`min_align_bytes=32` 是片上存储对齐提示，当前不会自动补齐输入、保证所有指针对齐或取消尾部 mask。昆仑芯的 Torch `cuda` 设备命名取自 FlagGems 上游设备描述，仍需核对比赛环境。

未放行模板记录在 `TritonProgram.refusals` 中，`complete=False`。这是显式拒绝，不是自动切换到 Torch 的降级执行。生成的 NPU host 导入 `torch_npu`，在运行时检查输入设备；这些检查不证明底层指令兼容或不会发生设备故障。

## 使用与实验性放行

```python
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform

# actual_vector_core_count 必须来自当前设备属性。
profile = resolve_platform("ascend-910b", device_cores=actual_vector_core_count)
result = compile_source(dsl_code, platform_config=profile)
```

需要探测 GEMM 等模板时，可以在 Python API 中显式指定 `allowed_kernel_kinds=("pointwise", "matmul")`。这只是实验性代码生成许可，不表示已经验证硬件支持；默认预设仍只放行逐元素模板。Hybrid 库调用路径仅用于 CUDA 本地实验。

CUDA GEMM 现在将 `num_warps` 和 `num_stages` 传入实际 launch；`group_m` 按每个 kernel 的字面值传递，避免同一个生成模块中不同配置共用全局变量。参数不是自动调优器，也不能承诺在其他芯片上的含义与 CUDA 相同。

```sh
.venv/bin/python main.py examples/mapping_pointwise.fgpu --emit triton --platform cuda --block-size 256 --num-warps 8
# GEMM 可指定 --group-m 8 --num-stages 3 --num-warps 4
```

## 正式参赛证据的归属

`private evidence` 是独立手写的 Triton 适配器。该文件对应正式七芯片通过回执，平均加速比 0.78×。它不是上述 DSL lowering 的产物，不能据此宣称 `ascend-910b` 或 `kunlun-r300` 配置的 GEMM、归约或逐元素流水线已通过官方评测。

正式回执 (private evidence, not included)、逐元素 312 组 A800 检查 (private evidence, not included)、[逐元素映射说明](pointer-mapping notes) 分别保留自己的适用范围。取得具体型号、逐尺寸日志及设备编译执行结果后，才能回填硬件参数或扩大默认放行范围。

后续手写 `fused_sigmoid_mul` 与 `rmsnorm_hf` 也已正式7/7通过，平均2.87×与4.44×。编译器已接入它们的语义及 pointwise/rowwise 输入 stride 映射，但仅新增A800执行证据；默认国产设备rowwise白名单未因此扩大。详见 [接入与证据](private benchmark evidence)。

## 评测反馈迁移（2026-10-02）

新增 `transpose_mapping=tiled|output_gather`。CUDA默认tiled，Kunlun预设偏好output_gather，但默认transpose仍被白名单拒绝；需Python API显式实验性放行。输出gather携带外部输入stride并连续写回。220组A800执行与578项CPU回归通过，不证明昆仑设备编译或性能。A800大尺寸gather慢于tiled，不能统一所有平台映射。详见[本轮结果](private mapping follow-up)。
