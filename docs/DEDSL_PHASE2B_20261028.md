# 去 DSL 中心化 Phase 2b：Triton importer + backend 收敛（2026-10-28）

Phase 2b 让**外部已有 Triton kernel** 进入 FlowGPU 的核心 IR/契约/优化，并完成
backend 对 frontend import 路径的收敛。

## 1. Triton importer

`flowgpu/compiler/frontends/triton_importer.py`：

```python
from flowgpu.compiler.frontends.triton_importer import import_triton
from flowgpu.compiler.compile import compile_pipelines

imported = import_triton(source,
                         input_types={"x_ptr": "fp32[N]", "y_ptr": "fp32[N]"},
                         output_types={"out_ptr": "fp32[N]"})
result = compile_pipelines(imported.pipelines)
```

支持的 Phase 2b 子集：

- 单个 `@triton.jit` 函数；无控制流/循环/atomic；
- `tl.load(ptr + offsets, mask=...)` 视为读取一个连续输入张量；
- `tl.store(ptr + offsets, value, mask=...)` 视为一个 output；
- `+ - * /`、`tl.maximum/minimum/exp/log/sqrt/abs/sigmoid/tanh/...`；
- 比较（`< <= > >= == !=`）与 `tl.where`；比较/where 的数值常量通过
  `zeros_like + add` materialise 成 tile（语言没有 scalar-accepting broadcast 算子）；
- 多个 `tl.store` → 多个 output；
- `@triton.autotune(...)` / `@triton.jit` 装饰器栈（只取 jit 函数体）；
- `tl.sum(x, axis=k)` / `tl.max(x, axis=k)` → `sum_axis`/`max_axis`；
- 数值常量（如 `x * 2.0`）转成**显式 rank-0 标量输入**，因为 core 语言
  没有浮点字面量；`imported.constants` 给出名字和值，调用方按普通标量传参。

不支持的形态会报 `ImportError`（本模块自己的 ValueError 子类）：多 kernel、
控制流、未声明的指针、未 store 的 output、address/mask 之外的未知值等。
高-level 的 shape/dtype 必须由调用方提供——Triton kernel 只有指针，没有
shape/dtype，这是诚实的 importer 边界，而不是隐藏假设。

## 2. Backend/compile import 收敛

- `flowgpu/compiler/backend/*` 不再 import `flowgpu.compiler.frontend.*`，改用
  `flowgpu.compiler.core.ast/types`；
- `flowgpu/compiler/compile.py` 的 parser/ASTBuilder/LarkError 只在 parse 分支 lazy
  import；`import flowgpu.compiler.compile` 不加载 lark；
- 因此 `compile_pipelines(imported.pipelines)` 的完整编译路径不依赖 DSL 文本。

## 3. 验证

`test/test_triton_importer.py`（5 条）：

- pointwise kernel 导入后与 DSL 同一程序得到相同 dataflow op 序列；
- reduction（`tl.sum` → `sum_last`）；
- `x * 2.0` 生成显式标量输入并编译；
- 拒绝非 jit/错误指针/控制流/未 store 的 output；
- 子进程验证 importer + 完整编译后 `flowgpu.compiler.frontend.parser` 与 `lark` 都不在
  `sys.modules`。

`test/test_frontend_boundary.py` 新增 2 条：backend 全目录无 frontend import；
`flowgpu.compiler.compile` 模块级无 frontend import 且 import 不加载 lark。

全量：2902 tests / 0 failures / 6 skipped。

## 4. H100 数值差分（已验证）

`tools/verify_triton_import_h100.py` 在同一组 H100 输入上分别执行原始
`@triton.jit` kernel 与导入路径，逐元素比较。**8/8 case 全部 max_abs_diff = 0.0**：

| case | shape | dtype |
|---|---|---|
| add | [1024] | fp32 |
| add（mask 边界） | [1000] | fp16 |
| scale（常量转标量输入） | [1024] | fp32 |
| maximum | [777] | fp32 |
| sum reduction | [1] | fp32 |
| max reduction | [1] | fp32 |
| where + comparison | [1048576] | fp32 |
| multiple stores | 2×[1024] | fp32 |

结果：`private evidence`。

## 5. H100 性能 sanity check

`tools/benchmark_triton_import_h100.py` 对同一语义的原始 Triton 实现与导入实现
测端到端延迟（host+device，5 批 × 100 次中位数；两侧都正确 tiling/mask）：

| case | original µs | imported µs | ratio |
|---|---:|---:|---:|
| add N=1M | 11.89 | 16.89 | 1.42x |
| add N=4M | 62.58 | 58.96 | 0.94x |
| scale N=1M | 15.83 | 28.29 | 1.79x |
| where N=1M | 12.47 | 15.93 | 1.28x |
| multi-output N=1M | 12.35 | 31.32 | 2.54x |

读法：大 shape 的带宽型 add 基本持平（导入侧甚至略快，属于测量波动）；小
launch-bound case 和多 output case 的导入宿主要付额外的 shape/dtype 校验、输出
分配以及常量 materialise（`zeros_like + add`）开销。**这是 Phase 2b 的语义验证，
不是调优后的性能结论**；它说明下一步该优化生成宿主路径（或直接 ABI 导入），而不是
importer 的语义映射。

结果：`private evidence`。

## 5. 剩余工作

- importer 覆盖面：`tl.where`、多 output/store、动态 shape、`autotune` 包装；
- 在 H100 上对导入 kernel 做**数值差分执行**（原 Triton vs FlowGPU 参考后端），
  目前只有结构性/编译级验证；
- 仓库级 benchmark：同一 Triton kernel 经 importer 进入 FlowGPU 后与手写
  Triton 的性能对比；
- 课程作业仍以 DSL 为主路径，Triton importer 是架构/研究加分项。
