# 去 DSL 中心化 Phase 2a：Python eDSL（2026-10-28）

Phase 2a 的目标是让用户**不写 FlowGPU DSL 文本**也能使用完整编译器。新增一个
Python eDSL：它构建 `flowgpu.compiler.core.ast` 节点，然后走与 DSL 完全相同的
symbol/type inference、dataflow、fusion、liveness 和 Triton/reference 后端。

## 使用方式

```python
from flowgpu.compiler.frontends.python_edsl import program
from flowgpu.compiler.compile import compile_pipelines

with program("p") as p:
    x = p.input("x", "fp32[N]")
    y = p.relu(x)
    z = y + x
    p.output(z)

result = compile_pipelines(p.pipelines)
assert result.pipelines[0].triton.complete
```

支持的 Phase 2a 子集：

- `p.input(name, "fp32[N]")` / `"fp32"`（标量）；
- 运算符重载：`+`/`-`/`*`；
- 通用 `p.call(op, *args, out=...)` + `p.<operator>(...)` 便捷方法；
- `p.cast_to(x, "bf16")`、`p.int(...)`、`p.dtype(...)`、`p.shape(...)`；
- 单 output/multi-output 的运算符（多结果赋值暂未暴露）。

未知算子、重复名字、缺少 output 在构建期报 `ValueError`；类型/形状错误仍然由
共用的 `TypeInference` 报 `CompileError`，两个前端诊断一致。

## `compile_pipelines`：AST 入口

`flowgpu.compiler.compile.compile_pipelines(pipelines, **options)` 复用
`compile_source` 的**全部编译阶段**，只跳过 parse/ASTBuilder：

- `compile_source` 现在是 `parse → compile_pipelines` 的薄封装；
- parser/ASTBuilder/LarkError 改为 parse 分支内 lazy import；
- `symbol_table_builder`、`shape_constraints` 也移到 core 并保留 frontend
  re-export，所以 AST 入口不再需要 `flowgpu.compiler.frontend` 包。

`from flowgpu.compiler import compile_pipelines` 同样可用（lazy public API）。

## 验证

- `test/test_python_edsl.py`（6 条）：
  - eDSL 与 DSL 对同一程序得到相同的 dataflow op 序列和可发射的 Triton；
  - 运算符重载、标量 input、`T()` 类型助手；
  - 未知算子/坏 output/无 output 拒绝；
  - **子进程**中 eDSL 导入 + 完整编译后 `flowgpu.compiler.frontend.parser` 和 `lark`
    都不在 `sys.modules`。
- 全量：2895 tests / 0 failures / 6 skipped。

## 剩余 Phase 2b

- `flowgpu/compiler/backend/*` 仍通过 `frontend.ast_node` re-export 使用 AST 类型（同一
  core 对象，但 import 路径还可再收敛）；
- 真正的 Triton importer（pointwise/reduction 子集）尚未做；
- eDSL 多结果赋值、动态 shape、控制流不在本阶段范围；
- 课程作业仍以 DSL 为参考前端，eDSL 是第二条前端路径。
