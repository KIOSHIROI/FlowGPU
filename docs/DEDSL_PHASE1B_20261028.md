# 去 DSL 中心化 Phase 1b（2026-10-28）

Phase 1b 把 **program AST、type inference、symbol table、operator semantics 与
dtype rules** 全部移到 frontend-neutral core。结果是：`flowgpu.compiler.ir`、
`flowgpu.compiler.optimization`、`flowgpu.compiler.analysis` 不再 import `flowgpu.compiler.frontend`；
`DataflowBuilder` 可以直接消费 core AST 构图，不需要 lexer/parser/lark。

## 移动清单

| 模块 | 新位置 | 兼容层 |
|---|---|---|
| `PoisonType`/`Call`/`InputDecl`/`Assignment`/`OutputDecl`/`PipelineDecl`/`Symbol` | `flowgpu/compiler/core/ast.py` | `frontend/ast_node.py` re-export |
| `ShapeLiteral`/`DTypeLiteral`/`IntLiteral`/`Name` | `flowgpu/compiler/core/literals.py` | `frontend/ast_node.py` re-export |
| `ScalarType` | `flowgpu/compiler/core/scalars.py` | `frontend/scalars.py` |
| `DTYPE_BYTES`/dtype classes/精度规则 | `flowgpu/compiler/core/dtype_rules.py` | `frontend/dtype_rules.py` |
| `OperatorSpec`/`OPERATORS`/85 个算子类型规则 | `flowgpu/compiler/core/operators.py` | `frontend/operator.py` |
| `SymbolTable` | `flowgpu/compiler/core/symbol_table.py` | `frontend/symbol_table.py` |
| `TypeInference` | `flowgpu/compiler/core/type_inference.py` | `frontend/type_inference.py` |
| program-uniform scalar 规则 | `flowgpu/compiler/core/program_scalar.py` | `frontend/program_scalar.py` |

所有兼容层 re-export **同一对象**，`OPERATORS` 字典身份不变；通过
`flowgpu.compiler.frontend.operator` 注册的算子也会出现在 core 名称表里（注册方向仍是
frontend → core）。

## 现在的依赖方向

```
flowgpu.compiler.core.ast / types / literals / operators / dtype_rules
        ^                  ^
flowgpu.compiler.ir / analysis / optimization   (no flowgpu.compiler.frontend imports)
        ^
flowgpu.compiler.frontend.parser / ast_builder / symbol_table_builder
        ^
flowgpu.compiler.compile / CLI                    (application chooses the DSL frontend)
```

`flowgpu.compiler.frontend` 现在只剩**具体语法**相关部分：lexer、grammar、parser、
AST builder、symbol-table builder；这正是一个前端应该负责的东西。

## 验证

`test/test_frontend_boundary.py`（13 条）新增：

- `flowgpu.compiler.ir`/`optimization`/`analysis` 全目录无 `flowgpu.compiler.frontend` import；
- 子进程 `python -c` 直接 import core AST + `DataflowBuilder`，断言
  `flowgpu.compiler.frontend`/`lark` 不在 `sys.modules`；
- 用 core AST 手工构造 `PipelineDecl` + `SymbolTable`，直接构建并验证
  `['input','relu','output']` 图。

全量：2889 tests / 0 failures / 6 skipped。

## 仍未去中心化的部分（Phase 2）

- `flowgpu/compiler/backend/*` 和 `flowgpu/compiler/compile.py` 仍 import `frontend.ast_node` 的
  re-export 以及 DSL parser（应用层选择 DSL 前端是合理的，但 backend 不应当
  依赖 AST）；
- 真正的前端扩展仍是 Phase 2：Triton importer 子集 / Python eDSL，用 DSL 做
  差分测试；
- `flowgpu/compiler/frontends/programmatic.py` 目前直接构图，不经过 `DataflowBuilder`；
  Phase 2 可让它复用 builder 并接 Triton 后端。
