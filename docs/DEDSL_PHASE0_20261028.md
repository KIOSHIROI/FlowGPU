# 去 DSL 中心化 Phase 0（2026-10-28）

目标不是删除 DSL，而是把**前端无关的核心**从文本 DSL 中抽出来，让 DSL 成为
`flowgpu.compiler.core.frontend.Frontend` 协议的一个实现。课程交付仍然以 DSL 为参考前端。

## 本阶段做了什么

| 动作 | 位置 |
|---|---|
| 新增 frontend-neutral 核心包 | `flowgpu/compiler/core/` |
| 移动位置类型 | `flowgpu/compiler/core/position.py`（`frontend/position.py` 兼容 re-export） |
| 移动诊断类型 | `flowgpu/compiler/core/diagnostic.py`（`frontend/diagnostic.py` 兼容 re-export） |
| 移动 `TensorType` / shape 工具 | `flowgpu/compiler/core/types.py`（`frontend/ast_node.py` 兼容 re-export） |
| 前端协议 + 注册表 | `flowgpu/compiler/core/frontend.py`（`Frontend`、`FrontendInfo`、`register_frontend`） |
| DSL 适配器 | `flowgpu/compiler/frontend/dsl.py`（`DslFrontend`，复用现有 lexer/parser） |
| 依赖方向测试 | `test/test_frontend_boundary.py` |

兼容性：re-export 的是**同一个对象**（`is` 相等），所以 `except CompileError`、
isinstance、现有测试都不变；`from flowgpu.compiler.frontend.diagnostic import ...` 继续可用。
非 frontend 模块的 diagnostic/position import 已改成直接依赖 `flowgpu.compiler.core`。

## 依赖方向

```
flowgpu.compiler.core          (no flowgpu.compiler.frontend imports; no lark)
      ^
flowgpu.compiler.frontend.dsl  (calls core, owns its parser)
      ^
flowgpu.compiler.frontend.*    (lexer/parser/AST; still the reference frontend)
```

`flowgpu.compiler.ir` 仍会 import DSL 的 AST 节点类（Phase 1 处理），但已经不允许 import
lexer/parser/ast_builder；测试把这条边界钉住。

## 下一阶段（课程提交后）

1. **Phase 1**：把 `ir`/`analysis` 对 `frontend.ast_node` 的依赖降到只依赖
   `flowgpu.compiler.core` 的 IR 类型；把 operator 的语义/类型规则从 `frontend.operator`
   抽到核心算子表。
2. **Phase 2**：Triton importer（先 pointwise/reduction 子集）或 Python eDSL，
   用 DSL 路径做差分测试；此时 DSL 是回归/教学前端，而不是唯一入口。
3. 课程提交前不改变 DSL 主路径；本阶段只做零行为变更的边界抽取。

## 课程评分对应

见 `docs/COURSE_REQUIREMENTS_MAP_20261028.md`：词法、语法/AST、语义、代码生成和
DSL 设计文档全部保留；本阶段新增的是"关键抽象/设计取舍"的架构证据。
