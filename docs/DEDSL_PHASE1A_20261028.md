# 去 DSL 中心化 Phase 1a（2026-10-28）

Phase 1a 把**图 IR 本身**从 `compiler.ir` 移到 frontend-neutral core，并加入第一个
非 DSL 前端（programmatic），证明 core IR 可以在不经过 lexer/parser/AST 的情况下
被构造和被图级 pass 消费。

## 本阶段做了什么

| 动作 | 位置 |
|---|---|
| 移动 `Node`/`Edge`/`DataflowGraph`/`INPUT`/`OUTPUT`/`resolve_alias` | `compiler/core/ir.py` |
| 移动 view/permutation 契约 | `compiler/core/index_maps.py` |
| 算子**名称**词汇表（不是完整语义） | `compiler/core/operators.py` |
| DSL 注册算子名 | `compiler/frontend/operator.py` 的 `register_operator` 调用 core |
| 兼容 re-export | `compiler/ir/dataflow.py`、`compiler/ir/index_maps.py`（同一对象） |
| 非 DSL 前端 | `compiler/frontends/programmatic.py`（JSON/Python spec → `DataflowGraph`） |
| 边界测试 | `test/test_frontend_boundary.py` 新增 programmatic 构造/消费测试 |

`DataflowGraph.validate` 不再 import `compiler.frontend.operator`：它只查 core 的
算子名称词汇表；DSL 在前端注册时填充该表，programmatic 前端在构建时注册自己的
op 名称。这是 frontend → core 的注册方向，不是 core → frontend。

## Programmatic frontend

```python
from compiler.frontends.programmatic import ProgrammaticFrontend

graph = ProgrammaticFrontend().parse({
    "pipeline": "p",
    "inputs": {"x": "fp32[N]", "y": "fp32[N]"},
    "nodes": [
        {"op": "add", "inputs": ["x", "y"], "output": "z", "output_type": "fp32[N]"},
        {"op": "relu", "inputs": ["z"], "output": "w", "output_type": "fp32[N]"},
    ],
    "outputs": ["w"],
})
```

它显式要求 `output_type`（本阶段不做类型推断），注册 op 名称，构造并验证
`DataflowGraph`；测试里 `compiler.optimization.fusion.fuse(graph)` 能把它分成一个
融合组，说明图级 pass 不依赖 DSL 源文本。真正把它接到 Triton/参考后端是 Phase 2。

## 仍然存在的依赖（Phase 1b 目标）

- `compiler/ir/dataflow_builder.py` 仍消费 `frontend.ast_node` 节点和
  `frontend.type_inference`；
- `compiler/ir/semantic_ir.py`/`composite_ops.py` 仍 import `frontend.operator` 的
  完整语义（arity/type rules/fusion 分类）；
- `compiler/optimization/fusion.py` 从 DSL operator registry 读点状算子分类。

Phase 1b：把算子语义/类型规则抽到 core，IR builder 接受 frontend-neutral program
节点；Phase 2：Triton importer / Python eDSL，用 DSL 做差分测试。

## 真正的 import 隔离（Phase 1a hardening）

`compiler/__init__.py` 的公共 API（`compile_source` 等）改为 PEP 562 lazy
import：`import compiler.core` 不再间接 import DSL/lark。测试用子进程验证
`compiler.frontend` 和 `lark` 都不在 `sys.modules` 里；programmatic frontend
同样不会加载 DSL parser。没有这一步，"core 不依赖 DSL"只在模块 import 图上成立。

## 验证

`test/test_frontend_boundary.py`（10 条）覆盖 core 无 frontend 依赖、DSL/core
对象同一、DSL 满足协议、任意前端可注册、IR 不 import 语法层、programmatic 前端
无需 DSL 源文本即可构造并融合。全量测试 2885/0/6。
