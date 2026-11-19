"""Import a small Triton kernel subset into the frontend-neutral core AST.

Phase 2b's forcing function: external Triton kernels should be able to enter the
FlowGPU IR/contracts/optimizer without anybody rewriting them in the DSL.  This
is a deliberately small importer, not a Triton frontend:

* one ``@triton.jit`` function; no control flow, no loops, no atomics;
* contiguous ``tl.load(ptr + offsets, mask=...)`` reads an input tensor;
* ``tl.store(ptr + offsets, value, mask=...)`` writes one output;
* elementwise ``+ - * /`` and a whitelist of ``tl.*`` elementwise calls;
* ``tl.sum(x, axis=k)`` / ``tl.max(x, axis=k)`` reductions;
* numeric constants become implicit rank-0 scalar inputs (the core language has
  no float literals), so a caller supplies the same values it compiled into the
  original kernel.

The caller supplies the high-level input/output types because a Triton kernel
has pointers, not shapes/dtypes::

    import_triton(source, input_types={"x": "fp32[N]", "y": "fp32[N]"},
                  output_types={"out": "fp32[N]"})

Tiling, masks, program ids and pointer arithmetic are *lowering* details of the
original kernel and are dropped; the imported program keeps the logical values
and operations, which is what the FlowGPU backends re-tile anyway.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from compiler.core.ast import Assignment, Call, InputDecl, Name, OutputDecl, PipelineDecl
from compiler.core.literals import DTypeLiteral, IntLiteral, ShapeLiteral
from compiler.core.operators import OPERATORS
from compiler.core.types import TensorType

# elementwise tl.* calls that map directly to a registered FlowGPU operator
_ELEMENTWISE = {
    "where": "where",
    "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2",
    "sqrt": "sqrt", "rsqrt": "rsqrt", "abs": "abs", "sigmoid": "sigmoid",
    "tanh": "tanh", "erf": "erf", "relu": "relu",
    "maximum": "maximum", "minimum": "minimum",
}
_REDUCTIONS = {"sum": "sum_axis", "max": "max_axis"}
_COMPARE = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
            ast.Eq: "eq", ast.NotEq: "ne"}
_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class ImportError(ValueError):
    """The kernel is outside the Phase 2b subset."""


@dataclass
class ImportedKernel:
    pipelines: tuple[PipelineDecl, ...]
    source_name: str
    constants: dict[str, float] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    return ""


def _constexpr_arg(arg: ast.arg) -> bool:
    annotation = ast.unparse(arg.annotation) if arg.annotation is not None else ""
    return "constexpr" in annotation


def _base_pointer(expr) -> str | None:
    """The pointer name at the root of ``ptr + offsets`` / ``ptr[offsets]``."""
    while True:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.BinOp):
            # both `ptr + off` and `off + ptr` are common; the pointer is the
            # operand that is a declared argument, resolved by the caller later
            expr = expr.left
            continue
        if isinstance(expr, ast.Subscript):
            expr = expr.value
            continue
        return None


def _ptr_name(expr, pointer_args: set[str]) -> str:
    """Find the declared pointer name anywhere on the address's left spine."""
    node = expr
    while True:
        if isinstance(node, ast.Name) and node.id in pointer_args:
            return node.id
        if isinstance(node, ast.BinOp):
            found = _ptr_name(node.left, pointer_args) if isinstance(node.left, (ast.Name, ast.BinOp)) else None
            if found is None and isinstance(node.right, ast.Name) and node.right.id in pointer_args:
                return node.right.id
            if found is not None:
                return found
            node = node.left
            continue
        if isinstance(node, ast.Subscript):
            node = node.value
            continue
        raise ImportError(f"cannot find a declared pointer in address {ast.unparse(expr)!r}")


def _literal_value(node):
    """A numeric constant, or None for a non-constant expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        return None if inner is None else -inner
    return None


def import_triton(source: str, *, input_types: dict[str, str],
                  output_types: dict[str, str],
                  pipeline_name: str | None = None) -> ImportedKernel:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ImportError(f"not valid Python: {exc}") from exc
    def has_jit_decorator(fn) -> bool:
        for decorator in fn.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _dotted(target).endswith("jit"):
                return True
        return False

    kernels = [node for node in module.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and has_jit_decorator(node)]
    if len(kernels) != 1:
        raise ImportError(f"expected exactly one @triton.jit function, found {len(kernels)}")
    kernel = kernels[0]
    if isinstance(kernel, ast.AsyncFunctionDef):
        raise ImportError("async kernels are not supported")
    args = [arg.arg for arg in kernel.args.args]
    pointer_args = set(input_types) | set(output_types)
    unknown = pointer_args - set(args)
    if unknown:
        raise ImportError(f"declared pointer(s) not in kernel signature: {sorted(unknown)}")
    constexpr_args = {arg.arg for arg in kernel.args.args if _constexpr_arg(arg)} | {
        arg.arg for arg in kernel.args.kwonlyargs if _constexpr_arg(arg)
    }
    for pointer in set(output_types):
        if pointer in constexpr_args:
            raise ImportError(f"output {pointer!r} is declared constexpr")
    scalar_args = set(args) - pointer_args - constexpr_args
    # Runtime scalar args are allowed only in dropped address/mask expressions;
    # using one in tensor arithmetic hits resolve() and is rejected there.

    env: dict[str, str] = {name: name for name in input_types}  # loaded tensor values
    ignored: set[str] = set()          # offsets/masks/program ids
    constants: dict[str, float] = {}
    const_inputs: list[tuple[str, str, float]] = []
    statements: list[object] = []
    declared: set[str] = set(input_types) | set(output_types)
    counter = [0]
    stores: list[str] = []

    def fresh() -> str:
        while True:
            counter[0] += 1
            name = f"i{counter[0]}"
            if name not in declared and name not in input_types:
                declared.add(name)
                return name

    def broadcast_scalar(value, target):
        """Materialise a scalar constant as a tile so comparisons/where type."""
        if not (isinstance(value, Name) and value.name in constants):
            return value
        # The language has no scalar-accepting broadcast operator, but
        # `add(scalar, zeros_like(tensor))` is the canonical way to materialise
        # a scalar as a tile without introducing a new operator.
        zero = fresh()
        statements.append(Assignment([zero], Call("zeros_like", [target])))
        result = fresh()
        statements.append(Assignment([result], Call("add", [value, Name(zero)])))
        return Name(result)

    def resolve(node):
        if isinstance(node, ast.Name):
            if node.id in env:
                return Name(env[node.id])
            if node.id in ignored:
                raise ImportError(f"{node.id!r} is an address/mask value, not a tensor")
            if node.id in constexpr_args:
                raise ImportError(f"constexpr {node.id!r} is only supported as a static axis for reductions")
            raise ImportError(f"unknown value {node.id!r}")
        value = _literal_value(node)
        if value is not None:
            key = f"c{len(const_inputs)}"
            dtype = "int32" if isinstance(value, int) else "fp32"
            constants[key] = value
            const_inputs.append((key, dtype, value))
            declared.add(key)
            return Name(key)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ImportError("chained comparisons are not supported")
            op = type(node.ops[0])
            if op not in _COMPARE:
                raise ImportError(f"unsupported comparison {ast.unparse(node)!r}")
            left, right = resolve(node.left), resolve(node.comparators[0])
            left = broadcast_scalar(left, right)
            right = broadcast_scalar(right, left)
            result = fresh()
            statements.append(Assignment([result], Call(_COMPARE[op], [left, right])))
            return Name(result)
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINOPS:
                raise ImportError(f"unsupported binary operator {ast.unparse(node)!r}")
            left, right = resolve(node.left), resolve(node.right)
            result = fresh()
            statements.append(Assignment([result], Call(_BINOPS[type(node.op)], [left, right])))
            return Name(result)
        if isinstance(node, ast.Call):
            func = _dotted(node.func)
            short = func.rsplit(".", 1)[-1]
            if short in _ELEMENTWISE:
                if not node.args:
                    raise ImportError(f"{func} needs one operand")
                operands = [resolve(argument) for argument in node.args]
                if short == "where":
                    # where requires matching shapes: broadcast scalar branches
                    # to the first tensor operand, exactly as the language's
                    # explicit broadcast_like would.
                    target = next((operand for operand in operands
                                   if isinstance(operand, Name) and operand.name not in constants), None)
                    if target is not None:
                        operands = [broadcast_scalar(operand, target) for operand in operands]
                result = fresh()
                statements.append(Assignment([result], Call(_ELEMENTWISE[short], operands)))
                return Name(result)
            if short in _REDUCTIONS:
                if len(node.args) != 1:
                    raise ImportError(f"{func} needs exactly one tensor operand")
                operand = resolve(node.args[0])
                axis = 0
                for keyword in node.keywords:
                    if keyword.arg == "axis":
                        literal = _literal_value(keyword.value)
                        if literal is None or not isinstance(literal, int):
                            raise ImportError(f"{func}: axis must be an integer constant")
                        axis = literal
                result = fresh()
                statements.append(Assignment([result], Call(_REDUCTIONS[short], [operand, IntLiteral(axis)])))
                return Name(result)
            raise ImportError(f"unsupported Triton call {func!r}")
        raise ImportError(f"unsupported expression {ast.unparse(node)!r}")

    def load(node: ast.Call):
        if not node.args:
            raise ImportError("tl.load needs an address")
        pointer = _ptr_name(node.args[0], pointer_args)
        if pointer not in input_types:
            raise ImportError(f"tl.load reads {pointer!r}, which is not a declared input")
        env_var = None
        return pointer

    def ignored_assignment(node) -> bool:
        """True for the offset/mask/program-id computations we intentionally drop."""
        value = node.value
        calls = {_dotted(sub.func) for sub in ast.walk(value) if isinstance(sub, ast.Call)}
        if any(call in {"tl.program_id", "tl.arange"} for call in calls):
            return True
        if isinstance(value, ast.Compare):
            # A mask comparison over an address/arange is dropped; a value
            # comparison (used by tl.where) is lowered in resolve().
            names = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
            return bool(names & ignored)
        if isinstance(value, ast.BinOp) and any(
            isinstance(sub, ast.Name) and sub.id in ignored for sub in ast.walk(value)
        ):
            return True
        return False

    for statement in kernel.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ImportError("only single-name assignments are supported")
            target = statement.targets[0].id
            value = statement.value
            if isinstance(value, ast.Call) and _dotted(value.func).endswith("tl.load"):
                pointer = load(value)
                env[target] = pointer
                continue
            if ignored_assignment(statement):
                ignored.add(target)
                continue
            resolved = resolve(value)
            # resolve() emitted an intermediate; map the user's variable to it
            if isinstance(resolved, Name):
                env[target] = resolved.name
            else:
                raise ImportError(f"unsupported assignment value for {target!r}")
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and _dotted(statement.value.func).endswith("tl.store"):
            call = statement.value
            if len(call.args) < 2:
                raise ImportError("tl.store needs an address and a value")
            pointer = _ptr_name(call.args[0], pointer_args)
            if pointer not in output_types:
                raise ImportError(f"tl.store writes {pointer!r}, which is not a declared output")
            value = resolve(call.args[1])
            if not isinstance(value, Name):
                raise ImportError("tl.store value must be a tensor value")
            statements.append(OutputDecl(value.name))
            stores.append(pointer)
        elif isinstance(statement, ast.Pass):
            continue
        else:
            raise ImportError(f"unsupported statement in @triton.jit body: {type(statement).__name__}")

    missing = set(output_types) - set(stores)
    if missing:
        raise ImportError(f"output(s) never stored: {sorted(missing)}")
    declarations = [InputDecl(name, _tensor_type(type_text)) for name, type_text in input_types.items()]
    declarations += [InputDecl(name, TensorType(dtype=dtype, shape=[])) for name, dtype, _ in const_inputs]
    pipeline = PipelineDecl(pipeline_name or kernel.name, declarations + statements)
    return ImportedKernel(pipelines=(pipeline,), source_name=kernel.name,
                          constants=constants, inputs=dict(input_types),
                          outputs=dict(output_types))


def _tensor_type(text: str) -> TensorType:
    text = str(text).strip()
    if "[" not in text:
        return TensorType(dtype=text, shape=[])
    dtype, rest = text.split("[", 1)
    shape = [part.strip() for part in rest.rstrip("]").split(",") if part.strip()]
    return TensorType(dtype=dtype.strip(), shape=shape)
