"""The torch.library registration section for a generated module.

Design notes live with the emitter in triton_backend.py; this module exists
only to keep the section's text in one place. The block is pure rendering:
every decision (which symbolic dimensions exist, which shapes the outputs
have, whether an output aliases an input) was already made by the host
emission and the dataflow graph, and this text must not invent a second
opinion about any of them.
"""

from compiler.core.ast import InputDecl, OutputDecl
from compiler.ir.dataflow import resolve_alias


def render_custom_ops(pipeline, graph, parameters, config, dtype_to_torch, is_symbol) -> str:
    """One registration block per pipeline, or an explanatory comment.

    vLLM and SGLang consume operators as torch ops --
    `torch.ops.flowgpu.<pipeline>` -- so that is the contract this section
    registers. The real implementation is the host entry above it,
    unchanged; nothing is re-implemented. The fake implementation repeats
    the symbolic-dimension rule the host uses (a dimension symbol reads off
    the first input that declares it) and allocates each output with
    `new_empty`, which is all torch.compile needs to trace a shape-only
    call without touching a GPU.

    `mutates_args` is empty by construction: the host allocates every slot
    with torch.empty, so a caller's tensor is never written in place. An
    output that aliases an input (`y = x`) is returned, not mutated.

    Registration happens at import time and once per process: importing the
    same module twice in one process raises, because torch.library treats
    a re-registration of the same name as a program error, not a
    redefinition. A pipeline with no inputs or no outputs has nothing to
    call, so it registers nothing and says so.
    """
    declared = [s for s in pipeline.statements if isinstance(s, InputDecl)]
    outputs = [s.name for s in pipeline.statements if isinstance(s, OutputDecl)]
    entry = f"run_{pipeline.name}"
    prefix = f"flowgpu_{pipeline.name}"

    if not declared or not outputs:
        missing = "no inputs" if not declared else "no outputs"
        return (f"\n\n# pipeline '{pipeline.name}': {missing}, nothing to "
                f"register as a torch op\n")

    # A scalar input is a Python number in the torch op schema, not a tensor:
    # `torch.library` cannot declare a 0-d tensor parameter for a value the
    # host entry consumes as a float, and a rank-0 Tensor would make the op
    # contract disagree with the kernel ABI.
    float_dtypes = {"fp16", "bf16", "fp32"}

    def annotation(statement):
        if not statement.type.is_scalar:
            return "torch.Tensor"

        return "float" if statement.type.dtype in float_dtypes else "int"

    sig = ", ".join(
        f"{parameters[s.name]}: {annotation(s)}" for s in declared
    )
    call = ", ".join(parameters[s.name] for s in declared)
    resolved = [
        (value, resolve_alias(graph.aliases, value)) for value in outputs
    ]

    lines = [
        "",
        f"# torch.library registration for pipeline '{pipeline.name}'.",
        f"# torch.ops.flowgpu.{pipeline.name}({', '.join(parameters[s.name] for s in declared)})",
        "# -> the host entry above. mutates_args=(): slots are fresh",
        "# torch.empty allocations, a caller's tensor is never written.",
        "",
        f"def _{prefix}_impl({sig}) -> {returns_annotation(len(outputs))}:",
        '    """Real implementation: the host entry above, unchanged."""',
        *([f"    _flowgpu_result = {entry}({call})"] if len(outputs) > 1 else []),
        f"    return {returns_expression(entry, call, resolved)}",
        "",
        f"_{prefix}_op = torch.library.custom_op(",
        f'    "flowgpu::{pipeline.name}", _{prefix}_impl,',
        f'    mutates_args=(), device_types=("{config.device_type}",))',
        "",
        f"@_{prefix}_op.register_fake",
        f"def _{prefix}_meta({sig}) -> {returns_annotation(len(outputs))}:",
        '    """Shape/dtype-only: torch.compile traces without a GPU."""',
    ]

    # The same symbolic-dimension rule as the host: a dimension symbol is
    # read off the first input that declares it, in declaration order.
    bound = []
    for statement in declared:
        for axis, dimension in enumerate(statement.type.shape):
            if is_symbol(dimension) and dimension not in bound:
                bound.append(dimension)
                lines.append(
                    f"    {dimension} = {parameters[statement.name]}.shape[{axis}]"
                )
    if not bound:
        lines.append("    pass")

    # An output is allocated like some input: the first tensor one, because a
    # scalar has no `new_empty` and no device of its own. A pipeline with an
    # output but no tensor input cannot exist (the type rules produce tensors
    # only from tensors), so the search is an invariant, not a fallback.
    first_tensor = next(
        (statement for statement in declared if not statement.type.is_scalar),
        None,
    )
    if first_tensor is None:
        raise AssertionError(
            f"pipeline '{pipeline.name}' has outputs but no tensor input to "
            f"allocate them from"
        )
    first = parameters[first_tensor.name]
    empties = []
    for value, base in resolved:
        tensor_type = graph.values[base]
        shape = ", ".join(str(d) for d in tensor_type.shape)
        empties.append(
            f"{first}.new_empty(({shape}), dtype={dtype_to_torch[tensor_type.dtype]})"
        )
    if len(empties) == 1:
        lines.append(f"    return {empties[0]}")
    else:
        lines.append(f"    return ({', '.join(empties)},)")
    return "\n".join(lines) + "\n"


def returns_annotation(count) -> str:
    if int(count) == 1:
        return "torch.Tensor"
    # torch.library infers a fixed output schema; Ellipsis is not a Tensor.
    return "tuple[" + ", ".join(["torch.Tensor"] * int(count)) + "]"


def returns_expression(entry, call, resolved) -> str:
    container = f"{entry}({call})" if len(resolved) == 1 else "_flowgpu_result"
    gets = [f'{container}["{value}"]' for value, _base in resolved]
    if len(gets) == 1:
        return gets[0]
    return "(" + ", ".join(gets) + ")"
