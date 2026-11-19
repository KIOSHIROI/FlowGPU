"""`cast_to(value, dtype)`: an explicit cast whose target is written.

`cast_like(value, template)` needs a tensor to read the target dtype from, so
"cast to the output dtype" forced a `[1]` input whose only job was to carry a
dtype: one validated device tensor passed positionally on every call, and (when
the ABI pass proves it unused) a kernel formal that is pruned. The Task160
diagnosis measured that tensor at ~2.1 us to build plus ~1.5 us of validation
per call, against a whole eager reference call of 23 us.

This is the dtype-literal form. The target is written, so no tensor exists just
to hold it. The conversion contract is `cast_like`'s: same shape, the written
target dtype, bool targets refused as "not a number", and float-to-integer
refused because that direction's rounding is not specified here.

These tests are CPU-only: they check parsing, typing, the emitted Triton text,
the emitted reference program and its executable output, and the emitted host.
"""

import ast
import struct

import pytest

from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.ast_node import DTypeLiteral
from compiler.frontend.diagnostic import CompileError, DiagnosticKind
from compiler.frontend.parser import parse_ast


def compile_one(source, **options):
    return compile_source(source, **options).pipelines[0]


def signature_of(kernel):
    function = ast.parse(kernel.source).body[0]
    return [
        (argument.arg, ast.unparse(argument.annotation)
         if argument.annotation else None)
        for argument in function.args.args
    ]


def host_parameters(program):
    header = next(
        line for line in program.host.splitlines() if line.startswith("def ")
    )
    return [
        parameter.strip()
        for parameter in header.split("(")[1].split(")")[0].split(",")
        if parameter.strip()
    ]


def bf16_round(value):
    """The rounding the generated reference calls `bf16`: nearest even, 8 bits."""
    bits = struct.unpack("I", struct.pack("f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]


def fp16_round(value):
    return struct.unpack("e", struct.pack("e", value))[0]


BARE = "pipeline p { input x:fp32[M,N] y=cast_to(x, bf16) output y }"


# --- grammar and AST -------------------------------------------------------


def test_a_dtype_is_an_expression_atom():
    pipeline = parse_ast(
        "pipeline p { input x:fp32[M,N] y=cast_to(x, bf16) output y }"
    )[0]

    assignment = pipeline.statements[1]
    target = assignment.value.args[1]

    assert isinstance(target, DTypeLiteral)
    assert target.dtype == "bf16"
    assert str(target) == "bf16"
    # The position is kept so a misplaced dtype literal is reported where it
    # was written, the same way a shape literal is.
    assert target.position is not None


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int32", "int64"])
def test_every_non_bool_dtype_parses_as_a_cast_target(dtype):
    pipeline = parse_ast(
        f"pipeline p {{ input x:fp32[M,N] y=cast_to(x, {dtype}) output y }}"
    )[0]

    assert pipeline.statements[1].value.args[1].dtype == dtype


def test_declarations_still_use_dtype_tokens_normally():
    """Adding a dtype atom must not change how a declaration reads one."""
    pipeline = parse_ast("pipeline p { input a:fp16 input b:bf16[1] output b }")[0]

    scalar, tensor = pipeline.statements[0], pipeline.statements[1]
    assert scalar.type.dtype == "fp16" and scalar.type.shape == []
    assert tensor.type.dtype == "bf16" and tensor.type.shape == ["1"]


# --- type rule -------------------------------------------------------------


def test_cast_to_preserves_shape_and_takes_the_written_dtype():
    pipeline = compile_one(BARE)

    inferred = pipeline.symbols.lookup("y").type
    assert inferred.dtype == "bf16"
    assert inferred.shape == ["M", "N"]


def test_cast_to_accepts_a_widening_cast():
    pipeline = compile_one(
        "pipeline p { input x:fp16[M,N] y=cast_to(x, fp32) output y }"
    )

    assert pipeline.symbols.lookup("y").type.dtype == "fp32"


def test_cast_to_accepts_an_integer_to_float_cast():
    pipeline = compile_one(
        "pipeline p { input x:int32[M,N] y=cast_to(x, fp32) output y }"
    )

    assert pipeline.symbols.lookup("y").type.dtype == "fp32"


def test_cast_to_accepts_float_to_integer_with_defined_semantics():
    """Float -> integer is defined, not refused: truncate toward zero, wrap.

    It used to be a TYPE refusal ("semantics are not implemented").  The rule is
    now torch's `.to(int)` in both backends -- the Triton emitter already wrote
    `.to(dtype)` and the Python reference's `int8`/`int32`/`int64` helpers already
    truncated this way -- so the refusal was the only thing missing.  Rounding is
    a separate operator (`round`), which keeps one operator from owning both.
    """
    for target in ("int8", "uint8", "int32", "int64"):
        pipeline = compile_one(
            f"pipeline p {{ input x:fp32[M,N] y=cast_to(x, {target}) output y }}"
        )
        assert pipeline.symbols.lookup("y").type.dtype == target


def test_cast_to_refuses_a_bool_target():
    with pytest.raises(CompileError, match='olumn 34: Cannot cast a bool: bool is not a number') as caught:
        compile_one("pipeline p { input x:fp32[M,N] y=cast_to(x, bool) output y }")

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "bool is not a number" in caught.value.message


def test_cast_to_refuses_a_bool_operand():
    with pytest.raises(CompileError, match='olumn 48: Cannot cast a bool: bool is not a number') as caught:
        compile_one(
            "pipeline p { input x:fp32[M,N] y=isfinite(x) z=cast_to(y, fp32) output z }"
        )

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "bool is not a number" in caught.value.message


def test_cast_to_requires_a_dtype_literal_not_a_tensor():
    """A template tensor still belongs to cast_like; cast_to wants the dtype."""
    with pytest.raises(CompileError, match="olumn 53: cast_to's second argument must be a dtype literal such as bf16") as caught:
        compile_one(
            "pipeline p { input x:fp32[M,N] input like:bf16[1] "
            "y=cast_to(x, like) output y }"
        )

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "must be a dtype literal" in caught.value.message


def test_a_dtype_literal_is_refused_outside_cast_tos_second_argument():
    with pytest.raises(CompileError, match="olumn 41: a dtype literal cannot be an argument of 'add'; it only names cast_to's target dtype") as caught:
        compile_one("pipeline p { input x:fp32[M,N] y=add(x, bf16) output y }")

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "dtype literal" in caught.value.message
    assert "cast_to" in caught.value.message


def test_a_dtype_literal_is_not_a_value():
    with pytest.raises(CompileError, match="olumn 34: a dtype literal only names cast_to's target dtype; it is not a tensor") as caught:
        compile_one("pipeline p { input x:fp32[M,N] y=bf16 output y }")

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "not a tensor" in caught.value.message


def test_cast_like_still_takes_a_tensor_template():
    """The old spelling is unchanged: a dtype literal is not its argument."""
    with pytest.raises(CompileError, match="olumn 66: a dtype literal cannot be an argument of 'cast_like'; it only names cast_to's target dtype") as caught:
        compile_one(
            "pipeline p { input x:fp32[M,N] input like:bf16[1] "
            "y=cast_like(x, bf16) output y }"
        )

    assert caught.value.kind is DiagnosticKind.TYPE
    assert "cast_like" in caught.value.message

    pipeline = compile_one(
        "pipeline p { input x:fp32[M,N] input like:bf16[1] "
        "y=cast_like(x, like) output y }"
    )
    assert pipeline.symbols.lookup("y").type.dtype == "bf16"


# --- emitted Triton --------------------------------------------------------


def test_the_cast_is_emitted_and_no_dtype_tensor_is_an_argument():
    pipeline = compile_one(BARE)
    program = pipeline.triton

    assert program.complete
    (kernel,) = program.kernels
    assert kernel.ops == ("cast_to",)
    assert ".to(tl.bfloat16)" in kernel.source
    # One input pointer, one output pointer, two dimensions, four layout
    # constexprs for the one tensor, one tile: no template tensor anywhere.
    formals = dict(signature_of(kernel))
    assert formals["x_ptr"] is None and formals["y_ptr"] is None
    assert "like_ptr" not in formals
    assert set(formals) == {"x_ptr", "y_ptr", "M", "N", "x_contiguous",
                            "x_dim0", "x_stride0", "x_dim1", "x_stride1",
                            "FG_BLOCK"}


def test_the_host_signature_has_only_the_values_the_program_declares():
    program = compile_one(BARE).triton

    assert host_parameters(program) == ["x"]
    assert "like" not in program.host
    assert "torch.bfloat16" in program.host
    assert "torch.empty" in program.host


def test_a_cast_into_a_fused_region_is_still_one_rounding_boundary():
    """cast_to inside a pointwise chain is the declared dtype boundary."""
    program = compile_one(
        "pipeline p { input x:bf16[N] a=cast_to(x, fp32) y=mul(a, a) output y }"
    ).triton

    (kernel,) = program.kernels
    assert kernel.ops == ("cast_to", "mul")
    assert "(x_ptr)" not in kernel.source
    assert ".to(tl.float32)" in kernel.source


def test_cast_to_survives_abi_pruning():
    """The builder compiles with prune_kernel_abi=True; no formal is invented."""
    program = compile_one(BARE, prune_kernel_abi=True).triton

    assert program.complete
    (kernel,) = program.kernels
    # x_dim0 is pruned: a flat 2-D pointwise kernel addresses through the row
    # stride, so the extent of axis 0 is never read.
    assert kernel.parameters == (
        "x_ptr", "y_ptr", "M: tl.constexpr", "N: tl.constexpr",
        "x_contiguous: tl.constexpr", "x_stride0: tl.constexpr",
        "x_dim1: tl.constexpr", "x_stride1: tl.constexpr",
        "FG_BLOCK: tl.constexpr",
    )


def test_cast_to_is_lowered_by_both_backends():
    program = compile_one(BARE)

    assert program.reference.complete
    (reference_kernel,) = program.reference.kernels
    assert "bf16(" in reference_kernel.source
    assert "like" not in reference_kernel.source
    assert program.triton.kernels[0].source


# --- emitted reference: numbers -------------------------------------------


def test_the_reference_backend_rounds_to_the_written_dtype():
    program = compile_one(BARE).reference
    sizes = {"M": 2, "N": 2}
    values = inputs_for(program, sizes)[0][3]

    output = run_reference(program, sizes)["y"]

    assert output == [bf16_round(value) for value in values]


def test_the_reference_backend_matches_cast_like_on_the_same_values():
    """The conversion contract is cast_like's, so the two spellings agree."""
    written = compile_one(
        "pipeline p { input x:fp32[M,N] y=cast_to(x, bf16) output y }"
    ).reference
    templated = compile_one(
        "pipeline p { input x:fp32[M,N] input like:bf16[1] "
        "y=cast_like(x, like) output y }"
    ).reference
    sizes = {"M": 3, "N": 2}

    assert run_reference(written, sizes)["y"] == run_reference(templated, sizes)["y"]


def test_the_reference_backend_rounds_fp16_too():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] y=cast_to(x, fp16) output y }"
    ).reference
    sizes = {"M": 2, "N": 2}
    values = inputs_for(program, sizes)[0][3]

    output = run_reference(program, sizes)["y"]

    assert output == [fp16_round(value) for value in values]


# --- shell: both spellings together ---------------------------------------


def test_both_cast_spellings_compile_in_one_program():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input like:bf16[1] "
        "a=cast_like(x, like) y=cast_to(a, fp32) output y }"
    ).triton

    assert program.complete
    assert len(program.kernels) == 1
    assert "like_ptr" in program.kernels[0].source
