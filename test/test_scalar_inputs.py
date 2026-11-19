"""Declared scalars: `input alpha:fp32` is one number, not a tile.

The language had no scalar type, so a task parameter had to be declared as a
`[M,N]` tensor and expanded by the adapter into a stride-0 view. That costs a
device allocation and a host-to-device copy per call, plus four layout
constexprs per operand that describe strides that are all zero. A declared
scalar is rank 0: the host validates a Python number, the kernel takes it as a
plain argument, and no pointer or layout metadata exists for it.

These tests are CPU-only, like the compiler: they check parsing, typing, the
emitted signature/body/host text, the reference backend's executable output,
and the buffer plan. Nothing here runs a kernel.
"""

import ast
import re

import pytest

from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.ast_node import TensorType
from compiler.frontend.diagnostic import CompileError
from compiler.frontend.parser import parse_ast
from lark.exceptions import UnexpectedToken


ONE_SCALAR = (
    "pipeline p { input x:bf16[M,N] input a:fp32 y=mul(x,a) output y }"
)

# The Task160 shape: six [M,N] operands of which alpha/beta/limit/neg_limit
# become scalars, one [1] dtype template, and the value-layer arithmetic the
# task actually uses (the clamps are what make limit/neg_limit scalars).
TASK160_SHAPE = {
    "legacy": (
        "pipeline t { input gate:fp32[M,N] input up:fp32[M,N] "
        "input alpha:fp32[M,N] input beta:fp32[M,N] "
        "input limit:fp32[M,N] input neg_limit:fp32[M,N] "
        "input like:fp32[1] "
        "g=to_fp32(gate) u=to_fp32(up) a=alpha b=beta l=limit nl=neg_limit "
        "upper=minimum(g,l) lower=maximum(u,nl) bounded=minimum(lower,l) "
        "gated=mul(upper,sigmoid(mul(a,upper))) res=mul(gated,add(bounded,b)) "
        "out=cast_like(res,like) output out }"
    ),
    "scalar": (
        "pipeline t { input gate:fp32[M,N] input up:fp32[M,N] "
        "input alpha:fp32 input beta:fp32 "
        "input limit:fp32 input neg_limit:fp32 "
        "input like:fp32[1] "
        "g=to_fp32(gate) u=to_fp32(up) a=alpha b=beta l=limit nl=neg_limit "
        "upper=minimum(g,l) lower=maximum(u,nl) bounded=minimum(lower,l) "
        "gated=mul(upper,sigmoid(mul(a,upper))) res=mul(gated,add(bounded,b)) "
        "out=cast_like(res,like) output out }"
    ),
}


def compile_one(source, **options):
    return compile_source(source, **options).pipelines[0]


def signature_of(kernel):
    """The emitted one-line signature, parsed into its formal names/annotations."""
    function = ast.parse(kernel.source).body[0]
    formals = []
    for argument in function.args.args:
        formals.append(
            (argument.arg, ast.unparse(argument.annotation)
             if argument.annotation else None)
        )
    return formals


def kernel_launch_of(program):
    """The one kernel launch line of a generated host."""
    lines = [
        line.strip() for line in program.triton.host.splitlines()
        if re.search(r"_k\d+\[", line)
    ]
    assert len(lines) == 1, lines
    return lines[0]


def launch_arguments(program):
    """The positional arguments of the one kernel launch, as written."""
    launch = kernel_launch_of(program)
    body = launch[launch.index("](") + 2:launch.rindex(")")]
    arguments = [argument.strip() for argument in body.split(", ")]
    return [argument for argument in arguments if "=" not in argument]


# --- grammar and AST -------------------------------------------------------


def test_scalar_declaration_is_rank_zero():
    pipeline = parse_ast(ONE_SCALAR)[0]
    scalar = next(
        statement for statement in pipeline.statements
        if getattr(statement, "name", None) == "a"
    )

    assert isinstance(scalar.type, TensorType)
    assert scalar.type.dtype == "fp32"
    assert scalar.type.shape == []
    assert scalar.type.is_scalar
    assert str(scalar.type) == "fp32"


def test_bracketed_single_element_stays_a_tensor():
    pipeline = parse_ast(
        "pipeline p { input like:fp16[1] output like }"
    )[0]
    declared = pipeline.statements[0]

    assert declared.type.shape == ["1"]
    assert not declared.type.is_scalar
    assert str(declared.type) == "fp16[1]"


def test_scalar_dtype_is_required():
    # A bare name after the colon is a syntax error, not a silent rank-0 type.
    with pytest.raises(UnexpectedToken, match='Unexpected token Token'):
        parse_ast("pipeline p { input a:bad output a }")


# --- type checking ---------------------------------------------------------


def test_scalar_broadcasts_against_a_tile_and_keeps_its_dtype():
    program = compile_one(ONE_SCALAR)
    symbols = program.symbols.symbols

    assert symbols["a"].type.is_scalar
    assert str(symbols["y"].type) == "bf16[M,N]"
    assert program.triton.complete


@pytest.mark.parametrize("operator", ["add", "sub", "mul"])
def test_either_side_may_hold_the_scalar(operator):
    program = compile_one(
        f"pipeline p {{ input x:fp32[M,N] input a:fp32 "
        f"y={operator}(a,x) output y }}"
    )

    formals = dict(signature_of(program.triton.kernels[0]))
    assert formals["a"] is None
    assert "x_ptr" in formals
    assert "a_ptr" not in formals


def test_ordering_ops_take_a_scalar_operand_for_clamping():
    # Task160's clamps: minimum(tile, limit) with the task parameter declared
    # as a scalar. These are value-layer anchors, not pointwise operators, and
    # the primitive/indexed lowering must still pass the number through.
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input lo:fp32 input hi:fp32 "
        "a=maximum(x,lo) b=minimum(a,hi) output b }"
    )

    assert program.triton.complete, program.triton.messages()
    assert len(program.triton.kernels) == 1
    formals = dict(signature_of(program.triton.kernels[0]))
    assert formals["lo"] is None
    assert formals["hi"] is None
    assert "lo_ptr" not in formals
    assert "hi_contiguous" not in formals


def test_two_scalars_have_no_tile_to_broadcast_over():
    with pytest.raises(CompileError, match='olumn 42: Cannot multiply two scalars: a scalar input is a number, not a tensor, and a pipeline value must be a tensor') as caught:
        compile_source(
            "pipeline p { input a:fp32 input b:fp32 y=mul(a,b) output y }"
        )

    assert "two scalars" in str(caught.value)


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "bool"])
def test_a_scalar_cannot_declare_a_precision_the_host_cannot_carry(dtype):
    # A Python number reaches the device as fp32 or as an integer; a bf16
    # declaration would be a precision promise the host path cannot keep.
    with pytest.raises(CompileError, match='scalar would be carried at fp32 and its declared precision would not hold') as caught:
        compile_source(
            f"pipeline p {{ input x:fp32[M,N] input a:{dtype} "
            f"y=mul(x,a) output y }}"
        )

    message = str(caught.value)
    assert f"scalar input 'a' must be declared fp32, int32 or int64" in message
    assert dtype in message


def test_an_integer_scalar_is_supported():
    program = compile_one(
        "pipeline p { input x:int32[K] input a:int32 y=mul(x,a) output y }"
    )

    assert program.triton.complete
    assert "a = int(a)" in program.triton.host
    formals = dict(signature_of(program.triton.kernels[0]))
    assert formals["a"] is None


def test_a_scalar_cannot_be_indexed_by_a_tensor_operator():
    with pytest.raises(CompileError, match="olumn 58: operator 'rms_norm' needs tensor operands, but 'a' is a scalar \\(fp32\\); a scalar input can only mix with a tensor in elementwise arithmetic \\(add, sub, mul, maximum, minimum\\)") as caught:
        compile_source(
            "pipeline p { input x:fp32[M,N] input a:fp32 "
            "y=rms_norm(x,a) output y }"
        )

    message = str(caught.value)
    assert "rms_norm" in message
    assert "'a' is a scalar" in message
    assert "elementwise arithmetic" in message


def test_unary_operator_on_a_scalar_is_refused():
    with pytest.raises(CompileError, match="olumn 34: operator 'relu' needs tensor operands, but 'a' is a scalar \\(fp32\\); a scalar input can only mix with a tensor in elementwise arithmetic \\(add, sub, mul, maximum, minimum\\)") as caught:
        compile_source(
            "pipeline p { input a:fp32 y=relu(a) output y }"
        )

    assert "relu" in str(caught.value)
    assert "scalar" in str(caught.value)


def test_float_tile_with_integer_scalar_is_refused():
    with pytest.raises(CompileError, match='olumn 48: Cannot multiply a int32 scalar with a fp32 tensor: the scalar must be the same kind of number as the tensor') as caught:
        compile_source(
            "pipeline p { input x:fp32[M,N] input a:int32 y=mul(x,a) output y }"
        )

    assert "int32 scalar" in str(caught.value)
    assert "fp32 tensor" in str(caught.value)


def test_a_scalar_is_not_a_pipeline_output():
    with pytest.raises(CompileError, match="olumn 34: output 'a' is a scalar \\(fp32\\); a pipeline output must be a tensor") as caught:
        compile_source("pipeline p { input a:fp32 output a }")

    message = str(caught.value)
    assert "output 'a' is a scalar" in message
    assert "must be a tensor" in message


def test_an_alias_to_a_scalar_is_not_a_pipeline_output():
    with pytest.raises(CompileError, match="olumn 38: output 'y' is a scalar \\(fp32\\); a pipeline output must be a tensor") as caught:
        compile_source(
            "pipeline p { input a:fp32 y=a output y }"
        )

    assert "output 'y' is a scalar" in str(caught.value)


# --- pointwise emission ----------------------------------------------------


def test_scalar_is_one_plain_formal_with_no_layout_metadata():
    program = compile_one(ONE_SCALAR)
    kernel = program.triton.kernels[0]
    formals = signature_of(kernel)
    names = [name for name, _ in formals]

    assert ("a", None) in formals
    assert "a_ptr" not in names
    for layout_name in ("a_contiguous", "a_dim0", "a_stride0", "a_dim1", "a_stride1"):
        assert layout_name not in names
        assert layout_name not in kernel.source

    # Pointers first, then the scalar, then the constexpr/launch metadata.
    assert names[0] == "x_ptr"
    assert names[1] == "y_ptr"
    assert names[2] == "a"


def test_scalar_is_used_directly_in_the_expression():
    program = compile_one(ONE_SCALAR)
    kernel = program.triton.kernels[0]

    assert "tl.load(a" not in kernel.source
    assert "(t0 * a)" in kernel.source


def test_scalar_launch_argument_is_the_raw_number():
    program = compile_one(ONE_SCALAR)
    kernel = program.triton.kernels[0]
    launch = kernel_launch_of(program)

    assert "tensor(a)" not in launch
    assert "as_tensor" not in program.triton.host
    assert "as_strided" not in program.triton.host
    assert "a.is_contiguous()" not in launch
    assert "a.shape" not in launch
    # Host signature keeps the number, and the launch passes it positionally.
    assert program.triton.host.splitlines()[0] == "def run_p(x, a):"
    launched = launch.split("]", 1)[1]
    assert ", a, " in launched


def test_scalar_formal_order_matches_the_launch_order():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input a:fp32 input b:fp32 "
        "r=add(x,a) s=mul(r,b) output s }"
    )
    kernel = program.triton.kernels[0]
    formals = [name for name, _ in signature_of(kernel)]
    positional = launch_arguments(program)

    # The launch passes one argument per formal (plus launch options, which are
    # keyword arguments and come after the positional ones).
    assert len(positional) == len(kernel.parameters)

    # Scalars a and b are after the two pointers, before the count/constexprs.
    assert formals[:2] == ["x_ptr", "s_ptr"]
    assert formals[2:4] == ["a", "b"]
    assert positional[2:4] == ["a", "b"]
    # A fused group emits one kernel and one load per tensor, not per scalar.
    assert kernel.source.count("tl.load(") == 1


def test_a_scalar_named_like_an_abi_name_does_not_collide():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input numel:fp32 "
        "y=mul(x,numel) output y }"
    )
    formals = [name for name, _ in signature_of(program.triton.kernels[0])]

    assert len(formals) == len(set(formals))
    assert "numel" in formals and "numel_2" in formals


def test_single_element_tensor_keeps_its_pointer_and_layouts():
    program = compile_one(
        "pipeline p { input x:fp16[1] input a:fp32 y=mul(x,a) output y }"
    )
    formals = dict(signature_of(program.triton.kernels[0]))

    assert formals["x_ptr"] is None
    assert formals["a"] is None
    assert "a_ptr" not in formals
    assert formals["x_contiguous"] == "tl.constexpr"
    assert formals["x_dim0"] == "tl.constexpr"


def test_scalar_declared_but_unused_compiles():
    program = compile_one(
        "pipeline p { input a:fp32 input x:bf16[M,N] y=mul(x,x) output y }"
    )

    assert program.triton.complete
    assert "a" not in [name for name, _ in signature_of(program.triton.kernels[0])]


# --- host validation and launch -------------------------------------------


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *arguments, **options):
        self.calls.append((arguments, options))


class _FakeTensor:
    """Enough of a CUDA tensor for the emitted guards, without torch."""

    def __init__(self, shape, device="cuda:0"):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = "bfloat16-marker"
        self.is_cuda = True
        self.layout = "strided-marker"
        self.device = device

    def is_contiguous(self):
        return True

    def stride(self, axis):
        strides = []
        for index in range(len(self.shape)):
            stride = 1
            for dimension in self.shape[index + 1:]:
                stride *= dimension
            strides.append(stride)
        return strides[axis]


class _FakeTorch:
    bfloat16 = "bfloat16-marker"
    float32 = "float32-marker"
    strided = "strided-marker"

    def __init__(self):
        self.allocations = []

    def empty(self, shape, dtype=None, device=None):
        self.allocations.append((shape, dtype, device))
        return f"slot{len(self.allocations) - 1}"


def _host_namespace(program):
    kernel = program.triton.kernels[0]
    namespace = {
        "torch": _FakeTorch(),
        "triton": type("Triton", (), {"cdiv": staticmethod(lambda a, b: -(-a // b))}),
        "BLOCK": 1024,
        kernel.name: _RecordingKernel(),
    }
    return namespace, namespace[kernel.name]


def _call_host(program, *arguments):
    namespace, recorder = _host_namespace(program)
    exec(program.triton.host, namespace)
    result = namespace["run_p"](*arguments)
    return result, recorder


def test_host_passes_a_python_number_straight_to_the_kernel():
    program = compile_one(ONE_SCALAR)
    x = _FakeTensor((2, 3))

    returned, recorder = _call_host(program, x, 0.5)

    assert len(recorder.calls) == 1
    arguments, options = recorder.calls[0]
    assert arguments[0] is x
    assert arguments[1] == "slot0"
    assert arguments[2] == 0.5
    assert options == {}
    assert returned == {"y": "slot0"}


def test_host_converts_the_scalar_to_its_declared_dtype():
    program = compile_one(ONE_SCALAR)

    _returned, recorder = _call_host(program, _FakeTensor((2, 3)), 2)

    (arguments, _options) = recorder.calls[0]
    assert arguments[2] == 2.0
    assert isinstance(arguments[2], float)


def test_host_rejects_a_tensor_where_a_scalar_was_declared():
    program = compile_one(ONE_SCALAR)

    with pytest.raises(ValueError, match='input a is a scalar fp32; pass a Python int or float, not a tensor') as caught:
        _call_host(program, _FakeTensor((2, 3)), "not-a-number")

    message = str(caught.value)
    assert "input a is a scalar fp32" in message
    assert "not a tensor" in message


def test_host_rejects_a_bool_scalar():
    program = compile_one(ONE_SCALAR)

    with pytest.raises(ValueError, match='input a is a scalar fp32; pass a Python int or float, not a tensor') as caught:
        _call_host(program, _FakeTensor((2, 3)), True)

    assert "input a is a scalar fp32" in str(caught.value)


def test_host_has_no_rank_shape_dtype_device_guard_for_a_scalar():
    program = compile_one(ONE_SCALAR)
    host = program.triton.host

    assert "input a rank" not in host
    assert "input a shape" not in host
    assert "input a dtype" not in host
    assert "input a must be a CUDA tensor" not in host
    assert "input a must have a strided layout" not in host
    assert "input a must be on the same" not in host


# --- reference backend -----------------------------------------------------


def test_reference_metadata_reports_the_scalar_as_a_scalar():
    program = compile_one(ONE_SCALAR)
    reference = program.reference
    declared = dict(reference.inputs)

    assert declared["a"].is_scalar
    assert str(declared["a"]) == "fp32"
    assert str(declared["x"]) == "bf16[M,N]"


def test_reference_inputs_are_a_number_not_a_one_element_list():
    program = compile_one(ONE_SCALAR)
    values = {name: value for name, _shape, _dtype, value in inputs_for(
        program.reference, {"M": 2, "N": 3}
    )}

    assert values["x"] == [-0.5, 1.25, 0.25, -0.75, 1.0, 0.0]
    assert values["a"] == -0.5
    assert isinstance(values["a"], float)


def test_reference_kernel_uses_the_number_and_not_an_index():
    program = compile_one(ONE_SCALAR)
    kernel = program.reference.kernels[0]
    formals = dict(signature_of(kernel))

    assert formals["a"] is None
    assert "a[" not in kernel.source
    assert "(x[i] * a)" in kernel.source


def test_reference_runs_a_scalar_program():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input a:fp32 y=mul(x,a) output y }"
    )

    result = run_reference(program.reference, {"M": 2, "N": 3})

    assert result["y"] == [0.25, -0.625, -0.125, 0.375, -0.5, -0.0]


def test_reference_module_accepts_a_number():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input a:fp32 y=mul(x,a) output y }"
    )
    from compiler.backend.reference import module as reference_module

    namespace = {}
    exec(reference_module([program.reference]), namespace)
    run = namespace["run_p"]

    assert run([1.0, 2.0], 2.0, 1, 2) == {"y": [2.0, 4.0]}


def test_reference_fp32_mode_converts_the_scalar_itself():
    program = compile_one(
        "pipeline p { input x:fp32[M,N] input a:fp32 y=mul(x,a) output y }"
    )
    from compiler.backend.reference import module as reference_module

    code = reference_module([program.reference], numeric_mode="fp32")
    assert "a = fp32(a)" in code
    assert "a = [fp32(v) for v in a]" not in code

    namespace = {}
    exec(code, namespace)
    assert namespace["run_p"]([1.0, 2.0], 2.0, 1, 2) == {"y": [2.0, 4.0]}


# --- the buffer plan does not invent an allocation ------------------------


def test_scalar_never_gets_a_storage_slot():
    program = compile_one(ONE_SCALAR)
    memory = program.memory

    buffers = {buffer.value: buffer for buffer in memory.buffers}
    assert buffers["a"].tensor_type.is_scalar
    # The caller provides the number itself, so there is no buffer to
    # allocate and no slot to share.
    assert buffers["a"].storage == "input"
    assert not buffers["a"].allocated
    for slot in memory.slots:
        assert "a" not in slot.values
        for value in slot.values:
            assert not buffers[value].tensor_type.is_scalar


def test_json_metadata_marks_the_input_as_a_scalar(tmp_path, capsys):
    import json
    from compiler.cli import main

    path = tmp_path / "scalar.fgpu"
    path.write_text(ONE_SCALAR, encoding="utf-8")

    assert main([str(path), "--emit", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    inputs = {
        entry["name"]: entry
        for entry in payload["pipelines"][0]["inputs"]
    }

    assert inputs["a"]["type"] == {"dtype": "fp32", "shape": [], "scalar": True}
    assert "scalar" not in inputs["x"]["type"]


# --- the ABI pass and the compacted module --------------------------------


def test_kernel_abi_pruning_keeps_a_used_scalar():
    program = compile_one(ONE_SCALAR, prune_kernel_abi=True)
    kernel = program.triton.kernels[0]
    formals = dict(signature_of(kernel))

    assert formals["a"] is None
    assert "a_ptr" not in formals
    assert len(launch_arguments(program)) == len(kernel.parameters)


def test_compact_fixed_module_accepts_a_scalar_program():
    from compiler.backend.triton_backend import module

    program = compile_one(ONE_SCALAR, prune_kernel_abi=True)
    code = module([program.triton], compact_fixed=True)

    # The compacted module is still valid Python.
    ast.parse(code)
    assert "as_strided" not in code


# --- the Task160 parameter count ------------------------------------------


@pytest.mark.parametrize("prune,before,after", [(False, 44, 24), (True, 34, 18)])
def test_task160_parameter_count_before_and_after(prune, before, after):
    # Before: four scalars are [M,N] tensors, so each contributes a pointer and
    # five layout constexprs (contiguity + 2 dims + 2 strides).
    # After: four pointers and 20 layout constexprs become four plain formals.
    legacy = compile_one(TASK160_SHAPE["legacy"], prune_kernel_abi=prune)
    scalar = compile_one(TASK160_SHAPE["scalar"], prune_kernel_abi=prune)

    assert len(legacy.triton.kernels[0].parameters) == before
    assert len(scalar.triton.kernels[0].parameters) == after

    scalar_kernel = scalar.triton.kernels[0]
    formals = dict(signature_of(scalar_kernel))
    for name in ("alpha", "beta", "limit", "neg_limit"):
        assert formals[name] is None
        assert f"{name}_contiguous" not in formals
        assert f"{name}_dim0" not in formals
        assert f"{name}_stride0" not in formals

    # The launch passes the four raw numbers and no adapter-built tile.
    launch = kernel_launch_of(scalar)
    assert "alpha" in launch and "as_strided" not in launch
    assert "alpha.is_contiguous()" not in launch


def test_task160_scalar_program_compiles_to_one_kernel_everywhere():
    from compiler.backend.platform_configs import resolve_platform

    for platform in ("cuda", "ascend", "kunlunxin"):
        config = resolve_platform(
            platform, mapping_strategy="direct_grid", block_size=None,
            allowed_kernel_kinds=("pointwise", "primitive", "layout"),
            experimental_region_fusion=platform != "cuda",
        )
        program = compile_one(
            TASK160_SHAPE["scalar"], platform_config=config,
            prune_kernel_abi=True,
        )

        assert program.triton.complete, (platform, program.triton.messages())
        assert len(program.triton.kernels) == 1
        assert not program.reference.messages()
