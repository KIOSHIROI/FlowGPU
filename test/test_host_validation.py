"""The emitted validation block: every safety property, in fewer branches.

A generated host runs its contract checks on every call, before any allocation
or launch. The previous emission wrote one `if` per (input, property) and
compared `tuple(x.shape) != (M, N)`, which allocates twice per input: six
tensor inputs became ~24 branches. On the H100 Task160 diagnosis that block
measured 6.1 us of a 23.9 us host function.

The shape now is: one guard per tensor input for rank + device + layout, one
guard per tensor input for dtype + shape, and one guard for the whole scalar
ABI (whose failure path still names the scalar). Fewer branches and fewer
attribute reads -- but no property is dropped, which is what these tests
establish: a wrong rank, dtype, shape, device, layout, contiguity or scalar
type must still raise ValueError before the kernel runs.

The harness below executes the emitted host text against metadata-only fake
tensors, so a rejection is observed in the same order a device call would
reach it, without needing a GPU (the compiler itself imports neither torch nor
triton).
"""

import ast

import pytest

from flowgpu.compiler.compile import compile_source


# The Task160 shape, in the post-`cast_to` form the Lead's builder emits: two
# real tensor inputs and four declared scalars, no `[1]` dtype template.
TASK160 = (
    "pipeline t { input gate:bf16[M,N] input up:bf16[M,N] "
    "input alpha:fp32 input beta:fp32 input limit:fp32 input neg_limit:fp32 "
    "upper=minimum(gate,limit) lower=maximum(up,neg_limit) "
    "bounded=minimum(lower,limit) "
    "gated=mul(upper,sigmoid(mul(alpha,upper))) "
    "res=mul(gated,add(bounded,beta)) out=cast_to(res,bf16) output out }"
)

LITERAL = "pipeline p { input x:fp32[4,5] y=relu(x) output y }"
SQUARE = "pipeline p { input x:fp32[N,N] y=to_fp32(x) output y }"
MATMUL = (
    "pipeline p { input a:fp16[M,K] input b:fp16[K,N] y=matmul(a,b) output y }"
)
CAST_LIKE = (
    "pipeline p { input x:fp32[M,N] input like:bf16[1] "
    "y=cast_like(x, like) output y }"
)


# --- a fake enough torch ---------------------------------------------------


class Device:
    def __init__(self, kind, index=0):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return (isinstance(other, Device)
                and (self.type, self.index) == (other.type, other.index))

    def __hash__(self):
        return hash((self.type, self.index))

    def __str__(self):
        return f"{self.type}:{self.index}"


CUDA0 = Device("cuda", 0)
CUDA1 = Device("cuda", 1)
CPU = Device("cpu", 0)

DTYPES = {
    "fp16": "torch.float16",
    "bf16": "torch.bfloat16",
    "fp32": "torch.float32",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "bool": "torch.bool",
}


class FakeTorch:
    """The torch surface the host text uses: dtype/layout constants and empty."""

    float16 = DTYPES["fp16"]
    bfloat16 = DTYPES["bf16"]
    float32 = DTYPES["fp32"]
    int32 = DTYPES["int32"]
    int64 = DTYPES["int64"]
    bool = DTYPES["bool"]
    strided = "strided"

    def __init__(self):
        self.allocations = []

    def empty(self, shape, dtype=None, device=None):
        self.allocations.append((tuple(shape), dtype, device))
        return f"slot{len(self.allocations) - 1}"


class FakeTensor:
    """Metadata only: the host may read it, never compute with it."""

    def __init__(self, shape, dtype=DTYPES["bf16"], device=CUDA0,
                 layout="strided", cuda=True, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self._cuda = cuda
        self._contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self._cuda

    def is_contiguous(self):
        return self._contiguous

    def stride(self, axis):
        strides = []
        step = 1
        for size in reversed(self.shape):
            strides.append(step)
            step *= size
        return list(reversed(strides))[axis]


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*arguments, **options):
            self.calls.append((grid, arguments, options))
        return launch


def host_of(source):
    """The emitted host text plus a recorder for every kernel it may launch.

    Compiled the way the Task160 builder compiles (`prune_kernel_abi=True`):
    pruning rewrites kernel formals and the paired launch arguments, and
    nothing else, so the validation block under test is the shipped one.
    """
    program = compile_source(source, prune_kernel_abi=True).pipelines[0].triton
    assert program.complete, [refusal.message() for refusal in program.refusals]
    torch = FakeTorch()
    recorder = Recorder()
    namespace = {
        "torch": torch,
        "triton": type("Triton", (), {"cdiv": staticmethod(lambda a, b: -(-a // b))}),
        "_flowgpu_device_key": lambda index: "device-key",
        "BLOCK": 1024, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
        **{kernel.name: recorder for kernel in program.kernels},
    }
    exec(program.host, namespace)
    entry = next(value for name, value in namespace.items()
                 if name.startswith("run_"))
    return entry, recorder, program, torch


def valid_task160():
    return [FakeTensor((128, 1024)), FakeTensor((128, 1024)),
            0.7, 1.1, 5.0, -5.0]


# --- invariant 1: shape of the emitted block -------------------------------


def test_the_validation_block_is_branch_thinned_for_the_task160_shape():
    _entry, _recorder, program, _torch = host_of(TASK160)
    host = program.host

    branches = sum(isinstance(node, ast.If)
                   for node in ast.walk(ast.parse(host)))
    # Before the branch-thinning change: 18 `if` statements for gate/up + four
    # scalars + M,N.  The +1 is the shared 32-bit index-range guard (review
    # P1-8): one `if` for every symbolic input together, not one per tensor.
    assert branches == 12, host
    # No shape copy is allocated to compare against.
    assert "tuple(" not in host
    # The device is read once and reused by every later check.
    assert host.count("= gate.device") == 1
    # The scalar ABI is one branch whose failure path names the scalar.
    assert host.count("is a scalar fp32") == 4
    assert len(program.host.encode()) < 3000
    # The dtype-template tensor is gone, and it never was a kernel formal (the
    # ABI pass had already pruned it): the kernel keeps its 18 formals.
    assert "like" not in host
    assert len(program.kernels[0].parameters) == 18


def test_the_host_signature_carries_only_declared_inputs():
    _entry, _recorder, program, _torch = host_of(TASK160)

    assert program.host.splitlines()[0] == (
        "def run_t(gate, up, alpha, beta, limit, neg_limit):"
    )


def test_a_valid_call_validates_once_and_launches_once():
    entry, recorder, _program, torch = host_of(TASK160)

    returned = entry(*valid_task160())

    assert len(recorder.calls) == 1
    assert returned == {"out": "slot0"}
    assert len(torch.allocations) == 1
    assert torch.allocations[0][1] == DTYPES["bf16"]


# --- invariant 2: every rejection still happens, before the launch ---------


def _reject(entry, recorder, arguments, *fragments):
    launched = len(recorder.calls)
    with pytest.raises(ValueError, match='(?:input up must be a CUDA tensor with a strided layout and rank 2, on the same CUDA device as gate|input gate dtype or shape does not match its declaration|input up dtype or shape does not match its declaration|input gate must be a CUDA tensor with a strided layout and rank 2|tensor dimensions must be positive|input alpha is a scalar fp32; pass a Python int or float, not a tensor|input beta is a scalar fp32; pass a Python int or float, not a tensor|input limit is a scalar fp32; pass a Python int or float, not a tensor|input neg_limit is a scalar fp32; pass a Python int or float, not a tensor|input x dtype or shape does not match its declaration|input b dtype or shape does not match its declaration|input a must be contiguous for its consuming kernels \\(matmul do not carry layout metadata\\)|input like dtype or shape does not match its declaration)') as caught:
        entry(*arguments)
    assert len(recorder.calls) == launched, "a rejected call launched the kernel"
    message = str(caught.value)
    for fragment in fragments:
        assert fragment in message, message
    return message


@pytest.mark.parametrize("tensors, fragment", [
    # rank
    ([FakeTensor((128, 1024)), FakeTensor((128, 1024, 1))], "input up"),
    # dtype
    ([FakeTensor((128, 1024), dtype=DTYPES["fp16"]),
      FakeTensor((128, 1024))], "input gate"),
    # declared shape (the second input must match the first's extents)
    ([FakeTensor((128, 1024)), FakeTensor((128, 512))], "input up"),
    # device: a different CUDA device is not the caller's device
    ([FakeTensor((128, 1024)), FakeTensor((128, 1024), device=CUDA1)], "input up"),
    # a CPU tensor never matches the CUDA device anchor
    ([FakeTensor((128, 1024), device=CPU, cuda=False),
      FakeTensor((128, 1024))], "input gate"),
    # layout
    ([FakeTensor((128, 1024)), FakeTensor((128, 1024), layout="sparse")],
     "input up"),
    ([FakeTensor((128, 1024), layout="sparse"), FakeTensor((128, 1024))],
     "input gate"),
    # a symbol read off the first tensor is still checked to be positive
    ([FakeTensor((0, 1024)), FakeTensor((0, 1024))],
     "tensor dimensions must be positive"),
])
def test_every_tensor_rejection_still_raises_before_the_launch(tensors, fragment):
    entry, recorder, _program, _torch = host_of(TASK160)

    _reject(entry, recorder, tensors + [0.7, 1.1, 5.0, -5.0], fragment)


@pytest.mark.parametrize("scalars, fragment", [
    ([True, 1.1, 5.0, -5.0], "input alpha is a scalar fp32"),
    ([0.7, "beta", 5.0, -5.0], "input beta is a scalar fp32"),
    ([0.7, 1.1, object(), -5.0], "input limit is a scalar fp32"),
    ([0.7, 1.1, 5.0, [1.0]], "input neg_limit is a scalar fp32"),
])
def test_every_scalar_rejection_names_the_scalar(scalars, fragment):
    entry, recorder, _program, _torch = host_of(TASK160)

    _reject(entry, recorder,
           [FakeTensor((128, 1024)), FakeTensor((128, 1024))] + scalars,
           fragment)


def test_a_scalar_is_converted_to_its_declared_python_type():
    entry, recorder, _program, _torch = host_of(TASK160)

    entry(*valid_task160())

    (_grid, arguments, _options) = recorder.calls[0]
    scalars = [value for value in arguments if type(value) is float]
    assert sorted(scalars) == [-5.0, 0.7, 1.1, 5.0]


def test_an_integer_scalar_is_accepted_and_converted():
    entry, recorder, _program, _torch = host_of(TASK160)

    entry(FakeTensor((128, 1024)), FakeTensor((128, 1024)),
          0, 1, 5, -5)

    (_grid, arguments, _options) = recorder.calls[0]
    assert sorted(value for value in arguments if type(value) is float) == [-5.0, 0.0, 1.0, 5.0]


# --- the shape forms that replaced the tuple comparison --------------------


def test_a_literal_declaration_is_checked_as_a_whole_shape():
    entry, recorder, program, _torch = host_of(LITERAL)
    assert "shape != (4, 5,)" in program.host

    entry(FakeTensor((4, 5), dtype=DTYPES["fp32"]))
    assert len(recorder.calls) == 1

    _reject(entry, recorder, [FakeTensor((4, 6), dtype=DTYPES["fp32"])])


def test_a_repeated_symbol_in_one_declaration_stays_a_square_check():
    entry, recorder, _program, _torch = host_of(SQUARE)

    _reject(entry, recorder, [FakeTensor((2, 3), dtype=DTYPES["fp32"])])


def test_a_later_symbolic_extent_is_checked_against_its_binding():
    entry, recorder, _program, _torch = host_of(MATMUL)

    _reject(entry, recorder,
            [FakeTensor((2, 3), dtype=DTYPES["fp16"], contiguous=True),
             FakeTensor((4, 5), dtype=DTYPES["fp16"], contiguous=True)])


def test_contiguity_is_still_required_where_the_kernel_needs_it():
    entry, recorder, _program, _torch = host_of(MATMUL)

    _reject(entry, recorder,
            [FakeTensor((2, 3), dtype=DTYPES["fp16"], contiguous=False),
             FakeTensor((3, 5), dtype=DTYPES["fp16"], contiguous=True)],
            "input a must be contiguous")


# --- the old spelling keeps its own template checks ----------------------


def test_cast_like_still_validates_its_template_tensor():
    entry, recorder, _program, _torch = host_of(CAST_LIKE)

    _reject(entry, recorder,
            [FakeTensor((2, 3), dtype=DTYPES["fp32"]),
             FakeTensor((2,), dtype=DTYPES["bf16"])],
            "input like")

    entry(FakeTensor((2, 3), dtype=DTYPES["fp32"]),
          FakeTensor((1,), dtype=DTYPES["bf16"]))
    assert len(recorder.calls) == 1
