"""CPU execution double for the emitted 32-bit / 64-bit index forms.

These are not Triton's interpreter and not device evidence. They execute the
*actual emitted source* -- the index lines, the address expressions, the masks
and the stores -- against the small standard-library double in
``test/segmented_support.py``, so the int32 and int64 forms of one program can
be compared lane by lane on the same inputs. The host's own bound check is
executed separately: it is plain Python, so it runs directly.

What this does not model: integer overflow. Both forms therefore agree here
exactly when their address expressions agree over the values the inputs can
take; overflow safety is enforced by the emitted host bound, which has its own
tests.
"""
import ast
import math
import types

from test import segmented_support


MAX_INDEX32 = 2**31 - 1


class IndexedTL(segmented_support.TL):
    """The Triton language double, extended to the value-layer calls emitted
    by the indexed/fused lowering (``minimum``/``maximum``/``sigmoid``)."""

    def minimum(self, a, b):
        return segmented_support.apply(min, a, b)

    def maximum(self, a, b):
        return segmented_support.apply(max, a, b)

    def sigmoid(self, x):
        # Split at zero, like `reference.py`'s `sigmoid`. The device does not
        # raise when `exp` saturates -- it returns inf and the reciprocal is 0 --
        # so the CPU double must not raise on a large-magnitude lane either;
        # this is what lets the saturation extremes be executed and compared.
        def evaluate(value):
            if value >= 0.0:
                return 1.0 / (1.0 + math.exp(-value))
            exponential = math.exp(value)
            return exponential / (1.0 + exponential)

        return segmented_support.apply(evaluate, x)


def _load(code):
    """Exec one emitted module against the CPU doubles."""
    tl = IndexedTL()
    triton = segmented_support.Triton(tl)
    allocations = []

    def empty(shape, *, dtype, device):
        if isinstance(shape, int):
            shape = (shape,)
        buffer = segmented_support.Buffer([None] * math.prod(shape), shape, dtype)
        allocations.append(buffer)
        return buffer

    torch = types.SimpleNamespace(
        empty=empty,
        empty_like=lambda x: empty(x.shape, dtype=x.dtype, device=x.device),
        strided='strided',
        float16='fp16', bfloat16='bf16', float32='fp32',
        int32='int32', int64='int64', bool='bool',
    )
    tree = ast.parse(code)
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name != '_flowgpu_device_key']
    namespace = {'tl': tl, 'triton': triton, 'torch': torch}
    exec(compile(ast.fix_missing_locations(tree), '<emitted-cpu-double>', 'exec'),
         namespace)
    return namespace, triton, allocations


def execute_module(code, inputs, pipeline):
    """Run one emitted module on CPU; return ({name: list}, launches)."""
    namespace, triton, allocations = _load(code)
    result = namespace['run_' + pipeline](*inputs)
    for buffer in allocations:
        assert len(buffer.writes) == len(buffer.data), \
            ('unwritten CPU allocation', buffer.shape)
    return {key: buffer.data for key, buffer in result.items()}, triton.launches


def execute_module_expect_error(code, inputs, pipeline, match):
    """Run one emitted module that must be rejected before launching.

    Returns the launch list, which has to stay empty: the host bound is what
    makes a 32-bit index admissible, so it runs before the kernel does.
    """
    namespace, triton, allocations = _load(code)
    try:
        namespace['run_' + pipeline](*inputs)
    except ValueError as error:
        assert match in str(error), error
        return triton.launches
    raise AssertionError('the emitted host accepted an unsafe index extent')


class _StridedView:
    """Only what the host bound needs from a tensor: its runtime strides."""

    def __init__(self, strides):
        self._strides = tuple(strides)

    def stride(self, axis):
        return self._strides[axis]


def index_bound_statements(host):
    """The emitted bound assignment and its rejection branch, as a source block.

    Returns ``(source, target_name)``. Exactly one block is expected: a host
    that launches one int32 kernel.
    """
    tree = ast.parse(host)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith('run_')]
    assert len(functions) == 1, [node.name for node in functions]
    body = functions[0].body
    blocks = [node for node in body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id.endswith('_fg_index_bound')]
    assert len(blocks) == 1, [ast.unparse(node) for node in blocks]
    start = body.index(blocks[0])
    following = body[start + 1]
    assert isinstance(following, ast.If), ast.unparse(following)
    source = ast.unparse(blocks[0]) + '\n' + ast.unparse(following)
    return source, blocks[0].targets[0].id


def evaluate_index_bound(host, *, dimensions, strides):
    """Execute only the emitted bound check; return the computed bound.

    ``dimensions`` are the host's dimension locals (``{'M': 2}``) and
    ``strides`` maps each named tensor of the bound expression to its runtime
    strides, so no tensor is allocated at all. Raises ValueError exactly when
    the emitted host would reject the launch.
    """
    source, target = index_bound_statements(host)
    namespace = dict(dimensions)
    namespace.update({name: _StridedView(value) for name, value in strides.items()})
    exec(source, namespace)
    return namespace[target]


#: The Task160 arithmetic (swiglu_oai_split) as the typed submission programs
#: declare it: two same-shaped strided views, four fp32 scalars, one cast.
SWIGLU = (
    "pipeline swiglu { input gate:fp16[M,N] input up:fp16[M,N] "
    "input alpha:fp32 input beta:fp32 input limit:fp32 input neg_limit:fp32 "
    "g=to_fp32(gate) u=to_fp32(up) upper=minimum(g,limit) "
    "lower=maximum(u,neg_limit) bounded=minimum(lower,limit) "
    "gated=mul(upper,sigmoid(mul(alpha,upper))) res=mul(gated,add(bounded,beta)) "
    "out=cast_to(res,fp16) output out }"
)


def swiglu_inputs(m=2, n=3, *, stride_rows=None, row_width=None):
    """Two column views of one packed row-major buffer, as the adapter builds.

    ``gate = storage[:, :n]`` and ``up = storage[:, n:2n]`` are non-contiguous
    whenever ``n`` is not ``row_width``; the fake Buffer is the packed storage
    itself, so the emitted stride metadata is the real one.
    """
    width = row_width or 2 * n
    storage = [float(row * width + column) * 0.5
               for row in range(m) for column in range(width)]
    strides = (stride_rows if stride_rows is not None else width, 1)
    return (
        segmented_support.Buffer(storage, (m, n), 'fp16', strides),
        segmented_support.Buffer(storage, (m, n), 'fp16', strides),
        0.5, 0.1, 4.0, -4.0,
    )


def int32_index_line(program):
    """The emitted program-index statement of a one-kernel program.

    A grid-capped target (Ascend) rewrites ``program_id(0)`` to its stride-loop
    tile, so the statement may read ``idx = tile.to(tl.int32) * ...`` rather
    than naming ``program_id`` directly.
    """
    assert len(program.kernels) == 1
    for line in program.kernels[0].source.splitlines():
        stripped = line.strip()
        if not stripped.startswith(('idx ', 'row ', 'tile_row ', 'tile_col ', 'd ')):
            continue
        if '= ' in stripped and ('tl.program_id(0)' in stripped
                                 or '.to(tl.int' in stripped):
            return stripped
    raise AssertionError(program.kernels[0].source)
