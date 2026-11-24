"""A reference backend: the program, in Python that runs, with no dependencies.

Design.md 13.8 names Triton as v0.1's target. This is a second target, and it
exists because of what the first one cannot claim here: the generated Triton
can be read but not run, so nothing in the repository could say whether the
kernels compute the right numbers. This backend computes them.

What it is:

* **The same decisions, made visible.** One function per fusion group (13.6);
  a fused value held in a local rather than a buffer (13.7); one list per slot
  in the host function, so two values that share a slot are literally the same
  list.
* **The same index arithmetic.** A tensor is a flat row-major list, which is
  the layout 13.8 assumed for Triton, so `x[base + j]` here and `x_ptr + base
  + offsets` there are the same statement. That is most of the point: the bug
  a backend actually has is an index, and a wrong index here runs off the end
  of a list or produces a different number.
* **The same operator semantics.** `rms_norm` normalizes with the same eps,
  `softmax` subtracts the row maximum, `matmul` accumulates along the
  reduction axis and rounds its result to the declared dtype, `silu` is
  `x * sigmoid(x)`.

What it is not, and cannot be:

* **Not a simulator of the Triton module.** There are no blocks, no masks and
  no `tl.dot` here: every kernel walks its tensor one element at a time. So
  this checks what an operator computes and where each element is addressed,
  and says nothing about tiling, vectorization, or whether the Triton text
  implements the same thing. Comparing against a device is what closes that,
  and it needs a device.
* **Not bit-exact.** A computed value is rounded to its declared dtype, which
  is what a `tl` operation does, but everything between those points is double
  precision here and fp32 on a device, and `tl.sum` is a tree while these
  loops are sequential. The numbers agree to rounding, not to the last bit.

The convention this backend adds is that a tensor is a flat row-major `list`
of numbers and the caller passes the symbolic dimensions along with the data,
because a list does not carry a shape.

The shape of the generated code follows the anchor, the same way 13.6's rule
does. Where a value is consumed elementwise it is substituted into the
consumer's expression; where a reduction needs the whole axis first, the row
is materialized as a list. So a fused value disappears in both cases, but for
the reason the fusion was allowed in the first place.
"""

from dataclasses import dataclass

from flowgpu.compiler.core.ast import InputDecl, OutputDecl
from flowgpu.compiler.shared import dimension_symbols
from flowgpu.compiler.backend.backends import (
    Argument,
    Names,
    RMS_EPS,
    Refusal,
    SkippedKernel,
    group_operands,
    group_results,
    identifier,
    is_symbol,
    kernel_kind,
    product,
    refusal_for,
    scalar_operands,
    storage_of,
    tensor_operands,
    unique_formal,
)
from flowgpu.compiler.ir.dataflow import resolve_alias


# Pointwise operators are one Python expression. `{zero}` is the dtype's zero.
# These are the same decisions as the Triton backend's table, written for a
# target that has no tiles; reading the two side by side is the cheapest way
# to see what a backend contributes that the front end does not.
POINTWISE_EXPRESSION = {
    "add": "({a} + {b})",
    # row operand mapping happens at the operand address (see the emitters);
    # the arithmetic is an add
    "add_broadcast": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "relu": "max({a}, {zero})",
    "silu": "({a} * sigmoid({a}))",
    # Python's round() is banker's rounding, the same rule torch.round uses.
    "round": "_flowgpu_round({a})",
}

# How a declared dtype rounds a computed value. A dtype with no entry is
# already what Python does: fp64 is a Python float, and int32, int64 and bool
# hold whole numbers. fp32 is exact here in the sense that nothing this
# backend produces is wider than a double -- see the module docstring on
# precision.
ROUNDING = {
    "bf16": "bf16",
    "fp16": "fp16",
    "fp32": "fp32",
    "fp8e4m3": "fp8e4m3",
    "fp8e5m2": "fp8e5",
    "int8": "int8",
    "uint8": "uint8",
    "int32": "int32",
    "int64": "int64",
}

#: Names a generated reference kernel must not shadow with a program value.
#: The kernel bodies call these builtins directly (`range`, `len`, `float`,
#: `isinstance`, ...), so an input named like one has to be alpha-renamed in the
#: kernel signature as well as in the host.  One shared set keeps the two
#: renaming points identical (review 2026-10-26).
KERNEL_RESERVED_NAMES = frozenset({
    'math', 'struct', 'float', 'int', 'len', 'isinstance', 'ValueError',
    'product', 'range', 'str', 'list', 'tuple', 'dict', 'set', 'bool', 'abs',
    'min', 'max', 'sum', 'round', 'print', 'enumerate', 'zip',
})


def _value_variable(names, value):
    """A kernel variable for a program value, never shadowing a builtin.

    The mapping is cached on the Names instance: a kernel body reads the same
    program value several times, and a second `fresh` call would hand back a
    different spelling for the same formal (the reference host then fails with
    `range_value_2 is not defined`).
    """
    cache = getattr(names, "_reference_value_names", None)
    if cache is None:
        cache = {}
        names._reference_value_names = cache
    key = identifier(value)
    if key not in cache:
        cache[key] = (names.fresh(f"{key}_value")
                      if key in KERNEL_RESERVED_NAMES else names.reserve(key))
    return cache[key]


def _formal_hint(value):
    """The hint for a scalar formal, renamed away from a builtin name."""
    name = identifier(value)
    return f"{name}_value" if name in KERNEL_RESERVED_NAMES else name


PREAMBLE = '''# Generated by FlowGPU. Design.md 13.8: FlowGPU -> Python.
#
# This one runs. It is the reference implementation the generated Triton is
# checked against, and it needs nothing but the standard library.
#
# A tensor is a flat row-major list of numbers, and the symbolic dimensions
# come from the caller, because a list does not carry a shape.
#
# Between the points where the program declares a dtype, the arithmetic here
# is double precision, so this agrees with a device to rounding rather than to
# the last bit.

import math
import struct


def sigmoid(value):
    """silu's inner function, split at zero so exp cannot overflow."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))

    exponential = math.exp(value)

    return exponential / (1.0 + exponential)


def fp32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def fp8e4m3(value):
    """Round to float8_e4m3fn: 4 exponent bits, 3 mantissa bits, max finite 448.

    Python has no fp8, so the reference rounds the fp32 value the way the format
    does: nearest, ties to even, saturating at the largest finite value (the
    `_fn` in the name means the format has no infinity to overflow into).
    """
    if value != value:
        return value
    if value == 0.0:
        return 0.0
    sign = -1.0 if value < 0.0 else 1.0
    magnitude = abs(value)
    if magnitude >= 448.0:
        return sign * 448.0
    exponent = max(-6, min(8, math.floor(math.log2(magnitude))))
    scale = 2.0 ** (3 - exponent)      # quantise the mantissa to three bits
    quantised = round(magnitude * scale) / scale
    return sign * min(quantised, 448.0)


def fp8e5(value):
    """Round to float8_e5m2: 5 exponent bits, 2 mantissa bits, max finite 57344.

    e5m2 is IEEE-like (it has infinities), so the rounding below is the format's
    own: nearest, ties to even, subnormals down to 2**-16, and overflow past
    halfway between 57344 and the next value it would encode -- which is
    infinity, so anything at or above 61440 becomes an infinity rather than
    saturating.  Values in the competition cases stay far inside the range, but
    the reference states the whole rule anyway.
    """
    if value != value:
        return value
    if value == 0.0:
        return 0.0
    negative = value < 0.0
    sign = -1.0 if negative else 1.0
    magnitude = abs(value)
    if magnitude >= 61440.0:
        return -math.inf if negative else math.inf
    if magnitude >= 57344.0:
        return sign * 57344.0
    if magnitude < 2.0 ** -16:
        return sign * (2.0 ** -16 if magnitude >= 2.0 ** -17 else 0.0)
    if magnitude < 2.0 ** -14:
        step = 2.0 ** -16                      # subnormal grid below the smallest normal
        return sign * round(magnitude / step) * step
    exponent = min(15, math.floor(math.log2(magnitude)))
    scale = 2.0 ** (2 - exponent)              # quantise the mantissa to two bits
    return sign * (round(magnitude * scale) / scale)


def int8(value):
    return (int(value) + 2**7) % 2**8 - 2**7


def uint8(value):
    return int(value) % 2**8


def int32(value):
    return (int(value) + 2**31) % 2**32 - 2**31


def int64(value):
    return (int(value) + 2**63) % 2**64 - 2**63


def bf16(value):
    """Round to bfloat16: 7 fraction bits plus the implicit leading bit, nearest even.

    Python has no bfloat16, so this rounds the fp32 bits the way the hardware
    does, which is what lets the reference agree with a kernel to the
    precision the program declares.
    """
    try:
        bits = struct.unpack("I", struct.pack("f", value))[0]
    except OverflowError:
        return math.inf if value > 0.0 else -math.inf

    bits += 0x7FFF + ((bits >> 16) & 1)

    return struct.unpack("f", struct.pack("I", bits & 0xFFFF0000))[0]


def fp16(value):
    """Round to fp16, overflowing to infinity as GPU storage does."""
    try:
        return struct.unpack("e", struct.pack("e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)
'''


PREAMBLE += '''
def _flowgpu_round(value):
    """torch.round semantics, total on the float domain.

    Python's round raises on NaN and on infinities, while torch.round (and the
    Triton floor/where lowering) return them unchanged.  The reference backend
    is the executable oracle, so it has to survive the values the device
    passes through (review P1-13, 2026-10-26).
    """
    if value != value:
        return value
    if value in (math.inf, -math.inf):
        return value
    return float(round(value))
'''
PREAMBLE += '\n\ndef ieee_div(a, b):\n    if b == 0.0:\n        if a == 0.0 or math.isnan(a): return math.nan\n        return math.copysign(math.inf, a) * math.copysign(1.0, b)\n    return a / b\n\ndef safe_exp(a, base2=False):\n    try:\n        return 2.0 ** a if base2 else math.exp(a)\n    except OverflowError:\n        return math.inf\n'

# The elementary value layer's helpers (`_sqrt`, `_maximum`, `_softplus`,
# `_gelu`, ...). They live with the operator tables rather than here, so the
# generated expression and the helper it calls are read together; appending
# them after `sigmoid` keeps `_gelu_tanh`'s neighbour defined before it.
from flowgpu.compiler.backend.value_layer import PREAMBLE as VALUE_LAYER_PREAMBLE

PREAMBLE += VALUE_LAYER_PREAMBLE


@dataclass(frozen=True)
class ReferenceKernel:
    """One generated function, which computes what one fusion group needs."""

    name: str
    kind: str
    ops: tuple
    operands: tuple
    results: tuple
    parameters: tuple
    launch: tuple
    source: str

    @property
    def signature(self) -> str:
        return next(
            line
            for line in self.source.splitlines()
            if line.startswith("def ")
        )


@dataclass(frozen=True)
class ReferenceProgram:
    """The runnable form of one pipeline: one function per kernel, plus host."""

    pipeline: str
    kernels: tuple
    refusals: tuple
    skipped: tuple
    host: str
    inputs: tuple
    outputs: tuple
    dimensions: tuple
    numeric_mode: str = "fp64"

    @property
    def complete(self) -> bool:
        return not self.refusals

    def messages(self) -> tuple:
        """Only the refusals: a DCE'd kernel is not a failure.

        `--emit reference` and `--run` used to treat a dead-code skip as a
        compile failure, so any program with an unused intermediate exited 1
        on the reference path while the Triton path (correctly) exited 0
        (review 2026-10-26).
        """
        return tuple(refusal.message() for refusal in self.refusals)


def _expression(node, operands) -> str:
    return POINTWISE_EXPRESSION[node.op].format(
        **dict(zip("ab", operands)), zero="0.0"
    )


def _moved(index, first, rest) -> str:
    """One rotation link's load map, as a generated-Python expression.

    `j = (i % first) * rest + i // first` readdresses a linear index from a
    rotated space back to the space of the tensor the rotation read.
    """
    return f"(({index}) % ({first})) * ({rest}) + (({index}) // ({first}))"


MOVED_HELPER = '''

def _moved(index, first, rest):
    """One rotation link's load map: rotated-space index -> source index."""
    return (index % first) * rest + index // first

'''


def _rounded(graph, value, expression) -> str:
    """Store a computed value in its declared dtype, as a `tl` op does.

    This is why the reference writes `bf16(...)` where the Triton backend
    writes nothing: a `tl` operation rounds because its inputs are tiles of
    that dtype, and a Python expression would not.
    """
    rounding = ROUNDING.get(graph.values[value].dtype)

    return f"{rounding}({expression})" if rounding else expression


def _comment(value, group) -> str:
    if value in group.fused_values:
        return f"{value}: fused, 13.7 keeps it in registers"

    return str(value)


def _prologue_element(graph, names, group, leaf, value) -> str:
    """One element of a value, with the prologue region that computes it.

    A value the prologue computes has no buffer of its own -- it is 13.7's
    fused value -- so this substitutes the region's expression rather than
    reading something that was never written. `leaf` says where one element of
    an external operand is, which is the part each kernel knows differently.

    A prologue ``move_first_last`` is index arithmetic, not elementwise math:
    it readdresses where the element lives (13.6), so the load map composes
    into the ADDRESS chain, wrapping the leaf's position, never the value.
    """
    produced = {node.output: node for node in group.prologue}

    def element(name, address=None):
        """The element's value; `address` overrides leaf addressing.

        Walking down a rotation chain accumulates the mapped address; at the
        external operand the accumulated address is indexed directly.
        """
        if name not in produced:
            if address is None:
                return leaf(name)
            variable = _value_variable(names, name)
            return f"{variable}[{address}]"

        node = produced[name]

        if node.op == 'move_first_last':
            source = graph.values[node.inputs[0]]
            first, rest = source.shape[0], product(source.shape[1:])
            inner = address if address is not None else None
            if inner is None:
                # The chain starts here: map the leaf's position.
                inner = element_address(name)
            mapped = f"_moved({inner}, {first}, {rest})"
            return element(node.inputs[0], mapped)

        if node.op in {'cast_like', 'cast_to', 'to_fp32'}:
            # A cast is the declared-dtype boundary itself: rounding the
            # operand at the mapped address is the whole computation.
            operand = element(node.inputs[0], address)
            return _rounded(graph, name, operand)

        arguments = [element(argument, address) for argument in node.inputs]

        return _rounded(graph, name, _expression(node, arguments))

    def element_address(name):
        """The leaf address for a rotation rooted at this value."""
        # Rotations are rooted at external operands, whose leaf addressing
        # is `variable[base + position]`; recover the address part.
        leaf_text = leaf(name)
        return leaf_text[leaf_text.index('[') + 1:-1]

    return element(value)


def _dimension_parameters(names, pairs, dimensions) -> None:
    """Add the symbolic dimensions a kernel needs, in the order it uses them.

    A dimension that is an integer (8.3) is a literal in the generated code
    and needs no parameter; folding it is what makes `bf16[2,4]` lower to a
    length of 8 instead of an expression.
    """
    for dimension in dimensions:
        if not is_symbol(dimension):
            continue
        if any(parameter == str(dimension) for parameter, _ in pairs):
            continue

        pairs.append(
            (
                names.reserve(str(dimension)),
                Argument("value", str(dimension)),
            )
        )


def _pointwise_kernel(graph, group, memory, name) -> ReferenceKernel:
    """A kernel that reads operands and writes its result, elementwise.

    Every operator here sees the same index, so there is nothing to
    materialize: each one is a line, and the fused ones are lines no other
    kernel can see.

    A rank-0 operand is a Python number in this target's signature, exactly as
    it is a scalar argument in Triton's: no list, no index, no length check.
    """
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]
    numel = product(graph.values[result].shape)

    names = Names(graph)
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    pairs = [
        (_value_variable(names, value), Argument("tensor", value))
        for value in tensors
    ]
    pairs += [
        (names.reserve(f"{identifier(value)}_out"), Argument("tensor", value))
        for value in results
    ]
    count = names.fresh("numel")
    used_formals = {count} | dimension_symbols(graph)
    used_formals.update(parameter for parameter, _ in pairs)
    scalar_formals = {}
    for value in scalars:
        formal = unique_formal(_formal_hint(value), used_formals)
        names.reserve(formal)
        scalar_formals[value] = formal
        pairs.append((formal, Argument("scalar", value)))
    pairs.append((count, Argument("value", numel)))

    index = names.fresh("i")
    parameters = ", ".join(parameter for parameter, _ in pairs)
    lines = [f"    for {index} in range({count}):"]
    computed = {}

    def operand(value) -> str:
        if value in computed:
            return computed[value]

        if value in scalar_formals:
            # The number itself: a scalar is not stored in a list, so there is
            # no element to address.
            return scalar_formals[value]

        return f"{_value_variable(names, value)}[{index}]"

    from flowgpu.compiler.backend import cse
    subexpressions = {}
    for node in group.members:
        values = [operand(value) for value in node.inputs]
        # Same in-kernel hash-consing the device emitter does: the reference
        # kernel computes one element at a time, so a repeated expression is a
        # repeated Python call, and sharing the variable keeps the two backends
        # computing the same set of values from the same operands.
        cse_key = (cse.key(node.op, values, graph.values[node.output].dtype,
                           node.part, node.attributes)
                         if cse.eligible(node.op) else None)
        if cse_key is not None and cse_key in subexpressions:
            computed[node.output] = subexpressions[cse_key]
            continue
        variable = _value_variable(names, node.output)
        computed[node.output] = variable
        if cse_key is not None:
            subexpressions[cse_key] = variable
        lines.append(
            f"        {variable} = {_rounded(graph, node.output, _expression(node, values))}"
            f"    # {_comment(node.output, group)}"
        )

    for value in results:
        lines.append(
            f"        {names.reserve(f'{identifier(value)}_out')}[{index}]"
            f" = {computed[value]}"
        )

    return _kernel(name, "pointwise", group, operands, results, pairs, lines)


def _rowwise_kernel(graph, group, memory, name) -> ReferenceKernel:
    """One row at a time, reducing along the last axis.

    A reduction cannot consume its operand elementwise -- it needs the whole
    axis before any output element is known -- so the row becomes a list here.
    That is the same reason 13.6 cannot fuse a producer *into* an rms_norm.
    """
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    # A trailing rotation chain rooted at the anchor output only readdresses
    # where the sweep stores: fold it into the store index instead of
    # materializing each rotation (13.6 store rotation). The row decomposition
    # still comes from the anchor's own input: the reduction runs in the
    # rotated space, and the chain readdresses the store afterwards.
    from flowgpu.compiler.backend.backends import store_rotation_chain
    rotation_chain = store_rotation_chain(graph, group)
    result = rotation_chain[-1].output if rotation_chain else results[0]
    input_shape = graph.values[anchor.inputs[0]].shape
    row = input_shape[-1]
    leading = input_shape[:-1]

    names = Names(graph)
    pairs = [
        (_value_variable(names, value), Argument("tensor", value))
        for value in operands
    ]
    pairs += [
        (names.reserve(f"{identifier(value)}_out"), Argument("tensor", value))
        for value in results
    ]
    rows = names.fresh("rows")
    pairs.append(
        (rows, Argument("value", product(leading) if leading else "1"))
    )
    _dimension_parameters(names, pairs, (row,))
    # A load- or store-side rotation chain composes `_moved(first, rest)`
    # over its links' source shapes, introducing dimensions the kernel's
    # own indexing never mentions; they must join the parameter list.
    from flowgpu.compiler.backend.backends import load_rotation_roots
    chain_dims = set()
    for node in group.members:
        if node.op != 'move_first_last':
            continue
        source = graph.values[node.inputs[0]].shape
        chain_dims.update(str(d) for d in source)
    _dimension_parameters(names, pairs, sorted(
        (d for d in chain_dims if is_symbol(d)), key=lambda d: str(d),
    ))

    row_length = names.reserve(str(row)) if is_symbol(row) else str(row)
    index = names.fresh("row")
    position = names.fresh("j")
    base = names.fresh("base")

    lines = [f"    for {index} in range({rows}):"]
    lines.append(f"        {base} = {index} * {row_length}")

    computed = {}
    prologue = {node.output for node in group.prologue}

    def leaf(value) -> str:
        """Where one element of an external operand is.

        A 1-D operand such as rms_norm's weight is indexed by the position
        *within the row*; indexing it by the position in the tensor would read
        past its end for every row but the first.
        """
        if len(graph.values[value].shape) == 1:
            return f"{_value_variable(names, value)}[{position}]"

        return f"{_value_variable(names, value)}[{base} + {position}]"

    def element(value) -> str:
        return _prologue_element(graph, names, group, leaf, value)

    def row_of(value) -> str:
        """The row as a list, because a reduction needs the whole axis at once.

        This is the one place the reference has to materialize something a
        pointwise consumer would not: the anchor is exactly the operator that
        cannot consume its operand elementwise, which is the same fact that
        stops 13.6 fusing a producer *into* it.
        """
        if value in computed:
            return computed[value]

        variable = (
            _value_variable(names, value)
            if value in prologue
            else names.fresh(f"{identifier(value)}_row")
        )
        computed[value] = variable
        lines.append(
            f"        {variable} = [{element(value)} for {position} "
            f"in range({row_length})]    # {_comment(value, group)}"
        )

        return variable

    source = row_of(anchor.inputs[0])

    if anchor.op in {"rms_norm", "rmsnorm_hf"}:
        total = names.fresh("total")
        mean = names.fresh("mean")
        scale = names.fresh("scale")
        lines.append(f"        {total} = 0.0")
        lines.append(f"        for {position} in range({row_length}):")
        lines.append(
            f"            {total} += {source}[{position}] * {source}[{position}]"
        )
        lines.append(f"        {mean} = {total} / {row_length}")
        lines.append(
            f"        {scale} = 1.0 / math.sqrt({mean} + {RMS_EPS})"
            "    # rms_norm: rsqrt(mean + eps)"
        )
        weight = element(anchor.inputs[1])
        normalized = f"{source}[{position}] * {scale}"
        if anchor.op == "rmsnorm_hf":
            normalized = _rounded(graph, anchor.inputs[0], normalized)
        body = f"({normalized}) * {weight}"
    else:
        largest = names.fresh("largest")
        exponentials = names.fresh("exponentials")
        total = names.fresh("total")
        lines.append(f"        {largest} = max({source})")
        lines.append(
            f"        {exponentials} = [math.exp(value - {largest}) "
            f"for value in {source}]    # softmax: exp of the row"
        )
        lines.append(f"        {total} = sum({exponentials})")
        body = f"{exponentials}[{position}] / {total}"

    variable = _value_variable(names, anchor.output)
    lines.append(
        f"        {variable} = [{_rounded(graph, anchor.output, body)} "
        f"for {position} in range({row_length})]"
        f"    # {_comment(anchor.output, group)}"
    )
    computed[anchor.output] = variable

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        if node in rotation_chain:
            continue
        arguments = [
            (
                f"{computed[value]}[{position}]"
                if value in computed
                else element(value)
            )
            for value in node.inputs
        ]
        previous = sharing.previous(node, arguments, graph.values[node.output].dtype)
        if previous is not None:
            computed[node.output] = previous
            continue
        variable = _value_variable(names, node.output)
        lines.append(
            f"        {variable} = "
            f"[{_rounded(graph, node.output, _expression(node, arguments))} "
            f"for {position} in range({row_length})]"
            f"    # {_comment(node.output, group)}"
        )
        computed[node.output] = variable
        sharing.record(node, arguments, graph.values[node.output].dtype, variable)

    # The rotation chain composes into the store index: each link readdresses
    # a linear position with the same formula the load side uses, applied in
    # chain order, so the sweep writes the rotated result directly. The value
    # stored for the rotated result is the anchor's own output. Every other
    # result keeps the plain row store the emitter always used.
    anchor_result = anchor.output
    non_rotation = [node for node in group.epilogue if node not in rotation_chain]
    if non_rotation:
        anchor_result = non_rotation[-1].output
    destination = names.reserve(f"{identifier(result)}_out")
    store_index = f"{base} + {position}"
    # The load map of one rotation sends an index of its OUTPUT to an index of
    # its input. A store needs the inverse: given the anchor's linear element
    # j, which rotated-result position receives it. Each link's inverse has
    # the closed form (j % rest)*d0 + j//rest over that link's source shape,
    # applied in chain order from the anchor outward.
    for node in rotation_chain:
        source = graph.values[node.inputs[0]]
        first = source.shape[0]
        rest = product(source.shape[1:])
        store_index = (
            f"(({store_index}) % ({rest})) * ({first})"
            f" + (({store_index}) // ({rest}))"
        )
    lines.append(f"        for {position} in range({row_length}):")
    lines.append(
        f"            {destination}[{store_index}] = "
        f"{computed[anchor_result]}[{position}]"
    )

    for value in results[1:]:
        destination = names.reserve(f"{identifier(value)}_out")
        source_value = value
        if value in computed:
            source_value = value
        lines.append(f"        for {position} in range({row_length}):")
        lines.append(
            f"            {destination}[{base} + {position}] = "
            f"{computed[source_value]}[{position}]"
        )

    return _kernel(name, "rowwise", group, operands, results, pairs, lines)


def _matmul_kernel(graph, group, memory, name) -> ReferenceKernel:
    """A dense matmul, accumulating along the reduction axis.

    Two access patterns where the pointwise model has one, which is the
    concrete reason a reduction cannot absorb just any neighbour: the
    iteration spaces differ.
    """
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]

    left, right = anchor.inputs
    rows = graph.values[left].shape[0]
    depth = graph.values[left].shape[1]
    columns = graph.values[right].shape[1]

    names = Names(graph)
    pairs = [
        (_value_variable(names, value), Argument("tensor", value))
        for value in operands
    ]
    pairs += [
        (names.reserve(f"{identifier(value)}_out"), Argument("tensor", value))
        for value in results
    ]
    _dimension_parameters(names, pairs, (rows, depth, columns))

    row_index = names.fresh("i")
    column_index = names.fresh("j")
    depth_index = names.fresh("k")
    accumulator = names.fresh("accumulator")
    rows_name = names.reserve(str(rows)) if is_symbol(rows) else str(rows)
    depth_name = names.reserve(str(depth)) if is_symbol(depth) else str(depth)
    columns_name = (
        names.reserve(str(columns)) if is_symbol(columns) else str(columns)
    )

    prologue = {node.output: node for node in group.prologue}

    def element(value, index: str) -> str:
        """One element of a matmul operand, with its prologue applied.

        An operand a prologue region computes has no buffer of its own: it is
        13.7's fused value, so this substitutes the region's expression
        instead of reading something that was never written.
        """
        if value not in prologue:
            return f"{_value_variable(names, value)}[{index}]"

        node = prologue[value]
        arguments = [element(argument, index) for argument in node.inputs]

        return _rounded(graph, value, _expression(node, arguments))

    a_index = f"{row_index} * {depth_name} + {depth_index}"
    b_index = f"{depth_index} * {columns_name} + {column_index}"

    lines = [f"    for {row_index} in range({rows_name}):"]
    lines.append(f"        for {column_index} in range({columns_name}):")
    lines.append(f"            {accumulator} = 0.0")
    lines.append(f"            for {depth_index} in range({depth_name}):")
    lines.append(
        f"                {accumulator} += {element(left, a_index)} * "
        f"{element(right, b_index)}    # fp32 accumulation"
    )

    computed = {}
    dtype = graph.values[anchor.output].dtype
    variable = _value_variable(names, anchor.output)
    computed[anchor.output] = variable
    fused = (
        ", fused by 13.6: no buffer"
        if anchor.output in group.fused_values
        else ""
    )
    lines.append(
        f"            {variable} = "
        f"{_rounded(graph, anchor.output, accumulator)}"
        f"    # {anchor.output}: fp32 accumulation, {dtype} result{fused}"
    )

    # `matmul(a, b)` followed by a bias add is the canonical epilogue, and the
    # bias is neither an operand of the matmul nor a value the group computed:
    # it is addressed like the result, which is a third access pattern in a
    # kernel that already had two.
    result_index = f"{row_index} * {columns_name} + {column_index}"

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        arguments = [
            computed[value] if value in computed else element(value, result_index)
            for value in node.inputs
        ]
        previous = sharing.previous(node, arguments, graph.values[node.output].dtype)
        if previous is not None:
            computed[node.output] = previous
            continue
        variable = _value_variable(names, node.output)
        computed[node.output] = variable
        lines.append(
            f"            {variable} = "
            f"{_rounded(graph, node.output, _expression(node, arguments))}"
            f"    # {_comment(node.output, group)}"
        )
        sharing.record(node, arguments, graph.values[node.output].dtype, variable)

    destination = names.reserve(f"{identifier(result)}_out")
    lines.append(
        f"            {destination}[{result_index}] = {computed[result]}"
    )

    return _kernel(name, "matmul", group, operands, results, pairs, lines)


def _transpose_kernel(graph, group, memory, name) -> ReferenceKernel:
    """A 2D index permutation, 13.3's own restriction on `transpose`."""
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]

    source = anchor.inputs[0]
    rows, columns = graph.values[source].shape

    names = Names(graph)
    pairs = [
        (_value_variable(names, value), Argument("tensor", value))
        for value in operands
    ]
    pairs += [
        (names.reserve(f"{identifier(value)}_out"), Argument("tensor", value))
        for value in results
    ]
    _dimension_parameters(names, pairs, (rows, columns))

    row_index = names.fresh("i")
    column_index = names.fresh("j")
    rows_name = names.reserve(str(rows)) if is_symbol(rows) else str(rows)
    columns_name = (
        names.reserve(str(columns)) if is_symbol(columns) else str(columns)
    )

    prologue = {node.output: node for node in group.prologue}

    def element(value, index: str) -> str:
        if value not in prologue:
            return f"{_value_variable(names, value)}[{index}]"

        node = prologue[value]
        arguments = [element(argument, index) for argument in node.inputs]

        return _rounded(graph, value, _expression(node, arguments))

    lines = [f"    for {row_index} in range({rows_name}):"]
    lines.append(f"        for {column_index} in range({columns_name}):")

    computed = {}
    source_index = f"{row_index} * {columns_name} + {column_index}"
    variable = _value_variable(names, anchor.output)
    computed[anchor.output] = variable
    lines.append(
        f"            {variable} = {element(source, source_index)}"
        f"    # {_comment(anchor.output, group)}"
    )

    result_index = f"{column_index} * {rows_name} + {row_index}"

    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        arguments = [
            computed[value] if value in computed else element(value, result_index)
            for value in node.inputs
        ]
        previous = sharing.previous(node, arguments, graph.values[node.output].dtype)
        if previous is not None:
            computed[node.output] = previous
            continue
        variable = _value_variable(names, node.output)
        computed[node.output] = variable
        lines.append(
            f"            {variable} = "
            f"{_rounded(graph, node.output, _expression(node, arguments))}"
            f"    # {_comment(node.output, group)}"
        )
        sharing.record(node, arguments, graph.values[node.output].dtype, variable)

    destination = names.reserve(f"{identifier(result)}_out")
    lines.append(
        f"            {destination}[{result_index}] = {computed[result]}"
    )

    return _kernel(name, "transpose", group, operands, results, pairs, lines)


def _kernel(name, kind, group, operands, results, pairs, lines) -> ReferenceKernel:
    """Assemble a kernel from its parameter pairs and body lines."""
    ops = " -> ".join(node.op for node in group.members)
    fused = ", ".join(group.fused_values)
    header = f"# {ops}"

    if fused:
        header += f"   ({fused} fused by 13.6: no buffer, 13.7)"

    parameters = ", ".join(parameter for parameter, _ in pairs)
    source = (
        f"{header}\n"
        f"def {name}({parameters}):\n"
        + "\n".join(lines)
        + "\n"
    )

    return ReferenceKernel(
        name=name,
        kind=kind,
        ops=tuple(node.op for node in group.members),
        operands=operands,
        results=results,
        parameters=tuple(parameter for parameter, _ in pairs),
        launch=pairs,
        source=source,
    )


def _rope_kernel(graph, group, memory, name) -> ReferenceKernel:
    anchor = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    result = results[0]
    shape = graph.values[anchor.inputs[0]].shape
    names = Names(graph)
    variables = {v: names.fresh(identifier(v)) for v in operands + results}
    pairs = [(variables[v], Argument("tensor", v)) for v in operands + results]
    _dimension_parameters(names, pairs, shape)
    index, partner, first = (names.fresh(n) for n in ("i", "partner", "first_half"))
    width = str(shape[-1])
    prologue = {node.output: node for node in group.prologue}

    def element(value, offset):
        if value not in prologue:
            return f"{variables[value]}[{offset}]"
        node = prologue[value]
        return _rounded(graph, value, _expression(node, [element(v, offset) for v in node.inputs]))

    lines = [f"    for {index} in range({product(shape)}):",
             f"        {first} = {index} % {width} < {width} // 2",
             f"        {partner} = {index} + {width} // 2 if {first} else {index} - {width} // 2"]
    x, cosine, sine = anchor.inputs
    rotated = f"(-({element(x, partner)}) if {first} else ({element(x, partner)}))"
    expression = f"({element(x, index)} * {element(cosine, index)} + {rotated} * {element(sine, index)})"
    computed = {anchor.output: names.fresh(identifier(anchor.output))}
    lines.append(f"        {computed[anchor.output]} = {_rounded(graph, anchor.output, expression)}")
    from flowgpu.compiler.backend import cse
    sharing = cse.Sharing()
    for node in group.epilogue:
        args = [computed[v] if v in computed else element(v, index) for v in node.inputs]
        previous = sharing.previous(node, args, graph.values[node.output].dtype)
        if previous is not None:
            computed[node.output] = previous
            continue
        variable = names.fresh(identifier(node.output))
        lines.append(f"        {variable} = {_rounded(graph, node.output, _expression(node, args))}")
        computed[node.output] = variable
        sharing.record(node, args, graph.values[node.output].dtype, variable)
    lines.append(f"        {variables[result]}[{index}] = {computed[result]}")
    return _kernel(name, "rope", group, operands, results, pairs, lines)


def _attention_kernel(graph, group, memory, name) -> ReferenceKernel:
    """Independent CPU implementation; round once after the attention region.

    One emitter for every attention the language accepts. 2D is the rank
    reduction of 4D (B=H=G=1), so the loop nest is written once for the 4D
    shape and the 2D program walks it with the outer three loops bound to
    literal 1 -- the same flat row-major linear order either way. GQA is
    the index arithmetic: query head h reads key/value head h // (H // G),
    which is exactly the repeat_interleave semantics of grouping. Causal
    is the mask bound: query position i attends to keys 0..i+(S-L), the
    kv-cache offset convention, so a 4D program may have L != S.
    """
    assert len(group.members) == 1
    anchor = group.anchor
    causal = anchor.op == "causal_attention"
    windowed = anchor.op == "window_attention"
    operands, results = group_operands(graph, group), group_results(graph, group)
    names = Names(graph)
    variables = {v: names.fresh(identifier(v)) for v in operands + results}
    pairs = [(variables[v], Argument("tensor", v)) for v in operands + results]
    q, k, v = anchor.inputs[:3]

    rank = len(graph.values[q].shape)
    if rank == 2:
        q_rows, q_depth = map(str, graph.values[q].shape)
        k_keys, k_depth = map(str, graph.values[k].shape)
        v_keys, v_width = map(str, graph.values[v].shape)
        batch = heads = groups = "1"
        length = q_rows
        _dimension_parameters(names, pairs, (q_rows, q_depth, k_keys, v_width))
    elif rank == 3:
        heads, length, q_depth = map(str, graph.values[q].shape)
        groups, k_keys, k_depth = map(str, graph.values[k].shape)
        v_groups, v_keys, v_width = map(str, graph.values[v].shape)
        batch = "1"
        _dimension_parameters(names, pairs, (heads, length, q_depth, k_keys, v_width, groups))
    else:
        batch, heads, length, q_depth = map(str, graph.values[q].shape)
        k_batch, groups, k_keys, k_depth = map(str, graph.values[k].shape)
        v_batch, v_groups, v_keys, v_width = map(str, graph.values[v].shape)
        # groups joins the parameters: the GQA divisibility check names it,
        # and head_ratio's host-side expression is written with it.
        _dimension_parameters(names, pairs, (batch, heads, length, q_depth, k_keys, v_width, groups))

    qv, kv, vv = (variables[x] for x in anchor.inputs[:3])
    out = variables[anchor.output]
    b, h, i, j, d, c = [names.fresh(x) for x in ("b", "h", "i", "j", "d", "c")]
    scores, largest, weights, total = [names.fresh(x) for x in
        ("scores", "largest", "weights", "total")]

    # 2D is the 4D shape with the outer three loops bound to literal 1, so
    # the index expressions drop the terms the loops would contribute.
    is_4d = rank == 4
    is_3d = rank == 3
    # GQA head mapping: query head h belongs to key/value group h // ratio.
    # A literal ratio folds to an integer; a symbolic one is a parameter the
    # host passes as the expression itself, and its divisibility is checked
    # at run time because the checker cannot divide two names.
    if heads == groups:
        group_of_head = f"{h}" if (is_4d or is_3d) else ""
        ratio = "1"
    else:
        ratio = f"({heads} // {groups})" if heads.isdigit() and groups.isdigit() else "head_ratio"
        if ratio == "head_ratio":
            pairs.append((names.reserve("head_ratio"), Argument("value", f"({heads} // {groups})")))
        group_of_head = f"({h} // {ratio})"

    # Offsets into the flat row-major tensors: only the axes that have a
    # loop contribute a term, so the 2D program indexes like the old 2D
    # kernel did and the 4D one adds the batch/head strides in front.
    q_base = (f"{b} * {heads} * {length} * {q_depth} + " if is_4d else "") + \
             (f"{h} * {length} * {q_depth} + " if (is_4d or is_3d) else "") + f"{i} * {q_depth}"
    kv_base = (f"{b} * {groups} * {k_keys} * {k_depth} + " if is_4d else "") + \
              (f"{group_of_head} * {k_keys} * {k_depth} + " if group_of_head else "") + \
              f"{j} * {k_depth}"
    v_base = (f"{b} * {groups} * {v_keys} * {v_width} + " if is_4d else "") + \
             (f"{group_of_head} * {v_keys} * {v_width} + " if group_of_head else "") + \
             f"{j} * {v_width}"
    out_base = (f"{b} * {heads} * {length} * {v_width} + " if is_4d else "") + \
               (f"{h} * {length} * {v_width} + " if (is_4d or is_3d) else "") + \
               f"{i} * {v_width}"

    if causal:
        # The mask bound with the kv-cache offset: query i sees keys up to
        # i + (S - L) inclusive. With L == S this is the plain triangle; with
        # a cache it is the S - L prefix. S < L has no causal meaning, and
        # the run-time check says so instead of computing nonsense.
        window = f"{i} + {k_keys} - {length} + 1"
    elif windowed:
        window = f"{variables[anchor.inputs[3]]}[{b}]" if rank == 4 else f"{variables[anchor.inputs[3]]}[0]"
    else:
        window = k_keys

    lines = []
    if is_4d:
        lines.append(f"    for {b} in range({batch}):")
        lines.append(f"        for {h} in range({heads}):")
        indent = "            "
    elif is_3d:
        lines.append(f"    for {h} in range({heads}):")
        indent = "        "
    else:
        indent = "    "

    checks = []
    if heads != groups and not (heads.isdigit() and groups.isdigit()):
        checks += [f"    if {heads} % {groups}:",
                   f"        raise ValueError('GQA: the key head count must divide the query head count')"]
    if causal:
        checks += [f"    if {k_keys} < {length}:",
                   f"        raise ValueError('causal attention requires the key length to be at least the query length')"]
    lines = checks + lines

    inner = [
        f"{indent}for {i} in range({length}):",
        f"{indent}    {scores} = [",
        f"{indent}        sum({qv}[{q_base} + {d}] * {kv}[{kv_base} + {d}] for {d} in range({q_depth})) / math.sqrt({q_depth})",
        f"{indent}        for {j} in range({window})]",
        f"{indent}    {largest} = max({scores})",
        f"{indent}    {weights} = [math.exp(s - {largest}) for s in {scores}]",
        f"{indent}    {total} = sum({weights})",
        f"{indent}    for {c} in range({v_width}):",
        f"{indent}        {out}[{out_base} + {c}] = {_rounded(graph, anchor.output, f'sum({weights}[{j}] * {vv}[{v_base} + {c}] for {j} in range({window})) / {total}')}",
    ]
    lines += inner

    return _kernel(name, "attention", group, operands, results, pairs, lines)


def _layout_kernel(graph, group, memory, name) -> ReferenceKernel:
    """A layout operator on flat row-major lists.

    The reference's tensors are flat lists in row-major order, which is the
    same linear order a reshape reinterprets: reshape is an element-count
    check and nothing else here. split and concat remap indices within each
    row, so the loops carry the leading dimensions' product; the widths are
    literals because the language layer refuses a symbolic last axis there.
    """
    anchor = group.anchor
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    result = results[0]
    names = Names(graph)
    pairs = [
        (_value_variable(names, value), Argument("tensor", value))
        for value in operands
    ]
    pairs += [
        (names.reserve(f"{identifier(value)}_out"), Argument("tensor", value))
        for value in results
    ]

    source = pairs[0][0]
    destination = pairs[-1][0]
    leading = graph.values[anchor.output].shape[:-1]

    if anchor.op == "reshape":
        # The runtime half of the conservation contract, the same check the
        # Triton host emits: literal-only products were proven at the type
        # layer, a symbolic one is only decidable with the value in hand.
        expected = product(graph.values[anchor.output].shape)
        _dimension_parameters(names, pairs, graph.values[anchor.output].shape)
        lines = [
            f"    if len({source}) != ({expected}):",
            f"        raise ValueError('reshape does not conserve elements')",
            f"    {destination}[:] = {source}",
        ]
        return _kernel(name, "layout", group, operands, results, pairs, lines)

    # split / concat: one row at a time, the last axis inside it.
    rows = names.fresh("rows")
    pairs.append((rows, Argument("value", product(leading) if leading else "1")))
    row, column = names.fresh("row"), names.fresh("column")
    _dimension_parameters(names, pairs, ())

    if anchor.op == "split":
        width = int(graph.values[anchor.output].shape[-1])
        length = int(graph.values[anchor.inputs[0]].shape[-1])
        low = width * anchor.part
        lines = [
            f"    for {row} in range({rows}):",
            f"        for {column} in range({width}):",
            f"            {destination}[{row} * {width} + {column}] = "
            f"{source}[{row} * {length} + {low} + {column}]",
        ]
    elif anchor.op == "interleave":
        a, b = (pair[0] for pair in pairs[:2])
        width = int(graph.values[anchor.inputs[0]].shape[-1])
        total = int(graph.values[anchor.output].shape[-1])
        lines = [
            f"    for {row} in range({rows}):",
            f"        for {column} in range({width}):",
            f"            {destination}[{row} * {total} + 2 * {column}] = "
            f"{a}[{row} * {width} + {column}]",
            f"            {destination}[{row} * {total} + 2 * {column} + 1] = "
            f"{b}[{row} * {width} + {column}]",
        ]
    else:
        a, b = (pair[0] for pair in pairs[:2])
        first = int(graph.values[anchor.inputs[0]].shape[-1])
        second = int(graph.values[anchor.inputs[1]].shape[-1])
        total = int(graph.values[anchor.output].shape[-1])
        # b indexes with ITS OWN last-axis width: after the join the row is
        # `total` wide, but b's rows are only `second` wide, and using the
        # joined stride reads past b's end on every row but the first.
        lines = [
            f"    for {row} in range({rows}):",
            f"        for {column} in range({total}):",
            f"            {destination}[{row} * {total} + {column}] = "
            f"({a}[{row} * {first} + {column}] if {column} < {first} "
            f"else {b}[{row} * {second} + {column} - {first}])",
        ]
    return _kernel(name, "layout", group, operands, results, pairs, lines)


from flowgpu.compiler.backend.primitives import reference_kernel as _primitive_kernel
from flowgpu.compiler.backend.cache_store_backend import reference_kernel as _cache_store_reference_kernel
from flowgpu.compiler.backend.cache_write_backend import cache_write_reference_kernel as _cache_write_reference_kernel
from flowgpu.compiler.backend.cache_write_hsd_backend import cache_write_hsd_reference_kernel as _cache_write_hsd_reference_kernel
from flowgpu.compiler.backend.cache_write_batch_backend import cache_write_batch_reference_kernel as _cache_write_batch_reference_kernel
from flowgpu.compiler.backend.cache_write_range_backend import cache_write_range_reference_kernel as _cache_write_range_reference_kernel
from flowgpu.compiler.backend.pad_cat_backend import reference_kernel as _pad_cat_reference_kernel
from flowgpu.compiler.backend.scan_backend import reference_kernel as _scan_reference_kernel
from flowgpu.compiler.backend.patchify_backend import reference_kernel as _patchify_reference_kernel
from flowgpu.compiler.backend.segmented_matmul_backend import reference_kernel as _segmented_reference_kernel
from flowgpu.compiler.backend.scatter_backend import reference_kernel as _scatter_reference_kernel
from flowgpu.compiler.backend.store_at_backend import reference_kernel as _store_at_reference_kernel
from flowgpu.compiler.backend.segment_rank_backend import reference_kernel as _segment_rank_reference_kernel
from flowgpu.compiler.backend.recurrence_backend import reference_kernel as _recurrence_reference_kernel
from flowgpu.compiler.backend.sort_backend import sort_reference_kernel as _sort_reference_kernel
from flowgpu.compiler.backend.mean_backend import mean_reference_kernel as _mean_reference_kernel
from flowgpu.compiler.backend.layer_norm_backend import layer_norm_reference_kernel as _layer_norm_reference_kernel
from flowgpu.compiler.backend.conv2d_backend import conv2d_reference_kernel as _conv2d_reference_kernel
from flowgpu.compiler.backend.conv2d_same_backend import conv2d_same_reference_kernel as _conv2d_same_reference_kernel
from flowgpu.compiler.backend.conv2d_stride2_backend import conv2d_stride2_reference_kernel as _conv2d_stride2_reference_kernel
from flowgpu.compiler.backend.conv2d_stride2_backend import conv2d_stride2_same_reference_kernel as _conv2d_stride2_same_reference_kernel
from flowgpu.compiler.backend.conv2d_ex_backend import conv2d_ex_reference_kernel as _conv2d_ex_reference_kernel
from flowgpu.compiler.backend.topk_backend import (
    topk_reference_kernel as _topk_reference_kernel,
    gather_reference_kernel as _gather_last_reference_kernel)
from flowgpu.compiler.backend.reduce_with_index_backend import (
    reference_kernel as _reduce_with_index_reference_kernel)

def _linear_decode_reference_kernel(graph, group, memory, name):
    operands = group_operands(graph, group); results = group_results(graph, group)
    q, k, v, cache, slope, slots = operands; out, cache_out = results
    names = Names(graph)
    variables = {x: _value_variable(names, x) for x in operands + results}
    pairs = [(variables[x], Argument('tensor', x)) for x in operands + results]
    b, h, d = (graph.values[q].shape[i] for i in (0, 1, 3))
    _dimension_parameters(names, pairs, [b, h, d])
    # `out[:] = 0` raises "can only assign an iterable"; zero it element by
    # element instead (review P1-2/P1-11, 2026-10-26).  The cache output is a
    # plain slice copy, which Python accepts.
    zero_index = names.fresh("i")
    lines = [f'    for {zero_index} in range(len({variables[out]})):',
             f'        {variables[out]}[{zero_index}] = 0.0',
             f'    {variables[cache_out]}[:] = {variables[cache]}[:]']
    lines += [f'    for bi in range({b}):', f'        slot = {variables[slots]}[bi]',
             f'        if slot < 0 or slot >= len({variables[cache]}) // ({h} * {d} * {d}):',
             '            continue', f'        for hi in range({h}):', f'            decay = math.exp(-float({variables[slope]}[hi]))', f'            for i in range({d}):', f'                for j in range({d}):', f'                    old = {variables[cache]}[slot * ({h}) * ({d}) * ({d}) + hi * ({d}) * ({d}) + i * ({d}) + j]', f'                    kval = {variables[k]}[bi * ({h}) * ({d}) + hi * ({d}) + i]', f'                    vval = {variables[v]}[bi * ({h}) * ({d}) + hi * ({d}) + j]', f'                    state = float(kval) * float(vval) + decay * float(old)', f'                    {variables[cache_out]}[slot * ({h}) * ({d}) * ({d}) + hi * ({d}) * ({d}) + i * ({d}) + j] = state', f'                    {variables[out]}[bi * ({h}) * ({d}) + hi * ({d}) + j] += float({variables[q]}[bi * ({h}) * ({d}) + hi * ({d}) + i]) * state']
    return _kernel(name, 'linear_decode', group, operands, results, pairs, lines)

EMITTERS = {
    "primitive": _primitive_kernel,
    "pointwise": _pointwise_kernel,
    "rowwise": _rowwise_kernel,
    "matmul": _matmul_kernel,
    "transpose": _transpose_kernel,
    "rope": _rope_kernel,
    "attention": _attention_kernel,
    "layout": _layout_kernel,
    "cache_store": _cache_store_reference_kernel,
    "cache_write": _cache_write_reference_kernel,
    "cache_write_hsd": _cache_write_hsd_reference_kernel,
    "cache_write_hsd_batch": _cache_write_batch_reference_kernel,
    "cache_write_hsd_range": _cache_write_range_reference_kernel,
    "linear_decode": _linear_decode_reference_kernel,
    "pad_cat": _pad_cat_reference_kernel,
    "scan": _scan_reference_kernel,
    "patchify": _patchify_reference_kernel,
    "segmented_matmul": _segmented_reference_kernel,
    "scatter_by_index": _scatter_reference_kernel,
    "store_at": _store_at_reference_kernel,
    "topk": _topk_reference_kernel,
    "sort": _sort_reference_kernel,
    "mean": _mean_reference_kernel,
    "layer_norm": _layer_norm_reference_kernel,
    "conv2d": _conv2d_reference_kernel,
    "conv2d_same": _conv2d_same_reference_kernel,
    "conv2d_stride2": _conv2d_stride2_reference_kernel,
    "conv2d_stride2_same": _conv2d_stride2_same_reference_kernel,
    "conv2d_ex": _conv2d_ex_reference_kernel,
    "axis_gather": _gather_last_reference_kernel,
    "reduce_with_index": _reduce_with_index_reference_kernel,
    "index_table": _segment_rank_reference_kernel,
    "recurrence": _recurrence_reference_kernel,
}


def lower(pipeline, graph, fusion, memory, *, numeric_mode="fp64") -> ReferenceProgram:
    """Emit one runnable function per fusion group, plus the host code."""
    graph.validate()
    if numeric_mode not in {"fp64", "fp32"}:
        raise ValueError("numeric_mode must be 'fp64' or 'fp32'")
    kernels = []
    refusals = []
    skipped = []

    for index, group in enumerate(fusion.groups):
        name = f"{pipeline.name}_k{index}"

        refusal = refusal_for(
            graph, group, POINTWISE_EXPRESSION, "the reference backend"
        )

        if refusal is not None:
            refusals.append(refusal)
            continue

        program_scalars = [node for node in group.members
                           if getattr(node.attributes, "program_scalar", None)
                           is not None]
        if program_scalars:
            # A program-uniform scalar is defined by the *launch mapping* (which
            # program computes which element). The reference backend walks one
            # element at a time and has no program domain to define it against,
            # so it says so instead of inventing one. The numeric oracle for
            # this value category is the CPU Triton double, which executes the
            # actual emitted source with a real grid.
            refusals.append(Refusal(
                group.anchor.op if group.anchor is not None else "program_scalar",
                tuple(group_results(graph, group)),
                "a program-uniform scalar is defined by the launch mapping; the "
                "reference backend has no program domain and does not invent "
                "one (execute the emitted kernel on the CPU Triton double)",
                "the reference backend"))
            continue

        results = group_results(graph, group)
        dead = [
            value
            for value in results
            if storage_of(memory, value) is None
        ]
        # All-dead groups are dead code and are eliminated before the refusal
        # classification (review 2026-10-26), exactly as in the Triton lowering.
        if dead and len(dead) == len(results):
            skipped.append(
                SkippedKernel(
                    name=name,
                    values=tuple(dead),
                    reason="13.7 gives a value nobody reads no buffer, so "
                    "there is nowhere to write it",
                )
            )
            continue

        # Same rule as the Triton backend: a multi-result operator whose emitters
        # store each live result keeps its kernel when only some results are
        # dead; anything else refuses rather than skipping a kernel that a live
        # result depends on (review P0-1, 2026-10-26).
        from flowgpu.compiler.backend.triton_backend import RESULT_OPTIONAL_OPS
        if dead:
            result_optional = (group.anchor is not None
                               and group.anchor.op in RESULT_OPTIONAL_OPS)
            if not result_optional:
                refusals.append(Refusal(
                    group.anchor.op if group.anchor is not None else 'pointwise',
                    tuple(results),
                    "only some results of this operator are used, and this "
                    "lowering cannot emit a subset",
                    "the reference backend"))
                continue

        kind = kernel_kind(group.anchor)
        if group.indexed and kind in {"pointwise", "unknown"}:
            kind = "primitive"
        scalar_leaves = scalar_operands(graph, group_operands(graph, group))
        if scalar_leaves and kind not in {"pointwise", "primitive",
                                          "segmented_matmul", "layer_norm"}:
            refusals.append(Refusal(
                group.anchor.op if group.anchor else 'pointwise',
                tuple(group_results(graph, group)),
                'a scalar input is only supported by the pointwise, '
                'primitive and segmented_lora_matmul reference lowerings; '
                'this kernel kind would index the number as a tensor',
                'the reference backend'))
            continue

        emitter = EMITTERS[kind]
        kernel = emitter(graph, group, memory, name)
        if not kernel.source.strip():
            refusals.append(Refusal(
                group.anchor.op if group.anchor else 'pointwise',
                tuple(group_results(graph, group)),
                'reference emitter has no executable numerical implementation',
                'the reference backend'))
            continue
        kernels.append(kernel)

    return ReferenceProgram(
        pipeline=pipeline.name,
        kernels=tuple(kernels),
        refusals=tuple(refusals),
        skipped=tuple(skipped),
        host=_host(pipeline, graph, memory, kernels),
        inputs=tuple(
            (statement.name, statement.type)
            for statement in pipeline.statements
            if isinstance(statement, InputDecl)
        ),
        # `output y` names a value; its type is the graph's, since a name is
        # not what holds the data (`y = x` emits no node, 13.4).
        outputs=tuple(
            (
                statement.name,
                graph.values[resolve_alias(graph.aliases, statement.name)],
            )
            for statement in pipeline.statements
            if isinstance(statement, OutputDecl)
        ),
        dimensions=_symbols(graph),
        numeric_mode=numeric_mode,
    )


def _symbols(graph) -> tuple:
    """The symbolic dimensions (8.3) a caller has to supply, in order."""
    found = []

    for value in graph.values.values():
        for dimension in value.shape:
            if is_symbol(dimension) and str(dimension) not in found:
                found.append(str(dimension))

    return tuple(found)


def _host(pipeline, graph, memory, kernels) -> str:
    """The launch code: one list per slot, one call per kernel."""
    name = pipeline.name

    declared = [
        statement
        for statement in pipeline.statements
        if isinstance(statement, InputDecl)
    ]
    outputs = [
        statement.name
        for statement in pipeline.statements
        if isinstance(statement, OutputDecl)
    ]

    names = Names(graph)
    # Names the host body itself uses.  An input with one of these names would
    # shadow it (`def run_p(float)` breaks the slot lists below, `torch`
    # shadows a module), so the parameter yields -- the launch arguments are
    # written with the parameter mapping, not with the source name.
    host_reserved = {'math', 'struct', 'float', 'int', 'len', 'isinstance',
                     'ValueError', 'product', 'range', 'str', 'list', 'tuple'}
    # An input whose name is also a dimension symbol yields to the dimension:
    # the launch arguments are written with the dimension variables, so the
    # parameter is renamed (the same rule the Triton host uses).  Without it the
    # generated signature was `def run_p(x, N, N)` -- a SyntaxError.
    dimension_symbols = {
        str(dimension)
        for value in graph.values.values()
        for dimension in value.shape
        if is_symbol(dimension)
    }
    parameters = {
        statement.name: (
            names.fresh(identifier(statement.name))
            if identifier(statement.name) in host_reserved | dimension_symbols
            else names.reserve(identifier(statement.name))
        )
        for statement in declared
    }
    slots = {
        f"slot{slot.index}": names.fresh(f"slot{slot.index}")
        for slot in memory.slots
    }

    lines = []
    for dimension in _symbols(graph):
        lines += [f"    if not isinstance({dimension}, int) or isinstance({dimension}, bool) or {dimension} <= 0:",
                  f"        raise ValueError('dimension {dimension} must be a positive integer')"]
    # A rank-0 declaration is a host number in this target too: not a list, so
    # it has no element count and no slot. Only tensors carry the length guard.
    scalar_declared = [
        statement for statement in declared if statement.type.is_scalar
    ]
    tensor_declared = [
        statement for statement in declared if not statement.type.is_scalar
    ]
    for statement in tensor_declared:
        variable = parameters[statement.name]
        expected = product(statement.type.shape)
        lines += [f"    if len({variable}) != ({expected}):",
                  f"        raise ValueError('input {statement.name} has the wrong number of elements')"]
    for statement in scalar_declared:
        variable = parameters[statement.name]
        dtype = statement.type.dtype
        if dtype in {"fp16", "bf16", "fp32"}:
            lines += [
                f"    if isinstance({variable}, bool) or not isinstance({variable}, (int, float)):",
                f"        raise ValueError('input {statement.name} is a scalar {dtype}; "
                f"pass a Python int or float, not a tensor')",
                f"    {variable} = float({variable})",
            ]
        else:
            lines += [
                f"    if type({variable}) is not int:",
                f"        raise ValueError('input {statement.name} is a scalar {dtype}; "
                f"pass a Python int, not a tensor')",
                f"    {variable} = int({variable})",
            ]
    for node in graph.nodes:
        if node.op == "rope":
            width = str(graph.values[node.inputs[0]].shape[-1])
            lines += [f"    if {width} % 2:", "        raise ValueError('rope requires an even last dimension')"]

    # 13.7's slots become lists. Two values that share a slot share this list,
    # so a wrong reuse decision shows up as a wrong number instead of staying
    # invisible. A cell nobody wrote stays NaN, so reading one is loud.
    for slot in memory.slots:
        buffer = next(
            entry for entry in memory.buffers if entry.value == slot.values[0]
        )
        size = product(buffer.tensor_type.shape)
        lines.append(
            f"    {slots[f'slot{slot.index}']} = [float(\"nan\")] * ({size})"
            f"    # {', '.join(slot.values)}"
        )

    def variable(value) -> str:
        where = resolve_alias(graph.aliases, value)

        if storage_of(memory, where) == "input":
            return parameters.get(where, identifier(where))

        for slot in memory.slots:
            if where in slot.values:
                return slots[f"slot{slot.index}"]

        raise AssertionError(f"{value!r} has no storage to name")

    for kernel in kernels:
        arguments = [
            variable(argument.text)
            if argument.kind in {"tensor", "scalar"}
            else argument.text
            for _, argument in kernel.launch
        ]
        index = kernel.name.rsplit("_k", 1)[-1]
        lines.append(f"\n    # k{index}: {' -> '.join(kernel.ops)}")
        lines.append(f"    {kernel.name}({', '.join(arguments)})")

    if not lines:
        lines.append("    pass")

    returned = ", ".join(f'"{value}": {variable(value)}' for value in outputs)
    signature = ", ".join(
        [parameters[statement.name] for statement in declared]
        + list(_symbols(graph))
    )

    return (
        f"def run_{name}({signature}):\n"
        f'    """{name}: run the kernels of the pipeline, in order.\n'
        "\n"
        "    The tensors are flat row-major lists, and the dimensions follow\n"
        "    them because a list does not carry a shape.\n"
        '    """\n'
        + "\n".join(lines)
        + f"\n\n    return {{{returned}}}\n"
    )


def module(programs, *, numeric_mode=None, emulate_fp32=False) -> str:
    """The whole generated file: the runtime helpers, then the pipelines."""
    if numeric_mode is None:
        numeric_mode = "fp32" if emulate_fp32 else (programs[0].numeric_mode if programs else "fp64")
    if numeric_mode not in {"fp64", "fp32"}:
        raise ValueError("numeric_mode must be 'fp64' or 'fp32'")
    parts = [PREAMBLE, MOVED_HELPER]
    if numeric_mode == "fp32":
        from flowgpu.compiler.backend.fp32_reference import PREAMBLE as FP32_PREAMBLE
        parts.append(FP32_PREAMBLE)

    for program in programs:
        for kernel in program.kernels:
            parts.append("\n" + kernel.source)

        host = program.host
        if numeric_mode == "fp32":
            import ast
            function = ast.parse(host).body[0]
            conversions = []
            for arg, (_, tensor_type) in zip(function.args.args, program.inputs):
                dtype = tensor_type.dtype
                if dtype not in {'fp16', 'bf16', 'fp32'}:
                    continue
                if tensor_type.is_scalar:
                    # A scalar is one number, not a list: promote the Python
                    # value to the declared precision directly.
                    conversions.append(f'    {arg.arg} = {dtype}({arg.arg})')
                else:
                    conversions.append(
                        f'    {arg.arg} = [{dtype}(v) for v in {arg.arg}]'
                    )
            lines = host.splitlines()
            lines[1:1] = conversions
            host = '\n'.join(lines) + '\n'
        parts.append("\n" + host)

    return "".join(parts)


# Data for `run`. Quarters with both signs: exact in every float dtype the
# language has, small enough that a matmul of them stays where it started, and
# negative half the time so relu and softmax have something to do.
def _data(dtype, count):
    if dtype == "bool":
        return [index % 2 == 0 for index in range(count)]

    if dtype in ("int32", "int64"):
        return [(index * 7 + 3) % 11 - 5 for index in range(count)]

    return [((index * 7 + 3) % 11 - 5) / 4 for index in range(count)]


def _size_of(shape, sizes) -> int:
    total = 1

    for dimension in shape:
        total *= sizes[dimension] if is_symbol(dimension) else int(dimension)

    return total


def inputs_for(program, sizes) -> list:
    """The data this program is run on, as `(name, shape, dtype, values)`.

    Exposed because more than one target wants to run the same program: a
    caller running the generated Triton on a device needs exactly these
    numbers, or the two results cannot be compared.

    A scalar input is a number, not a list: its shape is empty and its value is
    the one number it contributes, so `run` can pass it straight to the host
    entry and a device harness can pass the same number as a Triton scalar.
    """
    inputs = []
    for name, tensor_type in program.inputs:
        if tensor_type.is_scalar:
            inputs.append((name, [], tensor_type.dtype,
                           _data(tensor_type.dtype, 1)[0]))
            continue
        inputs.append(
            (
                name,
                [
                    sizes[dimension] if is_symbol(dimension) else int(dimension)
                    for dimension in tensor_type.shape
                ],
                tensor_type.dtype,
                _data(tensor_type.dtype, _size_of(tensor_type.shape, sizes)),
            )
        )

    return inputs


def run(program, sizes=None, size=2, *, numeric_mode=None, emulate_fp32=False) -> dict:
    """Generate this program, run it, and return what it produced.

    This is the only piece of the compiler that executes anything, and what it
    executes is the reference backend rather than the Triton one: the text
    being run is Python this compiler just wrote. Every symbolic dimension
    takes the same value `size` unless `sizes` names them individually, and the
    inputs are a fixed function of the dimensions, so the result is
    reproducible.
    """
    if sizes is None:
        sizes = {dimension: size for dimension in program.dimensions}

    namespace = {}
    if numeric_mode is not None and numeric_mode not in {"fp64", "fp32"}:
        raise ValueError("numeric_mode must be 'fp64' or 'fp32'")
    if emulate_fp32 and numeric_mode is not None and numeric_mode != "fp32":
        raise ValueError("emulate_fp32 conflicts with numeric_mode='fp64'")
    mode = ("fp32" if emulate_fp32 else numeric_mode)
    mode = mode or getattr(program, "numeric_mode", "fp64")
    exec(compile(module([program], numeric_mode=mode), "<reference>", "exec"), namespace)

    tensors = [
        values for _, _, _, values in inputs_for(program, sizes)
    ]
    host = namespace[f"run_{program.pipeline}"]

    return host(*tensors, *[sizes[dimension] for dimension in program.dimensions])
