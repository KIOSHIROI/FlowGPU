"""Operator semantics: arity, type rules and fusion classifications.

Phase 1b moved this registry out of ``flowgpu.compiler.frontend.operator``.  It is
frontend-neutral: rules take ``TensorType``/literal values and raise
``RuleError``; they never see source text.  The frontend module of the same name
re-exports it for compatibility.

The module also owns the operator-name vocabulary used by ``DataflowGraph``:
registering an operator registers its name, so core IR validation needs no
frontend import.
"""
_KNOWN: set[str] = set()


def register_operator_name(name: str) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("operator name must be a non-empty string")
    _KNOWN.add(name)


def is_known_operator(name: str) -> bool:
    return name in _KNOWN


def known_operator_names() -> frozenset[str]:
    return frozenset(_KNOWN)


def clear_operator_names() -> None:
    """Test helper; applications register operators once at import."""
    _KNOWN.clear()


from dataclasses import dataclass
from typing import Callable

from flowgpu.compiler.core.literals import DTypeLiteral, ShapeLiteral
from flowgpu.compiler.core.types import TensorType, format_shape
from flowgpu.compiler.core.diagnostic import DiagnosticKind
from flowgpu.compiler.core.dtype_rules import (dtype_class, require_arithmetic,
                                       require_numeric, RuleError)


@dataclass
class OperatorSpec:
    """Everything the compiler knows about one operator.

    `pointwise` means same-index arithmetic, as used by add/sub/mul,
    silu/relu. RoPE reads a partner from another column and is a separate
    anchor. It is not a scalar expression that can attach to every neighbor.

    It is a required argument rather than a default, so that registering an
    operator forces the decision. A default would let a new operator be
    silently treated as non-pointwise, which reads as "there is no fusion
    here" rather than as "nobody said". `None` means there is nothing to
    place, which is true only of a reserved operator: it has no computation to
    fuse, because it is refused before it can reach a graph.

    `multi` marks an operator that produces several values, so its call can
    only appear under a multi-name assignment; the number of names is the
    number of results, which is the language's only arity selector for it.
    """

    name: str
    arity: int
    infer: Callable
    pointwise: bool | None
    acc: str | None
    multi: bool = False


OPERATORS = {}


def register_operator(name, arity, infer, *, pointwise, acc, multi=False):
    OPERATORS[name] = OperatorSpec(
        name=name,
        arity=arity,
        infer=infer,
        pointwise=pointwise,
        acc=acc,
        multi=multi,
    )
    register_operator_name(name)
    # Phase 1a: the frontend-neutral IR validates op names from core; the DSL
    # is the first frontend to populate that vocabulary.
    register_operator_name(name)


def reject(kind, message):
    """A type rule refusing its arguments.

    The rule states the category, because the rule is what knows whether the
    dtype or the shape was wrong; it says nothing about *where* the call was
    written, because it never sees the program. See diagnostic.py.
    """
    return RuleError(kind, message)


def require_same_dtype(x, y, verb):
    """One rule, used by every elementwise operator.

    add, sub and mul had three byte-identical copies of this check and the
    one below; the only difference between them was the verb in the message.
    """
    require_arithmetic(x, verb)
    require_arithmetic(y, verb)

    if x.dtype != y.dtype:
        raise reject(
            DiagnosticKind.TYPE,
            f"Cannot {verb} {x.dtype} and {y.dtype}",
        )


def require_same_shape(x, y, verb):
    if x.shape != y.shape:
        raise reject(
            DiagnosticKind.SHAPE,
            f"Cannot {verb} tensors with shapes "
            f"{format_shape(x.shape)} and {format_shape(y.shape)}",
        )


# Operators where a rank-0 declared input is a legal operand beside a tensor.
# A scalar has no shape to compare against the tile's, so the two operand
# checks above do not apply to it: the device converts it to the tile's dtype,
# exactly as it types a Python float written in the expression (`bf16_tile *
# 2.0` is bf16 in both Triton and PyTorch). The declaration still governs the
# host-side value -- `_host` validates it and converts it to the declared
# dtype -- it just does not widen the tile's arithmetic. Two scalars have no
# tile to broadcast over: that result would itself be a scalar, and a pipeline
# value is a tensor, so it is refused by name rather than emitted.
#
# `maximum`/`minimum` are here for the same reason as add/sub/mul even though
# they are non-pointwise anchors: `clamp(x, limit)` is an elementwise
# comparison against a scalar, which is exactly the shape a task parameter
# takes. `abs` is unary and is not: `abs(scalar)` would be a scalar value.
SCALAR_BROADCAST_OPERATORS = frozenset(
    {"add", "sub", "mul", "div", "maximum", "minimum",
     # Integer masks and shift amounts are exactly this shape: `bits & 0x7FFFFF`
     # and `bits >> 23` are a tile against a number.  The integer class rule
     # keeps a float scalar out of a bit pattern.
     "bitand", "bitor", "bitxor", "shl", "shr"}
)

# `div` is in the set because a scalar divisor is the same elementwise mix as a
# scalar multiplier, and normalisation by a row extent is the case that needs it:
# `dynamic_quant`'s scale is `max(|x|) / 127` and the quantiser then divides by
# that scale.  Leaving it out forced a reciprocal the language cannot compute.

# Operators whose *type rule* explicitly takes a rank-0 operand as a value
# rather than as a tile: the rule decides what the scalar means, so the generic
# "a scalar can only broadcast over elementwise arithmetic" gate must not fire
# first. `segmented_lora_matmul`'s `heads` is the head count that flattens
# (token, head) rows, i.e. an integer the kernel needs as a number.
SCALAR_ARGUMENT_OPERATORS = frozenset({"segmented_lora_matmul", "layer_norm", "batch_norm"})


def broadcast_elementwise(x, y, verb):
    """The tensor operand that decides an elementwise op's result type.

    Either operand may be the rank-0 scalar; the other must be a tensor and
    supplies both the dtype and the shape of the result. When both operands
    are tensors this is exactly the old same-dtype/same-shape check, so every
    existing message is unchanged.
    """
    require_arithmetic(x, verb)
    require_arithmetic(y, verb)

    if not x.shape or not y.shape:
        tile = x if x.shape else y
        if not tile.shape:
            raise reject(
                DiagnosticKind.SHAPE,
                f"Cannot {verb} two scalars: a scalar input is a number, not "
                f"a tensor, and a pipeline value must be a tensor",
            )
        scalar = y if tile is x else x
        # The scalar is converted to the tile's dtype before the operation, so
        # a float tile with an integer scalar (or the reverse) would silently
        # change the arithmetic class. Keep the two on the same side of
        # 21.6's float/integer split; widths within a class are fine.
        if dtype_class(scalar.dtype) != dtype_class(tile.dtype):
            raise reject(
                DiagnosticKind.TYPE,
                f"Cannot {verb} a {scalar.dtype} scalar with a {tile.dtype} "
                f"tensor: the scalar must be the same kind of number as the "
                f"tensor",
            )
        return tile

    if x.dtype != y.dtype:
        raise reject(
            DiagnosticKind.TYPE,
            f"Cannot {verb} {x.dtype} and {y.dtype}",
        )

    if x.shape != y.shape:
        raise reject(
            DiagnosticKind.SHAPE,
            f"Cannot {verb} tensors with shapes "
            f"{format_shape(x.shape)} and {format_shape(y.shape)}",
        )

    return x


def infer_unary_same_type(args):
    """Preserve dtype and shape for numeric unary operators.

    ReLU uses this directly. SiLU and Softmax additionally require floating
    storage. RoPE validates three inputs in its own rule.
    """
    x = args[0]

    require_arithmetic(x, "apply this operator to")

    return TensorType(
        dtype=x.dtype,
        shape=x.shape.copy(),
    )


def require_float(x, operator):
    require_arithmetic(x, operator)
    if x.dtype not in {"bf16", "fp16", "fp32"}:
        raise reject(DiagnosticKind.TYPE, f"{operator} requires floating-point tensors")


def infer_float_unary(args):
    require_float(args[0], "silu/softmax")
    return infer_unary_same_type(args)


def infer_rope(args):
    x, cosine, sine = args
    require_float(x, "rope")
    for table in (cosine, sine):
        require_same_dtype(x, table, "rotate")
        require_same_shape(x, table, "rotate")
    width = x.shape[-1]
    if width.isdigit() and int(width) % 2:
        raise reject(DiagnosticKind.SHAPE, "rope requires an even last dimension")
    return TensorType(x.dtype, x.shape.copy())


def infer_add(args):
    x, y = args

    tile = broadcast_elementwise(x, y, "add")

    return TensorType(
        dtype=tile.dtype,
        shape=tile.shape.copy(),
    )


def infer_sub(args):
    x, y = args

    tile = broadcast_elementwise(x, y, "subtract")

    return TensorType(
        dtype=tile.dtype,
        shape=tile.shape.copy(),
    )


def infer_mul(args):
    x, y = args

    tile = broadcast_elementwise(x, y, "multiply")

    return TensorType(
        dtype=tile.dtype,
        shape=tile.shape.copy(),
    )


def infer_rms_norm(args):
    x, weight = args
    require_float(x, "rms_norm")

    # Normalization runs over the last dimension, so any rank works as long
    # as the weight matches that dimension: bf16[N,D] and bf16[B,H,S,D] are
    # both valid (Design.md 12.3 normalizes 4D q and k). This used to be
    # restricted to 2D, which is why that example did not compile.
    if weight.shape != [x.shape[-1]]:
        raise reject(
            DiagnosticKind.SHAPE,
            f"rms_norm weight shape {format_shape(weight.shape)} does not "
            f"match the last dimension of {format_shape(x.shape)}",
        )

    require_arithmetic(x, "rms_norm")

    if x.dtype != weight.dtype:
        raise reject(
            DiagnosticKind.TYPE,
            f"rms_norm dtype mismatch: {x.dtype} and {weight.dtype}",
        )

    return TensorType(
        dtype=x.dtype,
        shape=x.shape.copy(),
    )


def infer_matmul(args):
    left, right = args

    require_arithmetic(left, "matmul")
    require_float(left, "matmul")

    if len(left.shape) != 2 or len(right.shape) != 2:
        raise reject(
            DiagnosticKind.SHAPE,
            "matmul currently supports only 2D tensors",
        )

    if left.dtype != right.dtype:
        raise reject(
            DiagnosticKind.TYPE,
            f"matmul dtype mismatch: {left.dtype} and {right.dtype}",
        )

    if left.shape[1] != right.shape[0]:
        raise reject(
            DiagnosticKind.SHAPE,
            f"matmul shape mismatch: "
            f"{format_shape(left.shape)} and {format_shape(right.shape)}",
        )

    return TensorType(
        dtype=left.dtype,
        shape=[
            left.shape[0],
            right.shape[1],
        ],
    )


def infer_transpose(args):
    x = args[0]

    if len(x.shape) != 2:
        raise reject(
            DiagnosticKind.SHAPE,
            f"transpose currently expects a 2D tensor, "
            f"got {format_shape(x.shape)}",
        )

    return TensorType(
        dtype=x.dtype,
        shape=[
            x.shape[1],
            x.shape[0],
        ],
    )


def infer_attention(args):
    """Non-causal, dropout-free scaled dot-product attention.

    Two ranks, one operator. 2D is the v0.1 contract: q:[L,D], k:[S,D],
    v:[S,V] -> [L,V]. 4D is the batched-heads contract real models lower to:
    q:[B,H,L,D], k:[B,G,S,D], v:[B,G,S,V] -> [B,H,L,V], where G may divide
    H (GQA: grouped-query attention, k/v shared by H//G query heads) or
    equal it (MHA). The divisibility is checked where the checker can see
    it -- both head counts literal -- and deferred to the host's run-time
    check otherwise, the same boundary reshape's symbolic conservation
    uses: a symbolic H//G is arithmetic this language cannot write.

    causal_attention takes the same shapes with a causal mask: query i
    attends to keys 0..i+(S-L), the kv-cache offset convention. Causal is
    a separate operator NAME because the language has no boolean literal
    to put in a call -- an attention(q,k,causal) would need a third type
    of argument the grammar has no rule for, and rms_norm/rmsnorm_hf is
    the precedent: two conventions, two names.
    """
    query, key, value = args
    for tensor in args:
        require_float(tensor, "attention")
    if len({tensor.dtype for tensor in args}) != 1:
        raise reject(DiagnosticKind.TYPE, "attention requires matching floating-point dtypes")
    ranks = {len(tensor.shape) for tensor in args}
    if len(ranks) != 1:
        raise reject(
            DiagnosticKind.SHAPE,
            "attention operands must share one rank: 2D [L,D], 3D [H,L,D] or 4D [B,H,L,D]",
        )
    (rank,) = ranks
    if rank == 2:
        if query.shape[1] != key.shape[1] or key.shape[0] != value.shape[0]:
            raise reject(DiagnosticKind.SHAPE, "attention requires q:[L,D], k:[S,D], v:[S,V]")
        return TensorType(query.dtype, [query.shape[0], value.shape[1]])
    if rank == 3:
        h, l, d = query.shape
        g, s, kd = key.shape
        vg, vs, width = value.shape
        if g != vg or s != vs or d != kd:
            raise reject(DiagnosticKind.SHAPE,
                         "3D attention requires q:[H,L,D], k:[G,S,D], v:[G,S,V]")
        if h != g and h.isdigit() and g.isdigit() and int(h) % int(g):
            raise reject(DiagnosticKind.SHAPE,
                         f"GQA requires the key head count {g} to divide the query head count {h}")
        return TensorType(query.dtype, [h, l, width])
    if rank != 4:
        raise reject(DiagnosticKind.SHAPE,
                     "attention supports 2D, 3D or 4D operands, not rank %d" % rank)
    b, h, l, d = query.shape
    kb, g, s, kd = key.shape
    vb, vg, vs, v_width = value.shape
    if not (b == kb == vb and g == vg and s == vs and d == kd):
        raise reject(
            DiagnosticKind.SHAPE,
            "4D attention requires q:[B,H,L,D], k:[B,G,S,D], v:[B,G,S,V]",
        )
    # GQA: G names k/v's head count. When it differs from H the counts must
    # divide, checked statically only if both are literal.
    if h != g:
        if h.isdigit() and g.isdigit() and int(h) % int(g):
            raise reject(
                DiagnosticKind.SHAPE,
                f"GQA requires the key head count {g} to divide the query "
                f"head count {h}",
            )
    return TensorType(query.dtype, [b, h, l, v_width])


def infer_causal_attention(args):
    """Causal masking with the kv-cache offset: query i sees keys 0..i+(S-L)."""
    return infer_attention(args)


def infer_window_attention(args):
    """Attention limited to a per-batch valid-key count.

    q/k/v follow ``attention`` shapes. ``valid`` is int32 ``[B]`` for rank-4
    operands, or int32 ``[1]`` for rank-2/3. Scores for key positions
    ``>= valid`` are masked before the softmax.
    """
    if len(args) != 4:
        raise reject(DiagnosticKind.TYPE, "window_attention takes q, k, v, valid")
    out = infer_attention(args[:3])
    valid = args[3]
    if len(valid.shape) != 1 or valid.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE, "window_attention valid must be a 1-D integer tensor")
    rank = len(args[0].shape)
    if rank == 4:
        expected = args[0].shape[0]
    else:
        expected = "1"
    if valid.shape[0] != expected:
        raise reject(DiagnosticKind.SHAPE,
                     "window_attention valid must be [B] for 4-D attention, [1] otherwise")
    return out


def split_part_symbol(dimension, count):
    """The derived shape symbol for one part of a symbolic split.

    A literal last axis gives a literal part width (`16 / 2 == 8`). A symbolic
    one has no value at compile time, so the part width is derived at launch
    from the source's runtime last dimension (`N // parts`), exactly the way
    the shipped `split_half_inputs` host already does it. The relation has to
    live somewhere the host can see it; it lives in the shape symbol's name:
    the parts of `N` cut into `count` equal pieces are named `N__<count>`.
    That is a naming convention inside the compiler, not user syntax: users
    still write `gate, up = split(x)`, and no dimension arithmetic enters the
    grammar.

    The convention is parsed in two places only -- the host binding that emits
    `N__2 = N // 2` (and its exact-division guard), and the tests below -- so
    it cannot drift. `__` is not otherwise meaningful in a shape symbol.
    """
    return f'{dimension}__{count}'


def is_derived_split_symbol(symbol):
    """True for a `N__2`-style symbol produced by :func:`split_part_symbol`."""
    base, separator, parts = str(symbol).rpartition('__')
    return bool(separator) and base and parts.isdigit()


def infer_split(args, count, *, allow_symbolic=False):
    """Split a tensor into equal parts along the last axis.

    The number of parts is the number of names on the assignment's left
    side -- exactly the `q, k, v = split(x)` shape the v0.1 refusal named
    as the missing language feature. The axis is the last one, which is the
    language's row axis: every reduction and every rowwise kernel reads it,
    so a split there is the only one whose result rows keep the same
    contiguous-memory meaning.

    A literal last axis must be divisible by the part count; a non-divisible
    split has no equal-parts semantics to define, and inventing ragged padding
    would be a silent layout change.

    With ``allow_symbolic`` (the opt-in `split_views` path) a symbolic last
    axis is admitted too: the part width is derived at launch from the
    source's runtime last axis and passed as a constexpr. The host enforces the
    exact-division contract before the kernel runs. Without the opt-in the
    refusal is unchanged, so the default language and every default emission
    stay exactly as they were.
    """
    (x,) = args
    # see reshape: moving elements is not arithmetic
    require_arithmetic(x, "split")
    if len(x.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "split requires at least one dimension")
    last = x.shape[-1]
    if not last.isdigit():
        if not allow_symbolic:
            raise reject(
                DiagnosticKind.SHAPE,
                f"split needs a literal last axis, got {x}: the part width "
                f"cannot be written from a symbolic dimension",
            )
        part = x.shape[:-1] + [split_part_symbol(last, count)]
    else:
        width = int(last)
        if width % count:
            raise reject(
                DiagnosticKind.SHAPE,
                f"split cannot cut a last axis of {width} into {count} equal parts",
            )
        part = x.shape[:-1] + [str(width // count)]
    return [TensorType(dtype=x.dtype, shape=part) for _ in range(count)]


def infer_reshape(args):
    """Reshape to a shape literal with element-count conservation.

    The v0.1 refusal said the new shape "needs a shape literal in argument
    position"; the grammar now has one. Conservation is checked where the
    language can decide it: when both sides are fully literal, the products
    must match. When symbols appear, the products stay symbolic and cannot
    be compared -- the language has no dimension arithmetic (EBNF NOTES) --
    so the check defers to the host's runtime shape validation, exactly the
    way input declarations already defer symbolic positivity to run time.
    A -1 placeholder is refused: exactly one free dimension is expressible
    (make the user write the symbolic name), and a bare placeholder in a
    language without dimension arithmetic is a second type system in disguise.
    """
    x, shape = args
    # A layout operator moves elements; it does not compute them. `transpose`
    # and `move_first_last` have always accepted integers; refusing them here
    # was an inconsistency, and it blocked the MoE dispatch family, whose whole
    # job is to index with a *table* that then has to be reshaped or sliced.
    require_arithmetic(x, "reshape")
    # The type layer routes a shape literal raw to this rule, so it arrives
    # as the AST node; a tensor in this position is the caller's mistake.
    if not isinstance(shape, ShapeLiteral):
        raise reject(DiagnosticKind.SHAPE, "reshape requires a shape literal, like [N,32,64]")
    dims = [str(d) for d in shape.shape]
    # Compare numerically, not as strings: `[N,00]` and `[N,000]` are zero too,
    # and a string comparison accepted them until run time (review P2,
    # 2026-10-26).
    if any(d.lstrip('-').isdigit() and int(d) <= 0 for d in dims):
        raise reject(DiagnosticKind.SHAPE, "reshape dimensions must be positive; write the symbolic name instead of -1")
    literal_in = all(d.isdigit() for d in x.shape)
    literal_out = all(d.isdigit() for d in dims)
    if literal_in and literal_out:
        product_in = 1
        for d in x.shape:
            product_in *= int(d)
        product_out = 1
        for d in dims:
            product_out *= int(d)
        if product_in != product_out:
            raise reject(
                DiagnosticKind.SHAPE,
                f"reshape cannot turn {x} into {format_shape(dims)}: "
                f"{product_in} elements cannot become {product_out}",
            )
    return TensorType(dtype=x.dtype, shape=dims)


def infer_concat(args):
    """Concatenate two tensors along the last axis.

    The v0.1 refusal said the concatenated axis "is not specified"; making
    it an argument would need an integer literal in expression position,
    which the grammar has no type for. It is the last axis by definition,
    symmetric with split: the pair is the minimum the language can express,
    other axes would need non-row-major indexing the backends do not carry,
    and a concat whose axis had to be invented would be exactly the silent
    wrongness the v0.1 reserved operators existed to prevent.
    """
    a, b = args
    # see reshape: moving elements is not arithmetic
    require_arithmetic(a, "concat")
    require_arithmetic(b, "concat")
    if a.dtype != b.dtype:
        raise reject(DiagnosticKind.TYPE, "concat requires matching dtypes")
    if len(a.shape) < 1 or len(b.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "concat requires at least one dimension")
    if a.shape[:-1] != b.shape[:-1]:
        raise reject(
            DiagnosticKind.SHAPE,
            f"concat requires matching leading dimensions: {a} and {b}",
        )
    last_a, last_b = a.shape[-1], b.shape[-1]
    # A symbolic last axis would make the joined width "64+D" -- a string no
    # input ever declares, so no host could allocate the buffer or bind the
    # dimension. The arithmetic is not writable in this language, exactly
    # the reason split refuses a symbolic last axis: symmetry is the point.
    if not (last_a.isdigit() and last_b.isdigit()):
        raise reject(
            DiagnosticKind.SHAPE,
            f"concat requires literal last axes: the joined width of "
            f"{a} and {b} cannot be written from a symbolic dimension",
        )
    total = int(last_a) + int(last_b)
    return TensorType(dtype=a.dtype, shape=a.shape[:-1] + [str(total)])


def _reserved(name, reason):
    """An operator Design.md lists but v0.1 does not implement.

    Refusing is deliberate. These used to return a wrong-but-plausible type
    (split returned its input shape, concat ignored the concatenated axis,
    reshape could not be called at all because its shape argument would have
    to be an expression). A silently wrong type travels into code generation
    as fact, which is worse than a compile error.
    """

    def infer(args):
        raise reject(
            DiagnosticKind.OPERATOR,
            f"operator '{name}' is reserved, not implemented in v0.1: "
            f"{reason}",
        )

    return infer


register_operator("silu", 1, infer_float_unary, pointwise=True, acc=None)
register_operator("relu", 1, infer_unary_same_type, pointwise=True, acc=None)
register_operator("round", 1, infer_float_unary, pointwise=True, acc=None)
register_operator("add", 2, infer_add, pointwise=True, acc=None)
register_operator("sub", 2, infer_sub, pointwise=True, acc=None)
register_operator("mul", 2, infer_mul, pointwise=True, acc=None)
register_operator("rms_norm", 2, infer_rms_norm, pointwise=False, acc="fp32")
register_operator("rmsnorm_hf", 2, infer_rms_norm, pointwise=False, acc="fp32")
register_operator("matmul", 2, infer_matmul, pointwise=False, acc="fp32")
register_operator("softmax", 1, infer_float_unary, pointwise=False, acc="fp32")
register_operator("transpose", 1, infer_transpose, pointwise=False, acc=None)
register_operator("rope", 3, infer_rope, pointwise=False, acc=None)
register_operator("attention", 3, infer_attention, pointwise=False, acc="fp32")
register_operator("causal_attention", 3, infer_causal_attention, pointwise=False, acc="fp32")
register_operator("window_attention", 4, infer_window_attention, pointwise=False, acc="fp32")

# v0.2: the three formerly-reserved operators, now with signatures their
# v0.1 refusals named the missing pieces of. split produces several values,
# so it is registered multi and can only appear under a multi-name
# assignment; its part count is the number of names. reshape takes a shape
# literal in argument position. concat is along the last axis by definition.
register_operator("reshape", 2, infer_reshape, pointwise=False, acc=None)
register_operator("split", 1, infer_split, pointwise=False, acc=None, multi=True)
register_operator("concat", 2, infer_concat, pointwise=False, acc=None)


# ---------------------------------------------------------------------------
# v0.4: padded temporal concatenation.
#
#     out = pad_cat(x, cache, w_l, w_r, h_t, h_b, t_front, t_back)
#
# is `F.pad(cat([cache, x], dim=2), (w_l, w_r, h_t, h_b, t_front - Tc, t_back))`:
# the cached frames occupy the front of the temporal padding, so the effective
# front pad is `t_front - Tc` and must be non-negative.
#
# The output extents are `T + t_front + t_back` x `H + h_t + h_b` x
# `W + w_l + w_r`. This language has no dimension arithmetic (EBNF NOTES), so a
# symbolic input extent could not be written back as an output extent -- the same
# refusal `concat` already makes for a symbolic last axis. The op therefore
# requires all five input extents and all six pads to be literals, which is also
# what makes its output buffer allocatable and its kernel fully specialized.
def infer_pad_cat(args):
    from flowgpu.compiler.core.literals import IntLiteral

    x, cache = args[0], args[1]
    require_float(x, "pad_cat")
    require_float(cache, "pad_cat")
    if x.dtype != cache.dtype:
        raise reject(DiagnosticKind.TYPE, "pad_cat requires matching dtypes")
    if len(x.shape) != 5 or len(cache.shape) != 5:
        raise reject(DiagnosticKind.SHAPE, "pad_cat requires rank-5 [B,C,T,H,W] inputs")
    pads = []
    for literal in args[2:]:
        if not isinstance(literal, IntLiteral):
            raise reject(DiagnosticKind.TYPE, "pad_cat requires static integer padding literals")
        pads.append(literal.value)
    if len(pads) != 6:
        raise reject(DiagnosticKind.TYPE, "pad_cat requires exactly six padding literals")
    w_l, w_r, h_t, h_b, t_front, t_back = pads
    for name, value in (("width", w_l + w_r), ("height", h_t + h_b),
                        ("temporal", t_front + t_back)):
        if value < 0:
            raise reject(DiagnosticKind.SHAPE, f"pad_cat would crop the {name} axis")
    tc = cache.shape[2]
    if not tc.isdigit():
        raise reject(DiagnosticKind.SHAPE,
                     "pad_cat requires a literal cached-frame count: the front pad "
                     "is t_front - Tc and cannot be written from a symbolic extent")
    depth = t_front - int(tc)
    if depth < 0:
        raise reject(DiagnosticKind.SHAPE,
                     f"pad_cat front pad {t_front} is smaller than the {tc} cached frames")

    dims = []
    for axis, name in enumerate("BCTHW"):
        left, right = x.shape[axis], cache.shape[axis]
        if axis == 2:
            if not (left.isdigit() and right.isdigit()):
                raise reject(DiagnosticKind.SHAPE,
                             "pad_cat requires literal temporal extents so the output "
                             "extent T + t_front + t_back can be written")
            dims.append(str(int(left) + t_front + t_back))
            continue
        if left != right:
            raise reject(DiagnosticKind.SHAPE,
                         f"pad_cat requires matching {name} extents: {left} and {right}")
        if not left.isdigit():
            raise reject(DiagnosticKind.SHAPE,
                         f"pad_cat requires a literal {name} extent so the padded output "
                         f"extent can be written")
        add = (w_l + w_r) if axis == 4 else (h_t + h_b) if axis == 3 else 0
        dims.append(str(int(left) + add))
    return TensorType(dtype=x.dtype, shape=dims)


register_operator("pad_cat", 8, infer_pad_cat, pointwise=False, acc=None)


# v0.4: cumulative sum along the last axis, and its axis surface.
#
# A scan is not pointwise (each output depends on every earlier lane) and not a
# reduction (it keeps the axis), so it is its own kernel kind. The accumulator is
# fp32 and the result is rounded at the operator boundary, which is the same
# precision contract the reductions keep (21.6).
def infer_cumsum(args):
    require_float(args[0], "cumsum")
    return TensorType(args[0].dtype, args[0].shape.copy())


def infer_add_broadcast(args):
    """``add_broadcast(x, row)``: add a per-row value to every element of its row.

    `row` has the same leading extents as `x` and a trailing extent of exactly
    1 (the shape `sum_last`/`cumsum_totals` already use for a row result). A
    same-index `add` cannot express this -- this language's pointwise operators
    deliberately have no implicit broadcasting -- but a two-level prefix sum
    needs exactly this shape: offsets are per row, the scan is per element.

    The operator is pointwise (it can fuse into a region) and is the one
    broadcast a pointwise region may consume without materialising the larger
    operand first.
    """
    x, row = args
    require_float(x, "add_broadcast")
    require_same_dtype(x, row, "add_broadcast")
    expected = x.shape[:-1] + [str(1)]
    if [str(d) for d in row.shape] != expected:
        raise reject(DiagnosticKind.SHAPE,
                     f"add_broadcast needs a row operand of shape "
                     f"{expected}, got {[str(d) for d in row.shape]}")
    return TensorType(x.dtype, x.shape.copy())


register_operator("add_broadcast", 2, infer_add_broadcast, pointwise=True,
                  acc=None)


register_operator("cumsum", 1, infer_cumsum, pointwise=False, acc="fp32")


def infer_cumsum_totals(args, count):
    """``running, totals = cumsum_totals(x)``: the scan and its carry, in one kernel.

    A two-level prefix sum needs both halves: the within-row scan, and the row's
    total to build the row-to-row offsets.  Computing the total with a separate
    `sum_last` re-reads the whole input; this operator carries the same running
    state the scan lowering already keeps, so the total is free.  `totals` keeps
    the reduced axis as a 1, the same shape convention `sum_last` uses.
    """
    (x,) = args
    require_float(x, "cumsum_totals")
    if len(x.shape) < 1:
        raise reject(DiagnosticKind.SHAPE,
                     "cumsum_totals requires a rank >= 1 operand")
    if not all(str(d).isdigit() for d in x.shape[:-1]):
        raise reject(DiagnosticKind.SHAPE,
                     "cumsum_totals requires literal leading extents to launch "
                     "one program per row")
    totals = TensorType(x.dtype, x.shape[:-1] + ["1"])
    return [TensorType(x.dtype, x.shape.copy()), totals]


register_operator("cumsum_totals", 1, infer_cumsum_totals, pointwise=False,
                  acc="fp32", multi=True)


# v0.14: the carried-tuple scan family (P1 of the iteration region, see
# docs/ITERATION_REGION.md). `cumsum` carries one fp32 total; a prefix
# min/max must also carry *which index* attained it, so the state becomes a
# tuple and the operator has two results.
#
# The index dtype is int64 because that is what the host ABI and the CANN
# `Cummin` schema (values, indices with int64 indices, matching torch.cummin)
# specify. Tie-breaking is part of the contract: equal values keep the
# **earliest** index, which is what makes the merge function associative
# (`a if a.value <= b.value else b`, left-biased) and therefore scannable.
def infer_prefix_extreme(args, count, operator):
    x, = args
    require_float(x, operator)
    if len(x.shape) < 1:
        raise reject(DiagnosticKind.SHAPE,
                     f"{operator} requires a rank >= 1 operand")
    if not all(str(d).isdigit() for d in x.shape[:-1]):
        raise reject(DiagnosticKind.SHAPE,
                     f"{operator} requires literal leading extents to launch one "
                     f"program per row; the last axis may stay symbolic")
    return [TensorType(x.dtype, x.shape.copy()),
            TensorType("int64", x.shape.copy())]


def infer_cummin_last(args, count):
    """``values, indices = cummin_last(x)``: prefix min with earliest index."""
    return infer_prefix_extreme(args, count, "cummin_last")


def infer_cummax_last(args, count):
    """``values, indices = cummax_last(x)``: prefix max with earliest index."""
    return infer_prefix_extreme(args, count, "cummax_last")


register_operator("cummin_last", 1, infer_cummin_last, pointwise=False,
                  acc="fp32", multi=True)
register_operator("cummax_last", 1, infer_cummax_last, pointwise=False,
                  acc="fp32", multi=True)


# v0.6: segmented cumulative sum -- the reset scan of Task153 `kda_prologue`.
#
#     y[..., t] = cumsum(x)[..., t] - (cumsum(x)[..., s[..., t] - 1] if s[..., t] > 0 else 0)
#
# `seg_start` is per position: `s[..., t]` is the index of the first element of
# the segment that contains `t`, measured along the last axis. Rank and dtype
# are the input's; the scan accumulates in fp32 and rounds at its operator
# boundary exactly like `cumsum` (21.6), and the segmented correction subtracts
# two already-rounded prefix values in the declared dtype.
#
# This operator is a **composition, not a kernel**. `flowgpu/compiler/ir/composite_ops.py`
# expands it before fusion into operators the compiler already lowers:
#
#     prefix = cumsum(x)                     # the existing scan lowering
#     index  = max(sub(seg_start, 1), 0)     # the identity's own -1 is clamped
#     origin = eq(seg_start, 0)              # first segment subtracts 0
#     y      = sub(prefix, where(origin, 0, gather_last(prefix, index)))
#
# so the scan and the gather keep their one implementation each, their kernel
# kinds, their platform gates and their reference lowerings. The clamp is the
# fail-closed part of the identity: `gather_rows`' boundary rule normalises a
# negative index once by adding the extent and reads zero beyond it, so a raw
# `seg_start - 1` at `seg_start == 0` would read `cumsum[..., -1]` -- the row
# total -- and be silently wrong. The expansion proves the gathered index is in
# `[0, N)` at emission time instead; no access relies on the out-of-range
# convention.
#
# Boundaries, stated rather than discovered:
#
#   * last axis only, like `cumsum` (the axis surface is not mirrored: the
#     per-position `seg_start` tensor would have to rotate with x, which this
#     version does not implement; put the scan axis last in the program);
#   * every extent literal: the component `cumsum` and `gather_last` lowerings
#     need static rows/tiles and grids. A symbolic shape is a shape diagnostic,
#     not a wrong answer;
#   * `seg_start` is int32/int64 with exactly x's shape. Valid data has
#     `seg_start[..., 0] == 0`, is non-decreasing along the axis, and satisfies
#     `seg_start[..., t] <= t`. A length-0 segment has no position to be written
#     at, so it cannot be expressed; segments are the maximal runs of equal
#     start values, and a segment always contains at least its first position;
#   * the compiler never reads device data, so values outside that set are a
#     contract violation it cannot diagnose. What it does guarantee is that the
#     emitted access cannot leave the row (the clamp above) and that the
#     first-segment lane subtracts zero; the result for invalid data is
#     undefined and is not claimed.
def infer_segmented_cumsum(args):
    x, seg_start = args[0], args[1]
    require_float(x, "segmented_cumsum")
    if not x.shape:
        raise reject(DiagnosticKind.SHAPE, "segmented_cumsum needs rank >= 1")
    if len(seg_start.shape) != len(x.shape) or seg_start.shape != x.shape:
        raise reject(DiagnosticKind.SHAPE,
                     "segmented_cumsum requires seg_start to have exactly the input shape: "
                     f"{seg_start.shape} and {x.shape}")
    if seg_start.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE,
                     "segmented_cumsum requires an integer seg_start (int32 or int64)")
    symbolic = [dimension for dimension in x.shape if not str(dimension).isdigit()]
    if symbolic:
        raise reject(DiagnosticKind.SHAPE,
                     "segmented_cumsum requires literal extents: it is lowered by "
                     "composition into cumsum and gather_last, whose static rows/tiles "
                     f"cannot be written for symbolic {', '.join(map(str, symbolic))}")
    return TensorType(x.dtype, x.shape.copy())


register_operator("segmented_cumsum", 2, infer_segmented_cumsum,
                  pointwise=False, acc="fp32")

# Operators the language defines by composition: they are registered so their
# type and precision contracts are checked like any other operator, but no
# backend emits them. flowgpu/compiler/ir/composite_ops.py expands them into registered
# operators before fusion; `classification_gaps` reads this tuple so the
# "every operator has a backend decision" invariant stays honest (the decision
# is "composed away", not "silently missing").
COMPOSED_OPERATORS = ("segmented_cumsum", "layer_norm", "batch_norm")


# v0.4: static patch geometry.
#
#     out = patchify(x, ph, pw)   ==   b c (h ph) (w pw) -> b (h w) (c ph pw)
#
# This is a *declarative layout operator*, not general indexing: the index map
# is fixed by the two literal patch extents, the way `reshape`'s is fixed by its
# shape literal. It is the gap that kept the Task154 operator out of the
# language, and it is deliberately static -- the source address is a permutation
# of the output index, so the emitted plan's access shape is a permutation too,
# which is what `analysis/access_shape` reports and what collapses on 昆仑芯.
def infer_patchify(args):
    from flowgpu.compiler.core.literals import IntLiteral

    x = args[0]
    require_float(x, "patchify")
    if len(x.shape) != 4:
        raise reject(DiagnosticKind.SHAPE, "patchify requires a rank-4 [B,C,H,W] input")
    extents = []
    for literal in args[1:]:
        if not isinstance(literal, IntLiteral):
            raise reject(DiagnosticKind.TYPE, "patchify requires static integer patch extents")
        if literal.value <= 0:
            raise reject(DiagnosticKind.SHAPE, "patchify patch extents must be positive")
        extents.append(literal.value)
    if len(extents) != 2:
        raise reject(DiagnosticKind.TYPE, "patchify takes exactly two patch extents")
    ph, pw = extents
    b, c, h, w = x.shape
    for name, extent in (("B", b), ("C", c), ("H", h), ("W", w)):
        if not str(extent).isdigit():
            raise reject(DiagnosticKind.SHAPE,
                         f"patchify requires a literal {name} extent: the output "
                         f"extents are computed from it and no dimension "
                         f"arithmetic exists in this language")
    height, width = int(h), int(w)
    if height % ph or width % pw:
        raise reject(DiagnosticKind.SHAPE,
                     f"patchify patch {ph}x{pw} does not divide {height}x{width}")
    return TensorType(dtype=x.dtype,
                      shape=[str(b), str((height // ph) * (width // pw)),
                             str(int(c) * ph * pw)])


register_operator("patchify", 3, infer_patchify, pointwise=False, acc=None)


# v0.4: a segment descriptor, and the grouped matmul that needs it.
#
#     out = segmented_matmul(x, w, indptr)
#
#   x: [T, K]              tokens, in segment order
#   w: [S, K, N]           one weight matrix per segment
#   indptr: [S+1] int32    segment s owns rows [indptr[s], indptr[s+1])
#   out: [T, N]
#
# This is the `segment` capability the catalogue survey found blocking thirteen
# tasks: the sgmv/LoRA and grouped-GEMM family is exactly "a matmul whose weight
# is selected per segment". It is *declarative*, not general indexing: the
# segment boundaries are an explicit tensor input, the token axis may stay
# symbolic because no arithmetic is performed on it, and the weight index is the
# segment id rather than a computed address.
#
# The reduction `K` and the column `N` may be literals or symbols. A literal
# extent makes the product a compile-time tile (the original emission, and the
# one whose bytes the H100 validated). A symbol makes them launch-time values:
# the emitted kernel walks them in blocks, which is what lets one module serve
# whatever `kv_lora_rank`/`rank` the Task157 harness chooses -- a module whose
# declaration says `[3,16,8]` can only ever serve 16 and 8, and a wrong answer
# is 0/7 on every chip. `S` and the indptr length are never constexpr (the grid
# and the row loop read them at run time), so they need no literal form either.
def infer_segmented_matmul(args):
    x, w, indptr = args[0], args[1], args[2]
    require_float(x, "segmented_matmul")
    require_float(w, "segmented_matmul")
    if x.dtype != w.dtype:
        raise reject(DiagnosticKind.TYPE, "segmented_matmul requires matching dtypes")
    if len(x.shape) != 2 or len(w.shape) != 3:
        raise reject(DiagnosticKind.SHAPE,
                     "segmented_matmul takes x:[T,K], w:[S,K,N] and indptr:[S+1]")
    if len(indptr.shape) != 1 or indptr.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE,
                     "segmented_matmul requires a rank-1 integer indptr")
    if x.shape[1] != w.shape[1]:
        raise reject(DiagnosticKind.SHAPE,
                     f"segmented_matmul reduction extents differ: {x.shape[1]} and {w.shape[1]}")
    segments = w.shape[0]
    if str(segments).isdigit():
        expected = str(int(segments) + 1)
        if str(indptr.shape[0]).isdigit() and str(indptr.shape[0]) != expected:
            raise reject(DiagnosticKind.SHAPE,
                         f"segmented_matmul indptr must have {expected} entries for "
                         f"{segments} segments, got {indptr.shape[0]}")
    return TensorType(dtype=x.dtype, shape=[x.shape[0], w.shape[2]])


register_operator("segmented_matmul", 3, infer_segmented_matmul,
                  pointwise=False, acc="fp32")


# v0.5: the LoRA form of the grouped GEMM -- the weight is not pre-stacked per
# segment but *selected* from a shared adapter stack by an index tensor.
#
#     out = segmented_lora_matmul(x, a, indptr, weight_indices, lora_ranks, heads)
#
#   x:      [T, K]        flat (token, head) rows, contiguous in segment order
#   a:      [L, R, K]     every adapter's A matrix, `a[w][n, k] = A_w[n, k]`
#   indptr: [S+1] int     *token*-space segment boundaries (multiplied by `heads`
#                         inside the kernel; the adapter does not touch them)
#   weight_indices: [S] int   adapter index per segment
#   lora_ranks:     [L] int   valid rank per adapter; 0 means the task skips it
#   heads:          int32     the head count that flattens (token, head) rows
#   out:    [T, R]        dtype of `x`
#
# This is the Task157/sgmv shape and nothing more general: the segment id and
# the adapter index are both data, the reduction is over the last axis of both
# operands, and there is no computed address. It exists as its own operator
# because the alternative -- building `[S, K, R]` stacked weights on the host
# with `index_select` + `transpose` + `contiguous` and zeroing rank-0 adapters
# with `masked_fill` -- is five torch launches per call that the kernel can
# absorb for free: the adapter index is one scalar load, the rank test is one
# comparison, and the transposed weight tile is a strided load. On the official
# harness the call is the score, so those five launches are the difference
# between a mid-pack answer and a competitive one.
#
# `lora_ranks[w] == 0` means "this adapter has no delta": the reference skips
# the segment, so those rows must be exactly zero. The kernel writes zeros for
# them rather than skipping the store, because the destination buffer is
# uninitialized.
def infer_segmented_lora_matmul(args):
    x, a, indptr, weight_indices, lora_ranks, heads = args
    require_float(x, "segmented_lora_matmul")
    require_float(a, "segmented_lora_matmul")
    if x.dtype != a.dtype:
        raise reject(DiagnosticKind.TYPE,
                     "segmented_lora_matmul requires matching dtypes")
    if len(x.shape) != 2 or len(a.shape) != 3:
        raise reject(DiagnosticKind.SHAPE,
                     "segmented_lora_matmul takes x:[T,K], a:[L,R,K] and indptr:[S+1]")
    for name, value in (("indptr", indptr), ("weight_indices", weight_indices),
                        ("lora_ranks", lora_ranks)):
        if len(value.shape) != 1 or value.dtype not in {"int32", "int64"}:
            raise reject(DiagnosticKind.TYPE,
                         f"segmented_lora_matmul requires a rank-1 integer {name}")
    if not heads.is_scalar or heads.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE,
                     "segmented_lora_matmul requires an integer `heads` scalar")
    if x.shape[1] != a.shape[2]:
        raise reject(DiagnosticKind.SHAPE,
                     f"segmented_lora_matmul reduction extents differ: {x.shape[1]} and {a.shape[2]}")
    segments = weight_indices.shape[0]
    if str(segments).isdigit() and str(indptr.shape[0]).isdigit():
        expected = str(int(segments) + 1)
        if str(indptr.shape[0]) != expected:
            raise reject(DiagnosticKind.SHAPE,
                         f"segmented_lora_matmul indptr must have {expected} entries for "
                         f"{segments} segments, got {indptr.shape[0]}")
    if str(a.shape[0]).isdigit() and str(lora_ranks.shape[0]).isdigit():
        if str(a.shape[0]) != str(lora_ranks.shape[0]):
            raise reject(DiagnosticKind.SHAPE,
                         f"segmented_lora_matmul needs one rank per adapter: "
                         f"{lora_ranks.shape[0]} ranks for {a.shape[0]} adapters")
    return TensorType(dtype=x.dtype, shape=[x.shape[0], a.shape[1]])


register_operator("segmented_lora_matmul", 6, infer_segmented_lora_matmul,
                  pointwise=False, acc="fp32")


# v0.4: declared indexed write.
#
#     out = scatter_by_index(base, index, values)
#
#   base:   [N, ...]            the destination, copied through
#   index:  int32/int64 [M]     destination row for each of M value rows
#   values: [M, ...]            the rows to write
#
#   out = base, with out[index[i]] = values[i] for every i whose index is inside
#   [0, N). **Out-of-bounds entries are discarded**, the mirror image of
#   `gather_rows` returning zero for an out-of-bounds read: rejecting would need
#   a compile-time bound proof that a data-dependent index cannot supply, and
#   clamping would silently duplicate a write onto a neighbouring row.
#
# Duplicate indices are last-write-wins *unordered*: two values targeting one row
# race in the lowering, so a program that relies on the order is not expressible
# this way. That is stated here, in the type rule, rather than discovered later.
#
# This is a declared operator, not general indexing: the index is an explicit
# tensor input (`ir/indexing_ir`'s preparatory ScatterWriteSpec written as a
# surface operator), the ranks and the row shape are static, and the address is
# the index value rather than an expression the language computes.
def infer_scatter_by_index(args):
    base, index, values = args[0], args[1], args[2]
    require_float(base, "scatter_by_index")
    require_float(values, "scatter_by_index")
    if base.dtype != values.dtype:
        raise reject(DiagnosticKind.TYPE, "scatter_by_index requires matching dtypes")
    if len(base.shape) < 1 or len(values.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "scatter_by_index needs rank >= 1 operands")
    if len(index.shape) != 1 or index.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE,
                     "scatter_by_index requires a rank-1 integer index")
    if values.shape[1:] != base.shape[1:]:
        raise reject(DiagnosticKind.SHAPE,
                     f"scatter_by_index row shapes differ: {base.shape[1:]} and {values.shape[1:]}")
    if values.shape[0] != index.shape[0]:
        raise reject(DiagnosticKind.SHAPE,
                     f"scatter_by_index needs one index per value row: {values.shape[0]} "
                     f"rows and {index.shape[0]} indices")
    for name, extent in (("base", base.shape[0]), ("index", index.shape[0])):
        if not str(extent).isdigit():
            raise reject(DiagnosticKind.SHAPE,
                         f"scatter_by_index requires a literal {name} extent: the launch "
                         f"grid is computed from it")
    return TensorType(dtype=base.dtype, shape=base.shape.copy())


register_operator("scatter_by_index", 3, infer_scatter_by_index,
                  pointwise=False, acc=None)

# v0.3: general indexing, explicit broadcasting, casts and last-axis reductions.
# These are composable language operations, not competition-task builtins.
def infer_numeric_unary(args):
    return infer_unary_same_type(args)


def infer_to_fp32(args):
    require_arithmetic(args[0], 'cast')
    return TensorType('fp32', args[0].shape.copy())


def infer_finite(args):
    require_float(args[0], 'isfinite')
    return TensorType('bool', args[0].shape.copy())


def infer_where(args):
    condition, a, b = args
    if condition.dtype != 'bool':
        raise reject(DiagnosticKind.TYPE, 'where requires a bool condition')
    require_same_shape(condition, a, 'select')
    require_same_shape(a, b, 'select')
    if a.dtype != b.dtype:
        raise reject(DiagnosticKind.TYPE, 'where requires matching branch dtypes')
    return TensorType(a.dtype, a.shape.copy())


def infer_broadcast(args):
    x, target = args
    if len(x.shape) > len(target.shape):
        raise reject(DiagnosticKind.SHAPE, 'broadcast_like cannot lower rank')
    padded = ['1'] * (len(target.shape)-len(x.shape)) + x.shape
    if any(a != '1' and a != b for a, b in zip(padded, target.shape)):
        raise reject(DiagnosticKind.SHAPE, 'broadcast_like requires matching or singleton dimensions')
    return TensorType(x.dtype, target.shape.copy())


def infer_cast_like(args):
    x, target = args
    # `require_numeric`, not `require_arithmetic`: a cast is the only way a
    # storage dtype legitimately enters or leaves a program.
    require_numeric(x, 'cast')
    require_numeric(target, 'cast')
    return TensorType(target.dtype, x.shape.copy())


def infer_cast_to(args):
    """`cast_to(x, dtype)`: an explicit cast whose target is written, not read.

    `cast_like` needs a tensor template, so "cast to the output dtype" forced a
    `[1]` input whose only job was to carry a dtype -- a validated device tensor
    and one positional argument on every call, and (when the emitter proves it
    unused) a kernel formal the ABI pass then prunes. The target here is a dtype
    literal, so the pipeline declares what it means and no tensor exists solely
    for its dtype.

    The conversion contract is `cast_like`'s, unchanged: same shape, target
    dtype, bool targets refused as "not a number".

    Float -> integer is defined as **truncation toward zero, wrapping at the
    target width** -- exactly torch's `tensor.to(int)` and the hardware's own
    `.to(tl.int8)`, so the emitted expression and the Python reference agree.
    It is deliberately *not* rounding: a quantiser that needs round-half-to-even
    says so with the `round` operator, and one that needs saturation says so with
    `minimum`/`maximum`.  Folding either into the cast would give one operator
    two jobs and make the program unable to state either.
    """
    x, target = args
    require_numeric(x, 'cast')
    if not isinstance(target, DTypeLiteral):
        raise reject(DiagnosticKind.TYPE,
                     "cast_to's second argument must be a dtype literal such as bf16")
    if dtype_class(target.dtype) == 'bool':
        raise reject(DiagnosticKind.TYPE, f'Cannot cast a {target.dtype}: bool is not a number')
    return TensorType(target.dtype, x.shape.copy())


def infer_interleave(args):
    """Alternate two tensors on the last axis: out[..., 2i] = a[..., i].

    The inverse of ``split`` for two parts.  It exists because a permutation
    *within* a word (AWQ's nibble order) has no expression through
    reshape/split/concat: those keep last-axis blocks contiguous, while the
    shuffle needs a new axis whose positions come from two operands at once.
    Both operands must have the same shape; a symbolic last axis is refused for
    the same reason concat refuses one (the joined width cannot be written).
    """
    a, b = args
    # see reshape: moving elements is not arithmetic
    require_arithmetic(a, "interleave")
    require_arithmetic(b, "interleave")
    if a.dtype != b.dtype:
        raise reject(DiagnosticKind.TYPE, "interleave requires matching dtypes")
    if len(a.shape) < 1 or len(b.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "interleave requires at least one dimension")
    if list(a.shape) != list(b.shape):
        raise reject(DiagnosticKind.SHAPE,
                     f"interleave requires matching shapes: {a} and {b}")
    if not a.shape[-1].isdigit():
        raise reject(
            DiagnosticKind.SHAPE,
            f"interleave requires a literal last axis: the joined width of {a} "
            "cannot be written from a symbolic dimension",
        )
    total = 2 * int(a.shape[-1])
    return TensorType(dtype=a.dtype, shape=a.shape[:-1] + [str(total)])


def infer_move_first_last(args):
    (x,) = args
    return TensorType(x.dtype, x.shape[1:] + x.shape[:1])


def infer_reduce_last(args):
    (x,) = args
    require_float(x, 'reduce_last')
    return TensorType('fp32', x.shape[:-1] + ['1'])


def infer_gather_row_offset(args):
    x, index = args
    if len(x.shape) != 2 or len(index.shape) != 1:
        raise reject(DiagnosticKind.SHAPE, 'gather_row_offset needs a 2D tensor and 1D indices')
    if index.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.TYPE, 'gather_row_offset requires integer indices')
    return TensorType(x.dtype, index.shape.copy())


def infer_gather_rows(args):
    x, index = args
    if not x.shape or len(index.shape) != 1:
        raise reject(DiagnosticKind.SHAPE, 'gather_rows needs a rank >= 1 tensor and 1D indices')
    if index.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.TYPE, 'gather_rows requires integer indices')
    return TensorType(x.dtype, index.shape.copy() + x.shape[1:])


def infer_linear_decode_forward(args, count):
    if count != 2 or len(args) != 7:
        raise reject(DiagnosticKind.SHAPE, 'linear_decode_forward returns output and cache and takes six arguments plus BLOCK_SIZE')
    q, k, v, cache, slope, slots, block = args
    if not hasattr(block, 'value') or block.value <= 0:
        raise reject(DiagnosticKind.SHAPE, 'linear_decode_forward BLOCK_SIZE must be a positive literal')
    if block.value & (block.value - 1):
        # the kernel's v-lane tile is `tl.arange(0, BLOCK_SIZE)`, and Triton
        # requires a power of two (review P1-9, 2026-10-26)
        raise reject(DiagnosticKind.SHAPE,
                     'linear_decode_forward BLOCK_SIZE must be a power of two')
    if len(q.shape) != 4 or str(q.shape[2]) != '1' or k.shape != q.shape or v.shape != q.shape:
        raise reject(DiagnosticKind.SHAPE, 'linear_decode_forward q/k/v must be [B,H,1,D]')
    if len(cache.shape) != 4 or cache.shape[1:] != [q.shape[1], q.shape[3], q.shape[3]]:
        raise reject(DiagnosticKind.SHAPE, 'linear_decode_forward cache must be [S,H,D,D]')
    if slope.shape != [q.shape[1]] or slots.shape != [q.shape[0]] or slots.dtype != 'int32':
        raise reject(DiagnosticKind.TYPE, 'linear_decode_forward slope/slot shapes or dtype are invalid')
    if q.dtype not in {'fp16', 'bf16', 'fp32'} or k.dtype != q.dtype or v.dtype != q.dtype or cache.dtype != q.dtype:
        raise reject(DiagnosticKind.TYPE, 'linear_decode_forward tensors must share floating dtype')
    heads, head_dim = q.shape[1], q.shape[3]
    if str(heads).isdigit() and str(head_dim).isdigit():
        # a literal declaration has no H/D host variables to derive HD from,
        # so fold the product here (review 2026-10-26)
        output_width = str(int(heads) * int(head_dim))
    else:
        output_width = 'HD'
    return [TensorType(q.dtype, [q.shape[0], output_width]),
            TensorType(cache.dtype, cache.shape.copy())]


def infer_store_cache_4d(args, count):
    if count != 2 or len(args) != 6:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d returns exactly two outputs and takes five tensors plus page_size')
    k_view, v_view, cache_k, cache_v, loc, page_size = args
    if not hasattr(page_size, 'value') or page_size.value <= 0:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d page_size must be a positive literal')
    # Rank first: reading shape[1] before the rank check turned a rank-1 view
    # into an IndexError instead of a diagnostic (review 2026-10-26).
    if len(k_view.shape) != 4 or len(v_view.shape) != 4:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d views must be rank 4')
    if str(k_view.shape[1]).isdigit() and int(k_view.shape[1]) != page_size.value:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d page_size must match view page dimension')
    if k_view.shape != v_view.shape:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d K/V views must have matching shapes')
    if len(cache_k.shape) != 3 or len(cache_v.shape) != 3:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d caches must be rank 3')
    if cache_k.shape != cache_v.shape:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d caches must have matching shapes')
    if cache_k.shape[1:] != k_view.shape[2:]:
        raise reject(DiagnosticKind.SHAPE, 'store_cache_4d cache head/dim must match views')
    if loc.shape != [cache_k.shape[0]] or loc.dtype != 'int64':
        raise reject(DiagnosticKind.TYPE, 'store_cache_4d loc must be int64[N]')
    if k_view.dtype not in {'fp16', 'bf16', 'fp32'} or v_view.dtype != k_view.dtype:
        raise reject(DiagnosticKind.TYPE, 'store_cache_4d views must have matching floating dtype')
    if cache_k.dtype not in {'fp16', 'bf16', 'fp32'} or cache_v.dtype not in {'fp16', 'bf16', 'fp32'}:
        raise reject(DiagnosticKind.TYPE, 'store_cache_4d caches must be floating dtype')
    return [TensorType(k_view.dtype, k_view.shape.copy()), TensorType(v_view.dtype, v_view.shape.copy())]


for name in ('exp', 'exp2', 'log', 'log2', 'neg_inf_like'):
    register_operator(name, 1, infer_float_unary, pointwise=False, acc=None)
for name in ('ones_like', 'zeros_like'):
    register_operator(name, 1, infer_numeric_unary, pointwise=False, acc=None)
register_operator('to_fp32', 1, infer_to_fp32, pointwise=False, acc=None)
register_operator('isfinite', 1, infer_finite, pointwise=False, acc=None)
register_operator('where', 3, infer_where, pointwise=False, acc=None)
register_operator('broadcast_like', 2, infer_broadcast, pointwise=False, acc=None)
register_operator('cast_like', 2, infer_cast_like, pointwise=False, acc=None)
register_operator('cast_to', 2, infer_cast_to, pointwise=False, acc=None)
register_operator('move_first_last', 1, infer_move_first_last, pointwise=False, acc=None)
register_operator('gather_row_offset', 2, infer_gather_row_offset, pointwise=False, acc=None)
register_operator('gather_rows', 2, infer_gather_rows, pointwise=False, acc=None)
register_operator('store_cache_4d', 6, infer_store_cache_4d, pointwise=False, acc=None, multi=True)


def infer_cache_write(args):
    cache, token, loc = args
    for value in (cache, token):
        require_float(value, 'cache_write')
    if len(cache.shape) != 3:
        from flowgpu.compiler.core.diagnostic import reject
        raise reject(DiagnosticKind.SHAPE, 'cache_write cache must be [S,H,D]')
    if cache.shape[1:] != token.shape:
        raise reject(DiagnosticKind.SHAPE, 'cache_write token must be [H,D]')
    if loc.shape != ['1'] and loc.shape != [1] or loc.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.SHAPE, 'cache_write loc must be an integer [1]')
    return TensorType(cache.dtype, cache.shape.copy())


register_operator('cache_write', 3, infer_cache_write, pointwise=False, acc=None)


def infer_cache_write_hsd(args):
    cache, token, loc = args
    for value in (cache, token):
        require_float(value, 'cache_write_hsd')
    if len(cache.shape) != 3:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd cache must be [H,S,D]')
    if cache.shape[0] != token.shape[0] or cache.shape[2:] != token.shape[1:]:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd token must be [H,D]')
    if loc.shape != ['1'] and loc.shape != [1] or loc.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd loc must be an integer [1]')
    return TensorType(cache.dtype, cache.shape.copy())


register_operator('cache_write_hsd', 3, infer_cache_write_hsd, pointwise=False, acc=None)


def infer_cache_write_hsd_batch(args):
    cache, token, loc = args
    for value in (cache, token):
        require_float(value, 'cache_write_hsd_batch')
    if len(cache.shape) != 4:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_batch cache must be [B,H,S,D]')
    if len(token.shape) != 3 or cache.shape[:2] != token.shape[:2] or cache.shape[3] != token.shape[2]:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_batch token must be [B,H,D]')
    if loc.shape != [cache.shape[0]] or loc.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_batch loc must be an integer [B]')
    return TensorType(cache.dtype, cache.shape.copy())


register_operator('cache_write_hsd_batch', 3, infer_cache_write_hsd_batch, pointwise=False, acc=None)


def infer_cache_write_hsd_range(args):
    cache, new, loc = args
    for value in (cache, new):
        require_float(value, 'cache_write_hsd_range')
    if len(cache.shape) != 3 or len(new.shape) != 3:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_range operands must be [H,S,D]')
    if cache.shape[0] != new.shape[0] or cache.shape[2] != new.shape[2]:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_range head/depth mismatch')
    if loc.shape != ['1'] and loc.shape != [1] or loc.dtype not in {'int32', 'int64'}:
        raise reject(DiagnosticKind.SHAPE, 'cache_write_hsd_range loc must be an integer [1]')
    return TensorType(cache.dtype, cache.shape.copy())


register_operator('cache_write_hsd_range', 3, infer_cache_write_hsd_range, pointwise=False, acc=None)
register_operator('linear_decode_forward', 7, infer_linear_decode_forward, pointwise=False, acc='fp32', multi=True)
def infer_div(args):
    for x in args:
        require_float(x, 'div')
    return infer_mul(args)

register_operator('div', 2, infer_div, pointwise=False, acc=None)
for name in ('sum_last', 'max_last'):
    register_operator(name, 1, infer_reduce_last, pointwise=False, acc='fp32')

def infer_mean_last(args):
    (x,) = args
    require_float(x, 'mean_last')
    return TensorType('fp32', x.shape[:-1] + ['1'])


register_operator('mean_last', 1, infer_mean_last, pointwise=False, acc='fp32')


def infer_layer_norm(args):
    x, weight, bias, eps = args
    require_float(x, 'layer_norm')
    require_float(weight, 'layer_norm')
    require_float(bias, 'layer_norm')
    require_float(eps, 'layer_norm')
    if not x.shape:
        raise reject(DiagnosticKind.SHAPE, 'layer_norm needs rank >= 1')
    if len(weight.shape) != 1 or weight.shape != x.shape[-1:]:
        raise reject(DiagnosticKind.SHAPE, 'layer_norm weight must have shape [D]')
    if bias.shape != weight.shape:
        raise reject(DiagnosticKind.SHAPE, 'layer_norm bias must match weight shape')
    if eps.shape:
        raise reject(DiagnosticKind.SHAPE, 'layer_norm eps must be a rank-0 scalar')
    return TensorType(x.dtype, x.shape.copy())


register_operator('layer_norm', 4, infer_layer_norm, pointwise=False, acc='fp32')


def infer_batch_norm(args):
    x, mean, var, weight, bias, eps = args
    for value in (x, mean, var, weight, bias, eps):
        require_float(value, 'batch_norm')
    if len(x.shape) != 2:
        raise reject(DiagnosticKind.SHAPE, 'batch_norm currently supports rank-2 [N, C] tensors')
    channel = x.shape[1]
    for name, value in (('mean', mean), ('var', var), ('weight', weight), ('bias', bias)):
        if len(value.shape) != 1 or value.shape != [channel]:
            raise reject(DiagnosticKind.SHAPE, f'batch_norm {name} must have shape [C]')
    if eps.shape:
        raise reject(DiagnosticKind.SHAPE, 'batch_norm eps must be a rank-0 scalar')
    return TensorType(x.dtype, x.shape.copy())


register_operator('batch_norm', 6, infer_batch_norm, pointwise=False, acc='fp32')


def infer_conv2d(args):
    x, weight, bias = args
    require_float(x, 'conv2d')
    require_float(weight, 'conv2d')
    require_float(bias, 'conv2d')
    if len(x.shape) != 4 or len(weight.shape) not in (2, 4):
        raise reject(DiagnosticKind.SHAPE,
                     'conv2d requires x [N,H,W,C] and weight [O,C] or [O,C,KH,KW]')
    if not all(str(d).isdigit() for d in x.shape + weight.shape + bias.shape):
        raise reject(DiagnosticKind.SHAPE, 'conv2d currently requires literal extents')
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = (1, 1) if len(weight.shape) == 2 else (int(weight.shape[2]), int(weight.shape[3]))
    if int(x.shape[3]) != in_channels:
        raise reject(DiagnosticKind.SHAPE, 'conv2d input channels and weight in-channels must match')
    if int(x.shape[1]) < kh or int(x.shape[2]) < kw:
        raise reject(DiagnosticKind.SHAPE, 'conv2d kernel is larger than the input')
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise reject(DiagnosticKind.TYPE, 'conv2d operands must share a floating dtype')
    if bias.shape != [str(out_channels)] and bias.shape != [out_channels]:
        raise reject(DiagnosticKind.SHAPE, 'conv2d bias must have shape [O]')
    return TensorType(x.dtype, [x.shape[0], str(int(x.shape[1]) - kh + 1),
                                str(int(x.shape[2]) - kw + 1), str(out_channels)])


register_operator('conv2d', 3, infer_conv2d, pointwise=False, acc='fp32')


def infer_conv2d_same(args):
    x, weight, bias = args
    require_float(x, 'conv2d_same')
    require_float(weight, 'conv2d_same')
    require_float(bias, 'conv2d_same')
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise reject(DiagnosticKind.SHAPE,
                     'conv2d_same requires x [N,H,W,C] and weight [O,C,KH,KW]')
    if not all(str(d).isdigit() for d in x.shape + weight.shape + bias.shape):
        raise reject(DiagnosticKind.SHAPE, 'conv2d_same currently requires literal extents')
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if kh != kw or kh % 2 == 0:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_same requires an odd square kernel')
    if int(x.shape[3]) != in_channels:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_same input channels must match weight in-channels')
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise reject(DiagnosticKind.TYPE, 'conv2d_same operands must share a floating dtype')
    if bias.shape != [str(out_channels)] and bias.shape != [out_channels]:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_same bias must have shape [O]')
    return TensorType(x.dtype, [x.shape[0], x.shape[1], x.shape[2], str(out_channels)])


register_operator('conv2d_same', 3, infer_conv2d_same, pointwise=False, acc='fp32')


def infer_conv2d_stride2(args):
    x, weight, bias = args
    require_float(x, 'conv2d_stride2')
    require_float(weight, 'conv2d_stride2')
    require_float(bias, 'conv2d_stride2')
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise reject(DiagnosticKind.SHAPE,
                     'conv2d_stride2 requires x [N,H,W,C] and weight [O,C,KH,KW]')
    if not all(str(d).isdigit() for d in x.shape + weight.shape + bias.shape):
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2 currently requires literal extents')
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if int(x.shape[3]) != in_channels:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2 channels must match')
    if int(x.shape[1]) < kh or int(x.shape[2]) < kw:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2 kernel is larger than the input')
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise reject(DiagnosticKind.TYPE, 'conv2d_stride2 operands must share a dtype')
    if bias.shape != [str(out_channels)] and bias.shape != [out_channels]:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2 bias must have shape [O]')
    out_h = (int(x.shape[1]) - kh) // 2 + 1
    out_w = (int(x.shape[2]) - kw) // 2 + 1
    return TensorType(x.dtype, [x.shape[0], str(out_h), str(out_w), str(out_channels)])


register_operator('conv2d_stride2', 3, infer_conv2d_stride2, pointwise=False, acc='fp32')


def infer_conv2d_stride2_same(args):
    x, weight, bias = args
    require_float(x, 'conv2d_stride2_same')
    require_float(weight, 'conv2d_stride2_same')
    require_float(bias, 'conv2d_stride2_same')
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise reject(DiagnosticKind.SHAPE,
                     'conv2d_stride2_same requires x [N,H,W,C] and weight [O,C,3,3]')
    if not all(str(d).isdigit() for d in x.shape + weight.shape + bias.shape):
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2_same currently requires literal extents')
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if (kh, kw) != (3, 3):
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2_same requires a 3x3 kernel')
    if int(x.shape[3]) != in_channels:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2_same channels must match')
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise reject(DiagnosticKind.TYPE, 'conv2d_stride2_same operands must share a dtype')
    if bias.shape != [str(out_channels)] and bias.shape != [out_channels]:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_stride2_same bias must have shape [O]')
    out_h = (int(x.shape[1]) + 2 - 3) // 2 + 1
    out_w = (int(x.shape[2]) + 2 - 3) // 2 + 1
    return TensorType(x.dtype, [x.shape[0], str(out_h), str(out_w), str(out_channels)])


register_operator('conv2d_stride2_same', 3, infer_conv2d_stride2_same, pointwise=False, acc='fp32')


def infer_conv2d_ex(args):
    from flowgpu.compiler.core.literals import IntLiteral
    x, weight, bias, stride, padding = args
    require_float(x, 'conv2d_ex')
    require_float(weight, 'conv2d_ex')
    require_float(bias, 'conv2d_ex')
    if not isinstance(stride, IntLiteral) or not isinstance(padding, IntLiteral):
        raise reject(DiagnosticKind.TYPE, 'conv2d_ex stride and padding must be integer literals')
    if stride.value <= 0 or padding.value < 0:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_ex needs stride > 0 and padding >= 0')
    if len(x.shape) != 4 or len(weight.shape) != 4:
        raise reject(DiagnosticKind.SHAPE,
                     'conv2d_ex requires x [N,H,W,C] and weight [O,C,KH,KW]')
    if not all(str(d).isdigit() for d in x.shape + weight.shape + bias.shape):
        raise reject(DiagnosticKind.SHAPE, 'conv2d_ex currently requires literal extents')
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if int(x.shape[3]) != in_channels:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_ex channels must match')
    if int(x.shape[1]) + 2 * padding.value < kh or int(x.shape[2]) + 2 * padding.value < kw:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_ex kernel is larger than the padded input')
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise reject(DiagnosticKind.TYPE, 'conv2d_ex operands must share a dtype')
    if bias.shape != [str(out_channels)] and bias.shape != [out_channels]:
        raise reject(DiagnosticKind.SHAPE, 'conv2d_ex bias must have shape [O]')
    out_h = (int(x.shape[1]) + 2 * padding.value - kh) // stride.value + 1
    out_w = (int(x.shape[2]) + 2 * padding.value - kw) // stride.value + 1
    return TensorType(x.dtype, [x.shape[0], str(out_h), str(out_w), str(out_channels)])


register_operator('conv2d_ex', 5, infer_conv2d_ex, pointwise=False, acc='fp32')


def infer_sort_last(args):
    (x,) = args
    require_float(x, 'sort_last')
    if len(x.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, 'sort_last needs at least one dimension')
    if not all(str(d).isdigit() for d in x.shape):
        raise reject(DiagnosticKind.SHAPE, 'sort_last currently requires literal extents')
    return TensorType(x.dtype, x.shape.copy())


register_operator('sort_last', 1, infer_sort_last, pointwise=False, acc=None)


AXIS_OPERATORS = {'sum_axis': 'sum_last', 'max_axis': 'max_last',
                  'cumsum_axis': 'cumsum',
                  'softmax_axis': 'softmax', 'rms_norm_axis': 'rms_norm'}


def normalize_axis(x, literal):
    from flowgpu.compiler.core.literals import IntLiteral
    if not isinstance(literal, IntLiteral):
        raise reject(DiagnosticKind.TYPE, 'axis requires a static integer literal')
    rank = len(x.shape)
    if not -rank <= literal.value < rank:
        raise reject(DiagnosticKind.SHAPE, f'axis {literal.value} out of range for rank {rank}')
    return literal.value % rank


def axis_infer(operation):
    def infer(args):
        x, literal = args[0], args[-1]
        axis = normalize_axis(x, literal)
        rotated = TensorType(x.dtype, x.shape[axis+1:] + x.shape[:axis+1])
        result = OPERATORS[operation].infer([rotated] + args[1:-1])
        count = len(x.shape) - axis - 1
        return TensorType(result.dtype, result.shape[count:] + result.shape[:count])
    return infer


for surface, operation in AXIS_OPERATORS.items():
    register_operator(surface, OPERATORS[operation].arity + 1, axis_infer(operation), pointwise=False, acc='fp32')


# v0.5: the elementary value layer (tables in flowgpu/compiler/backend/value_layer.py).
#
# The 2026-10-20 content census classified all 115 tasks by what their reference
# implementations call, and found the largest gap was not addressing but this:
# 49 tasks compare values, 40 call a function the registry did not have. The
# registry had `relu`, `silu` and even `fused_sigmoid_mul`, while `where` -- the
# language's only select -- could be driven by exactly one boolean producer,
# `isfinite`. A language that can form a mask only by asking "is this finite"
# cannot write `where(x > 0, x, 0)` at all.
#
# These are registered `pointwise=False` with `acc=None`, exactly like
# `exp`/`log`: one expression, no accumulation. Fusing them is a separate,
# measurable decision (`silu` fuses, `exp` does not) and there is no device
# available to measure it today, so the conservative position is taken and
# stated rather than guessed.
def infer_comparison(args):
    """A value comparison: the language's only producer of `bool` besides
    `isfinite`.

    Same shape and same dtype, like every other elementwise operator: a
    same-index operation has no broadcasting, and `broadcast_like` is where a
    broadcast is declared. `bool` operands are refused by
    `require_arithmetic` -- comparing two truth values is a logical operator,
    which this language does not claim to have.
    """
    x, y = args[0], args[1]
    require_same_dtype(x, y, "compare")
    require_same_shape(x, y, "compare")
    return TensorType(dtype="bool", shape=x.shape.copy())


def infer_ordering(args):
    """`abs`/`maximum`/`minimum`: meaningful on integers, so no float rule.

    `maximum`/`minimum` accept a rank-0 operand beside a tensor, for the
    clamp-with-a-task-parameter shape (`minimum(x, limit)`); `abs` is unary, so
    a scalar there would produce a scalar and is refused upstream.
    """
    if len(args) == 1:
        require_arithmetic(args[0], "abs")
        return TensorType(dtype=args[0].dtype, shape=args[0].shape.copy())
    x, y = args[0], args[1]
    tile = broadcast_elementwise(x, y, "compare")
    return TensorType(dtype=tile.dtype, shape=tile.shape.copy())


def float_unary(operator):
    """A same-dtype float-in/float-out rule that names its own operator.

    `infer_float_unary` hardcodes "silu/softmax" in its refusal, and a message
    that names the wrong operator is worse than no message: it sends the reader
    to the wrong rule.
    """
    def infer(args):
        require_float(args[0], operator)
        return TensorType(dtype=args[0].dtype, shape=args[0].shape.copy())
    return infer


for name in ('lt', 'le', 'gt', 'ge', 'eq', 'ne'):
    register_operator(name, 2, infer_comparison, pointwise=False, acc=None)

register_operator('abs', 1, infer_ordering, pointwise=False, acc=None)
register_operator('neg', 1, infer_ordering, pointwise=False, acc=None)
for name in ('maximum', 'minimum'):
    register_operator(name, 2, infer_ordering, pointwise=False, acc=None)

for name in ('sqrt', 'rsqrt', 'tanh', 'sigmoid', 'softplus', 'erf',
             'gelu', 'gelu_tanh'):
    register_operator(name, 1, float_unary(name), pointwise=False, acc=None)

def infer_pow(args):
    x, y = args[0], args[1]
    require_float(x, 'pow')
    require_float(y, 'pow')
    require_same_dtype(x, y, 'pow')
    require_same_shape(x, y, 'pow')
    return TensorType(dtype=x.dtype, shape=x.shape.copy())


register_operator('pow', 2, infer_pow, pointwise=False, acc=None)


# 2026-10-24: integer bit manipulation (value_layer.BITWISE / BITS).
#
# The concrete need: three competition chips cannot compile an FP8 conversion at
# all, so the quantiser has to round and pack e4m3/e5m2 bytes itself -- which is
# integer work on the fp32 bit pattern.  `float_bits` and `bits_float` are the
# reinterpretations that make the pattern reachable; they are not casts (a cast
# converts the value, this changes its type without touching the bits), which is
# why they are separate operators with their own dtype rule.
def integer_bitwise(operator):
    """Both operands are integers; the result keeps the tile's dtype.

    `broadcast_elementwise` supplies the same-dtype/same-shape contract (and the
    one rank-0 operand allowance), and the class check keeps a float out: `&` on
    a float tile would be a type error on the device, so it is one here.
    """
    def infer(args):
        x, y = args[0], args[1]
        for operand, side in ((x, 'first'), (y, 'second')):
            if dtype_class(operand.dtype) != 'integer':
                raise reject(DiagnosticKind.TYPE,
                             f"{operator} requires integer operands; the {side} "
                             f"operand is {operand.dtype}")
        tile = broadcast_elementwise(x, y, f"{operator}-combine")
        return TensorType(tile.dtype, tile.shape.copy())
    return infer


def infer_float_bits(args):
    """`float_bits(x)`: the fp32 bit pattern as int32.  fp32 in, int32 out."""
    (x,) = args
    if x.dtype != 'fp32':
        raise reject(DiagnosticKind.TYPE,
                     "float_bits requires an fp32 operand; it reinterprets the "
                     "32-bit pattern rather than converting the value")
    return TensorType('int32', x.shape.copy())


def infer_bits_float(args):
    """`bits_float(b)`: the fp32 whose bit pattern is the int32 `b`."""
    (b,) = args
    if b.dtype != 'int32':
        raise reject(DiagnosticKind.TYPE,
                     "bits_float requires an int32 bit pattern")
    return TensorType('fp32', b.shape.copy())


for name in ('bitand', 'bitor', 'bitxor', 'shl', 'shr'):
    register_operator(name, 2, integer_bitwise(name), pointwise=False, acc=None)

register_operator('float_bits', 1, infer_float_bits, pointwise=False, acc=None)
register_operator('bits_float', 1, infer_bits_float, pointwise=False, acc=None)


# v0.5: which of these is biggest, and fetch by it.
#
# The content census put this family first among the remaining gaps (10 tasks).
# The router/gate shape needs both halves: `topk_last` gives the expert ids, and
# `gather_last` fetches the routing weights *from the full-expert softmax*, which
# is what those tasks specify ("weights come from the global softmax, not
# renormalised over the selected topk"). A top-k of a softmax over a top-k slice
# is a different function, so the second operator is not optional.
def infer_topk_last(args, count):
    """``values, indices = topk_last(x, k)``.

    Descending by value; equal values go to the **lower column index** (torch's
    `argsort` leaves tie order unspecified, so the language fixes it rather than
    inheriting an unspecified one). NaN and -inf are never selected, and a row
    with fewer than `k` candidates pads with value 0 and index 0.

    `k` is a static integer literal for the same reason `pad_cat`'s pads are: the
    output extent `[..., k]` has to be writable, and this language cannot write a
    computed extent.
    """
    from flowgpu.compiler.core.literals import IntLiteral

    if count != 2:
        raise reject(DiagnosticKind.SHAPE,
                     "topk_last returns values and indices, so it needs two names")
    x, literal = args
    require_float(x, "topk_last")
    if not isinstance(literal, IntLiteral):
        raise reject(DiagnosticKind.TYPE, "topk_last requires a static integer k")
    k = literal.value
    if k <= 0:
        raise reject(DiagnosticKind.SHAPE, "topk_last requires k > 0")
    if len(x.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "topk_last requires at least one dimension")
    width = x.shape[-1]
    if not width.isdigit():
        raise reject(DiagnosticKind.SHAPE,
                     "topk_last requires a literal last axis: the row tile is a "
                     "constexpr and the grid is computed from the extents")
    if k > int(width):
        raise reject(DiagnosticKind.SHAPE, f"topk_last k={k} exceeds the last axis ({width})")
    if not all(str(d).isdigit() for d in x.shape[:-1]):
        raise reject(DiagnosticKind.SHAPE,
                     "topk_last requires literal leading extents so the grid is static")
    shape = x.shape[:-1] + [str(k)]
    return [TensorType(x.dtype, shape.copy()), TensorType("int32", shape.copy())]


def infer_gather_last(args):
    """``out[..., j] = x[..., index[..., j]]``.

    The last-axis counterpart of `gather_rows`, and not a composition of it: the
    row is selected by a value *inside* the row, which an axis-0 row gather
    cannot express. Boundary convention is the row gather's -- a negative index is
    normalised once by adding the extent, and one still out of bounds reads as
    zero -- so the two behave the same way at the edges of a table.
    """
    x, index = args
    require_float(x, "gather_last")
    if index.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE, "gather_last requires an integer index")
    if len(index.shape) != len(x.shape):
        raise reject(DiagnosticKind.SHAPE,
                     f"gather_last needs an index of the same rank as x: "
                     f"{len(index.shape)} and {len(x.shape)}")
    if index.shape[:-1] != x.shape[:-1]:
        raise reject(DiagnosticKind.SHAPE,
                     f"gather_last leading extents differ: {index.shape[:-1]} and "
                     f"{x.shape[:-1]}")
    for name, extent in (("x", x.shape), ("index", index.shape)):
        if not all(str(d).isdigit() for d in extent):
            raise reject(DiagnosticKind.SHAPE,
                         f"gather_last requires literal {name} extents: the tile and "
                         f"the grid are computed from them")
    return TensorType(x.dtype, index.shape.copy())


register_operator("topk_last", 2, infer_topk_last, pointwise=False, acc=None, multi=True)
register_operator("gather_last", 2, infer_gather_last, pointwise=False, acc=None)


# v0.5: the index-carrying reduction -- ONE structural feature under two names.
#
# `argmax_last` and `argmin_last` are **one language feature**, not two. They
# share one type rule (modulo direction), one lowering (`reduce_with_index_backend`),
# and one kernel kind (`reduce_with_index`). The DSL has no string-literal mode
# argument, so the direction is spelled by the operator name; a mode argument
# would be a different spelling of the same capability. The audit's denominator
# therefore counts this as 1.
#
# What the reduction decides, rather than inherits:
#
#   * **Last axis only.** The result removes the last axis.
#   * **Ties go to the smallest index.** `torch.argmax`/`torch.argmin` do not
#     guarantee a tie order, so the language states its own instead of inheriting
#     an unspecified one. `test/test_reduce_with_index.py` pins it with a row of
#     equal values.
#   * **NaN is never selected.** An all-NaN row has no winner; a row whose values are
#     all NaN returns the direction's identity (-inf for max, +inf for min) with
#     index 0. This matches `topk_last`'s "NaN is never a candidate" rule.
#   * **+-inf follow the normal max/min rules.** Unlike `topk_last` (where -inf
#     is padding and never selected), an infinite extreme is a real candidate here:
#     +inf can win `argmax_last`, -inf can win `argmin_last`.
#   * **values keep the input dtype; indices are int32.**
#   * **Static extents.** The last axis must be a literal and non-empty, and every
#     leading extent must be literal, because the tile, the chunk loop and the
#     grid are all computed from them. The rejection style is `topk_last`'s.
def _infer_index_reduction(args, count, name):
    """``values, indices = argmax_last(x)`` / ``argmin_last(x)``.

    The direction is a *parameter of this one rule's lowering*, not a second
    rule: the two registered names are two spellings of the same language
    feature, so the type rule itself is identical for both.
    """
    if count != 2:
        raise reject(DiagnosticKind.SHAPE,
                     f"{name} returns values and indices, so it needs two names")

    (x,) = args
    require_float(x, name)
    if len(x.shape) < 2:
        raise reject(
            DiagnosticKind.SHAPE,
            f"{name} requires at least two dimensions: reducing the last axis "
            f"of a rank-1 input gives a scalar, and a pipeline value is a tensor")
    width = x.shape[-1]
    if not width.isdigit():
        raise reject(DiagnosticKind.SHAPE,
                     f"{name} requires a literal last axis: the row tile is a "
                     f"constexpr and the grid is computed from the extents")
    if int(width) == 0:
        raise reject(DiagnosticKind.SHAPE,
                     f"{name} requires a non-empty last axis")
    if not all(str(d).isdigit() for d in x.shape[:-1]):
        raise reject(DiagnosticKind.SHAPE,
                     f"{name} requires literal leading extents so the grid is static")
    shape = x.shape[:-1]
    return [TensorType(x.dtype, shape.copy()), TensorType("int32", shape.copy())]


def infer_argmax_last(args, count):
    """``values, indices = argmax_last(x)``: the max-with-index reduction."""
    return _infer_index_reduction(args, count, "argmax_last")


def infer_argmin_last(args, count):
    """``values, indices = argmin_last(x)``: the min-with-index reduction."""
    return _infer_index_reduction(args, count, "argmin_last")


register_operator("argmax_last", 1, infer_argmax_last, pointwise=False, acc=None, multi=True)
register_operator("argmin_last", 1, infer_argmin_last, pointwise=False, acc=None, multi=True)


# v0.5: the destination table a MoE dispatch is built from.
#
# The census called this capability `index_table` (four tasks). Task 69 states it
# as `offset = atomic_add(masked_m[expert], 1); src2dst[i] = expert * m_max +
# offset`, which is a *stable rank within a segment* -- each slot's destination is
# its expert's base plus the number of earlier slots with the same expert. Written
# that way it needs no atomic and no compaction primitive, and it composes with
# `scatter_by_index` to move the rows.
#
# The operator builds the table and nothing else: a table can be inspected,
# reused by several launches, and tested against the task's own reference, which a
# fused "dispatch" cannot.
def infer_segment_rank(args, count):
    """``counts, positions = segment_rank(ids, num_experts, m_max)``.

    `positions[i] = ids[i] * m_max + rank within its own expert`, -1 for a
    negative (padded) id, and `counts[e]` is the number of slots routed to `e`.
    The rank is stable: an earlier slot keeps the lower position, which is what
    the task's atomic increment produces.
    """
    from flowgpu.compiler.core.literals import IntLiteral

    if count != 2:
        raise reject(DiagnosticKind.SHAPE,
                     "segment_rank returns counts and positions, so it needs two names")
    ids, experts, m_max = args
    if ids.dtype not in {"int32", "int64"}:
        raise reject(DiagnosticKind.TYPE, "segment_rank requires an integer id tensor")
    if len(ids.shape) != 1:
        raise reject(DiagnosticKind.SHAPE, "segment_rank takes a flat [N] id tensor")
    for name, literal in (("num_experts", experts), ("m_max", m_max)):
        if not isinstance(literal, IntLiteral):
            raise reject(DiagnosticKind.TYPE,
                         f"segment_rank requires a static integer {name}: the output "
                         f"extent and the destination space are computed from it")
        if literal.value <= 0:
            raise reject(DiagnosticKind.SHAPE, f"segment_rank requires {name} > 0")
    if not ids.shape[0].isdigit():
        raise reject(DiagnosticKind.SHAPE,
                     "segment_rank requires a literal slot count: the launch grid is "
                     "computed from it")
    experts_value, m_max_value = experts.value, m_max.value
    if int(ids.shape[0]) > experts_value * m_max_value:
        raise reject(DiagnosticKind.SHAPE,
                     f"segment_rank destination space is {experts_value} x {m_max_value} "
                     f"= {experts_value * m_max_value} but there are {ids.shape[0]} slots: "
                     f"two slots of one expert would collide")
    return [TensorType("int32", [str(experts_value)]),
            TensorType("int32", ids.shape.copy())]


register_operator("segment_rank", 3, infer_segment_rank, pointwise=False, acc=None,
                  multi=True)


# v0.6: the first *temporal* operator, and the only one in the language.
#
#     output, final_state = gated_delta_recurrence(q, k, v, g, beta, state)
#
#     for t in 0..T-1:
#         state      = state * exp(g_t)
#         pred       = sum_k(state[v,k] * k_t[k])
#         correction = (v_t - pred) * beta_t
#         state      = state + correction[v] * k_t[k]
#         output_t   = sum_k(state[v,k] * q_t[k])
#
# This is Task18 `fused_recurrent_gdn`. Every step is elementwise or a reduction,
# all of which the language already has; what it could not express is the loop --
# `state` has to stay in registers across the whole time axis (materialising
# [B,T,H,V,K] is ~10^10 bytes at a real shape, and output_t reads the state *after*
# step t). That is a temporal construct, so the language grows one declared
# recurrence and leaves "no loops in the DSL" (Design.md 2) intact: the loop lives
# in the emitted kernel, under a kernel kind of its own.
#
# The task also takes `scale`, optional L2 norm of q/k and a GQA ratio. All three
# are *composition*, and are left to the DSL program rather than baked into a
# kernel only that task could reuse: scale is one `mul`, L2 norm is
# `div(x, sqrt(sum_axis(mul(x,x), -1) + eps))` with the value layer's operators,
# and the ratio needs `repeat_interleave` -- the census's `row_repeat` row.
def infer_gated_delta_recurrence(args, count):
    """One (batch, head) program per grid cell; the state is `[V, K]` in registers."""
    if count != 2:
        raise reject(DiagnosticKind.SHAPE,
                     "gated_delta_recurrence returns output and final state, so it "
                     "needs two names")
    q, k, v, gate, beta, state = args
    for tensor, name in ((q, "q"), (k, "k"), (v, "v"), (state, "state")):
        require_float(tensor, f"gated_delta_recurrence ({name})")
    for tensor, name in ((q, "q"), (k, "k"), (v, "v"), (state, "state")):
        if tensor.dtype != q.dtype:
            raise reject(DiagnosticKind.TYPE,
                         f"gated_delta_recurrence requires matching dtypes: q is "
                         f"{q.dtype} and {name} is {tensor.dtype}")
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        if len(tensor.shape) != 4:
            raise reject(DiagnosticKind.SHAPE,
                         f"gated_delta_recurrence takes rank-4 {name}: [B,T,H,D]")
    if len(state.shape) != 4:
        raise reject(DiagnosticKind.SHAPE,
                     "gated_delta_recurrence takes a rank-4 initial state [B,H,V,K]")
    for tensor in (q, k, v, gate, beta, state):
        if not all(str(dimension).isdigit() for dimension in tensor.shape):
            raise reject(DiagnosticKind.SHAPE,
                         "gated_delta_recurrence requires literal extents: the state "
                         "tile is a constexpr and the time loop is bounded by shape")
    batch, steps, heads, width = (int(d) for d in q.shape)
    if [int(d) for d in k.shape] != [batch, steps, heads, width]:
        raise reject(DiagnosticKind.SHAPE, "gated_delta_recurrence k must match q")
    # The GQA ratio: q/k carry H heads, v/gate/beta/state carry HV = H * ratio,
    # and head hv of the state reads source head hv // ratio. That mapping is a
    # declared index relation, so it belongs inside the operator -- it is exactly
    # what `attention` does for its own GQA, and it means the task does not need
    # `repeat_interleave` at all.
    out_heads = int(v.shape[2])
    if [int(d) for d in v.shape[:2]] != [batch, steps]:
        raise reject(DiagnosticKind.SHAPE,
                     "gated_delta_recurrence v must share q's batch and time extents")
    if out_heads % heads:
        raise reject(DiagnosticKind.SHAPE,
                     f"gated_delta_recurrence needs the value head count ({out_heads}) "
                     f"to be a multiple of q's ({heads})")
    channel = int(v.shape[3])
    if [int(d) for d in state.shape] != [batch, out_heads, channel, width]:
        raise reject(DiagnosticKind.SHAPE,
                     f"gated_delta_recurrence state must be [B,HV,V,K] = "
                     f"[{batch},{out_heads},{channel},{width}], got {state.shape}")
    if [int(d) for d in gate.shape] != [batch, steps, out_heads]:
        raise reject(DiagnosticKind.SHAPE,
                     f"gated_delta_recurrence gate must be [B,T,HV] = "
                     f"[{batch},{steps},{out_heads}], got {gate.shape}")
    for tensor, name, shape in ((beta, "beta", [batch, steps, out_heads]),):
        if [int(d) for d in tensor.shape] != shape:
            raise reject(DiagnosticKind.SHAPE,
                         f"gated_delta_recurrence {name} must be [B,T,H] = {shape}")
    return [TensorType(q.dtype, [str(batch), str(steps), str(out_heads), str(channel)]),
            TensorType(q.dtype, [str(batch), str(out_heads), str(channel), str(width)])]


register_operator("gated_delta_recurrence", 6, infer_gated_delta_recurrence,
                  pointwise=False, acc="fp32", multi=True)

# v0.6.1: the layout operator that makes an in-word permutation
# (AWQ nibble shuffle) expressible; see infer_interleave.
register_operator("interleave", 2, infer_interleave, pointwise=False, acc=None)
