"""The elementary value layer: one table per backend, one place to decide.

Why this module exists at all.

The 2026-10-20 content census
classified all 115 season-2 tasks by what their **reference implementation
calls**, instead of by their names. The result inverted the earlier name-based
ranking: the largest gap was not addressing (`moe_dispatch`, `topk`) but the
*value* layer. 49 of 115 reference implementations compare values (`x > 0`,
`mask == 0`) and 40 call an elementary function the registry did not have
(`sqrt`, `rsqrt`, `tanh`, `sigmoid`, `softplus`, `erf`, `gelu`, `clamp`).

The gap was easy to miss because the registry is not thin everywhere: it had
`relu`, `silu` and even the oddly specific `fused_sigmoid_mul`, while
`where` -- the language's only select -- could be driven by exactly one
boolean producer, `isfinite`. A language that can form a mask only by asking
"is this finite" cannot express `where(x > 0, x, 0)`.

What is decided here, and why it is decided in one table:

Four sites emit an elementwise expression -- `primitives` and `indexed_backend`,
each in Triton and in the reference -- and they had drifted into four chains of
`elif op == ...` with a `tl.{op}(...)` fallback. The fallback is why
`tanh` could not simply be added: `tl.tanh` does not exist, and neither does
`tl.math.tanh` -- measured on the H100 with Triton 3.3.1, `tl.math.tanh` raises
`AttributeError: module 'triton.language.math' has no attribute 'tanh'` while
`tl.extra.cuda.libdevice.tanh` is bit-exact against `torch.tanh`. A
name-derivation rule cannot know that, and it cannot tell a widened float
function from an exact integer one either. So the two expression tables live
here, the four sites import them, and adding an operator is a row.

Every row below was executed on an H100 (Triton 3.3.1, 2026-10-20) against the
torch function it claims to be, not inferred from documentation
(`tools/verify_value_layer.py`). Measured: `tanh`, `erf`, `abs`, `maximum`,
`minimum`, `sigmoid` and the comparison operators are **bit-exact**;
`sqrt`/`rsqrt` agree to **one ulp** (torch and libdevice need not choose the
same instruction); `sqrt(negative) -> NaN` and `rsqrt(0) -> +inf` are exact.
A bf16 chain accumulates one bf16 ulp per operator boundary (3.9e-3 over six
operators), which is the rounding rule doing its job, not a codegen error.

Three conventions the tables keep, each a contract rather than a style:

* **Float functions are computed in fp32 and rounded once at the boundary.**
  That is 21.6's split, and it is why every template widens with
  `.to(tl.float32)`: `tl.math.erf` has no bf16 overload, the same reason
  `silu` and `fused_sigmoid_mul` were the first entries in `WIDER_ON_DEVICE`.
  The reference gets the same shape for free -- it computes in double and
  `_rounded` writes the declared dtype.
* **Integer operators stay integer.** `abs`, `maximum` and `minimum` are the
  three that are meaningful on int32/int64, so their templates do not widen;
  widening them would silently turn a clamp into a float round-trip.
* **`maximum`/`minimum` propagate NaN, like torch and like `tl.maximum`.** A
  Python `max(a, b)` returns whichever operand it meets first when one is NaN,
  which would make the reference disagree with the device on exactly the values
  a rounding test uses. The `_maximum`/`_minimum` helpers state it instead.

Exactly one row is not a fixed spelling: `sigmoid` is selected by the platform
policy ``PlatformConfig.sigmoid_impl`` (``native`` | ``exp2``, validated in
``platform_configs.SIGMOID_IMPLS``). The two forms are algebraically the same
function --

    1 / (1 + exp(-x))  ==  1 / (1 + exp2(-log2(e) * x)),   log2(e) = 1.4426950408889634

-- so the choice is a lowering decision, not a semantics decision. `native` is
``tl.sigmoid``, the form every already-scoring target emits. `exp2` exists for a
backend whose libdevice `exp` is emulated elementwise rather than mapped to a
native vector instruction: a base-2 exponential is more often native there, and
Task160 measured 0.79x on 华为 with the one fused kernel that reads 4-11x on the
CUDA-class chips. Both spellings live in ``TRITON_SIGMOID`` below; the
``REFERENCE`` table is untouched, so the generated reference module keeps
computing the double-precision torch-equivalent function on every platform.

Semantics that are *not* IEEE-maximumNumber but the platform's own choice:

* `sqrt` of a negative is NaN, not an error, and `rsqrt(0)` is +inf -- the
  values `tl.sqrt`/`tl.rsqrt` produce. The reference renders those with
  `_sqrt`/`_rsqrt` rather than `math.sqrt`, which raises.
* `softplus` is evaluated in the overflow-free form
  `max(x, 0) + log(1 + exp(-|x|))`, not as `log(1 + exp(x))`. The two agree to
  fp32 rounding for |x| < 17 and the former does not overflow for large x.
  torch's `softplus` switches to the identity above a threshold of 20; this
  form agrees with it to rounding, and the difference is stated rather than
  assumed because `test_value_layer.py` pins the reference to it.
* `gelu` is the exact erf formula and `gelu_tanh` is the tanh approximation,
  because the catalogue contains both and they are not interchangeable
  (Task29 requires `approximate="none"`, Task71 requires `"tanh"`).
"""

from flowgpu.compiler.shared import (  # noqa: E402  (shared owns the groups)
    BITS, BITWISE, COMPARISONS, ELEMENTARY, ORDERING,
)

NAMES = COMPARISONS + ORDERING + ELEMENTARY + BITWISE + BITS

#: Templates are formatted with `a` (first operand), `b` (second operand) and
#: `zero` (the result dtype's zero, for `relu`'s neighbours).
#:
#: The bindings are not uniform, which is the concrete reason this table exists
#: instead of a name-derivation rule: `tl.math.erf` exists but `tl.math.tanh`
#: does not, and `tl.tanh`/`tl.erf` do not exist at all. Measured on Triton
#: 3.3.1: `tl.extra.cuda.libdevice.tanh` is the bit-exact tanh.
TRITON = {
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({a} > {b})",
    "ge": "({a} >= {b})",
    "eq": "({a} == {b})",
    "ne": "({a} != {b})",
    "abs": "tl.abs({a})",
    "neg": "(-{a})",
    "maximum": "tl.maximum({a}, {b})",
    "minimum": "tl.minimum({a}, {b})",
    "sqrt": "tl.sqrt({a}.to(tl.float32))",
    "rsqrt": "tl.rsqrt({a}.to(tl.float32))",
    "tanh": "tl.extra.cuda.libdevice.tanh({a}.to(tl.float32))",
    "sigmoid": "tl.sigmoid({a}.to(tl.float32))",
    "softplus": ("(tl.maximum({a}.to(tl.float32), 0.0) + "
                 "tl.log(1.0 + tl.exp(-tl.abs({a}.to(tl.float32)))))"),
    "erf": "tl.math.erf({a}.to(tl.float32))",
    "gelu": ("(0.5 * {a}.to(tl.float32) * (1.0 + "
             "tl.math.erf({a}.to(tl.float32) * 0.7071067811865476)))"),
    "gelu_tanh": ("(0.5 * {a}.to(tl.float32) * (1.0 + "
                  "tl.extra.cuda.libdevice.tanh(0.7978845608028654 * "
                  "({a}.to(tl.float32) + 0.044715 * {a}.to(tl.float32) * "
                  "{a}.to(tl.float32) * {a}.to(tl.float32)))))"),
    # Integer rows.  Triton's `&`, `|`, `^`, `<<`, `>>` are elementwise on
    # integer tiles and take a tile or a scalar shift/mask on the right.
    "bitand": "({a} & {b})",
    "bitor": "({a} | {b})",
    "bitxor": "({a} ^ {b})",
    "shl": "({a} << {b})",
    "shr": "({a} >> {b})",
    # `bitcast=True` is a reinterpretation: the emitted group's final
    # `.to(result dtype)` is then a no-op conversion between identical types.
    "float_bits": "({a}).to(tl.int32, bitcast=True)",
    "bits_float": "({a}).to(tl.float32, bitcast=True)",
}

#: The two spellings of the `sigmoid` row, keyed by the platform policy value
#: (`PlatformConfig.sigmoid_impl`, whose allowed values are validated in
#: `platform_configs.SIGMOID_IMPLS`). The native entry *is* the table row above,
#: so the bytes every already-scoring target emits cannot drift from it; `exp2`
#: is the same function written with a base-2 exponential:
#:
#:     exp(-x) == exp2(-log2(e) * x),   log2(e) = 1.4426950408889634
#:
#: The widened operand is the same `.to(tl.float32)` the native row uses, so
#: both forms round identically at the declared dtype boundary. `exp2` is not a
#: faster-language hack: it computes the same float function, and the only extra
#: rounding is one fp32 multiply of the exponent plus a base-2 rather than a
#: natural exponential -- see `test_sigmoid_policy.py` for the measured bound.
TRITON_SIGMOID = {
    "native": TRITON["sigmoid"],
    "exp2": "1 / (1 + tl.exp2(-1.4426950408889634 * {a}.to(tl.float32)))",
}

#: The same operators as generated Python. The reference computes in double and
#: rounds at the declared boundary, so no template widens or narrows here.
REFERENCE = {
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({a} > {b})",
    "ge": "({a} >= {b})",
    "eq": "({a} == {b})",
    "ne": "({a} != {b})",
    "abs": "abs({a})",
    "neg": "(-{a})",
    "maximum": "_maximum({a}, {b})",
    "minimum": "_minimum({a}, {b})",
    "sqrt": "_sqrt({a})",
    "rsqrt": "_rsqrt({a})",
    "tanh": "math.tanh({a})",
    "sigmoid": "sigmoid({a})",
    "softplus": "_softplus({a})",
    "erf": "math.erf({a})",
    "gelu": "_gelu({a})",
    "gelu_tanh": "_gelu_tanh({a})",
    "bitand": "({a} & {b})",
    "bitor": "({a} | {b})",
    "bitxor": "({a} ^ {b})",
    "shl": "({a} << {b})",
    "shr": "({a} >> {b})",
    "float_bits": "_float_bits({a})",
    "bits_float": "_bits_float({a})",
}

#: Helpers the reference templates call. Rendered into the generated module's
#: preamble, after `sigmoid` (which `_gelu`'s neighbour `_gelu_tanh` needs).
PREAMBLE = '''

def _sqrt(value):
    """`math.sqrt` raises on a negative; the device returns NaN."""
    return math.nan if value < 0.0 else math.sqrt(value)


def _rsqrt(value):
    if value < 0.0:
        return math.nan
    return math.inf if value == 0.0 else 1.0 / math.sqrt(value)


def _maximum(a, b):
    """NaN-propagating, like `tl.maximum` and `torch.maximum`."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a >= b else b


def _minimum(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a <= b else b


def _softplus(value):
    """`log(1 + exp(x))`, written so neither end of the range overflows."""
    if math.isnan(value):
        return math.nan
    return _maximum(value, 0.0) + math.log(1.0 + math.exp(-abs(value)))


def _gelu(value):
    """The exact erf formula, not a tanh approximation."""
    return 0.5 * value * (1.0 + math.erf(value * 0.7071067811865476))


def _gelu_tanh(value):
    """`approximate="tanh"`, as `torch.nn.functional.gelu` spells it."""
    inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
    return 0.5 * value * (1.0 + math.tanh(inner))


def _float_bits(value):
    """The fp32 bit pattern of a value, as an int32.

    `struct.pack("f", ...)` rounds to fp32 nearest-even and handles inf/NaN, so
    this is the reference for `tl.int32(bitcast=True)`.  `fp32()` first keeps the
    double-to-fp32 rounding in one place.
    """
    return struct.unpack("i", struct.pack("f", fp32(value)))[0]


def _bits_float(bits):
    """The fp32 value of an int32 bit pattern; the inverse of `_float_bits`."""
    return struct.unpack("f", struct.pack("i", int(bits)))[0]
'''


def triton_expression(operator, operand_names, *, sigmoid_impl="native") -> str:
    """The Triton expression for one value-layer operator.

    `sigmoid_impl` is the platform policy (`PlatformConfig.sigmoid_impl`):
    `native` keeps `tl.sigmoid`, `exp2` spells the same function with a base-2
    exponential. Every other row has exactly one spelling, so the argument is
    ignored for them -- the policy is a property of `sigmoid`, not a second
    emission mechanism. An unknown value is refused here as well as at
    `PlatformConfig` construction, so a hand-built config cannot silently emit
    the native form while claiming another.
    """
    if operator == "sigmoid":
        try:
            template = TRITON_SIGMOID[sigmoid_impl]
        except KeyError:
            raise ValueError(
                f"unknown sigmoid_impl: {sigmoid_impl!r}; choose from "
                f"{', '.join(TRITON_SIGMOID)}") from None
    else:
        template = TRITON[operator]
    return template.format(a=operand_names[0],
                           b=(operand_names + ("0",))[1], zero="0")


def reference_expression(operator, operand_names) -> str:
    """The generated-Python expression for one value-layer operator."""
    return REFERENCE[operator].format(a=operand_names[0],
                                      b=(operand_names + ("0",))[1], zero="0")
