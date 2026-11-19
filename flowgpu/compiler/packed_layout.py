"""Packed storage layouts: descriptors and reference codecs (v2).

v1 (MXFP8) is unchanged: E4M3 values, one E8M0 scale per 32-value block.
v2 adds the two 4-bit layouts the extern ledger names as candidates:

* **NVFP4** (FlagOS task 172): E2M1 codes, block 32, one **UE8M0** exponent per
  block, two codes packed per byte (even code in the low nibble), four block
  exponents packed little-endian into one int32 per 128-value token.
* **AWQ int4** (FlagOS task 161): 4-bit unsigned weights packed eight per int32
  in AWQ's shuffled order ``[0, 4, 1, 5, 2, 6, 3, 7]``, dequantised with a
  per-group fp16 scale and an int4 zero point.

The module stays data-only: no compiler imports, no torch, no device.  The
``docs/PACKED_LAYOUT_V2.md`` page records the semantics and the byte-level
agreement tests in ``test/test_packed_layout_v2.py``.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

__all__ = [
    "PackedLayout",
    "MXFP8", "MXFP8_E5M2", "NVFP4", "AWQ_INT4",
    "E4M3_MAX", "E2M1_MAX",
    "e4m3_byte", "e4m3_value", "e5m2_byte", "e5m2_value",
    "e2m1_code", "e2m1_value",
    "e8m0_scale", "e8m0_value",
    "ue8m0_ceil_exp", "ue8m0_scale", "ue8m0_value",
    "pack_nibbles", "unpack_nibbles",
    "pack_scale_exponents_le", "unpack_scale_exponents_le",
    "quantize_mxfp8", "dequantize_mxfp8",
    "quantize_nvfp4_indexer", "dequantize_nvfp4_indexer",
    "awq_dequantize",
]

#: The largest finite E4M3 value (the _fn format has no infinity).
E4M3_MAX = 448.0
#: The largest finite E2M1 value.
E2M1_MAX = 6.0
#: E2M1 magnitude ladder: the code is the number of thresholds |v| exceeds.
E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
#: AWQ's nibble order inside one int32 (the inverse of the packing shuffle).
AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

_STORAGE_RULES = {
    "e4m3": {"pack_bits": 8, "scale": "e8m0", "scale_dtype": "uint8",
             "nibble_order": "low_first", "container_bits": 8},
    "e5m2": {"pack_bits": 8, "scale": "e8m0", "scale_dtype": "uint8",
             "nibble_order": "low_first", "container_bits": 8},
    "e2m1": {"pack_bits": 4, "scale": "ue8m0", "scale_dtype": "int32",
             "nibble_order": "low_first", "container_bits": 8},
    "uint4": {"pack_bits": 4, "scale": "affine", "scale_dtype": "fp16",
              "nibble_order": "awq_shuffled", "container_bits": 32},
}


@dataclass(frozen=True)
class PackedLayout:
    """How logical values map to storage bytes/words.

    Every field is checked against a named storage rule, so a descriptor can
    never silently describe a layout no codec implements.
    """

    storage: str = "e4m3"
    block: int = 32
    scale: str = "e8m0"
    scale_dtype: str = "uint8"
    pack_bits: int = 8
    nibble_order: str = "low_first"
    container_bits: int = 8
    #: One scale per ``block`` values (kernel-block layout), one per row, or one
    #: for the whole tensor.  v1/v2 only exercised ``block``; the field makes the
    #: other two layouts expressible without a second descriptor class.
    scale_mode: str = "block"

    def __post_init__(self):
        if self.storage not in _STORAGE_RULES:
            raise ValueError(f"unsupported packed storage: {self.storage!r}")
        rule = _STORAGE_RULES[self.storage]
        for field, expected in rule.items():
            actual = getattr(self, field)
            if actual != expected:
                raise ValueError(
                    f"{self.storage} requires {field}={expected!r}, got {actual!r}")
        if type(self.block) is not int or self.block <= 0 or self.block & (self.block - 1):
            raise ValueError("block must be a positive power of two")
        if self.scale_mode not in {"block", "row", "tensor"}:
            raise ValueError(f"unsupported scale mode: {self.scale_mode!r}")
        if self.storage in {"uint4", "e2m1"} and self.scale_mode != "block":
            raise ValueError(f"{self.storage} requires scale_mode='block'")

    @property
    def elements_per_byte(self) -> int:
        if self.pack_bits == 8:
            return 1
        return 2

    @property
    def values_per_container(self) -> int:
        return self.container_bits // self.pack_bits

    def scale_shape(self, shape) -> list:
        last = int(shape[-1])
        if last % self.block:
            raise ValueError(
                f"last extent {last} is not divisible by the block size {self.block}")
        if self.scale_mode == "tensor":
            return [1]
        if self.scale_mode == "row":
            return [1] * (len(shape) - 1) + [1]
        return list(shape[:-1]) + [last // self.block]


# The v2 block extent is a *format* constant taken from the task statements
# (128-value token / four scale groups = 32 values per group), not a scheduling
# choice: nothing tunes it and no platform policy selects it.
_FORMAT_GROUP = 32

MXFP8 = PackedLayout()

#: E5M2 twin of MXFP8: same block/scales, different 8-bit value format.
MXFP8_E5M2 = PackedLayout(storage="e5m2")

NVFP4 = PackedLayout(storage="e2m1", block=_FORMAT_GROUP, scale="ue8m0",
                     scale_dtype="int32", pack_bits=4,
                     nibble_order="low_first", container_bits=8)

AWQ_INT4 = PackedLayout(storage="uint4", block=_FORMAT_GROUP, scale="affine",
                        scale_dtype="fp16", pack_bits=4,
                        nibble_order="awq_shuffled", container_bits=32)


# --- E4M3 ----------------------------------------------------------------

def e4m3_byte(value: float) -> int:
    """The E4M3 byte for a Python float: RTNE, saturating at 448."""
    bits = struct.unpack("i", struct.pack("f", value))[0]
    sign = (bits >> 31) & 1
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude >> 23
    mantissa = magnitude & 0x7FFFFF
    if exponent == 0xFF:
        return (sign << 7) | 0x7E
    if exponent < 117:
        return sign << 7
    if exponent >= 121:
        e4 = exponent - 120
        full = mantissa + 0x7FFFF + ((mantissa >> 20) & 1)
        m3 = full >> 20
        e4 += m3 >> 3
        m3 &= 7
        if e4 > 15 or (e4 == 15 and m3 == 7):
            return (sign << 7) | 0x7E
        return (sign << 7) | (e4 << 3) | m3
    shift = 141 - exponent
    full = 0x800000 + mantissa
    k = (full + ((1 << (shift - 1)) - 1) + ((full >> shift) & 1)) >> shift
    return (sign << 7) | (8 if k == 8 else (k & 7))


def e4m3_value(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0 ** -6
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


# --- E5M2 ----------------------------------------------------------------

def e5m2_byte(value: float) -> int:
    """The E5M2 byte for a Python float: RTNE, subnormals to 2**-16.

    IEEE-like (5 exponent bits, 2 mantissa bits): past the range it rounds to
    infinity rather than saturating.  Written from the bit layout independently
    of ``tools/fp8_encoder`` and the backend preamble; the tests pin all three.
    """
    bits = struct.unpack("i", struct.pack("f", value))[0]
    sign = (bits >> 31) & 1
    magnitude = bits & 0x7FFFFFFF
    exponent = magnitude >> 23
    mantissa = magnitude & 0x7FFFFF
    if exponent == 0xFF:
        return (sign << 7) | 0x7B          # NaN/inf saturate to the top code
    if exponent <= 109:                    # underflow to signed zero
        return sign << 7
    if exponent >= 113:                    # normal
        e5 = exponent - 112
        full = mantissa + 0xFFFFF + ((mantissa >> 21) & 1)
        m2 = full >> 21
        e5 += m2 >> 2
        m2 &= 3
        if e5 >= 31:                       # overflow rounds to infinity
            return (sign << 7) | 0x7C
        return (sign << 7) | (e5 << 2) | m2
    shift = 134 - exponent                 # subnormal grid
    full = 0x800000 + mantissa
    k = (full + ((1 << (shift - 1)) - 1) + ((full >> shift) & 1)) >> shift
    return (sign << 7) | (4 if k == 4 else (k & 3))


def e5m2_value(byte: int) -> float:
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> 2) & 0x1F
    mantissa = byte & 0x03
    if exponent == 0:
        return sign * (mantissa / 4.0) * 2.0 ** -14
    if exponent == 0x1F:
        return sign * math.inf if mantissa == 0 else math.nan
    return sign * (1.0 + mantissa / 4.0) * 2.0 ** (exponent - 15)


# --- E8M0 (signed exponent, MXFP8) ---------------------------------------

def e8m0_scale(amax: float) -> int:
    amax = max(float(amax), 1e-30)
    biased = math.ceil(math.log2(amax / E4M3_MAX)) + 127
    return max(0, min(254, biased))


def e8m0_value(byte: int) -> float:
    return 2.0 ** (int(byte) - 127)


# --- E2M1 (NVFP4 values) -------------------------------------------------

def e2m1_code(value: float) -> int:
    """The 4-bit E2M1 code: magnitude ladder position + sign bit.

    Matches FlagOS task 172's definition exactly: ``min(|v|, 6)`` compared
    against ``(0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)``, sign bit only when the
    magnitude code is non-zero (negative zero is not representable as a code).
    """
    magnitude = min(abs(float(value)), E2M1_MAX)
    code = sum(1 for threshold in E2M1_THRESHOLDS if magnitude > threshold)
    if value < 0.0 and code:
        code |= 0x8
    return code


def e2m1_value(code: int) -> float:
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    value = magnitudes[code & 0x7]
    return -value if code & 0x8 else value


# --- UE8M0 (unsigned exponent, NVFP4 scales) -----------------------------

def ue8m0_ceil_exp(scale_value: float) -> int:
    """The clamped UE8M0 exponent: fp32 exponent, +1 when the mantissa is set."""
    bits = struct.unpack("i", struct.pack("f", float(scale_value)))[0]
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exponent += 1 if mantissa else 0
    return max(1, min(254, exponent))


def ue8m0_scale(exponent: int) -> float:
    """The pure power of two the stored exponent names."""
    return struct.unpack("f", struct.pack("i", (int(exponent) & 0xFF) << 23))[0]


def ue8m0_value(exponent: int) -> float:
    return 2.0 ** (int(exponent) - 127)


# --- packing helpers -----------------------------------------------------

def pack_nibbles(codes, order="low_first") -> list:
    """Pack 4-bit codes two per byte.

    ``low_first`` puts the even code in the low nibble (NVFP4 / task 172);
    ``high_first`` is the opposite convention, which other 4-bit kernels use
    and which a descriptor can now state instead of forcing a rewrite.
    """
    if order not in {"low_first", "high_first"}:
        raise ValueError(f"unsupported nibble order: {order!r}")
    codes = [int(code) & 0xF for code in codes]
    if len(codes) % 2:
        raise ValueError("an even number of nibbles is required")
    first = (lambda value: value) if order == "low_first" else (lambda value: value << 4)
    second = (lambda value: value << 4) if order == "low_first" else (lambda value: value)
    return [first(codes[index]) | second(codes[index + 1])
            for index in range(0, len(codes), 2)]


def unpack_nibbles(packed) -> list:
    values = []
    for byte in packed:
        values.extend((int(byte) & 0xF, (int(byte) >> 4) & 0xF))
    return values


def pack_scale_exponents_le(exponents) -> list:
    """Pack groups of four exponents into little-endian int32 words."""
    exponents = [int(exponent) for exponent in exponents]
    if len(exponents) % 4:
        raise ValueError("scale exponents must come in groups of four")
    words = []
    for index in range(0, len(exponents), 4):
        word = 0
        for offset in range(4):
            word |= (exponents[index + offset] & 0xFF) << (8 * offset)
        words.append(word)
    return words


def unpack_scale_exponents_le(words) -> list:
    exponents = []
    for word in words:
        exponents.extend((int(word) >> (8 * offset)) & 0xFF for offset in range(4))
    return exponents


# --- MXFP8 reference oracle (v1) -----------------------------------------

def quantize_mxfp8(values, shape=None):
    flat = [float(value) for value in values]
    if shape is None:
        raise ValueError("quantize_mxfp8 needs the logical shape")
    total = 1
    for extent in shape:
        total *= int(extent)
    if total != len(flat):
        raise ValueError(f"shape {shape} does not match {len(flat)} values")
    block = MXFP8.block
    columns = int(shape[-1])
    if columns % block:
        raise ValueError("the last extent must be divisible by the block size")
    rows = total // columns
    codes, scales = [], []
    for row in range(rows):
        for block_index in range(columns // block):
            start = row * columns + block_index * block
            chunk = flat[start:start + block]
            amax = max(abs(value) for value in chunk) if chunk else 0.0
            scale_byte = e8m0_scale(amax)
            descale = e8m0_value(scale_byte)
            scales.append(scale_byte)
            for value in chunk:
                scaled = max(-E4M3_MAX, min(E4M3_MAX, value / descale))
                codes.append(e4m3_byte(scaled))
    return codes, scales


def dequantize_mxfp8(codes, scales, shape):
    block = MXFP8.block
    columns = int(shape[-1])
    rows = len(codes) // columns
    values = []
    for row in range(rows):
        for block_index in range(columns // block):
            descale = e8m0_value(scales[row * (columns // block) + block_index])
            start = row * columns + block_index * block
            values.extend(e4m3_value(code) * descale for code in codes[start:start + block])
    return values


# --- NVFP4 reference oracle (v2) -----------------------------------------

def _f32(value: float) -> float:
    """Round a Python double to the nearest fp32 value (the statement's type)."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def quantize_nvfp4_indexer(values, shape=None):
    """FlagOS task 172's NVFP4 quantiser, byte for byte.

    Returns ``(packed_bytes, scale_words)``: two E2M1 codes per byte (even code
    in the low nibble) and one int32 per 128-value token holding the four
    block exponents little-endian.
    """
    flat = [float(value) for value in values]
    if shape is None:
        raise ValueError("quantize_nvfp4_indexer needs the logical shape")
    total = 1
    for extent in shape:
        total *= int(extent)
    if total != len(flat):
        raise ValueError(f"shape {shape} does not match {len(flat)} values")
    columns = int(shape[-1])
    block = NVFP4.block
    if columns % block:
        raise ValueError(f"the last extent {columns} must be divisible by {block}")
    groups_per_row = columns // block
    if groups_per_row % 4:
        raise ValueError("the scale word packs four groups, so the row needs a "
                         "multiple of 128 values")
    rows = total // columns
    packed, words = [], []
    for row in range(rows):
        row_codes = []
        row_exponents = []
        for group in range(groups_per_row):
            start = row * columns + group * block
            # The statement computes in fp32 (torch): round the inputs and the
            # scale division to fp32 before the integer/threshold logic, or a
            # double-rounding difference can flip the scale exponent at a
            # power-of-two boundary.
            chunk = [_f32(flat[start + index]) for index in range(block)]
            amax = max(abs(value) for value in chunk) if chunk else 0.0
            sf = _f32(max(amax / E2M1_MAX, 1e-4))
            exponent = ue8m0_ceil_exp(sf)
            row_exponents.append(exponent)
            scale = ue8m0_scale(exponent)
            row_codes.extend(e2m1_code(_f32(value / scale)) for value in chunk)
        packed.extend(pack_nibbles(row_codes))
        words.extend(pack_scale_exponents_le(row_exponents))
    return packed, words


def dequantize_nvfp4_indexer(packed, scale_words, shape):
    """Inverse of ``quantize_nvfp4_indexer``; used by the tolerance checks."""
    columns = int(shape[-1])
    block = NVFP4.block
    groups_per_row = columns // block
    rows = len(packed) * 2 // columns
    values = []
    for row in range(rows):
        row_codes = unpack_nibbles(packed[row * (columns // 2):(row + 1) * (columns // 2)])
        word_start = row * (groups_per_row // 4)
        exponents = unpack_scale_exponents_le(
            scale_words[word_start:word_start + groups_per_row // 4])
        for group, exponent in enumerate(exponents):
            scale = ue8m0_value(exponent)
            start = group * block
            values.extend(e2m1_value(code) * scale
                          for code in row_codes[start:start + block])
    return values


def awq_dequantize(qweight, scales, zeros, shape):
    """FlagOS task 161's AWQ dequantiser.

    ``qweight``: ``[K, M // 8]`` int32 words (AWQ nibble order), ``zeros``:
    ``[K // G, M // 8]`` int32 words, ``scales``: ``[K // G, M]`` fp16.  Returns
    a flat row-major list of K*M dequantised values.
    """
    rows, columns = int(shape[0]), int(shape[1])
    words_per_row = columns // 8
    if columns % 8 or len(qweight) != rows * words_per_row:
        raise ValueError("qweight must be K rows of M/8 words")
    scale_rows = len(scales) // columns
    if scale_rows <= 0 or rows % scale_rows:
        raise ValueError("scales rows must divide K")
    group_size = rows // scale_rows
    if group_size <= 0:
        raise ValueError("group size must be positive")
    result = []
    for row in range(rows):
        group_index = row // group_size
        row_values, row_zeros = [], []
        for word_index in range(words_per_row):
            raw = [(int(qweight[row * words_per_row + word_index]) >> (4 * shift)) & 0xF
                   for shift in range(8)]
            raw_zero = [(int(zeros[group_index * words_per_row + word_index]) >> (4 * shift)) & 0xF
                        for shift in range(8)]
            row_values.extend(raw[index] for index in AWQ_ORDER)
            row_zeros.extend(raw_zero[index] for index in AWQ_ORDER)
        scale_base = group_index * columns
        for column in range(columns):
            result.append((row_values[column] - row_zeros[column])
                          * float(scales[scale_base + column]))
    return result
