"""Packed layout v2: NVFP4 (E2M1 + UE8M0) and AWQ int4, byte for byte.

The oracles in ``compiler/packed_layout.py`` are compared with independent
translations of the two competition statements (FlagOS tasks 172 and 161),
written here with plain integer/bit operations rather than by calling the
module under test.  Both tasks are graded exact, so the tests compare packed
bytes / words exactly, not approximately.
"""
from __future__ import annotations

import math
import random
import struct

import pytest

from compiler.packed_layout import (
    AWQ_INT4, AWQ_ORDER, E2M1_MAX, E2M1_THRESHOLDS, MXFP8, NVFP4,
    PackedLayout, awq_dequantize, e2m1_code, e2m1_value,
    dequantize_nvfp4_indexer, pack_nibbles, pack_scale_exponents_le,
    quantize_nvfp4_indexer, ue8m0_ceil_exp, ue8m0_scale, ue8m0_value,
    unpack_nibbles, unpack_scale_exponents_le)

THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
AWQ_REVERSE = [0, 4, 1, 5, 2, 6, 3, 7]


# --- independent statement translations ----------------------------------

def _ref_ceil_ue8m0(value: float) -> int:
    bits = struct.unpack("i", struct.pack("f", value))[0]
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if mantissa:
        exponent += 1
    return max(1, min(254, exponent))


def _ref_e2m1(value: float) -> int:
    magnitude = min(abs(value), 6.0)
    code = 0
    for threshold in THRESHOLDS:
        if magnitude > threshold:
            code += 1
    if value < 0 and code:
        code |= 8
    return code


def _ref_nvfp4(values, rows, columns=128):
    codes, words = [], []
    for row in range(rows):
        exponents = []
        for group in range(columns // 32):
            start = row * columns + group * 32
            chunk = values[start:start + 32]
            amax = max(abs(v) for v in chunk)
            sf = max(amax / 6.0, 1e-4)
            exponent = _ref_ceil_ue8m0(sf)
            exponents.append(exponent)
            scale = struct.unpack("f", struct.pack("i", exponent << 23))[0]
            codes.extend(_ref_e2m1(v / scale) for v in chunk)
        for index in range(0, len(exponents), 4):
            word = 0
            for offset in range(4):
                word |= exponents[index + offset] << (8 * offset)
            words.append(word)
    packed = [codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2)]
    return packed, words


def _ref_awq(qweight, scales, zeros, rows, columns):
    words_per_row = columns // 8
    scale_rows = len(scales) // columns
    group_size = rows // scale_rows
    result = []
    for row in range(rows):
        group = row // group_size
        qrow, zrow = [], []
        for word_index in range(words_per_row):
            shifts = range(0, 32, 4)
            raw = [(qweight[row * words_per_row + word_index] >> shift) & 0xF
                   for shift in shifts]
            raw_z = [(zeros[group * words_per_row + word_index] >> shift) & 0xF
                     for shift in shifts]
            qrow.extend(raw[i] for i in AWQ_REVERSE)
            zrow.extend(raw_z[i] for i in AWQ_REVERSE)
        for column in range(columns):
            result.append((qrow[column] - zrow[column]) * scales[group * columns + column])
    return result


# --- descriptor ----------------------------------------------------------

def test_the_v2_storage_rules_are_named_and_checked():
    assert (NVFP4.storage, NVFP4.block, NVFP4.scale) == ("e2m1", 32, "ue8m0")
    assert NVFP4.values_per_container == 2
    assert (AWQ_INT4.storage, AWQ_INT4.pack_bits, AWQ_INT4.nibble_order) == (
        "uint4", 4, "awq_shuffled")
    assert AWQ_INT4.values_per_container == 8
    assert MXFP8.storage == "e4m3"                       # v1 unchanged

    with pytest.raises(ValueError, match="e2m1 requires pack_bits=4"):
        PackedLayout(storage="e2m1")
    with pytest.raises(ValueError, match="uint4 requires container_bits=32"):
        PackedLayout(storage="uint4", pack_bits=4, scale="affine",
                     scale_dtype="fp16", nibble_order="awq_shuffled")
    with pytest.raises(ValueError, match="last extent 48 is not divisible"):
        NVFP4.scale_shape([2, 48])


# --- codecs --------------------------------------------------------------

def test_e2m1_is_the_threshold_ladder_and_negative_zero_is_not_a_code():
    assert [e2m1_code(v) for v in (0.0, 0.25, 0.26, 0.75, 0.76, 3.4, 5.1, 6.0)] == [
        0, 0, 1, 1, 2, 5, 7, 7]
    assert e2m1_code(-0.1) == 0          # sign only when the magnitude is non-zero
    assert e2m1_code(-0.3) == 0x8 | 1
    assert e2m1_code(100.0) == 7          # clamped to 6 -> top code
    assert [e2m1_value(code) for code in (0, 1, 7, 0x8, 0x8 | 7)] == [
        0.0, 0.5, 6.0, -0.0, -6.0]
    for code in range(16):
        if code == 0x8:                  # negative zero is not a code the encoder emits
            assert _ref_e2m1(e2m1_value(code)) == 0
            continue
        assert _ref_e2m1(e2m1_value(code)) == code
    for value in (0.0, -0.2, 0.3, 1.2, 2.4, 4.9, -5.9, 6.0, -7.0):
        assert e2m1_code(value) == _ref_e2m1(value)
    assert E2M1_MAX == 6.0 and len(E2M1_THRESHOLDS) == 7


def test_ue8m0_ceil_rounds_exact_powers_up_only_when_the_mantissa_is_set():
    for value in (1.0, 2.0, 4.0, 0.5, 2.0 ** -13):
        assert ue8m0_ceil_exp(value) == _ref_ceil_ue8m0(value)
        assert math.log2(ue8m0_value(ue8m0_ceil_exp(value))) == math.log2(value)
    for value in (1.5, 3.0, 1e-4, 0.7):
        assert ue8m0_ceil_exp(value) == _ref_ceil_ue8m0(value)
        assert ue8m0_scale(ue8m0_ceil_exp(value)) >= value
    assert ue8m0_ceil_exp(0.0) == 1 or ue8m0_ceil_exp(0.0) == 1  # clamped low
    assert ue8m0_ceil_exp(1e30) <= 254


def test_nibble_and_scale_packing_round_trip():
    codes = [random.Random(3).randrange(16) for _ in range(256)]
    packed = pack_nibbles(codes)
    assert unpack_nibbles(packed) == codes
    exponents = [1, 2, 127, 254, 3, 4, 5, 6]
    words = pack_scale_exponents_le(exponents)
    assert len(words) == 2
    assert unpack_scale_exponents_le(words) == exponents
    assert words[0] == 1 | (2 << 8) | (127 << 16) | (254 << 24)


# --- NVFP4 oracle vs the statement ---------------------------------------

@pytest.mark.parametrize("seed", [1, 17, 99])
def test_nvfp4_matches_an_independent_statement_translation(seed):
    generator = random.Random(seed)
    values = [generator.uniform(-8.0, 8.0) for _ in range(4 * 128)]
    values[0:5] = [0.0, -0.0, 1e-9, 1e9, -6.0]        # boundary soup in one group
    packed, words = quantize_nvfp4_indexer(values, [4, 128])
    expected_packed, expected_words = _ref_nvfp4(values, 4)
    assert packed == expected_packed
    assert words == expected_words


def test_nvfp4_dequantises_within_one_e2m1_step():
    generator = random.Random(5)
    values = [generator.uniform(-3.0, 3.0) for _ in range(2 * 128)]
    packed, words = quantize_nvfp4_indexer(values, [2, 128])
    recovered = dequantize_nvfp4_indexer(packed, words, [2, 128])
    scales = [ue8m0_value(exponent)
              for word in words for exponent in unpack_scale_exponents_le([word])]
    for index, (original, got) in enumerate(zip(values, recovered)):
        scale = scales[index // 32]
        assert abs(original - got) <= 1.1 * scale, (index, original, got, scale)


# --- AWQ oracle vs the statement -----------------------------------------

@pytest.mark.parametrize("shape, scale_rows", [((4, 16), 2), ((8, 32), 1), ((16, 64), 4)])
def test_awq_matches_an_independent_statement_translation(shape, scale_rows):
    rows, columns = shape
    generator = random.Random(rows * 100 + columns)
    qweight = [generator.getrandbits(32) for _ in range(rows * columns // 8)]
    zeros = [generator.getrandbits(32) for _ in range(scale_rows * columns // 8)]
    scales = [round(generator.uniform(0.01, 2.0), 4) for _ in range(scale_rows * columns)]
    expected = _ref_awq(qweight, scales, zeros, rows, columns)
    actual = awq_dequantize(qweight, scales, zeros, list(shape))
    assert len(actual) == len(expected) == rows * columns
    assert actual == pytest.approx(expected, rel=0, abs=1e-12)


def test_awq_rejects_a_malformed_weight():
    with pytest.raises(ValueError, match="qweight must be"):
        awq_dequantize([0], [1.0] * 16, [0], [4, 16])
