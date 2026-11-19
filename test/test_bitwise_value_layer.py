"""Integer bit manipulation in the value layer (`float_bits`, masks, shifts).

Added 2026-10-24 for a concrete need the competition produced: three FlagOS
chips cannot compile an FP8 conversion, so a quantiser has to round to e4m3/e5m2
itself from the fp32 bit pattern.  The same operators are what an int4 unpack
(`(packed >> (4*i)) & 0xF`) needs.

They are value-layer rows, not kernel kinds: one expression each, no
accumulation.  `float_bits`/`bits_float` are *not* casts -- a cast converts the
value, these reinterpret the same bits -- so they have their own dtype rules.
"""
import pytest

from compiler.backend import reference
from compiler.backend.value_layer import BITWISE, BITS, NAMES
from compiler.compile import compile_source
from compiler.diagnostic import CompileError

BITS_PROGRAM = '''
pipeline bits {
    input x : fp32[2,4]
    input mask : int32
    input shift : int32
    b = float_bits(x)
    m = bitand(b, mask)
    e = shr(m, shift)
    o = bitor(e, shl(e, shift))
    y = bitxor(e, o)
    output y
}
'''


def kernels(source):
    return compile_source(source).pipelines[0].triton.kernels


def test_the_operators_are_part_of_the_value_layer():
    for name in BITWISE + BITS:
        assert name in NAMES


def test_bitwise_operands_must_be_integers():
    with pytest.raises(CompileError, match="requires integer operands"):
        kernels("pipeline p { input x : fp32[4] y = bitand(x, x) output y }")
    with pytest.raises(CompileError, match="requires integer operands"):
        kernels("pipeline p { input x : fp32[4] input y : int32[4] "
                "z = bitand(x, y) output z }")


def test_bitcast_dtypes_are_fixed():
    with pytest.raises(CompileError, match="requires an fp32 operand"):
        kernels("pipeline p { input x : int32[4] y = float_bits(x) output y }")
    with pytest.raises(CompileError, match="requires an int32 bit pattern"):
        kernels("pipeline p { input x : fp32[4] y = bits_float(x) output y }")
    kernels("pipeline p { input x : fp32[4] y = float_bits(x) output y }")
    kernels("pipeline p { input b : int32[4] y = bits_float(b) output y }")


def test_masks_and_shifts_take_a_scalar():
    """The reason the operators are in SCALAR_BROADCAST_OPERATORS."""
    source = ("pipeline p { input x : fp32[4] input mask : int32 input shift : int32 "
              "b = float_bits(x) m = bitand(b, mask) e = shr(m, shift) output e }")
    kernel = list(kernels(source))[0]
    assert "&" in kernel.source and ">>" in kernel.source


def test_emission_reinterprets_rather_than_converts():
    source = list(kernels(BITS_PROGRAM))[0].source
    assert "bitcast=True" in source
    assert " >> " in source and " << " in source
    assert " & " in source and " | " in source and " ^ " in source


def test_the_reference_matches_python_integers():
    import struct

    pipeline = compile_source(BITS_PROGRAM).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    values = [1.0, -2.0, 3.5, 0.001, -0.0, 448.0, 57344.0, 1e-9]
    mask, shift = 0x7FFFFFFF, 23
    got = namespace['run_bits'](values, mask, shift)['y']
    expected = []
    for value in values:
        bits = struct.unpack('i', struct.pack('f', value))[0]
        e = (bits & mask) >> shift
        o = e | (e << shift)
        expected.append(e ^ o)
    assert got == expected


def test_the_two_bitcast_directions_round_trip():
    source = ('pipeline p { input x : fp32[4] b = float_bits(x) '
              'y = bits_float(b) output y }')
    pipeline = compile_source(source).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    values = [0.0, 1.0, -2.5, 3.4e38]
    assert namespace['run_p'](values)['y'] == pytest.approx(values)
