import pytest

from flowgpu.compiler.frontend.ast_node import IntLiteral
from flowgpu.compiler.frontend.scalars import ScalarType, TypedScalar, static_int


def test_axis_literal_has_static_int32_type():
    literal = IntLiteral(-2)
    assert literal.type == ScalarType("int32")
    assert literal.value == -2


def test_static_scalar_contract_supports_future_slice_and_topk_parameters():
    scalar = static_int(7)
    assert scalar == TypedScalar(7, ScalarType("int32"))
    assert scalar.type.static is True


@pytest.mark.parametrize("dtype", ["bool", "fp16", "bf16"])
def test_unsupported_scalar_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError, match='unsupported scalar dtype:'):
        ScalarType(dtype)


def test_runtime_scalars_are_not_silently_enabled():
    with pytest.raises(ValueError, match='runtime scalar types are not enabled yet'):
        ScalarType("int32", static=False)


def test_integer_scalar_does_not_accept_float():
    with pytest.raises(TypeError, match='integer scalar needs a Python int value'):
        TypedScalar(1.0, ScalarType("int32"))
