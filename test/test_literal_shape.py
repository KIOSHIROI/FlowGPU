"""Literal-declaration rewriting used by LiteralShapeRunner."""
from flowgpu.compiler.runtime.literal_shape import rewrite_input_shapes

SOURCE = """
pipeline p {
  input a:fp16[1024]
  input b:fp16[1024]
  out = add(a, b)
  output out
}
"""


def test_rewrites_only_named_input_declarations():
    text = rewrite_input_shapes(SOURCE, {"a": (2048,), "b": (2048,)})
    assert "input a:fp16[2048]" in text
    assert "input b:fp16[2048]" in text
    assert "out = add(a, b)" in text


def test_unknown_input_is_an_error():
    try:
        rewrite_input_shapes(SOURCE, {"missing": (1,)})
    except ValueError as error:
        assert "could not rewrite" in str(error)
    else:
        raise AssertionError("missing input must fail")
