"""One SwiGLU source compiles fused or fully materialized."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

SOURCE = open("examples/llama3_mlp_swiglu.fgpu", encoding="utf-8").read()
SIZES = {"T": 2, "H": 8, "I": 16}


def test_materialize_all_splits_the_fused_epilogue():
    fused = compile_source(SOURCE, shape_bindings=SIZES).pipelines[0]
    material = compile_source(SOURCE, shape_bindings=SIZES,
                              materialize_all=True).pipelines[0]
    assert len(fused.triton.kernels) == 3
    assert len(material.triton.kernels) == 4
    assert any(kernel.ops == ("matmul", "silu", "mul")
               for kernel in fused.triton.kernels)
    assert any(kernel.ops == ("silu", "mul")
               for kernel in material.triton.kernels)


def test_both_forms_match_the_same_reference():
    fused = compile_source(SOURCE, shape_bindings=SIZES).pipelines[0]
    material = compile_source(SOURCE, shape_bindings=SIZES,
                              materialize_all=True).pipelines[0]
    one = reference.run(fused.reference, sizes=SIZES)["y"]
    other = reference.run(material.reference, sizes=SIZES)["y"]
    assert one == other
