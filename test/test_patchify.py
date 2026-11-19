"""`patchify`: a static element permutation, and the reason it closes the loop.

Task154's operator is a normalize followed by `b c (h ph) (w pw) -> b (h w) (c ph pw)`.
The language could not express the rearrangement, so the operator could not be
compiled and its official 昆仑芯 failure (0.02x output-indexed, Correctness Failed
in a 2-D masked form) could not be predicted from the IR. With `patchify` as a
declarative layout operator it can: the plan's access shape is a permutation of
the output index, `analysis/access_shape` reports exactly that, and
`platform_conflicts` refuses it on the target that collapses -- which is the same
judgement the official board made, now available before emission.
"""
import pytest

from compiler.analysis.access_shape import PERMUTED, plan_shape, platform_conflicts
from compiler.backend.platform_configs import PLATFORMS
from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.diagnostic import CompileError


def program(x="bf16[1,4,8,8]", ph=2, pw=2):
    source = ("pipeline patch_demo {\n"
              f" input x:{x}\n"
              f" out = patchify(x, {ph}, {pw})\n"
              " output out\n"
              "}\n")
    return compile_source(source).pipelines[0]


def test_the_output_extents_are_the_patch_geometry():
    """b c (h ph) (w pw) -> b (h w) (c ph pw): 8x8 into 2x2 patches is 16 tiles
    of 4x4 = 16 lanes."""
    (name, tensor_type) = program().reference.outputs[0]
    assert name == "out"
    assert tensor_type.shape == ["1", "16", "16"]
    assert tensor_type.dtype == "bf16"


def test_the_kernel_stores_densely_and_loads_a_permutation():
    kernels = list(program().triton.kernels)
    assert len(kernels) == 1
    kernel = kernels[0]
    assert kernel.kind == "patchify"
    assert kernel.ops == ("patchify",)
    source = kernel.source
    assert source.count("tl.load(") == 1 and source.count("tl.store(") == 1
    # the store address is the flat output index; the load address is computed
    assert "tl.store(out_ptr + e" in source
    assert "tl.load(x_ptr + src" in source or "tl.load(x_ptr + src)" in source


def test_fusion_does_not_fuse_pointwise_math_into_a_copy():
    """A copy kernel has no registers to fuse into, so the elementwise producer
    keeps its own kernel (the same rule the layout operators follow)."""
    pipeline = compile_source("pipeline p {\n input x:bf16[1,4,8,8]\n"
                              " centered = mul(x, x)\n out = patchify(centered, 2, 2)\n"
                              " output out\n}\n").pipelines[0]
    kinds = [kernel.kind for kernel in pipeline.triton.kernels]
    assert kinds == ["pointwise", "patchify"]


@pytest.mark.parametrize("ph,pw", [(3, 2), (6, 8)])  # 3 and 6 do not divide 8
def test_a_patch_that_does_not_divide_is_refused(ph, pw):
    with pytest.raises(CompileError, match='patchify patch') as caught:
        program(ph=ph, pw=pw)
    assert "divide" in str(caught.value) or "patch" in str(caught.value)


def test_a_symbolic_extent_is_refused():
    with pytest.raises(CompileError, match='the output extents are computed from it and no dimension arithmetic exists in this language') as caught:
        program(x="bf16[B,4,8,8]")
    assert "literal" in str(caught.value)


def test_the_reference_lowering_matches_an_independent_oracle():
    pipeline = program()
    reference = pipeline.reference
    produced = run_reference(reference)["out"]
    (values,) = [tensor for _, _, _, tensor in inputs_for(reference, {})]
    batch, channels, height, width = 1, 4, 8, 8
    ph = pw = 2
    tiles = (height // ph) * (width // pw)
    expected = []
    for b in range(batch):
        for tile in range(tiles):
            for channel in range(channels):
                for row in range(ph):
                    for col in range(pw):
                        h = (tile // (width // pw)) * ph + row
                        w = (tile % (width // pw)) * pw + col
                        expected.append(values[((b * channels + channel) * height + h) * width + w])
    assert produced == expected


def test_the_ir_calls_this_plan_a_permutation_and_refuses_it_on_kunlunxin():
    """The research loop closing: the same failure the official board measured is
    now predicted from the plan's access shape, before any text exists."""
    from types import SimpleNamespace

    plan = [SimpleNamespace(ops=("patchify",))]
    assert plan_shape(plan).kind == PERMUTED
    assert len(platform_conflicts(plan, PLATFORMS["kunlunxin"])) == 1
    assert platform_conflicts(plan, PLATFORMS["cuda"]) == []
