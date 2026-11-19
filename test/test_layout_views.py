"""P0: layout-view fusion removes split/concat/reshape materialisation kernels."""
from __future__ import annotations

from flowgpu.compiler import compile_source

SPLIT = "pipeline p { input x:fp32[32,64] a,b=split(x) out=add(a,b) output out }"
RESHAPE = "pipeline p { input x:fp32[32,64] y=reshape(x,[64,32]) out=relu(y) output out }"
CONCAT = ("pipeline p { input a:fp32[32,64] input b:fp32[32,64] "
          "c=concat(a,b) out=relu(c) output out }")


def _compile(source, **options):
    return compile_source(source, **options).pipelines[0]


def test_explicit_off_materialises_layout_anchors():
    assert [kernel.kind for kernel in _compile(SPLIT, layout_views=False).triton.kernels] == [
        "layout", "layout", "pointwise"]
    assert [kernel.kind for kernel in _compile(RESHAPE, layout_views=False).triton.kernels] == [
        "layout", "pointwise"]
    assert [kernel.kind for kernel in _compile(CONCAT, layout_views=False).triton.kernels] == [
        "layout", "pointwise"]


def test_cuda_default_fuses_layout_anchors():
    for source in (SPLIT, RESHAPE, CONCAT, INTERLEAVE):
        pipeline = _compile(source)
        assert len(pipeline.triton.kernels) == 1, source
        assert pipeline.triton.kernels[0].kind == "primitive"
        assert pipeline.triton.complete


def test_non_cuda_default_keeps_materialisation():
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    pipeline = _compile(SPLIT, platform_config=resolve_platform("kunlunxin"))
    # kunlunxin refuses the layout lowering, but the fusion decision is still
    # the materialising one: no indexed layout-view group on a non-CUDA default.
    assert all(not group.indexed for group in pipeline.fusion.groups)


def test_split_view_fusion_is_one_indexed_kernel():
    pipeline = _compile(SPLIT, layout_views=True)
    assert len(pipeline.triton.kernels) == 1
    assert pipeline.triton.kernels[0].kind == "primitive"
    assert pipeline.fusion.groups[0].indexed
    assert pipeline.triton.kernels[0].ops == ("split", "split", "add")
    assert len(pipeline.reference.kernels) == 1
    assert not pipeline.triton.refusals


def test_reshape_view_fusion_is_one_indexed_kernel():
    pipeline = _compile(RESHAPE, layout_views=True)
    assert len(pipeline.triton.kernels) == 1
    assert pipeline.triton.kernels[0].ops == ("reshape", "relu")
    assert len(pipeline.reference.kernels) == 1


def test_concat_view_fusion_is_one_indexed_kernel_and_masks_operands():
    pipeline = _compile(CONCAT, layout_views=True)
    assert len(pipeline.triton.kernels) == 1
    kernel = pipeline.triton.kernels[0]
    assert kernel.ops == ("concat", "relu")
    assert "mask=(valid) & (col < 64)" in kernel.source
    assert "mask=(valid) & (col >= 64)" in kernel.source
    assert len(pipeline.reference.kernels) == 1


def test_layout_views_does_not_change_unrelated_programs():
    source = "pipeline p { input x:fp32[1024] y=relu(x) z=add(y,x) output z }"
    default = _compile(source)
    views = _compile(source, layout_views=True)
    assert len(default.triton.kernels) == len(views.triton.kernels) == 1
    assert [k.ops for k in default.triton.kernels] == [k.ops for k in views.triton.kernels]


INTERLEAVE = ("pipeline p { input a:fp32[32,64] input b:fp32[32,64] "
              "c=interleave(a,b) out=relu(c) output out }")


def test_interleave_view_fusion_is_one_indexed_kernel():
    pipeline = _compile(INTERLEAVE, layout_views=True)
    assert len(pipeline.triton.kernels) == 1
    kernel = pipeline.triton.kernels[0]
    assert kernel.ops == ("interleave", "relu")
    assert "col % 2 == 0" in kernel.source
    assert len(pipeline.reference.kernels) == 1


def test_transpose_is_already_fused_by_its_own_emitter():
    source = "pipeline p { input x:fp32[32,64] t=transpose(x) out=relu(t) output out }"
    default = _compile(source)
    views = _compile(source, layout_views=True)
    assert len(default.triton.kernels) == len(views.triton.kernels) == 1


def test_n_way_split_view_uses_source_width_not_two_parts():
    # The fp4 example splits [4,4] into four [4,1] parts.  The generic mapping
    # must use the source width (4), not assume two halves.
    source = ("pipeline p { input x:fp32[4,4] a,b,c,d=split(x) "
              "s=add(add(a,b),add(c,d)) out=relu(s) output out }")
    pipeline = _compile(source, layout_views=True)
    assert len(pipeline.triton.kernels) == 1
    kernel = pipeline.triton.kernels[0]
    assert kernel.ops == ("split", "split", "split", "split", "add", "add", "add", "relu")
    # source width 4 appears in the flattened split index
    assert "* (4)" in kernel.source


def test_cost_policy_compiles_the_measured_good_split_case():
    pipeline = _compile(SPLIT, layout_views="cost")
    assert len(pipeline.triton.kernels) == 1
    assert pipeline.triton.kernels[0].kind == "primitive"
