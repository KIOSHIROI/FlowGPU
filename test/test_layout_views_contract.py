"""Cross-backend layout-view contract (P0 default boundary)."""
from __future__ import annotations

import pytest

from flowgpu.compiler import compile_source
from flowgpu.compiler.backend.platform_configs import PLATFORMS, resolve_platform
from flowgpu.compiler.frontend.diagnostic import CompileError

SPLIT = "pipeline p { input x:fp32[32,64] a,b=split(x) out=add(a,b) output out }"


def test_only_cuda_declares_a_validated_layout_view_contract():
    assert PLATFORMS["cuda"].layout_views is True
    assert PLATFORMS["cuda-tuned"].layout_views is True
    assert PLATFORMS["ascend"].layout_views is False
    assert PLATFORMS["ascend-wide"].layout_views is False
    assert PLATFORMS["kunlunxin"].layout_views is False


def test_platform_default_uses_the_capability_not_the_name():
    cuda = compile_source(SPLIT).pipelines[0]
    assert len(cuda.triton.kernels) == 1
    kunlunxin = compile_source(SPLIT, platform_config=resolve_platform("kunlunxin")).pipelines[0]
    assert all(not group.indexed for group in kunlunxin.fusion.groups)


def test_explicit_layout_views_on_an_unvalidated_target_fails_closed():
    with pytest.raises(CompileError, match="no validated layout-view fusion contract"):
        compile_source(SPLIT, platform_config=resolve_platform("kunlunxin"),
                      layout_views=True)
    # explicit off is still the supported materialising path
    pipeline = compile_source(SPLIT, platform_config=resolve_platform("kunlunxin"),
                              layout_views=False).pipelines[0]
    assert all(not group.indexed for group in pipeline.fusion.groups)


def test_experimental_override_is_explicit_and_not_a_platform_default():
    # The diagnostic build path may emit a non-CUDA indexed region deliberately;
    # the platform capability itself stays False.
    config = resolve_platform(
        "kunlunxin",
        allowed_kernel_kinds=("pointwise", "rowwise", "primitive", "layout",
                              "transpose", "rope"),
        indexed_rows=1, experimental_region_fusion=True)
    pipeline = compile_source(SPLIT, platform_config=config,
                              layout_views="experimental").pipelines[0]
    assert any(group.indexed for group in pipeline.fusion.groups)
    assert PLATFORMS["kunlunxin"].layout_views is False


def test_kunlun_indexed_reduction_uses_a_rank_one_tile():
    from flowgpu.compiler import compile_source
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    config = resolve_platform(
        "kunlunxin",
        allowed_kernel_kinds=("pointwise", "rowwise", "primitive", "layout",
                              "transpose", "rope"),
        indexed_rows=1, experimental_region_fusion=True)
    source = compile_source(
        "pipeline p { input x:fp16[R,128] input inv:fp32 "
        "xr=reshape(x,[R,128]) mx=max_last(abs(xr)) total=max_axis(mx,0) "
        "s=mul(total,inv) output s }",
        platform_config=config, layout_views="experimental").pipelines[0]
    kernel = source.triton.kernels[0].source
    assert "tl.max(reduce_tile, 1)" in kernel
    assert "tl.max(reduce_tile, 0)" not in kernel


def test_concat_operand_addresses_are_clamped_for_masked_lanes():
    from flowgpu.compiler import compile_source
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    config = resolve_platform(
        "kunlunxin",
        allowed_kernel_kinds=("pointwise", "rowwise", "primitive", "layout",
                              "transpose", "rope"),
        indexed_rows=1, experimental_region_fusion=True)
    source = compile_source(
        "pipeline p { input a:fp16[R,64] input b:fp16[R,64] input inv:fp32 "
        "xr=concat(a,b) mx=max_last(abs(xr)) total=max_axis(mx,0) "
        "s=mul(total,inv) output s }",
        platform_config=config, layout_views="experimental").pipelines[0]
    kernel = source.triton.kernels[0].source
    assert "tl.minimum(col_2" in kernel and "tl.maximum((col_2)" in kernel


def test_concat_of_the_two_split_parts_folds_to_the_source():
    from flowgpu.compiler import compile_source
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    config = resolve_platform(
        "kunlunxin",
        allowed_kernel_kinds=("pointwise", "rowwise", "primitive", "layout",
                              "transpose", "rope"),
        indexed_rows=1, experimental_region_fusion=True)
    source = compile_source(
        "pipeline p { input x:fp16[R,128] input inv:fp32 a,b=split(x) "
        "xr=concat(a,b) mx=max_last(abs(xr)) total=max_axis(mx,0) "
        "s=mul(total,inv) output s }",
        platform_config=config, layout_views="experimental").pipelines[0]
    kernel = source.triton.kernels[0].source
    assert "split_index" not in kernel
