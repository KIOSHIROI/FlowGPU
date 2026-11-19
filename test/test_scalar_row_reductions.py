from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.backend.platform_configs import resolve_platform
import pytest

SOURCE='''pipeline p {
 input x : fp32[B,C,T]
 m=max_last(x) mb=broadcast_like(m,x)
 e=exp(sub(x,mb)) z=sum_last(e)
 output m output z
}'''

def test_explicit_single_row_reduction_uses_one_vector_axis():
    config=resolve_platform(indexed_rows=1,indexed_columns=128)
    p=compile_source(SOURCE,platform_config=config).pipelines[0]
    assert p.triton.complete
    assert len(p.triton.kernels)==1
    source=p.triton.kernels[0].source
    assert '[:, None]' not in source
    assert '[None, :]' not in source
    assert 'tl.full((), ' in source
    assert 'tl.sum(reduce_tile' in source

def test_multirow_layout_keeps_second_vector_axis():
    config=resolve_platform(indexed_rows=4,indexed_columns=128)
    source=compile_source(SOURCE,platform_config=config).pipelines[0].triton.kernels[0].source
    assert '[:, None]' in source
    assert '[None, :]' in source

def test_single_row_coordinate_shortcut_also_stays_rank_one():
    config=resolve_platform(indexed_rows=1,indexed_columns=128,coalesce_reduction_sweeps=True)
    source=compile_source(SOURCE,platform_config=config).pipelines[0].triton.kernels[0].source
    assert '[None, :]' not in source
    assert 'row_coordinate' in source

def test_autotuning_can_vary_rows_even_with_single_row_starting_config():
    config=resolve_platform(indexed_rows=1,indexed_columns=128)
    kernel=compile_source(SOURCE,platform_config=config,autotune_fold=True).pipelines[0].triton.kernels[0]
    assert '@triton.autotune' in kernel.source
    assert '[:, None]' in kernel.source
    assert '[None, :]' in kernel.source
    assert 'META' in kernel.grid

def test_opt_in_result_layout_is_carried_by_low_level_reduction_abi():
    config=resolve_platform(indexed_rows=1,indexed_columns=128,strided_reduction_outputs=True)
    kernel=compile_source(SOURCE,platform_config=config).pipelines[0].triton.kernels[0]
    assert 'm_stride0' in kernel.source and 'z_stride0' in kernel.source
    assert 'm_address' in kernel.source and 'z_address' in kernel.source
    assert any(arg.kind=='stride' and arg.text=='m:0' for arg in kernel.launch)

def test_vendor_resource_option_is_explicit_and_target_checked():
    with pytest.raises(ValueError,match='explicit Kunlun'):
        resolve_platform(xpu_buffer_size_limit=2048)
    config=resolve_platform('kunlunxin',mapping_strategy='direct_grid',allowed_kernel_kinds=('primitive','pointwise','layout'),
        indexed_rows=1,indexed_columns=128,strided_reduction_outputs=True,xpu_buffer_size_limit=2048,
        experimental_region_fusion=True)
    program=compile_source(SOURCE,platform_config=config).pipelines[0].triton
    assert program.complete
    kernel=program.kernels[0]
    assert ('buffer_size_limit',2048) in kernel.launch_options

def test_unsupported_result_layout_emitter_refuses_instead_of_ignoring_flag():
    config=resolve_platform('kunlunxin',mapping_strategy='direct_grid',allowed_kernel_kinds=('primitive',),
        indexed_rows=1,indexed_columns=128,strided_reduction_outputs=True)
    program=compile_source(SOURCE,platform_config=config).pipelines[0].triton
    assert not program.complete
    assert any('strided reduction outputs require' in msg for msg in program.messages())

def test_visible_max_is_published_before_dependent_exponential_sweep():
    config=resolve_platform(indexed_rows=1,indexed_columns=128,strided_reduction_outputs=True,eager_row_result_stores=True)
    source=compile_source(SOURCE,platform_config=config).pipelines[0].triton.kernels[0].source
    assert source.index('tl.store(m_ptr') < source.index('tl.exp(')
    assert source.count('tl.store(m_ptr')==1
    assert source.count('tl.store(z_ptr')==1

def test_default_keeps_deferred_publication():
    config=resolve_platform(indexed_rows=1,indexed_columns=128)
    source=compile_source(SOURCE,platform_config=config).pipelines[0].triton.kernels[0].source
    assert source.index('tl.exp(') < source.index('tl.store(m_ptr')
