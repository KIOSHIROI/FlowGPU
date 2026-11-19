"""Program-index width is a platform policy with a host-enforced range proof.

The Task160 evidence: on the same module and the same seven chips, the
CUDA-class targets read 4-11x while 昆仑芯/华为 read 1.49x / 0.78x. The device
work and the host path are identical across the seven; the emitted kernel
derives every address from one program index, and on those two targets 64-bit
index arithmetic is the cost. So the index dtype becomes a policy field:

* cuda keeps int64 -- the emitted source of the already-scoring chips must not
  change at all;
* ascend/kunlunxin ask for int32, but a 32-bit index is only emitted together
  with a bound the host enforces from runtime shape/stride metadata, and an
  unbounded kernel stays int64 rather than run unchecked.

These tests execute the emitted source on the CPU double in
``test/segmented_support.py`` for the int32/int64 equality checks; that double
does not model integer overflow, which is exactly why the host bound has its own
tests here.
"""
import math

import pytest

from compiler.backend.platform_configs import (
    MAX_INDEX32, PlatformConfig, emitted_index_dtype, resolve_platform)
from compiler.backend.triton_backend import module
from compiler.compile import compile_source
from test import segmented_support
from test.index_width_support import (
    SWIGLU, evaluate_index_bound, execute_module, execute_module_expect_error,
    int32_index_line, swiglu_inputs)


def config_for(name, **overrides):
    """A profile that can emit the value-layer/indexed lowering under test."""
    options = dict(allowed_kernel_kinds=('pointwise', 'primitive', 'layout'),
                   # Same explicit builder choice as the Task160 package: the
                   # one-kernel value-layer region is what the submission ships.
                   experimental_region_fusion=(name != 'cuda'))
    if name.startswith('ascend'):
        options.update(mapping_strategy='direct_grid', device_cores=48)
    options.update(overrides)
    return resolve_platform(name, **options)


def program(name='cuda', source=SWIGLU, **overrides):
    return compile_source(source, platform_config=config_for(name, **overrides)
                          ).pipelines[0].triton


# --- policy resolution ------------------------------------------------------

def test_the_default_table_keeps_cuda_class_targets_on_int64():
    assert resolve_platform('cuda').index_dtype == 'int64'
    for name in ('ascend', 'ascend-910b'):
        assert resolve_platform(name, device_cores=48).index_dtype == 'int32'
    for name in ('kunlunxin', 'kunlun-r300'):
        assert resolve_platform(name).index_dtype == 'int32'


def test_index_dtype_override_wins_and_unknown_values_are_rejected():
    assert resolve_platform('cuda', index_dtype='auto').index_dtype == 'auto'
    assert resolve_platform('kunlunxin', index_dtype='int64').index_dtype == 'int64'
    for value in ('int16', 'wide', ''):
        with pytest.raises(ValueError, match='index_dtype'):
            resolve_platform('cuda', index_dtype=value)
        with pytest.raises(ValueError, match='index_dtype'):
            PlatformConfig('probe', index_dtype=value)


def test_int32_is_never_selected_without_a_proof():
    """'int32'/'auto' ask for 32-bit arithmetic; only a bound may grant it."""
    kunlun = resolve_platform('kunlunxin')
    assert emitted_index_dtype(kunlun, proven=False) == 'int64'
    assert emitted_index_dtype(kunlun, proven=True) == 'int32'
    cuda = resolve_platform('cuda')
    assert emitted_index_dtype(cuda, proven=True) == 'int64'
    assert emitted_index_dtype(cuda, proven=False) == 'int64'
    auto = resolve_platform('cuda', index_dtype='auto')
    assert emitted_index_dtype(auto, proven=True) == 'int32'


# --- the Task160-shaped kernel ---------------------------------------------

def test_the_task160_cuda_build_keeps_the_unchanged_int64_index():
    p = program('cuda')
    kernel = p.kernels[0]
    assert kernel.index_dtype == 'int64'
    assert kernel.index_bound is None
    assert int32_index_line(p) == (
        'idx = tl.program_id(0).to(tl.int64) * FG_BLOCK + tl.arange(0, FG_BLOCK)')
    assert '_fg_index_bound' not in p.host
    assert 'int32 index bound' not in p.host


def test_the_task160_foreign_builds_emit_int32_with_a_host_bound():
    """Both slow targets keep the policy; the *shape* of the bound differs.

    昆仑芯 emits the flat index (this test pins its exact form); 华为 emits the
    2-D tile form, whose bound has its own expression and is pinned in
    ``test_vector_tile_2d.py``. What is asserted here is the part both share:
    int32 is never emitted without a host-enforced address bound.
    """
    for platform in ('ascend', 'kunlunxin'):
        p = program(platform)
        kernel = p.kernels[0]
        assert kernel.index_dtype == 'int32', platform
        assert kernel.index_bound is not None, platform
        assert kernel.index_bound.leaves == ('gate', 'up'), platform
        assert '_fg_index_bound = max(' in p.host, platform
        assert 'if _fg_index_bound > 2147483647:' in p.host, platform
        assert 'int32 index bound exceeded' in p.host, platform
    flat = program('kunlunxin').kernels[0]
    assert '.to(tl.int32) * FG_BLOCK' in int32_index_line(program('kunlunxin'))
    # The proof names the iteration space and every leaf the host must bound.
    assert flat.index_bound.count == 'M * N'
    # the masked tail of one block; 昆仑芯's block is its measured curve peak
    assert flat.index_bound.padding == config_for('kunlunxin').default_block_size - 1
    assert flat.index_bound.padding == 2047


def test_an_unprovable_kernel_stays_on_int64():
    """A paired-view read takes an address the one-dimensional proof excludes."""
    p = program('kunlunxin', paired_view_loads=True)
    assert p.kernels[0].index_dtype == 'int64'
    assert p.kernels[0].index_bound is None
    assert '.to(tl.int64) * FG_BLOCK' in int32_index_line(p)
    assert '_fg_index_bound' not in p.host


def test_the_policy_can_be_turned_off_per_platform():
    p = program('kunlunxin', index_dtype='int64')
    assert p.kernels[0].index_dtype == 'int64'
    assert '_fg_index_bound' not in p.host


def test_a_pointwise_cuda_kernel_keeps_its_natural_32_bit_index():
    """The pointwise/rowwise lowerings already index in 32 bits.

    Their source is not rewritten to int64 for cuda because that would perturb
    the targets that currently score; the policy only replaces a hard-coded
    int64, and there is none on that path.
    """
    source = ('pipeline point { input x:fp32[M,N] input y:fp32[M,N] '
              't=add(x,y) z=mul(t,y) output z }')
    p = compile_source(source, platform_config=resolve_platform('cuda')
                       ).pipelines[0].triton
    line = next(line.strip() for line in p.kernels[0].source.splitlines()
                if 'tl.program_id(0)' in line)
    assert line == 'offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)'
    assert '_fg_index_bound' not in p.host


# --- the bound itself -------------------------------------------------------

def test_the_emitted_bound_accepts_the_largest_safe_extent():
    """The check is evaluated exactly as emitted, at the acceptance boundary.

    A [2, 1] view with row stride 2**31-1 derives addresses up to exactly
    ``MAX_INDEX32`` (masked tail and element count are smaller), so the host
    must accept it; the first unsafe stride, 2**31, must be rejected.
    """
    host = program('kunlunxin').host
    safe = evaluate_index_bound(host, dimensions={'M': 2, 'N': 1},
                                strides={'gate': (MAX_INDEX32, 0),
                                         'up': (MAX_INDEX32, 0)})
    assert safe == MAX_INDEX32
    with pytest.raises(ValueError, match='int32 index bound exceeded'):
        evaluate_index_bound(host, dimensions={'M': 2, 'N': 1},
                             strides={'gate': (MAX_INDEX32 + 1, 0),
                                      'up': (MAX_INDEX32 + 1, 0)})


def test_the_bound_counts_the_strided_extent_not_the_element_count():
    """Task160's gate/up are column views: about twice their element count.

    M=8192, N=2048 sliced out of a [M, 2N] packed buffer gives row stride
    4096 and column stride 1, so the largest derived address is
    (M-1)*4096 + (N-1) = 33,552,383 -- not M*N = 16,777,216.
    """
    host = program('kunlunxin').host
    bound = evaluate_index_bound(
        host, dimensions={'M': 8192, 'N': 2048},
        strides={'gate': (4096, 1), 'up': (4096, 1)})
    assert bound == (8192 - 1) * 4096 + (2048 - 1)
    assert bound == 33_552_383
    assert bound > 8192 * 2048
    assert bound <= MAX_INDEX32
    # The int64 form has no bound to evaluate at all: its addresses are wide.
    assert '_fg_index_bound' not in program('cuda').host


def test_a_contiguous_view_is_bounded_by_its_element_count_not_more():
    host = program('kunlunxin').host
    bound = evaluate_index_bound(
        host, dimensions={'M': 8192, 'N': 2048},
        strides={'gate': (2048, 1), 'up': (2048, 1)})
    # element count plus the masked tail of one block (昆仑芯's curve peak 2048)
    assert config_for('kunlunxin').default_block_size == 2048
    assert bound == 8192 * 2048 + 2047


# --- the rejection path runs before the launch -----------------------------

def test_the_host_rejects_an_unsafe_view_before_any_launch():
    p = program('cuda', index_dtype='int32')
    m, n = 2, 2
    storage = [float(i) for i in range(m * n)]
    unsafe = segmented_support.Buffer(storage, (m, n), 'fp16', (MAX_INDEX32, 1))
    launches = execute_module_expect_error(
        module([p]), (unsafe, unsafe, 0.5, 0.1, 4.0, -4.0), 'swiglu',
        'int32 index bound exceeded')
    # Nothing was launched: the bound rejects the call before any kernel runs.
    assert launches == []


def test_the_safe_path_runs_and_matches_the_int64_form():
    inputs = swiglu_inputs()
    wide, wide_launches = execute_module(module([program('cuda')]), inputs, 'swiglu')
    narrow, narrow_launches = execute_module(
        module([program('cuda', index_dtype='int32')]), inputs, 'swiglu')
    assert wide_launches == narrow_launches == [('swiglu_k0', (1,))]
    assert wide == narrow
    assert all(value is not None for value in wide['out'])


def test_the_rebuilt_int32_module_runs_the_same_values_as_the_int64_module():
    """End to end through the *compacted* module the builder would ship."""
    inputs = swiglu_inputs(m=3, n=5)
    wide = compile_source(SWIGLU, platform_config=config_for('cuda')).pipelines[0].triton
    narrow = compile_source(SWIGLU, platform_config=config_for(
        'cuda', index_dtype='int32')).pipelines[0].triton
    wide_values, _ = execute_module(module([wide]), inputs, 'swiglu')
    narrow_values, _ = execute_module(module([narrow]), inputs, 'swiglu')
    assert wide_values == narrow_values


# --- every lowering that now emits the policy dtype ------------------------

def test_the_indexed_reduction_path_keeps_identical_values():
    source = ('pipeline red { input x:fp32[M,N] e=exp(x) y=sum_last(e) output y }')
    inputs = (segmented_support.Buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                       (2, 3), 'fp32', (3, 1)),)
    wide = compile_source(source, platform_config=config_for('cuda')).pipelines[0].triton
    narrow = compile_source(source, platform_config=config_for(
        'cuda', index_dtype='int32')).pipelines[0].triton
    assert '.to(tl.int64)' in wide.kernels[0].source
    assert '.to(tl.int32)' in narrow.kernels[0].source
    wide_values, _ = execute_module(module([wide]), inputs, 'red')
    narrow_values, _ = execute_module(module([narrow]), inputs, 'red')
    assert wide_values == narrow_values
    oracle = [math.exp(1.0) + math.exp(2.0) + math.exp(3.0),
              math.exp(4.0) + math.exp(5.0) + math.exp(6.0)]
    for got, want in zip(wide_values['y'], oracle):
        assert got == pytest.approx(want, rel=1e-4)


def test_the_cross_domain_path_keeps_identical_values():
    from pathlib import Path
    source = Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    bindings = {'R': 2, 'B': 1, 'H': 2, 'D': 3}
    recv = segmented_support.Buffer(
        [float(i) * 0.1 for i in range(12)], (2, 1, 2, 3), 'fp16', (6, 6, 3, 1))
    lse = segmented_support.Buffer([0.5, -0.5, 1.0, -1.0], (2, 1, 2), 'fp32', (2, 2, 1))
    wide = compile_source(source, platform_config=config_for('cuda'),
                          shape_bindings=bindings).pipelines[0].triton
    narrow = compile_source(source, platform_config=config_for('cuda', index_dtype='int32'),
                            shape_bindings=bindings).pipelines[0].triton
    assert '.to(tl.int64)' in wide.kernels[0].source
    assert '.to(tl.int32)' in narrow.kernels[0].source
    wide_values, _ = execute_module(module([wide]), (recv, lse), 'lse')
    narrow_values, _ = execute_module(module([narrow]), (recv, lse), 'lse')
    assert wide_values == narrow_values
