"""The measured-selection space must equal the geometry the emitter would emit.

``flowgpu/compiler/runtime/tile_selection.py`` decides what to try and computes each
candidate's geometry with ``tile_from_config`` rather than parsing the emitter.
That is only honest while the two agree, so these tests recover the real tiles
from the emitted launch text and compare.

If the emitter's formula changes, the selection would otherwise keep searching a
space no candidate names any more, and every measurement in it would be a number
about a configuration nobody can select -- the exact mistake that produced an
earlier candidate table in this repository.
"""
from pathlib import Path

import pytest

from flowgpu.compiler.backend.triton_backend import module as emit
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.optimization.strategy import candidates
from flowgpu.compiler.runtime.tile_selection import (
    MATERIALIZE, SPACE, assert_emitted_tiles, dimension_values, emitted_tiles, space_for,
    tile_dimensions, tile_from_config, tile_options)

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'batch8' / 'dcp_lse_combine.fgpu'
ARM_NAMES = ('recompute', 'recompute-r4096-d8-budget32768',
             'recompute-r4096-d16-budget65536', 'recompute-r2048-d16-budget65536')
SHAPES = [(4096, 1, 1, 1024), (2048, 1, 1, 64), (16384, 1, 1, 64),
          (512, 1, 2, 256), (1024, 2, 2, 128)]


@pytest.mark.parametrize('shape', SHAPES)
def test_the_selection_searches_exactly_the_geometry_that_gets_emitted(shape):
    source = EXAMPLE.read_text()
    table = {spec.name: spec for spec in candidates(recompute=True, large_tiles=True,
                                                    wide_columns=True)}
    for name in ARM_NAMES:
        if name == 'recompute':
            options = {'cross_domain_strategy': 'recompute'}
            expected = tile_from_config(None, None, None, None, shape)
        else:
            spec = table[name]
            options = spec.options()
            expected = tile_from_config(spec.columns, spec.rows, spec.budget, spec.warps, shape)
        code = emit([compile_source(source, **options).pipelines[0].triton])
        assert emitted_tiles(code, shape) == expected, name


@pytest.mark.parametrize('shape', SHAPES)
def test_a_named_geometry_turns_back_into_options_that_emit_it(shape):
    """The trip measured -> options -> emitted is what lets a choice be shipped."""
    source = EXAMPLE.read_text()
    for _, options, geometry in space_for(shape):
        if geometry is None:
            continue
        code = emit([compile_source(source, **tile_options(geometry)).pipelines[0].triton])
        assert emitted_tiles(code, shape) == geometry, geometry


def test_no_arm_names_a_geometry_another_arm_already_names():
    for shape in SHAPES:
        geometries = [g for _, _, g in space_for(shape) if g is not None]
        assert len(geometries) == len(set(geometries)), shape


def test_both_strategies_are_in_the_space_because_each_wins_somewhere():
    labels = {label for label, _, _ in space_for((16384, 1, 1, 64))}
    assert MATERIALIZE in labels
    assert any(label.startswith('recompute') for label in labels)


def test_the_default_spelling_of_the_materialize_program_is_one_arm_not_two():
    """'default' and 'materialize' are one arm, so one of them is not measured twice.

    They no longer compile to the same program: the compiler default resolves per
    program and this one resolves to recompute. What has not changed is that
    naming the materialize program twice must not create two arms.
    """
    labels = [label for label, _, _ in space_for((4096, 1, 1, 1024), names=('default', 'materialize'))]
    assert labels == [MATERIALIZE]
    source = EXAMPLE.read_text()
    assert (emit([compile_source(source, cross_domain_strategy='materialize').pipelines[0].triton])
            != emit([compile_source(source).pipelines[0].triton]))


def test_the_space_never_names_an_arm_outside_the_declared_list():
    for shape in SHAPES:
        assert {label for label, _, _ in space_for(shape)} <= set(SPACE)


def test_an_unknown_candidate_is_refused_instead_of_silently_skipped():
    with pytest.raises(ValueError, match='unknown candidate'):
        space_for((4096, 1, 1, 1024), names=('recompute-r1-d1-budget1',))


def test_a_dropped_configuration_is_a_failure_not_a_slower_candidate():
    """The invariant that closes the failure class which invalidated two tables."""
    source = EXAMPLE.read_text()
    shape = (4096, 1, 1, 1024)
    code = emit([compile_source(source, **tile_options((4096, 16, 8))).pipelines[0].triton])

    assert assert_emitted_tiles(code, shape, (4096, 16, 8)) == (4096, 16, 8)
    with pytest.raises(ValueError, match='requested tile'):
        assert_emitted_tiles(code, shape, (4096, 8, 8))


SYNTHETIC = '''
def run_lse(first, second, third):
    R = second.shape[1]
    D = third.shape[0]
    lse_k0[(1, 2)](first, second, third, R, D, min(4096, triton.next_power_of_2(R)), min(16, max(1, 65536 // (min(4096, triton.next_power_of_2(R)))), triton.next_power_of_2(D)), num_warps=8, enable_fp_fusion=False)
'''


class _Tensor:
    def __init__(self, shape):
        self.shape = shape


def test_the_reduced_dimension_is_read_from_the_source_not_from_argument_order():
    """The dimension can be declared on any input, and on any axis of it."""
    args = (_Tensor((9,)), _Tensor((3, 7)), _Tensor((11,)))

    assert dimension_values(SYNTHETIC, 'lse', args) == dict(R=7, D=11)
    assert tile_dimensions(SYNTHETIC) == ('R', 'D')
    # An argument-order guess would have answered 9 and 3.
    assert dimension_values(SYNTHETIC, 'lse', args)['R'] != args[0].shape[0]


def test_a_program_without_a_cross_domain_tile_reports_nothing_to_select():
    """Two kernels is already a refusal: selection is per kernel, not per module."""
    source = EXAMPLE.read_text()
    code = emit([compile_source(source, cross_domain_strategy='materialize').pipelines[0].triton])
    with pytest.raises(ValueError, match='expected one launched kernel'):
        tile_dimensions(code)


def test_a_single_kernel_program_with_no_tile_still_refuses():
    """A pointwise kernel launches once and has no cross-domain tile to offer."""
    source = 'pipeline p { input x : fp32[8,8] y = exp(x) output y }'
    code = emit([compile_source(source).pipelines[0].triton])
    with pytest.raises(ValueError, match='dimension|no cross-domain tile'):
        tile_dimensions(code)


def test_the_extents_space_matches_the_shape_space_where_both_apply():
    from flowgpu.compiler.runtime.tile_selection import space_for_extents
    shape = (4096, 1, 1, 1024)
    assert space_for_extents(shape[0], shape[3]) == space_for(shape)


def swapped_declaration():
    """The LSE pipeline with its two inputs declared in the other order."""
    lines = EXAMPLE.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.strip().startswith('input recv'))
    assert lines[first + 1].strip().startswith('input lse_in')
    lines[first], lines[first + 1] = lines[first + 1], lines[first]
    return '\n'.join(lines)


def test_the_metadata_follows_the_declaration_order_not_the_argument_position():
    """The case the old ``args[0].shape`` guess could not survive.

    Declaring ``lse_in`` first makes it the first parameter, so its shape has rank
    three and no axis 3. A selection that indexed ``args[0].shape`` would raise;
    reading the emitted bindings answers from the tensor that declares the
    dimension.
    """
    code = emit([compile_source(swapped_declaration()).pipelines[0].triton])
    assert tile_dimensions(code) == ('R', 'D')

    lse_first = _Tensor((4096, 1, 1))
    recv_second = _Tensor((4096, 1, 1, 1024))
    values = dimension_values(code, 'lse', (lse_first, recv_second))
    assert values == dict(R=4096, B=1, H=1, D=1024)
    with pytest.raises(IndexError, match='tuple index out of range'):
        lse_first.shape[3]


def test_both_declaration_orders_produce_the_same_selection_space():
    from flowgpu.compiler.runtime.tile_selection import space_for_extents
    original = emit([compile_source(EXAMPLE.read_text()).pipelines[0].triton])
    swapped = emit([compile_source(swapped_declaration()).pipelines[0].triton])
    here = dimension_values(original, 'lse', (_Tensor((4096, 1, 1, 1024)), _Tensor((4096, 1, 1))))
    there = dimension_values(swapped, 'lse', (_Tensor((4096, 1, 1)), _Tensor((4096, 1, 1, 1024))))
    assert space_for_extents(here['R'], here['D']) == space_for_extents(there['R'], there['D'])


def test_a_module_that_launches_several_kernels_is_refused_rather_than_guessed():
    from flowgpu.compiler.runtime.tile_selection import launched_kernel
    code = emit([compile_source(EXAMPLE.read_text(),
                                cross_domain_strategy='materialize').pipelines[0].triton])
    with pytest.raises(ValueError, match='expected one launched kernel'):
        launched_kernel(code)
