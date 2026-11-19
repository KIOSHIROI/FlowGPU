"""Reduction domain coverage: which split arrangements are legal.

The emitter contract is one program per reduction domain. These tests pin the
classifier that names the arrangements, and one structural check that the
rowwise emitter really sweeps rather than splitting the axis across programs.
"""
import pytest

from flowgpu.compiler.analysis.reduction_coverage import (
    ILLEGAL,
    ONLINE,
    SINGLE_PROGRAM,
    TWO_STAGE,
    classify,
    group_coverage,
    require_legal,
)
from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.compile import compile_source

RMS = """pipeline rms {
    input x : bf16[B,D]
    input weight : bf16[D]
    y = rmsnorm_hf(x, weight)
    output y
}"""


def test_a_single_program_owns_the_domain_and_needs_no_merge():
    coverage = classify(programs_along_domain=1)
    assert coverage.strategy == SINGLE_PROGRAM and coverage.legal
    assert 'sweep' in coverage.reason


def test_a_split_without_a_merge_contract_is_illegal():
    coverage = classify(programs_along_domain=4)
    assert coverage.strategy == ILLEGAL and not coverage.legal
    assert 'denominator' in coverage.reason and 'merge' in coverage.reason


def test_a_split_becomes_legal_once_a_merge_is_proven():
    assert classify(programs_along_domain=4, merge_proven=True).strategy == TWO_STAGE
    assert classify(programs_along_domain=4, online=True).strategy == ONLINE
    # A proven merge takes precedence over the online form.
    assert classify(programs_along_domain=4, merge_proven=True, online=True).strategy == TWO_STAGE


def test_require_legal_raises_with_the_arrangement_reason():
    require_legal(classify(programs_along_domain=1))
    with pytest.raises(ValueError, match='merge contract'):
        require_legal(classify(programs_along_domain=8))


def test_nonsense_program_counts_are_rejected():
    for bad in (0, -1, 1.0, True, '2'):
        with pytest.raises(ValueError, match='programs_along_domain must be an integer of at least one'):
            classify(programs_along_domain=bad)


def test_a_group_without_a_reduction_domain_has_nothing_to_cover():
    class Group:
        reduction_domain = ()
    assert group_coverage(Group()) == ()


def test_a_group_with_a_domain_reports_one_record():
    class Group:
        reduction_domain = ('B', 'D')
    (coverage,) = group_coverage(Group())
    assert coverage.strategy == SINGLE_PROGRAM
    assert coverage.as_dict()['programs_along_domain'] == 1


def test_the_rowwise_emitter_sweeps_instead_of_splitting_the_axis():
    """Structural check: one program per row, the row walked in chunks."""
    config = resolve_platform('cuda', mapping_strategy='direct_grid', block_size=1024,
                              allowed_kernel_kinds=('rowwise', 'primitive', 'pointwise'))
    (kernel,) = compile_source(RMS, platform_config=config).pipelines[0].triton.kernels
    assert kernel.kind == 'rowwise'
    # The reduction axis appears as a bounded loop inside the program ...
    assert 'for chunk in range' in kernel.source
    # ... and the launch is one dimensional: the grid counts rows, not parts of
    # a reduction domain. Parsing is exact where counting commas is not, since a
    # one-tuple also carries a comma.
    import ast
    dimensions = ast.parse(kernel.grid, mode='eval').body.elts
    assert len(dimensions) == 1, kernel.grid
    # Which is exactly the arrangement the classifier calls legal.
    assert classify(programs_along_domain=1).legal
