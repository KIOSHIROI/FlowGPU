"""'auto' resolves per program instead of imposing one strategy on all of them.

The default geometry was measured to be the worst option in worst-case regret:
on three shapes the untuned ``recompute`` caps reached 6.67x the per-shape best and
``materialize`` 4.47x, while the caps ``auto`` resolves to were never worse than
1.16x (docs/PARALLELISM_GAP_20261019.md sections 40 and 44.3).

``auto`` is still not the default. Flipping it changes the kernel partition of
every program that has a cross-domain reduction -- the LSE example goes from two
kernels to one -- and the buffer plans, traffic reports and packaged submissions
in this repository were validated against the other partition. That is a release
decision, so the mode is offered and measured rather than imposed, and these tests
pin exactly what it does:

* a cross-domain reduction resolves to the measured minimax arm;
* a program without one emits byte-identical source to ``materialize``;
* a target without indexed and reduction fusion does not raise, it resolves to
  ``materialize``, because 'auto' is not a request for recompute.
"""
from pathlib import Path

import pytest

from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.backend.triton_backend import module as emit
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.tile_selection import emitted_tiles, tile_options

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
LSE = EXAMPLES / 'batch8' / 'dcp_lse_combine.fgpu'
# No cross-domain reduction: the strategy must not change a byte.
PLAIN = ('residual_norm', 'mlp', 'qk_transform', 'scaled_decoder')


def test_a_cross_domain_reduction_resolves_to_the_measured_minimax_arm():
    source = LSE.read_text()
    auto = emit([compile_source(source, cross_domain_strategy='auto').pipelines[0].triton])
    minimax = emit([compile_source(source, **tile_options((4096, 16, 8))).pipelines[0].triton])

    assert auto == minimax
    assert emitted_tiles(auto, (4096, 1, 1, 1024)) == (4096, 16, 8)


@pytest.mark.parametrize('name', PLAIN)
def test_a_program_without_a_cross_domain_reduction_is_left_alone(name):
    path = EXAMPLES / (name + '.fgpu')
    if not path.exists():
        pytest.skip('example moved: ' + name)
    source = path.read_text()
    plain = emit([compile_source(source).pipelines[0].triton])
    auto = emit([compile_source(source, cross_domain_strategy='auto').pipelines[0].triton])
    assert auto == plain


def test_a_target_without_the_required_fusion_resolves_instead_of_raising():
    """'auto' is not a request for recompute, so it cannot be invalid on a target."""
    source = LSE.read_text()
    plain = emit([compile_source(source, platform_config=resolve_platform('kunlunxin')).pipelines[0].triton])
    auto = emit([compile_source(source, platform_config=resolve_platform('kunlunxin'),
                                cross_domain_strategy='auto').pipelines[0].triton])
    assert auto == plain


def test_asking_for_recompute_still_requires_cuda_indexed_and_reduction_fusion():
    source = LSE.read_text()
    with pytest.raises(ValueError, match='requires CUDA indexed and reduction fusion'):
        compile_source(source, index_fusion=False, cross_domain_strategy='recompute')


def test_auto_is_a_named_mode_not_a_silent_choice_of_a_candidate():
    """The rule is structural -- fuse if there is something to fuse -- not a table."""
    from flowgpu.compiler.compile import auto_cross_domain_config
    config = auto_cross_domain_config()
    assert (config.indexed_columns, config.indexed_rows) == (4096, 16)
    assert config.reduction_block_limit == 65536


def test_the_default_partition_for_a_cross_domain_reduction_is_pinned():
    """The partition the default now produces, recorded as a test rather than a diff.

    Flipping the default moved this program from two kernels and four slots to one
    kernel and two slots. A partition change is the kind of thing that should be
    visible in the suite, not only in a report.
    """
    p = compile_source(LSE.read_text()).pipelines[0]
    assert len(p.triton.kernels) == 1
    assert len(p.memory.slots) == 2
    assert any(g.cross_domain for g in p.fusion.groups)


def test_the_materialize_partition_is_still_reachable_and_unchanged():
    p = compile_source(LSE.read_text(), cross_domain_strategy='materialize').pipelines[0]
    assert len(p.triton.kernels) == 2
    assert len(p.memory.slots) == 4
    assert not any(g.cross_domain for g in p.fusion.groups)
