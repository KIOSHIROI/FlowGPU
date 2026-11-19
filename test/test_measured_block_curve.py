"""`measured_block_curve` decides the default block size (and nowhere else).

The curve on a platform is official-board evidence: 昆仑芯's
``((512, 0.39), (1024, 0.65), (2048, 1.0), (4096, 0.09))`` was measured on
Task144, and on Task160 the flat default 1024 read 1.58x while the curve's peak
2048 read 1.78-1.86x across 2026-10-21 submissions. This file pins the rule that
consumes it:

* ``resolve_platform`` makes the peak the default block when the caller does
  not pass ``block_size`` -- the platform's own measurement decides code
  generation instead of a builder hard-coding it;
* an explicit ``block_size=`` always wins;
* platforms without a curve are byte-for-byte unchanged (cuda's 1024 is
  correct: a 2048 block there made 天数 worse, 11.09x -> 8.36x);
* the peak rule lives in ONE helper, which ``block_candidates`` also reads, so
  the search window and the default cannot drift apart.
"""
import pytest

from compiler.backend.platform_configs import (PLATFORMS, PlatformConfig,
                                               block_candidates,
                                               measured_block_peak,
                                               resolve_platform,
                                               resolved_default_block)
from compiler.compile import compile_source

#: The resolved default after consuming the curve: only a platform with a
#: measured record moves.
RESOLVED_DEFAULTS = {
    'cuda': 1024,
    'ascend': 1024,
    'ascend-910b': 1024,
    'kunlunxin': 2048,      # the curve peak
    'kunlun-r300': 1024,    # no curve
}
MAPPING = dict(mapping_strategy='direct_grid')


def resolved(name, **overrides):
    return resolve_platform(name, **{**MAPPING, **overrides})


@pytest.mark.parametrize('name,expected', sorted(RESOLVED_DEFAULTS.items()))
def test_the_resolved_table_is_the_curve_peak_only_where_a_curve_exists(name, expected):
    assert resolved(name).default_block_size == expected


def test_kunlunxin_resolves_its_measured_peak():
    config = resolved('kunlunxin')
    assert config.measured_block_curve == (
        (512, 0.39), (1024, 0.65), (2048, 1.0), (4096, 0.09))
    assert config.default_block_size == 2048


def test_cuda_is_untouched():
    """No curve means no change; a 2048 block on 天数 was measured worse."""
    assert PLATFORMS['cuda'].measured_block_curve is None
    assert resolved('cuda').default_block_size == 1024


def test_platforms_without_a_curve_keep_their_stored_default():
    for name, stored in PLATFORMS.items():
        config = resolved(name)
        if stored.measured_block_curve is None:
            assert config.default_block_size == stored.default_block_size, name


@pytest.mark.parametrize('override', [256, 512, 1024, 2048, 4096])
def test_an_explicit_override_always_beats_the_curve(override):
    assert resolved('kunlunxin', block_size=override).default_block_size == override
    # ...including on a platform whose compiled default is not the override
    assert resolved('cuda', block_size=override).default_block_size == override
    assert resolved('ascend', device_cores=48,
                    block_size=override).default_block_size == override


def test_a_synthetic_two_point_curve_resolves_to_its_peak():
    """The rule is a function of the config, not of the platform's name.

    `PLATFORMS` is an immutable mapping, so the synthetic platform is fed to the
    same resolution helper `resolve_platform` itself calls.
    """
    base = PLATFORMS['kunlunxin']
    synthetic = PlatformConfig(
        'synthetic-xpu', device_type=base.device_type,
        allowed_kernel_kinds=('pointwise',),
        measured_block_curve=((256, 0.5), (512, 1.0)))
    assert measured_block_peak(synthetic.measured_block_curve) == 512
    assert resolved_default_block(synthetic) == 512
    assert block_candidates(synthetic) == (256, 512)
    # ...and the explicit-override path is unchanged for the real platforms.
    assert resolved('kunlunxin', block_size=256).default_block_size == 256


def test_no_curve_means_no_measured_peak():
    assert measured_block_peak(None) is None
    assert measured_block_peak(()) is None
    assert block_candidates(resolved('cuda')) == ()
    assert block_candidates(resolved('ascend')) == ()


def test_the_peak_rule_is_one_helper_shared_with_block_candidates():
    """The default and the search window must agree on where the peak is."""
    for name, stored in PLATFORMS.items():
        config = resolved(name)
        peak = measured_block_peak(config.measured_block_curve)
        candidates = block_candidates(config)
        if peak is None:
            assert candidates == ()
            assert config.default_block_size == stored.default_block_size
        else:
            assert config.default_block_size == peak
            assert peak in candidates
            # and it is the only point the curve marks as the optimum
            assert [size for size, relative in config.measured_block_curve
                    if relative == 1.0] == [peak]
    assert block_candidates(resolved('kunlunxin')) == (1024, 2048, 4096)


@pytest.mark.parametrize('curve', [
    ((512, 1.0), (1024, 1.0)),        # two peaks: ambiguous
    ((512, 0.4), (1024, 0.5)),        # no measured peak
])
def test_a_malformed_curve_cannot_be_consumed(curve):
    with pytest.raises(ValueError, match='measured_block_curve must have exactly one peak \\(relative 1\\.0\\), got'):
        measured_block_peak(curve)


SOURCE = 'pipeline p { input x:fp32[M,N] input y:fp32[M,N] z=mul(x,y) output z }'


def emitted_block(platform, **overrides):
    config = resolved(platform, **overrides)
    program = compile_source(SOURCE, platform_config=config).pipelines[0]
    host = program.triton.host
    launch = next(line for line in host.splitlines() if '_k0[' in line)
    return config.default_block_size, launch


def test_the_resolved_block_reaches_the_emitted_launch():
    """The table is not documentation: resolution changes the emitted literal.

    The emit path spells 1024 symbolically as ``BLOCK`` and any other value as a
    literal, so the resolution is visible either as the literal in the launch or
    as the symbol the module defines.
    """
    # 昆仑芯 defaults to the peak: a literal 2048 in grid and launch.
    block, launch = emitted_block('kunlunxin')
    assert block == 2048
    assert 'triton.cdiv(M * N, 2048)' in launch and ', 2048,' in launch
    # an explicit 1024 override falls back to the symbolic BLOCK spelling.
    block, launch = emitted_block('kunlunxin', block_size=1024)
    assert block == 1024
    assert 'cdiv(M * N, BLOCK)' in launch and ', BLOCK,' in launch
    # cuda has no curve: its default stays the symbolic BLOCK == 1024, while an
    # explicit 2048 would change the emitted code.
    block, launch = emitted_block('cuda')
    assert block == 1024
    assert 'cdiv(M * N, BLOCK)' in launch
    block, launch = emitted_block('cuda', block_size=2048)
    assert block == 2048
    assert 'triton.cdiv(M * N, 2048)' in launch and ', 2048,' in launch
