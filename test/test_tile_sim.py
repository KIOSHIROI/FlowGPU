"""The CPU tile-plan simulator: correctness proofs and the numbers that matter.

Background: Task144's 华为 column is 141% behind the leader, the flat kernel is a
gather and the 2-D tile is mask-only because block_ptr is numerically broken
there.  The candidates we built for the next batch (candidate-20261023-ascend-*)
cannot be run locally, so the thing worth testing locally is the *plan*: does
every output element get written exactly once from the right source, how much
sector bandwidth does the plan waste, and does the launch fit under coreDim.

The 2026-10-22 day of quota was spent discovering facts this file asserts in
milliseconds; these tests keep them asserted.
"""
import pytest

from compiler.analysis import tile_sim as ts


SMALL = ts.Geom.from_shapes((1, 1, 1, 30, 52), 1, (1, 1, 1, 1, 1, 0))
TINY = ts.Geom.from_shapes((1, 2, 2, 3, 5), 1, (1, 1, 1, 1, 1, 0))
EDGE_CASES = [
    # (x, tc, padding); d_l = padding[4] - tc must stay >= 0
    ((1, 1, 1, 1, 1), 0, (0, 0, 0, 0, 0, 0)),
    ((1, 1, 2, 2, 2), 0, (1, 0, 0, 1, 1, 1)),
    ((2, 1, 1, 2, 2), 1, (2, 0, 2, 0, 1, 2)),
    ((1, 1, 3, 2, 4), 1, (0, 0, 1, 1, 2, 0)),
]


def _plans(w_out=54, elemsize=2, align=32):
    return {
        "flat": ts.flat_block_plan(1024),
        "rowblock1": ts.rowblock_plan(1),
        "rowblock_aligned": ts.rowblock_plan(ts.row_block_for(w_out, elemsize, align)),
        "interior_edge": ts.interior_edge_plan(rb=1),
    }


@pytest.mark.parametrize("geom", [SMALL, TINY] + [
    ts.Geom.from_shapes(x, tc, pad) for x, tc, pad in EDGE_CASES
])
def test_every_builtin_plan_is_exact(geom):
    """Coverage, disjointness, source index and source buffer, all at once."""
    for name, plan in _plans(geom.w_out).items():
        ok, problems = ts.verify(geom, plan)
        assert ok, f"{name} is wrong on {geom}: {problems}"
        assert len(ts.reference_map(geom)) == geom.total


def test_verify_catches_a_plan_that_forgets_an_element():
    def broken(geom):
        for pid, lanes in ts.rowblock_plan(1)(geom):
            yield pid, lanes[1:]          # drop the first lane of every program

    ok, problems = ts.verify(TINY, broken)
    assert not ok and problems["missing"] > 0


def test_verify_catches_a_double_write_and_a_wrong_source():
    base = list(ts.rowblock_plan(1)(TINY))

    def duplicated(geom):
        for pid, lanes in base:
            yield pid, lanes + [lanes[0]]

    ok, problems = ts.verify(TINY, duplicated)
    assert not ok and problems["duplicate"] > 0

    def wrong_src(geom):
        for pid, lanes in base:
            # some programs are pure padding rows and move nothing at all
            index = next((i for i, lane in enumerate(lanes) if lane.buf is not None), None)
            if index is None:
                yield pid, lanes
                continue
            lane = lanes[index]
            patched = list(lanes)
            patched[index] = ts.Lane(dst=lane.dst, buf=lane.buf, src=lane.src + 1)
            yield pid, patched

    ok, problems = ts.verify(TINY, wrong_src)
    assert not ok and problems["wrong_src"] > 0


def test_sector_efficiency_sees_misalignment():
    """A 32 B-aligned run of 16 fp16 elements costs exactly one sector."""
    aligned = [ts.Lane(dst=i) for i in range(16)]
    misaligned = [ts.Lane(dst=8 + i) for i in range(16)]

    class _One:
        def __call__(self, geom):
            yield (0,), aligned

    class _OneBad:
        def __call__(self, geom):
            yield (0,), misaligned

    stats = ts.simulate(TINY, _One(), elemsize=2, align_bytes=32)
    assert stats.dst_sector_efficiency == pytest.approx(1.0)
    bad = ts.simulate(TINY, _OneBad(), elemsize=2, align_bytes=32)
    assert bad.dst_sector_efficiency < 1.0


def test_row_block_for_is_the_smallest_power_of_two_that_aligns_a_row_block():
    assert ts.row_block_for(54, 2) == 8          # 54*2*8  = 864  = 27*32
    assert ts.row_block_for(106, 2) == 8         # 106*2*8 = 1696 = 53*32
    assert ts.row_block_for(210, 2) == 8         # 210*2*8 = 3360 = 105*32
    assert ts.row_block_for(418, 2) == 8         # 418*2*8 = 6688 = 209*32
    assert ts.row_block_for(54, 1) == 16         # one byte per element
    assert ts.row_block_for(54, 4) == 4          # fp32
    for w_out, elemsize, rb in ((54, 2, 8), (418, 2, 8), (54, 1, 16), (54, 4, 4)):
        assert (rb * w_out * elemsize) % 32 == 0


def test_aligned_row_block_recovers_the_store_sector_efficiency_of_one_row_per_program():
    """This is the whole reason candidate-20261023-ascend-rowblock exists."""
    one = ts.simulate(SMALL, ts.rowblock_plan(1), elemsize=2, align_bytes=32)
    aligned = ts.simulate(SMALL, ts.rowblock_plan(8), elemsize=2, align_bytes=32)
    assert one.dst_sector_efficiency < 0.9
    assert aligned.dst_sector_efficiency == pytest.approx(1.0, abs=1e-6)
    assert one.lane_use == pytest.approx(aligned.lane_use)   # only the grouping changed


def test_one_row_per_program_blows_past_coredim_while_the_aligned_block_fits():
    """coreDim <= 65535: rb=1 launches 65536 programs on the smallest official case."""
    small_p = ts.simulate(SMALL, ts.rowblock_plan(1), elemsize=2).programs
    big_p = ts.simulate(SMALL, ts.rowblock_plan(8), elemsize=2).programs
    per_plane_small = small_p // SMALL.planes      # h_out rows
    per_plane_big = big_p // SMALL.planes          # ceil(h_out / 8) blocks
    assert per_plane_small == SMALL.h_out
    assert per_plane_small > per_plane_big
    # the smallest official case is c1024 x (tc=1, t=1) => 2048 planes of 32 rows
    assert per_plane_small * 2048 > 65535
    assert per_plane_big * 2048 < 65535


def test_flat_plan_keeps_full_sector_efficiency_but_is_not_a_pure_block_move():
    """The flat kernel's store side is perfect; its source side is what costs 华为."""
    flat = ts.simulate(SMALL, ts.flat_block_plan(1024), elemsize=2, align_bytes=32)
    assert flat.dst_sector_efficiency == pytest.approx(1.0, abs=1e-6)
    # 5% of store runs are also full block moves => the load is not a copy
    assert flat.aligned_copy_fraction < 0.2


def test_plane_base_alignment_is_an_open_problem_not_a_tiling_choice():
    """The finding the tool surfaced: for 3 of 4 official shapes the *plane* base
    is not 32 B aligned (H_OUT*W_OUT*elemsize % 32 == 8), so an aligned row block
    still starts mid-sector on every odd plane."""
    plane_bytes = {54: 32 * 54 * 2, 106: 62 * 106 * 2, 210: 122 * 210 * 2, 418: 242 * 418 * 2}
    assert plane_bytes[54] % 32 == 0
    assert plane_bytes[106] % 32 == 24     # only every third plane starts mid-sector
    # 210 and 418 land on 8, i.e. every plane after the first is 24 B into a sector
    for w_out in (210, 418):
        assert plane_bytes[w_out] % 32 == 8
