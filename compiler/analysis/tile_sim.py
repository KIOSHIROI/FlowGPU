"""tile_sim -- score a tiling plan on the CPU, before spending device or quota.

Why this module exists
----------------------
On 2026-10-22 the Task144 work spent a day of competition quota learning things
that were computable locally in milliseconds:

* the flat kernel writes every element exactly once but its *load* is a gather,
  which triton-ascend names "discrete memory access and inefficient scalar
  mapping" and which measured 0.07x on 华为;
* the 2-D tile fixes the mapping (0.26x) but Task144's output rows are
  108/212/420/836 bytes, so no row start is 32 B aligned;
* a program block of RB rows is aligned only when RB * W_OUT * elemsize is a
  multiple of 32, which for even W_OUT means RB = 8 (fp16), 16 (int8), 4 (fp32).

None of that needs a device.  What it needs is a model of *what each program
touches*.  This module is that model: a plan enumerates, for a proxy of the real
geometry, the (dst offset, src buffer + offset) of every lane it would execute;
the simulator then reports

    correctness    every dst element written exactly once, from the right source
    access shape   consecutive lanes with dst+1 and src+1 are one BLOCK run; dst
                   contiguous with a jumping source is a SCATTER store, the
                   reverse is a GATHER load
    sector cost    distinct 32 B sectors touched versus useful bytes -- this is
                   where misalignment becomes a number
    alignment      fraction of run starts that are aligned for a given chip
    grid           programs launched, which is the quantity coreDim caps

Plans are plain callables, so a new kernel costs one function, not a new tool.
`verify` makes the correctness half a test rather than a code review.

This is deliberately not a performance model.  It says "this plan issues a
gather" or "84% of the sectors it touches are useful", not "this will run in X
microseconds".  Those first two are the failure modes we actually hit.

Example
-------
    geom = Geom.from_shapes(x=(1, 256, 4, 240, 416), tc=1, padding=(1, 1, 1, 1, 1, 0))
    plan = rowblock_plan(rb=8)
    ok, problems = verify(geom, plan)
    stats = simulate(geom, plan, elemsize=2, align_bytes=32).as_dict()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

# Access-shape vocabulary, matching compiler/analysis/access_shape.
BLOCK = "block"      # dst and src both advance by 1 -> one vectorisable copy
SCATTER = "scatter"  # dst contiguous, src jumps
GATHER = "gather"    # src contiguous, dst jumps
STRIDED = "strided"  # constant stride > 1 on both sides
UNKNOWN = "unknown"  # anything else (non-affine per-element addressing)

SECTOR_BYTES = 32

# Default vector-access alignment per chip, in bytes.  Overridable: the analysis
# layer must not depend on the backend policy table, so the tool layer passes the
# real values in (tools/simulate_tile_plan.py reads platform_configs).
CHIP_ALIGN_BYTES = {
    "ascend": 32,      # Ascend vector unit: 32 B; cube-vector fusion: 512 B
    "kunlunxin": 32,
    "cuda": 16,        # 128-bit loads
    "card_a": 16,
    "card_b": 16,
}


@dataclass(frozen=True)
class Geom:
    """A copy-with-fill geometry: concat(cache, x) along time, then pad."""

    b: int
    c: int
    t: int
    h: int
    w: int
    tc: int
    w_l: int = 0
    w_r: int = 0
    h_t: int = 0
    h_b: int = 0
    d_l: int = 0
    d_r: int = 0

    @classmethod
    def from_shapes(cls, x: tuple[int, ...], tc: int, padding: tuple[int, ...]) -> "Geom":
        b, c, t, h, w = x
        w_l, w_r, h_t, h_b, d_l, d_r = (int(p) for p in padding)
        d_l = d_l - tc
        if d_l < 0:
            raise ValueError("front pad is smaller than the cached frames")
        return cls(b=b, c=c, t=t, h=h, w=w, tc=tc, w_l=w_l, w_r=w_r,
                   h_t=h_t, h_b=h_b, d_l=d_l, d_r=d_r)

    @property
    def t_out(self) -> int:
        return self.tc + self.t + self.d_l + self.d_r

    @property
    def h_out(self) -> int:
        return self.h + self.h_t + self.h_b

    @property
    def w_out(self) -> int:
        return self.w + self.w_l + self.w_r

    @property
    def planes(self) -> int:
        return self.b * self.c * self.t_out

    @property
    def total(self) -> int:
        return self.planes * self.h_out * self.w_out

    def bc_t(self, plane: int) -> tuple[int, int]:
        return divmod(plane, self.t_out)

    def src_at(self, plane: int, ho: int, wo: int) -> tuple[str, int] | None:
        """Reference answer for one dst coordinate: ('c'|'x', flat) or None (zero)."""
        bc, t = self.bc_t(plane)
        ts = t - self.d_l
        hs, ws = ho - self.h_t, wo - self.w_l
        if not (0 <= ts < self.tc + self.t and 0 <= hs < self.h and 0 <= ws < self.w):
            return None
        x_plane = self.h * self.w
        if ts < self.tc:
            return ("c", (bc * self.tc + ts) * x_plane + hs * self.w + ws)
        return ("x", (bc * self.t + (ts - self.tc)) * x_plane + hs * self.w + ws)


@dataclass(frozen=True)
class Lane:
    """One executed element: where it is written, and what value it moves."""

    dst: int
    buf: str | None = None   # "x", "c", or None for a pure filler store
    src: int = 0             # flat offset inside that buffer; ignored when buf is None

    @property
    def useful(self) -> bool:
        """A lane is useful when it moves a value rather than filling."""
        return self.buf is not None


# A plan yields (program_id, lanes) for every program it would launch.
Plan = Callable[[Geom], Iterator[tuple[tuple[int, ...], list[Lane]]]]


def reference_map(geom: Geom) -> dict[int, tuple[str, int] | None]:
    """dst flat offset -> ('c'|'x', flat src) or None.  The ground truth."""
    out: dict[int, tuple[str, int] | None] = {}
    for plane in range(geom.planes):
        base = plane * geom.h_out * geom.w_out
        for ho in range(geom.h_out):
            for wo in range(geom.w_out):
                out[base + ho * geom.w_out + wo] = geom.src_at(plane, ho, wo)
    return out


def _lane(geom: Geom, dst: int, plane: int, ho: int, wo: int) -> Lane:
    tag = geom.src_at(plane, ho, wo)
    if tag is None:
        return Lane(dst=dst, buf=None)
    return Lane(dst=dst, buf=tag[0], src=tag[1])


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------
def rowblock_plan(rb: int) -> Plan:
    """RB output rows per program; each row is one contiguous w_out-wide run."""

    def plan(geom: Geom) -> Iterator[tuple[tuple[int, ...], list[Lane]]]:
        nrb = -(-geom.h_out // rb)
        for plane in range(geom.planes):
            for block in range(nrb):
                lanes: list[Lane] = []
                for r in range(rb):
                    ho = block * rb + r
                    if ho >= geom.h_out:
                        continue
                    base = (plane * geom.h_out + ho) * geom.w_out
                    for wo in range(geom.w_out):
                        lanes.append(_lane(geom, base + wo, plane, ho, wo))
                yield (plane, block), lanes

    plan.__name__ = f"rowblock(rb={rb})"
    return plan


def interior_edge_plan(rb: int = 8) -> Plan:
    """Interior copied as one contiguous run; w-padding columns filled separately."""

    def plan(geom: Geom) -> Iterator[tuple[tuple[int, ...], list[Lane]]]:
        nrb = -(-geom.h_out // rb)
        for plane in range(geom.planes):
            for block in range(nrb):
                lanes: list[Lane] = []
                for r in range(rb):
                    ho = block * rb + r
                    if ho >= geom.h_out:
                        continue
                    row = (plane * geom.h_out + ho) * geom.w_out
                    for wo in list(range(geom.w_l)) + list(range(geom.w_l + geom.w, geom.w_out)):
                        lanes.append(Lane(dst=row + wo))
                    for wo in range(geom.w_l, geom.w_l + geom.w):
                        lanes.append(_lane(geom, row + wo, plane, ho, wo))
                yield (plane, block), lanes

    plan.__name__ = f"interior_edge(rb={rb})"
    return plan


def flat_block_plan(block: int) -> Plan:
    """The shipped flat kernel: one 1-D block over the whole output.

    Every store lane is contiguous (100% lane use), but the source offset comes
    from unpacking a flattened (plane, h, w) coordinate -- the "discrete access"
    pattern that costs 华为 3.7x.
    """

    def plan(geom: Geom) -> Iterator[tuple[tuple[int, ...], list[Lane]]]:
        total = geom.total
        plane_stride = geom.h_out * geom.w_out
        for b0 in range(-(-total // block)):
            lanes: list[Lane] = []
            for i in range(block):
                flat = b0 * block + i
                if flat >= total:
                    continue
                plane, rem = divmod(flat, plane_stride)
                ho, wo = divmod(rem, geom.w_out)
                lanes.append(_lane(geom, flat, plane, ho, wo))
            yield (b0,), lanes

    plan.__name__ = f"flat_block(block={block})"
    return plan


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
@dataclass
class PlanStats:
    plan_name: str
    programs: int = 0
    lanes: int = 0
    useful_lanes: int = 0
    loads: int = 0
    stores: int = 0
    dst_sectors: int = 0
    src_sectors: int = 0
    dst_runs: int = 0
    aligned_dst_runs: int = 0
    copy_runs: int = 0
    aligned_copy_runs: int = 0
    access_shapes: dict[str, int] = field(default_factory=dict)
    elemsize: int = 2
    align_bytes: int = SECTOR_BYTES

    @property
    def grid_product(self) -> int:
        """Programs launched -- the quantity coreDim caps (one point per program)."""
        return self.programs

    @property
    def dst_sector_efficiency(self) -> float:
        moved = self.dst_sectors * SECTOR_BYTES
        return (self.stores * self.elemsize / moved) if moved else 0.0

    @property
    def src_sector_efficiency(self) -> float:
        moved = self.src_sectors * SECTOR_BYTES
        return (self.loads * self.elemsize / moved) if moved else 0.0

    @property
    def lane_use(self) -> float:
        return self.useful_lanes / self.lanes if self.lanes else 0.0

    @property
    def aligned_dst_fraction(self) -> float:
        return self.aligned_dst_runs / self.dst_runs if self.dst_runs else 0.0

    @property
    def aligned_copy_fraction(self) -> float:
        return self.aligned_copy_runs / self.copy_runs if self.copy_runs else 0.0

    def as_dict(self) -> dict:
        return {
            "plan": self.plan_name,
            "programs": self.programs,
            "lanes": self.lanes,
            "lane_use": round(self.lane_use, 4),
            "loads": self.loads,
            "stores": self.stores,
            "dst_sector_eff": round(self.dst_sector_efficiency, 4),
            "src_sector_eff": round(self.src_sector_efficiency, 4),
            "dst_runs": self.dst_runs,
            "dst_run_align": round(self.aligned_dst_fraction, 4),
            "copy_runs": self.copy_runs,
            "copy_run_align": round(self.aligned_copy_fraction, 4),
            "access_shapes": dict(self.access_shapes),
        }


def _sector_count(offsets: list[int], elemsize: int) -> int:
    """Distinct SECTOR_BYTES sectors touched by a set of element offsets.

    Counting distinct sector indices (not the min-max span) is what makes a
    gather visible: a scattered set of offsets inside a narrow span still touches
    many sectors, and a monotone run touches only its own.
    """
    if not offsets:
        return 0
    per_sector = max(SECTOR_BYTES // elemsize, 1)
    return len({off // per_sector for off in offsets})


def _shape_of(prev: Lane | None, cur: Lane) -> str:
    if prev is None:
        return ""
    d_dst = cur.dst - prev.dst
    if cur.buf is None or prev.buf is None:
        if cur.buf != prev.buf:
            return UNKNOWN
        return BLOCK if d_dst == 1 else (SCATTER if d_dst > 0 else UNKNOWN)
    if cur.buf != prev.buf:
        return UNKNOWN
    d_src = cur.src - prev.src
    if d_dst == 1 and d_src == 1:
        return BLOCK
    if d_dst == 1:
        return SCATTER
    if d_src == 1:
        return GATHER
    if d_dst == d_src and d_dst > 1:
        return STRIDED
    return UNKNOWN


def _run_census(stats: PlanStats, lanes: list[Lane], align_bytes: int,
                 elemsize: int) -> None:
    """Count store-side runs and full copy runs, plus how many start aligned.

    Two different questions, deliberately kept apart:

    * a *store run* only needs the dst side to be contiguous -- it is what the
      write vectoriser can fuse, and its alignment is set by where the row starts;
    * a *copy run* needs dst and src contiguous over the same buffer -- only then
      is the element-wise copy replaced by one block move.
    """
    def aligned(off: int) -> bool:
        return (off * elemsize) % align_bytes == 0

    # store runs
    start = lanes[0]
    prev = lanes[0]
    for lane in lanes[1:]:
        if lane.dst != prev.dst + 1:
            stats.dst_runs += 1
            stats.aligned_dst_runs += aligned(start.dst)
            start = lane
        prev = lane
    stats.dst_runs += 1
    stats.aligned_dst_runs += aligned(start.dst)

    # copy runs
    start = lanes[0]
    prev = lanes[0]
    for lane in lanes[1:]:
        same_move = (lane.buf is not None and prev.buf is not None
                     and lane.buf == prev.buf
                     and lane.dst == prev.dst + 1 and lane.src == prev.src + 1)
        if not same_move:
            stats.copy_runs += 1
            stats.aligned_copy_runs += aligned(start.dst)
            start = lane
        prev = lane
    stats.copy_runs += 1
    stats.aligned_copy_runs += aligned(start.dst)


def simulate(geom: Geom, plan: Plan, name: str = "", elemsize: int = 2,
             align_bytes: int = SECTOR_BYTES) -> PlanStats:
    """Run a plan over a proxy geometry and measure what it would touch."""
    stats = PlanStats(plan_name=name or getattr(plan, "__name__", "plan"),
                      elemsize=elemsize, align_bytes=align_bytes)
    shapes: dict[str, int] = {}
    for _pid, lanes in plan(geom):
        stats.programs += 1
        if not lanes:
            continue
        stats.lanes += len(lanes)
        stats.useful_lanes += sum(1 for lane in lanes if lane.useful)
        store_offsets = [lane.dst for lane in lanes]
        stats.stores += len(store_offsets)
        stats.dst_sectors += _sector_count(store_offsets, elemsize)
        # Loads must be counted per source buffer: 'x' and 'c' are different
        # allocations, so pooling their offsets would collapse two sectors into
        # one and report a nonsensical efficiency above 1.
        by_buf: dict[str, list[int]] = {}
        for lane in lanes:
            if lane.buf is not None:
                by_buf.setdefault(lane.buf, []).append(lane.src)
        for offsets in by_buf.values():
            stats.loads += len(offsets)
            stats.src_sectors += _sector_count(offsets, elemsize)
        prev: Lane | None = None
        for lane in lanes:
            shape = _shape_of(prev, lane)
            if shape:
                shapes[shape] = shapes.get(shape, 0) + 1
            prev = lane
        _run_census(stats, lanes, align_bytes, elemsize)
    stats.access_shapes = shapes
    return stats


def row_block_for(w_out: int, elemsize: int, align_bytes: int = SECTOR_BYTES,
                  cap: int = 64) -> int:
    """Smallest power-of-two row block whose byte size is a multiple of align_bytes."""
    rb = 1
    while (rb * w_out * elemsize) % align_bytes != 0 and rb < cap:
        rb *= 2
    return rb


def verify(geom: Geom, plan: Plan) -> tuple[bool, dict[str, int]]:
    """Prove the plan writes every dst element exactly once from the right source.

    A non-empty counter dict always means a real bug -- a missing element, a
    double write, a wrong source, or a wrong source buffer.  Those are the ways a
    tiling plan is silently wrong on a device and expensive to debug there.
    """
    want = reference_map(geom)
    problems = {"missing": 0, "duplicate": 0, "wrong_src": 0, "wrong_buffer": 0}
    seen: set[int] = set()
    for _pid, lanes in plan(geom):
        for lane in lanes:
            if lane.dst in seen:
                problems["duplicate"] += 1
                continue
            seen.add(lane.dst)
            tag = want.get(lane.dst)
            if tag is None:
                if lane.buf is not None:
                    problems["wrong_buffer"] += 1
                continue
            if lane.buf is None:
                problems["wrong_src"] += 1
            elif lane.buf != tag[0]:
                problems["wrong_buffer"] += 1
            elif lane.src != tag[1]:
                problems["wrong_src"] += 1
    problems["missing"] = len(set(want) - seen)
    return (sum(problems.values()) == 0), problems
