"""The dependence guard on slot reuse: opt-in, and off by default.

`analysis/dependences.py` judges memory dependences between the kernels of a plan.
Its consumer here is the *slot allocator*, not fusion: under single assignment a
fusion group reads its inputs and writes one fresh value, so it cannot contain two
accesses to one buffer -- measured over every example program, zero intra-kernel
dependences. What can go wrong is sharing a slot between two buffers whose
accesses are not provably disjoint, which is exactly what an indexed access makes
possible.

Turning the guard on is a change of default, so it is off, and
`docs/COMPILER_ACCEPTANCE.md` requires a parent/candidate comparison before that
changes. These tests pin both halves: off changes nothing, on can only raise the
slot count.
"""
from pathlib import Path

import pytest

from flowgpu.compiler.analysis.liveness import plan
from flowgpu.compiler.compile import compile_source

ROOT = Path(__file__).resolve().parents[1]

PROGRAMS = (
    "examples/batch10/normalize_and_patchify.fgpu",
    "examples/batch10/routed_silu.fgpu",
    "examples/residual_norm.fgpu",
    "examples/mlp.fgpu",
)


def pipeline(source_path, **kwargs):
    return compile_source((ROOT / source_path).read_text(encoding="utf-8"), **kwargs).pipelines[0]


def test_the_guard_is_off_by_default():
    """The default plan must be byte-identical to the one compiled before the
    guard existed: same slot count and same assignment."""
    plain = compile_source((ROOT / PROGRAMS[0]).read_text(encoding="utf-8")).pipelines[0]
    explicit = pipeline(PROGRAMS[0], dependence_guard=False)
    assert [slot.index for slot in plain.memory.slots] == [slot.index for slot in explicit.memory.slots]


@pytest.mark.parametrize("source_path", PROGRAMS)
def test_the_guard_never_uses_fewer_slots_than_the_default(source_path):
    default = pipeline(source_path).memory
    guarded = pipeline(source_path, dependence_guard=True).memory
    assert len(guarded.slots) >= len(default.slots)


def test_a_precise_plan_is_unchanged_by_the_guard():
    """Every example addresses memory as a block or a strided view today, so the
    guard has nothing to refuse and the plans agree exactly."""
    default = pipeline("examples/mlp.fgpu").memory
    guarded = pipeline("examples/mlp.fgpu", dependence_guard=True).memory
    assert len(guarded.slots) == len(default.slots)


def test_the_flag_reaches_the_allocator_and_the_two_paths_differ_in_principle():
    """A direct call proves the flag is threaded: same graph, same fusion, only the
    allocator's policy differs. On a program whose accesses are all precise the
    plans coincide, which is why the default can stay on."""
    compiled = pipeline("examples/batch10/normalize_and_patchify.fgpu")
    guarded = plan(compiled.dataflow, compiled.fusion,
                   reuse=True, dependence_guard=True)
    assert [slot.index for slot in guarded.slots] == \
        [slot.index for slot in compiled.memory.slots]
