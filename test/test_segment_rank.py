"""`segment_rank`: the destination table a MoE dispatch is built from.

Task 69 states it as an atomic: `offset = atomic_add(masked_m[expert], 1);
src2dst[i] = expert * m_max + offset`. Expressed as a *stable rank within a
segment* it needs no atomic, no compaction primitive and no indexed addressing,
which is why it can be one declarative operator instead of a hardware trick --
and why it composes with `scatter_by_index`, which is what actually moves rows.
"""
import pytest

from compiler.analysis.access_shape import BLOCK, of_op
from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.operator import OPERATORS
from compiler.frontend.diagnostic import CompileError

PROGRAM = """
pipeline dispatch {
 input ids:int32[8]
 counts, positions = segment_rank(ids, 2, 4)
 output counts
 output positions
}
"""


def compile_program(source=PROGRAM):
    return compile_source(source).pipelines[0]


def feeds(pipeline):
    return {name: values for name, _, _, values in inputs_for(pipeline.reference, {})}


def test_segment_rank_is_a_two_result_operator_that_accumulates_nothing():
    assert OPERATORS["segment_rank"].multi is True
    assert OPERATORS["segment_rank"].pointwise is False
    assert OPERATORS["segment_rank"].acc is None


def test_the_result_extents_come_from_the_two_literals():
    pipeline = compile_program()
    counts = pipeline.symbols.lookup("counts").type
    positions = pipeline.symbols.lookup("positions").type
    assert counts.dtype == "int32" and counts.shape == ["2"]      # num_experts
    assert positions.dtype == "int32" and positions.shape == ["8"]  # one per slot


def test_a_symbolic_num_experts_is_refused():
    source = PROGRAM.replace("segment_rank(ids, 2, 4)", "segment_rank(ids, e, 4)")
    source = source.replace(" input ids:int32[8]", " input ids:int32[8] input e:int32[1]")
    with pytest.raises(CompileError, match='the output extent and the destination space are computed from it'):
        compile_program(source)


def test_a_colliding_destination_space_is_refused():
    """`num_experts * m_max` is where positions live: if there are more slots than
    room, two slots of one expert would be given the same destination."""
    with pytest.raises(CompileError, match='segment_rank destination space is 2 x 2') as caught:
        compile_program(PROGRAM.replace("segment_rank(ids, 2, 4)", "segment_rank(ids, 2, 2)"))
    assert "would collide" in str(caught.value)


def test_the_lowering_is_a_reset_then_a_rank():
    """Two launches, for the reason `scatter_by_index` has two: a padded slot
    belongs to no expert, so some launch must own its -1."""
    kernels = list(compile_program().triton.kernels)
    assert [kernel.name.rsplit("_", 1)[-1] for kernel in kernels] == ["reset", "rank"]
    reset, rank = kernels
    assert "tl.store(positions_ptr + " in reset.source.replace("_ptr", "_ptr")
    assert "-1" in reset.source
    # the rank launch is one program per expert and keeps a running count across
    # chunks, which is what makes the rank stable in slot order
    assert "tl.cumsum" in rank.source
    assert "tl.program_id(0)" in rank.source
    # the grid is a host expression, so it must be the literal; a kernel
    # constexpr here is a NameError at launch (caught on the H100, 2026-10-20)
    assert rank.grid == "(2,)"


def test_segment_rank_is_a_block_access_so_no_target_refuses_it():
    """The half of a MoE dispatch that is *computation*: the reads are `ids[i]` in
    lane order and the writes are at computed *values*, not computed addresses."""
    assert of_op("segment_rank").kind == BLOCK


def test_the_reference_matches_an_independent_oracle():
    pipeline = compile_program()
    produced = run_reference(pipeline.reference)
    ids = feeds(pipeline)["ids"]
    experts, m_max = 2, 4

    expected_counts = [sum(1 for value in ids if value == expert)
                       for expert in range(experts)]
    expected_positions, seen = [], {expert: 0 for expert in range(experts)}
    for value in ids:
        if 0 <= value < experts:
            expected_positions.append(value * m_max + seen[value])
            seen[value] += 1
        else:
            expected_positions.append(-1)

    assert list(produced["counts"]) == expected_counts
    assert list(produced["positions"]) == expected_positions


def test_an_out_of_range_id_is_padding_too():
    """An id outside `[0, num_experts)` is owned by no expert, so it can be no
    expert's slot: its destination is -1, the same value padding uses. The
    emitted reset launch gives every slot that value before any expert runs, so
    this holds without the ranking launch needing to know."""
    pipeline = compile_program()
    produced = run_reference(pipeline.reference)
    ids = feeds(pipeline)["ids"]
    out_of_range = [value for value, position in zip(ids, produced["positions"])
                    if not 0 <= value < 2]
    assert out_of_range, "the generated data must contain an out-of-range id"
    assert all(position == -1 for value, position in zip(ids, produced["positions"])
               if not 0 <= value < 2)


def test_the_table_composes_with_an_indexed_scatter():
    """What the operator is *for*: the table says where each token row goes, and
    `scatter_by_index` moves it. The composition is the dispatch, and keeping the
    two apart is why the table can be tested on its own."""
    source = """
pipeline move {
 input rows:fp32[4,3]
 input values:fp32[4,3]
 input ids:int32[4]
 counts, positions = segment_rank(ids, 2, 4)
 moved = scatter_by_index(rows, positions, values)
 output counts
 output moved
}
"""
    pipeline = compile_program(source)
    assert pipeline.triton.complete
    kinds = [kernel.kind for kernel in pipeline.triton.kernels]
    assert "index_table" in kinds and "scatter_by_index" in kinds


def test_the_dispatch_example_runs_and_the_table_is_a_plain_block():
    """The artefact `docs/OPERATOR_LOOP.md` asks for: a DSL program whose emitted
    module is the operator, not a task special case."""
    source = open("examples/batch10/moe_dispatch_index.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source).pipelines[0]
    kinds = [kernel.kind for kernel in pipeline.triton.kernels]
    assert "index_table" in kinds and "scatter_by_index" in kinds
    assert pipeline.triton.complete
    assert of_op("segment_rank").kind == BLOCK

    produced = run_reference(pipeline.reference)
    assert len(produced["positions"]) == 8
    assert len(produced["moved"]) == 32
