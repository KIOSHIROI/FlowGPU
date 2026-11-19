"""`gated_delta_recurrence`: the language's only temporal operator.

Task18 `fused_recurrent_gdn` was the last real expressibility gap in the census.
Every step of it is elementwise or a reduction, all of which the language has; the
missing thing was the loop -- a `[V,K]` state that stays in registers across the
whole time axis, because materialising `[B,T,H,V,K]` is ~10^10 bytes at a real
shape and because step t's output is a readout of the state *after* step t.
"""
import math

import pytest

from flowgpu.compiler.analysis.access_shape import BLOCK, of_op
from flowgpu.compiler.backend.platform_configs import PLATFORMS
from flowgpu.compiler.backend.reference import inputs_for, run as run_reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.operator import OPERATORS
from flowgpu.compiler.frontend.diagnostic import CompileError

EXAMPLE = "examples/batch10/fused_recurrent_gdn.fgpu"
SMALL = """
pipeline gdn {
 input q:fp32[1,4,2,8]
 input k:fp32[1,4,2,8]
 input v:fp32[1,4,2,8]
 input g:fp32[1,4,2]
 input beta:fp32[1,4,2]
 input state0:fp32[1,2,8,8]
 o, s = gated_delta_recurrence(q, k, v, g, beta, state0)
 output o
 output s
}
"""


def compile_program(source=SMALL):
    return compile_source(source).pipelines[0]


def feeds(pipeline):
    return {name: values for name, _, _, values in inputs_for(pipeline.reference, {})}


# --------------------------------------------------------------------------
# the registry: temporal, two results, and it declares its accumulator
# --------------------------------------------------------------------------
def test_the_recurrence_is_a_two_result_operator_that_accumulates_in_fp32():
    """`acc='fp32'` is the 21.6 declaration: the state is a sum over k that runs
    for T steps, so the accumulator is part of the contract, not an implementation
    detail."""
    assert OPERATORS["gated_delta_recurrence"].multi is True
    assert OPERATORS["gated_delta_recurrence"].pointwise is False
    assert OPERATORS["gated_delta_recurrence"].acc == "fp32"


def test_the_result_shapes_are_the_sequence_and_the_final_state():
    pipeline = compile_program()
    output = pipeline.symbols.lookup("o").type
    state = pipeline.symbols.lookup("s").type
    assert output.dtype == "fp32" and output.shape == ["1", "4", "2", "8"]
    assert state.dtype == "fp32" and state.shape == ["1", "2", "8", "8"]
    # the *example* is the ratio case: HV = 4 = H * 2
    example = compile_source(open(EXAMPLE, encoding="utf-8").read()).pipelines[0]
    assert example.symbols.lookup("output_seq").type.shape == ["1", "4", "4", "8"]
    assert example.symbols.lookup("state_final").type.shape == ["1", "4", "8", "8"]


def test_a_mismatched_state_shape_is_refused():
    source = SMALL.replace("input state0:fp32[1,2,8,8]", "input state0:fp32[1,2,8,4]")
    with pytest.raises(CompileError, match='gated_delta_recurrence state must be') as caught:
        compile_program(source)
    assert "state must be [B,HV,V,K]" in str(caught.value)


def test_a_gqa_ratio_is_expressed_by_the_operators_own_head_mapping():
    """`HV = H * ratio` is not a `repeat_interleave`: it is a declared index
    relation (`state head hv reads source head hv // ratio`), which is the same way
    `attention` handles its own GQA. The operator accepts it, one program per
    (batch, value head)."""
    source = SMALL.replace("input v:fp32[1,4,2,8]", "input v:fp32[1,4,4,8]") \
                   .replace("input g:fp32[1,4,2]", "input g:fp32[1,4,4]") \
                   .replace("input beta:fp32[1,4,2]", "input beta:fp32[1,4,4]") \
                   .replace("input state0:fp32[1,2,8,8]", "input state0:fp32[1,4,8,8]")
    pipeline = compile_program(source)
    kernel = pipeline.triton.kernels[0]
    assert kernel.grid == "(1, 4)"
    assert "head_source = h // 2" in kernel.source
    assert "= (b * T + t) * H * K + head_source * K" in kernel.source
    assert "= (b * T + t) * HV * V + h * V" in kernel.source


def test_a_value_head_count_that_is_not_a_multiple_is_refused():
    source = SMALL.replace("input v:fp32[1,4,2,8]", "input v:fp32[1,4,3,8]")
    with pytest.raises(CompileError, match='gated_delta_recurrence needs the value head count') as caught:
        compile_program(source)
    assert "multiple" in str(caught.value)


def test_a_symbolic_extent_is_refused_because_the_state_tile_is_a_constexpr():
    source = SMALL.replace("input q:fp32[1,4,2,8]", "input q:fp32[1,T,2,8]")
    with pytest.raises(CompileError, match='the state tile is a constexpr and the time loop is bounded by shape') as caught:
        compile_program(source)
    assert "literal extents" in str(caught.value)


# --------------------------------------------------------------------------
# emission: one kernel, one program per (batch, head), state in registers
# --------------------------------------------------------------------------
def test_the_whole_recurrence_is_one_kernel_with_a_time_loop():
    triton = compile_program().triton
    assert triton.complete
    assert [kernel.kind for kernel in triton.kernels] == ["recurrence"]
    kernel = triton.kernels[0]
    assert kernel.grid == "(1, 2)"
    assert "for t in range(T)" in kernel.source
    # the state is a register tile: it is loaded once and stored once, outside
    # the loop, and never materialised per step
    assert kernel.source.count("state_tile = tl.load") == 1
    assert kernel.source.count("tl.store") == 2      # one output row per step + final
    assert "state_tile = state_tile * decay" in kernel.source


def test_the_two_readouts_are_reductions_over_the_state_tile():
    kernel = compile_program().triton.kernels[0]
    assert kernel.source.count("tl.sum(state_tile * k_t[None, :], axis=1)") == 1
    assert kernel.source.count("tl.sum(state_tile * q_t[None, :], axis=1)") == 1


def test_the_recurrence_reads_by_block_and_the_temporal_cost_is_a_contract():
    """Access shape describes one access, not an iteration: every load is a
    contiguous `[K]`/`[V]` run. The loop's cost belongs to the platform contract
    (`store_inside_loop_ok`, measured 0.03x on 昆仑芯), which is why the kind is
    CUDA-only."""
    assert of_op("gated_delta_recurrence").kind == BLOCK
    for name in ("ascend", "ascend-910b", "kunlunxin", "kunlun-r300"):
        assert "recurrence" not in PLATFORMS[name].allowed_kernel_kinds
    assert "recurrence" in PLATFORMS["cuda"].allowed_kernel_kinds


def test_the_kind_is_refused_on_a_target_that_cannot_sustain_a_carried_store():
    pipeline = compile_source(open(EXAMPLE, encoding="utf-8").read(),
                              platform_config=PLATFORMS["kunlunxin"]).pipelines[0]
    assert not pipeline.triton.kernels
    assert any("recurrence" in refusal.message() or "pointwise" in refusal.message()
               for refusal in pipeline.triton.refusals)


def test_the_final_state_alone_is_enough_for_the_kernel_to_run():
    """The sequence may be dead (a caller that only wants the carried state). The
    kernel then stores once, and the live result still has a producer."""
    source = SMALL.replace(" output o\n", "")
    pipeline = compile_program(source)
    assert pipeline.triton.complete
    kernel = pipeline.triton.kernels[0]
    assert "output_seq_ptr" not in kernel.source.split("def ")[1].split("(")[1]
    produced = run_reference(pipeline.reference)
    assert len(produced["s"]) == 128


# --------------------------------------------------------------------------
# reference semantics against the task's own algorithm
# --------------------------------------------------------------------------
def test_the_reference_reproduces_the_tasks_reference_implementation():
    """A from-scratch reimplementation of the five steps, in double precision, on
    the same generated inputs."""
    pipeline = compile_program()
    produced = run_reference(pipeline.reference)
    data = feeds(pipeline)
    batch, steps, heads, width = 1, 4, 2, 8
    channel = 8

    def at(values, shape, *indices):
        flat = 0
        for index, extent in zip(indices, shape):
            flat = flat * extent + index
        return values[flat]

    state = {}
    for h in range(heads):
        for vv in range(channel):
            for kk in range(width):
                state[(h, vv, kk)] = float(at(data["state0"], (batch, heads, channel, width),
                                              0, h, vv, kk))
    expected_output = []
    for t in range(steps):
        for h in range(heads):
            decay = math.exp(float(at(data["g"], (batch, steps, heads), 0, t, h)))
            for vv in range(channel):
                read = 0.0
                for kk in range(width):
                    state[(h, vv, kk)] *= decay
                    read += state[(h, vv, kk)] * float(
                        at(data["k"], (batch, steps, heads, width), 0, t, h, kk))
                correction = (float(at(data["v"], (batch, steps, heads, channel), 0, t, h, vv))
                              - read) * float(at(data["beta"], (batch, steps, heads), 0, t, h))
                for kk in range(width):
                    state[(h, vv, kk)] += correction * float(
                        at(data["k"], (batch, steps, heads, width), 0, t, h, kk))
                expected_output.append(sum(
                    state[(h, vv, kk)] * float(at(data["q"], (batch, steps, heads, width),
                                                  0, t, h, kk))
                    for kk in range(width)))
    expected_state = [state[(h, vv, kk)] for h in range(heads)
                      for vv in range(channel) for kk in range(width)]

    for got, want in zip(produced["o"], expected_output):
        assert got == pytest.approx(want, rel=1e-5, abs=1e-5)
    for got, want in zip(produced["s"], expected_state):
        assert got == pytest.approx(want, rel=1e-5, abs=1e-5)


def test_the_recurrence_does_not_mutate_its_initial_state():
    """The reference keeps the working state in a local list; mutating the operand
    would corrupt the caller's tensor, and the first version of that lowering did
    exactly that (it also forgot to write the final state at all)."""
    pipeline = compile_program()
    before = list(feeds(pipeline)["state0"])
    run_reference(pipeline.reference)
    assert feeds(pipeline)["state0"] == before


def test_the_example_compiles_emits_and_runs():
    pipeline = compile_source(open(EXAMPLE, encoding="utf-8").read()).pipelines[0]
    assert pipeline.triton.complete
    assert [kernel.kind for kernel in pipeline.triton.kernels] == ["recurrence"]
    produced = run_reference(pipeline.reference)
    assert len(produced["output_seq"]) == 1 * 4 * 4 * 8
    assert len(produced["state_final"]) == 1 * 4 * 8 * 8
