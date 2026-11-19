"""`segmented_cumsum`: the reset scan of Task153, lowered by composition.

Task152's `cumsum` closed "accumulate along one axis"; it cannot express a scan
that *restarts* at each segment boundary, which is what the Task153
`kda_prologue` does per chunk. The language gains one first-class operator for
it, and this file pins the three decisions that make it an operator rather than
a task-specific shortcut:

* the registry contract: float input, same-shaped integer `seg_start`, fp32
  accumulator, literal extents (the stated v0.6 boundary);
* the lowering is a **composition**: `compiler/ir/composite_ops.py` expands the
  single operator into the existing `cumsum` / `gather_last` / elementwise
  lowerings before fusion, so no backend gains a second copy of the scan or the
  gather. The emitted kernel count is pinned here, and the next-step one-pass
  plan is in `docs/OPERATOR_LOOP.md`;
* the identity's own out-of-range index is fail-closed: a first segment starts
  at 0, so `seg_start - 1` is -1 there, and the gather family's boundary rule
  (normalise a negative index once by adding the extent) would silently read
  `cumsum[..., -1]`. The expansion clamps the index and masks that lane to
  subtract zero, so an all-zero table *is* the plain prefix sum.

The oracle below is written from the definition, not from the emitted code:
per row, a prefix sum, then subtract the prefix at the segment start (zero for
the first segment).
"""
import pytest

from compiler import reference
from compiler.backend.reference import module as reference_module
from compiler.compile import compile_source
from compiler.frontend.diagnostic import CompileError, DiagnosticKind
from compiler.frontend.operator import OPERATORS
from compiler.platform_configs import resolve_platform

ROWS, COLS = 3, 5


def source(dtype="fp32", rows=ROWS, cols=COLS, name="segment_scan",
           start_dtype="int32"):
    return (
        f"pipeline {name} {{\n"
        f" input x:{dtype}[{rows},{cols}]\n"
        f" input seg_start:{start_dtype}[{rows},{cols}]\n"
        f" y = segmented_cumsum(x, seg_start)\n"
        f" output y\n"
        f"}}\n"
    )


def compiled(**options):
    return compile_source(source(**options)).pipelines[0]


def reference_run(pipeline, x, starts):
    namespace = {}
    exec(reference_module([pipeline.reference]), namespace)
    return namespace["run_" + pipeline.pipeline.name](x, starts)["y"]


def oracle(x, starts, rows, cols):
    """The definition: prefix sum per row, minus the prefix before the segment."""
    expected = []
    for row in range(rows):
        prefix = []
        running = 0.0
        for t in range(cols):
            running += float(x[row * cols + t])
            prefix.append(running)
        for t in range(cols):
            start = int(starts[row * cols + t])
            assert 0 <= start <= t, (row, t, start)
            expected.append(prefix[t] - (0.0 if start == 0 else prefix[start - 1]))
    return expected


def numbers(count):
    return [((index * 7 + 3) % 11 - 5) / 4 for index in range(count)]


def flat(rows):
    return [value for row in rows for value in row]


CASES = {
    # name: (rows, cols, per-row seg_start tables)
    "single_segment": (2, 4, [[0, 0, 0, 0], [0, 0, 0, 0]]),
    "equal_segments": (2, 6, [[0, 0, 0, 3, 3, 3], [0, 0, 0, 3, 3, 3]]),
    "unequal_segments": (2, 6, [[0, 0, 0, 3, 4, 4], [0, 1, 1, 1, 4, 4]]),
    "a_segment_of_length_one": (1, 5, [[0, 0, 2, 3, 3]]),
    "every_position_its_own_segment": (1, 6, [[0, 1, 2, 3, 4, 5]]),
    "a_repeated_start_is_one_segment": (1, 5, [[0, 0, 0, 0, 0]]),
}


# --- the registry contract --------------------------------------------------

def test_the_operator_declares_the_scan_precision_contract():
    spec = OPERATORS["segmented_cumsum"]
    assert spec.acc == "fp32"
    assert spec.pointwise is False
    assert spec.arity == 2


def test_a_non_integer_segment_table_is_a_type_diagnostic():
    with pytest.raises(CompileError, match='olumn 6: segmented_cumsum requires an integer seg_start \\(int32 or int64\\)') as caught:
        compile_source(source(start_dtype="fp32"))
    assert caught.value.kind is DiagnosticKind.TYPE
    assert "integer seg_start" in caught.value.message


def test_a_symbolic_extent_is_refused_with_the_reason():
    symbolic = ("pipeline symbolic {\n"
                " input x:fp32[M,4]\n"
                " input seg_start:int32[M,4]\n"
                " y = segmented_cumsum(x, seg_start)\n"
                " output y\n"
                "}\n")
    with pytest.raises(CompileError, match='olumn 6: segmented_cumsum requires literal extents: it is lowered by composition into cumsum and gather_last, whose static rows/tiles cannot be written for symbolic M') as caught:
        compile_source(symbolic)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "literal extents" in caught.value.message
    assert "cumsum and gather_last" in caught.value.message


def test_seg_start_must_have_the_input_shape():
    mismatched = ("pipeline mismatched {\n"
                  " input x:fp32[2,4]\n"
                  " input seg_start:int32[2,3]\n"
                  " y = segmented_cumsum(x, seg_start)\n"
                  " output y\n"
                  "}\n")
    with pytest.raises(CompileError, match="olumn 6: segmented_cumsum requires seg_start to have exactly the input shape: \\['2', '3'\\] and \\['2', '4'\\]") as caught:
        compile_source(mismatched)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "exactly the input shape" in caught.value.message


def test_an_int64_segment_table_is_accepted():
    pipeline = compile_source(source(start_dtype="int64")).pipelines[0]
    assert pipeline.triton.complete
    values = numbers(ROWS * COLS)
    starts = flat([[0, 0, 2, 2, 4], [0, 0, 0, 3, 3], [0, 1, 2, 3, 4]])
    assert reference_run(pipeline, values, starts) == pytest.approx(
        oracle(values, starts, ROWS, COLS), rel=1e-6)


# --- the lowering is a composition -----------------------------------------

def test_normalization_expands_the_operator_into_existing_operators():
    pipeline = compiled()
    assert [node.op for node in pipeline.semantic.nodes] == [
        "input", "input", "segmented_cumsum", "output"]
    expanded = [node.op for node in pipeline.dataflow.nodes]
    assert "segmented_cumsum" not in expanded
    for op in ("cumsum", "gather_last", "maximum", "eq", "where", "sub"):
        assert op in expanded
    assert any(decision.value == "y" and decision.action == "expand_composition"
               for decision in pipeline.normalization)


def test_no_backend_receives_the_composed_operator():
    pipeline = compiled()
    assert pipeline.triton.complete and pipeline.reference.complete
    for kernel in pipeline.triton.kernels:
        assert "segmented_cumsum" not in kernel.ops
    assert pipeline.triton.kernels[0].ops == ("cumsum",)


def test_the_composition_is_two_device_passes_in_four_kernels():
    """Pinned shape of the v0.6 lowering; the one-pass plan is in the docs.

    scan + index prep + gather + masked subtract. The gather is the boundary
    between the two data passes over x-sized tensors: no operator fusion merges
    an opaque gather with what surrounds it.
    """
    kernels = compiled().triton.kernels
    assert [kernel.kind for kernel in kernels] == [
        "scan", "primitive", "axis_gather", "primitive"]
    assert kernels[0].ops == ("cumsum",)
    assert "tl.cumsum(" in kernels[0].source
    assert "gather_last" in kernels[2].ops
    assert "tl.load" in kernels[2].source


def test_the_gather_index_is_clamped_before_it_is_read():
    """The identity's -1 must not reach the gather's negative-index rule.

    The gather's index value is produced by `maximum`, i.e. the emitted access
    is proven to start at 0; the first-segment lanes are separated out by
    `eq`/`where` instead of relying on gather's out-of-range convention.
    """
    pipeline = compiled()
    producer = {node.output: node for node in pipeline.dataflow.nodes}
    gather, = [node for node in pipeline.dataflow.nodes
               if node.op == "gather_last"]
    index = pipeline.dataflow.nodes[
        [node.output for node in pipeline.dataflow.nodes].index(gather.inputs[1])]
    assert index.op == "maximum", index
    # maximum's operands: the shifted (seg_start - 1) and a zero tile
    shifted, zeros = (producer[value] for value in index.inputs)
    assert shifted.op == "sub"
    assert zeros.op == "zeros_like"
    select, = [node for node in pipeline.dataflow.nodes if node.op == "where"]
    mask, _zeros, _gathered = select.inputs
    assert producer[mask].op == "eq"


# --- numerics against the independent oracle -------------------------------

@pytest.mark.parametrize("case", sorted(CASES), ids=lambda name: name)
def test_reference_matches_the_definition(case):
    rows, cols, starts = CASES[case]
    pipeline = compile_source(source(rows=rows, cols=cols)).pipelines[0]
    values = numbers(rows * cols)
    flat_starts = flat(starts)
    expected = oracle(values, flat_starts, rows, cols)
    got = reference_run(pipeline, values, flat_starts)
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_an_all_zero_start_table_is_exactly_the_plain_prefix_sum():
    """The wrap bug this operator could have had: -1 would read the row total.

    If `seg_start - 1` reached the gather unclamped, the first-segment lanes
    would subtract `cumsum[..., -1]` (the gather convention adds the extent
    once), so the last output would be 0 instead of the row sum.
    """
    rows, cols = 1, 5
    pipeline = compile_source(source(rows=rows, cols=cols)).pipelines[0]
    values = [0.5, -1.25, 2.0, 0.25, 3.0]
    starts = [0, 0, 0, 0, 0]
    got = reference_run(pipeline, values, starts)
    running = []
    total = 0.0
    for value in values:
        total += value
        running.append(total)
    assert got == pytest.approx(running, rel=1e-6)
    assert got[-1] == pytest.approx(sum(values), rel=1e-6)


@pytest.mark.parametrize("dtype, rel, abs_", [
    ("fp32", 1e-6, 1e-7),
    ("fp16", 5e-3, 5e-3),
    ("bf16", 3e-2, 3e-2),
])
def test_every_input_dtype_matches_within_its_precision(dtype, rel, abs_):
    """The scan accumulates in fp32; only the declared-dtype roundings differ."""
    rows, cols, starts = CASES["unequal_segments"]
    pipeline = compile_source(source(dtype=dtype, rows=rows, cols=cols)
                              ).pipelines[0]
    values = [0.7, -1.3, 2.1, 0.4, -0.9, 1.6,
              -0.2, 1.1, -1.7, 0.3, 2.4, -0.6]
    flat_starts = flat(starts)
    expected = oracle(values, flat_starts, rows, cols)
    got = reference_run(pipeline, values, flat_starts)
    assert got == pytest.approx(expected, rel=rel, abs=abs_)


# --- platform gate ---------------------------------------------------------

@pytest.mark.parametrize("platform, options", [
    ("kunlunxin", {}),
    ("ascend", {"device_cores": 48, "mapping_strategy": "direct_grid"}),
])
def test_targets_without_the_component_lowerings_refuse_the_program(platform, options):
    """The gate is inherited from the components, not re-implemented.

    `segmented_cumsum` owns no kernel kind of its own: a target that does not
    admit `scan` or `axis_gather` cannot run it, and the refusal names the
    component whose lowering is missing (the same gate as writing them by hand).
    """
    pipeline = compile_source(source(), platform_config=resolve_platform(
        platform, **options)).pipelines[0]
    assert not pipeline.triton.complete
    refused = {refusal.operator for refusal in pipeline.triton.refusals}
    assert refused & {"cumsum", "gather_last"}, refused
    assert "segmented_cumsum" not in " ".join(
        kernel.ops[0] for kernel in pipeline.triton.kernels)


# --- the shipped example ---------------------------------------------------

def test_the_task153_shaped_example_compiles():
    from pathlib import Path
    path = Path("examples/batch10/segmented_cumsum.fgpu")
    pipeline = compile_source(path.read_text(encoding="utf-8")).pipelines[0]
    assert pipeline.pipeline.name == "segmented_cumsum"
    assert pipeline.triton.complete
    assert len([kernel for kernel in pipeline.triton.kernels
                if kernel.kind == "scan"]) == 1
