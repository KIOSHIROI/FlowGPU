"""`cumsum`: a scan is its own kernel kind, with the reduction precision contract.

Task152 was the operator the language could not express: a chunk-axis cumulative
sum is neither pointwise (each lane depends on every earlier lane) nor a
reduction (the axis is kept). These tests pin the three properties that made it
worth a first-class operation rather than borrowing `sum_axis`:
its own kernel kind, an fp32 accumulator rounded once at the boundary, and a
store under exactly the load mask (which is what 昆仑芯 needs).

The 2026-10-22 audit pre-check found two defects the original tests could not
see, both fixed here and pinned below:

* a row wider than `BLOCK` was **silently truncated** -- the kernel loaded
  `tl.arange(0, BLOCK)` of a wider row and stored the same slice, leaving the
  rest of the output uninitialised (CANN's `Cummin` cases are 16384 wide and
  one is 1000003 elements, so this was not an edge case).  The sweep now carries
  the running total between chunks;
* a fused pointwise neighbour made the lowering answer with a bare
  `AssertionError`.  A scan is now an opaque anchor: the neighbour gets its own
  kernel, which is also the correct partition.
"""
import pytest

from flowgpu.compiler.backend.reference import inputs_for, run as run_reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.diagnostic import CompileError


def program(body="out = cumsum_axis(x, 1)", dtype="fp32[8,16]"):
    source = ("pipeline scan_demo {\n"
              f" input x:{dtype}\n"
              f" {body}\n"
              " output out\n"
              "}\n")
    result = compile_source(source)
    assert len(result.pipelines) == 1
    return result.pipelines[0]


def test_a_scan_is_its_own_kernel_kind():
    pipeline = program()
    kernels = list(pipeline.triton.kernels)
    assert len(kernels) == 1
    assert kernels[0].kind == "scan"
    assert kernels[0].ops == ("cumsum",)


def test_the_emitted_kernel_accumulates_in_fp32_and_stores_under_the_load_mask():
    """Both halves matter: bf16 accumulation would round at every prefix step,
    and a broader store mask is the construct 昆仑芯 collapses."""
    source = list(program(dtype="bf16[8,16]").triton.kernels)[0].source
    assert "tl.cumsum(" in source
    assert ".to(tl.float32)" in source
    load_mask = source.split("tl.load(")[1].split("mask=")[1].split(",")[0].strip()
    store_mask = source.split("tl.store(")[1].split("mask=")[1].split(")")[0].strip()
    assert load_mask == store_mask


def test_both_ends_of_the_axis_are_reachable():
    """The last axis scans in place; any other axis is normalized to
    move_first_last / last-axis op / inverse rotation (Design.md, the axis
    operators), so it materializes twice and still scans once."""
    last = [kernel.kind for kernel in program("out = cumsum(x)").triton.kernels]
    first = [kernel.kind for kernel in program("out = cumsum_axis(x, 0)").triton.kernels]
    assert last == ["scan"]
    assert first == ["primitive", "scan", "primitive"]
    assert "cumsum" in [op for kernel in program("out = cumsum_axis(x, 0)").triton.kernels
                        for op in kernel.ops]


def test_the_reference_running_sum_matches_an_independent_oracle():
    pipeline = program(dtype="fp32[4,8]")
    reference = pipeline.reference
    produced = run_reference(reference)["out"]
    (values,) = [tensor for _, _, _, tensor in inputs_for(reference, {})]
    expected = []
    for row in range(4):
        running = 0.0
        for col in range(8):
            running += values[row * 8 + col]
            expected.append(running)
    assert len(produced) == len(expected)
    assert produced == pytest.approx(expected)


def test_a_scan_of_a_bad_rank_or_type_is_refused():
    with pytest.raises(CompileError, match='Unexpected token Token'):
        compile_source("pipeline p {\n input x:fp32[]\n out = cumsum(x)\n output out\n}")
    with pytest.raises(CompileError, match='cumsum requires floating-point tensors'):
        compile_source("pipeline p {\n input x:int32[4,4]\n out = cumsum(x)\n output out\n}")


def test_a_row_wider_than_the_block_is_swept_in_chunks_with_a_carry():
    """The sweep, not `tl.arange(0, BLOCK)`, is what covers a wide row.

    The old kernel loaded and stored only the first `BLOCK` lanes of every row;
    with the default block of 1024 a `[8, 4096]` cumsum returned a quarter of each
    row and left the rest uninitialised.  The carry is what makes chunk `k+1`
    continue chunk `k`'s prefix instead of restarting it.
    """
    source = list(program(dtype="fp32[8,4096]").triton.kernels)[0].source
    assert "for chunk in range(0, tl.cdiv(N, BLOCK)):" in source
    assert "carry = tl.full((), 0.0, tl.float32)" in source
    body = source.split("for chunk in range", 1)[1]
    assert "tl.load(" in body and "tl.store(" in body
    assert "tl.cumsum(" in body
    assert "carry = carry + tl.sum(values, axis=0)" in body
    # Both the load and the store are inside the sweep.
    for line in source.splitlines():
        if "tl.load(" in line or "tl.store(" in line:
            assert len(line) - len(line.lstrip()) == 8, line


def test_a_symbolic_row_width_uses_the_same_sweep():
    """`N` may stay symbolic now: the chunk count is `tl.cdiv(N, BLOCK)`."""
    pipeline = program(dtype="fp32[8,N]")
    kernel = list(pipeline.triton.kernels)[0]
    assert kernel.kind == "scan"
    assert "tl.cdiv(N" in kernel.source or "tl.cdiv(N_" in kernel.source


def test_a_fused_scan_neighbour_is_emitted_in_the_scan_kernel():
    """A pointwise consumer of a scan is folded into the scan kernel now.

    The scan lowering emits prologue/epilogue expressions (2026-10-25), so a
    pointwise consumer no longer becomes its own kernel: `z = add(cumsum(x), y)`
    is one scan kernel that writes the sum.  The reference oracle covers the
    same fused group, which is what makes the device form checkable.  Before the
    epilogue support, the fusion put the consumer inside the group and the
    lowering answered with a bare AssertionError; the audit of 2026-10-22 found
    it, and the fix was to emit the epilogue rather than to keep the anchor
    opaque.
    """
    from flowgpu.compiler.backend import reference

    epilogue = compile_source(
        "pipeline p { input x : fp32[8,16] y = cumsum(x) z = add(y, y) output z }"
    ).pipelines[0]
    assert [(kernel.kind, kernel.ops) for kernel in epilogue.triton.kernels] == [
        ("scan", ("cumsum", "add"))]
    prologue = compile_source(
        "pipeline p { input x : fp32[8,16] input s : fp32 "
        "q = mul(x, s) y = cumsum(q) output y }").pipelines[0]
    assert [kernel.kind for kernel in prologue.triton.kernels] == ["scan"]
    namespace = {}
    exec(reference.module([epilogue.reference]), namespace)
    produced = namespace["run_p"]([float(v) for v in range(128)])["z"]
    # z = y + y with y the row-local prefix sum
    expected = []
    for row in range(8):
        running = 0.0
        for column in range(16):
            running += float(row * 16 + column)
            expected.append(2.0 * running)
    assert [round(value, 6) for value in produced] == expected


def test_the_reference_matches_a_sequential_oracle_on_a_wide_row():
    """The reference is the device lowering's oracle, so it must cover the sweep.

    This is the half of the wide-row fix that can be proven on this machine: the
    reference loops the whole row (3000 > the 1024 block) and agrees with a
    sequential Python prefix sum to the last element.
    """
    pipeline = program(dtype="fp32[4,3000]")
    produced = run_reference(pipeline.reference)["out"]
    (values,) = [tensor for _, _, _, tensor in inputs_for(pipeline.reference, {})]
    expected = []
    for row in range(4):
        running = 0.0
        for col in range(3000):
            running += values[row * 3000 + col]
            expected.append(running)
    assert len(produced) == len(expected) == 12000
    assert produced == pytest.approx(expected, abs=1e-6)


def test_a_symbolic_leading_extent_is_refused_by_name():
    """The grid is one program per row, so a symbolic row count has no launch."""
    with pytest.raises(CompileError, match="literal leading extents"):
        compile_source("pipeline p { input x : fp32[R,16] y = cumsum(x) output y }")


def test_cumsum_totals_returns_the_scan_and_its_carry():
    """`running, totals = cumsum_totals(x)`: the row total the scan lowering
    already carries, exposed as a second result.

    This is what makes a two-level prefix sum cost one pass over the input: the
    earlier form computed the row totals with a separate `sum_last`, which read
    the whole tensor a second time (measured on GPU MODE: 2.564 ms -> the
    totals form removes ~1 GB of traffic per 2**28-element call).
    """
    from flowgpu.compiler.backend import reference

    source = ('pipeline p {\n'
              '    input x : fp32[3,5]\n'
              '    running, totals = cumsum_totals(x)\n'
              '    output running\n'
              '    output totals\n}\n')
    program = compile_source(source)
    pipeline = program.pipelines[0]
    assert len(pipeline.triton.kernels) == 1, [k.ops for k in pipeline.triton.kernels]
    assert 'tl.store' in pipeline.triton.kernels[0].source
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    values = [1.0, 2.0, 3.0, 4.0, 5.0,
              10.0, 20.0, 30.0, 40.0, 50.0,
              -1.0, -2.0, -3.0, -4.0, -5.0]
    produced = namespace['run_p'](values)
    # the scan is row-local (the trailing-axis cumsum); the cross-row offsets of
    # a two-level prefix sum are built from `totals`, which is the whole point of
    # carrying them: each row's total is its last running value.
    expected_running = [1, 3, 6, 10, 15,
                        10, 30, 60, 100, 150,
                        -1, -3, -6, -10, -15]
    assert [round(v, 6) for v in produced['running']] == expected_running
    assert [round(v, 6) for v in produced['totals']] == [15.0, 150.0, -15.0]


def test_a_two_level_prefix_sum_is_exact():
    """The composition the GPU MODE submission ships, checked on the reference
    oracle at a literal shape: row-local scan, row-carried totals, offsets."""
    from flowgpu.compiler.backend import reference

    source = ('pipeline p {\n'
              '    input x : fp32[2,4]\n'
              '    local, totals = cumsum_totals(x)\n'
              '    prefix = cumsum_axis(totals, 0)\n'
              '    offs = sub(prefix, totals)\n'
              '    off = broadcast_like(offs, local)\n'
              '    y = add(local, off)\n'
              '    output y\n}\n')
    pipeline = compile_source(source).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    produced = namespace['run_p']([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])['y']
    assert [round(v, 6) for v in produced] == [1.0, 3.0, 6.0, 10.0, 20.0, 40.0, 70.0, 110.0]
