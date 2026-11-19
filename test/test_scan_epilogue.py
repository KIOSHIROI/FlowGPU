"""P1 scan+row-total rewrite."""
from __future__ import annotations

from compiler import compile_source

SCAN = ("pipeline p { input x:fp32[8,128] t=sum_axis(x,1) c=cumsum_axis(x,1) "
        "out=add(c,broadcast_like(t,x)) output out }")


def test_scan_epilogue_folds_the_row_total_into_cumsum_totals():
    default = compile_source(SCAN).pipelines[0]
    rewritten = compile_source(SCAN, scan_epilogue=True).pipelines[0]
    assert len(default.triton.kernels) == 3
    assert len(rewritten.triton.kernels) == 1
    assert rewritten.triton.kernels[0].ops == ("cumsum_totals", "cumsum_totals", "add_broadcast")
    assert not rewritten.triton.refusals


def test_scan_epilogue_is_opt_in_and_does_not_change_other_programs():
    source = "pipeline p { input x:fp32[8,128] y=relu(x) out=add(y,x) output out }"
    default = compile_source(source).pipelines[0]
    rewritten = compile_source(source, scan_epilogue=True).pipelines[0]
    assert [k.ops for k in default.triton.kernels] == [k.ops for k in rewritten.triton.kernels]
