"""Kernels launch in dependency order, not in "first member index" order.

The two coincide for the shapes the compiler was first built against, but not in
general: a fused pointwise region can consume a value defined *after* the
region's first member.  The partitioner used to order groups by their first
member's graph index, so that producer's kernel launched later than its
consumer, which then read an unwritten buffer.

Found on 2026-10-24 by the integer-only FP8 encoder (its reference backend
consumed `float('nan')` from a buffer whose producer ran later).  The fix is a
stable topological sort over inter-group value dependencies
(`fusion.order_by_dependencies`), applied by every pass that rebuilds groups.

The default pipeline's indexed-region fusion happens to merge these shapes into
one kernel, which hides the bug; these tests disable it so the base partitioner
is the thing under test.
"""
import math

import pytest

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

#: The smallest shape that reproduces it: a pointwise chain (`add` -> `mul` ->
#: `mul` -> `add`) fused into one region whose two non-input operands (`q`, `r`)
#: are defined by separate anchor kernels that sit *between* the chain's first
#: member and its consumer.
REPRODUCER = '''
pipeline p {
    input x : fp32[4,8]
    input s : fp32
    a = add(x, s)
    q = exp(x)
    r = exp2(x)
    m1 = mul(a, q)
    m2 = mul(m1, r)
    b = add(m2, q)
    output b
}
'''

PLAIN_PARTITION = dict(index_fusion=False, reduction_fusion=False)


def _assert_topological(pipeline):
    """Every kernel that consumes a value runs after the kernel that wrote it."""
    produced_by = {}
    for index, kernel in enumerate(pipeline.triton.kernels):
        for value in kernel.results:
            produced_by[value] = index
    for index, kernel in enumerate(pipeline.triton.kernels):
        for value in kernel.operands:
            producer = produced_by.get(value)
            assert producer is None or producer < index, (
                f"kernel {index} ({kernel.ops}) consumes '{value}' whose producer "
                f"is kernel {producer}")


def test_the_partitioner_orders_the_reproducer_by_dependencies():
    pipeline = compile_source(REPRODUCER, **PLAIN_PARTITION).pipelines[0]
    assert len(pipeline.triton.kernels) == 3
    _assert_topological(pipeline)


def test_the_reference_does_not_read_an_unwritten_buffer():
    """Before the fix this raised `ValueError: cannot convert float NaN to
    integer`, because the fused group ran first and read q/r as NaN."""
    pipeline = compile_source(REPRODUCER, **PLAIN_PARTITION).pipelines[0]
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    produced = namespace['run_p']([2.0] * 32, 0.5)['b']
    expected = (2.0 + 0.5) * math.exp(2.0) * math.exp2(2.0) + math.exp(2.0)
    assert produced == pytest.approx([expected] * 32, rel=1e-6)


@pytest.mark.parametrize("options", [dict(), PLAIN_PARTITION])
def test_pipelines_are_always_topologically_ordered(options):
    """Both the default (indexed-region fused) and the plain partition."""
    pipeline = compile_source(REPRODUCER, **options).pipelines[0]
    _assert_topological(pipeline)
