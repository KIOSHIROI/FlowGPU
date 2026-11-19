"""In-kernel common subexpression elimination: the switch.

A fused region may compute the same expression twice through different value
names -- a helper recomputed at each use.  Two structurally identical
subexpressions with identical operands are one device instruction, so the
emitters hash-cons them **inside one kernel**.  This is register reuse, not a
buffer: the fusion partition is untouched (sharing across kernels would have to
materialise, which is the partitioner's decision, not this pass's).

Enabled by default.  The flag exists so a performance bisect or a bug report can
attribute an emitted kernel to this pass without editing the emitters; the tests
in `test/test_cse.py` exercise both sides.
"""

ENABLED = True


def key(op, operands, dtype, part, attributes):
    """The canonical identity of one emitted expression.

    Two nodes with the same operator, the same *operand registers* (not operand
    value names), the same result dtype, the same `part`, and the same static
    attributes compute the same expression.  Recursion makes this structural:
    identical subtrees yield identical operand registers, so they hash together
    without walking the graph twice.

    `attributes` is the `OperationAttributes` object every DSL node carries; its
    fields are default for a pure elementwise node, and `repr` keeps the key
    comparable.  A pass that changes an attribute changes the key, so the two
    expressions no longer merge.
    """
    return (op, tuple(operands), dtype, part, repr(attributes))


class Sharing:
    """The shared-expression table of one kernel's member loop.

    Emitters whose fused members each produce one line (rowwise/matmul/rope and
    attention epilogues, and the reference counterparts) use this instead of
    writing their own dict: `previous` is the variable already computing the same
    expression, `record` remembers a freshly bound one.  A hit means the member
    is bound to the earlier variable and no line is emitted.
    """

    def __init__(self):
        self.entries = {}

    def previous(self, node, operands, dtype):
        if not ENABLED:
            return None
        return self.entries.get(key(node.op, operands, dtype, node.part,
                                    node.attributes))

    def record(self, node, operands, dtype, variable):
        if ENABLED:
            self.entries[key(node.op, operands, dtype, node.part,
                             node.attributes)] = variable
        return variable


#: Operators whose in-kernel hash-consing is enabled.  The device and the
#: reference indexed lowerings used to disagree here (the reference merged any
#: structurally identical call, the device only this set), so a fused region
#: with two identical `exp` calls emitted two device instructions and one
#: reference function (review P2, 2026-10-26).  One predicate, both lowerings.
_OPS = None


def ops():
    global _OPS
    if _OPS is None:
        from compiler.backend.triton_backend import POINTWISE_EXPRESSION
        from compiler.backend.value_layer import NAMES as VALUE_LAYER
        _OPS = (frozenset(POINTWISE_EXPRESSION) | frozenset(VALUE_LAYER)
                | frozenset({'where', 'div', 'isfinite', 'cast_to', 'cast_like',
                             'to_fp32', 'ones_like', 'zeros_like', 'neg_inf_like'}))
    return _OPS


def eligible(op):
    """Whether this operator may share an earlier identical expression."""
    return ENABLED and op in ops()
