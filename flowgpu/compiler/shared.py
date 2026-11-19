"""Helpers every layer above `frontend` may use, owned by none of them.

The 2026-10-26 review found four places where the layer rules in
``ARCHITECTURE.md`` were broken by convenience imports:

    ir/reduction_stages.py          -> backend/backends
    analysis/traffic.py             -> backend/backends, optimization/*
    optimization/indexed_fusion.py  -> backend/primitives
    optimization/gather_*.py        -> runtime/cache
    frontend/{operator,type_inference,store_at_rule}.py -> analysis/precision

None of them was a real dependency: each wanted a *pure function of the graph
and the program's names* that happened to live next to a backend.  Those
functions live here instead.  This module imports only ``frontend.ast_node``
(a layer below everything it serves), so importing it never introduces a cycle
and no layer gains a dependency on a layer above it.
"""

import hashlib
from pathlib import Path

from flowgpu.compiler.frontend.ast_node import is_symbol

__all__ = [
    "identifier", "product", "used_names", "Names", "unique_formal",
    "group_operands", "tensor_operands", "scalar_operands", "group_results",
    "store_rotation_chain", "ROW_WISE", "is_pointwise", "row_key", "full_key",
    "REDUCTIONS", "PRIMITIVE_OPS", "VALUE_LAYER_OPS", "INDEX_OPS",
    "compiler_fingerprint", "is_symbol", "dimension_symbols",
]

#: Row-wise reduction anchors (one program per row).
ROW_WISE = ("rms_norm", "rmsnorm_hf", "softmax")

#: Row reductions that take part in the indexed/reduction fusion families.
REDUCTIONS = frozenset({'sum_last', 'max_last'})

#: The value-layer operator names, grouped by the dtype rule they share.
#: `backend/value_layer.py` builds its tables from these, so the front-end
#: registration, the backend tables and the optimization classifier cannot
#: drift about what the layer contains.
COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")
ORDERING = ("abs", "maximum", "minimum")
ELEMENTARY = ("sqrt", "rsqrt", "tanh", "sigmoid", "softplus", "erf",
              "gelu", "gelu_tanh")
BITWISE = ("bitand", "bitor", "bitxor", "shl", "shr")
BITS = ("float_bits", "bits_float")
VALUE_LAYER_OPS = COMPARISONS + ORDERING + ELEMENTARY + BITWISE + BITS

#: Primitive (non-fused) operator names: the fixed part; `backend/primitives.py`
#: appends the value layer.  Kept in the same tuple order the backend uses.
PRIMITIVE_OPS = ('exp', 'exp2', 'log', 'log2', 'to_fp32', 'isfinite', 'where',
                 'zeros_like', 'ones_like', 'neg_inf_like', 'broadcast_like',
                 'cast_like', 'cast_to', 'move_first_last',
                 'gather_row_offset', 'gather_rows', 'div',
                 'sum_last', 'max_last')

#: Operators the indexed fusion may absorb: primitives, the value layer and the
#: one layout op it rewrites.
INDEX_OPS = frozenset(PRIMITIVE_OPS) | frozenset(VALUE_LAYER_OPS) | {'reshape'}


def identifier(name) -> str:
    """A value name as a Python identifier.

    13.4 names temporaries `%0`, which no user can write and no target can
    declare, so anything derived from a value name is sanitized. Both
    backends need this, for the same reason.
    """
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(name)
    )


def product(dimensions) -> str:
    """The product of a shape, with integer factors folded.

    Folding is not cosmetic: `bf16[2,4]` and `bf16[N,D]` reach a backend as
    very different things, a size and an expression. Both backends have to
    make the same distinction, so it is made once, here.
    """
    factors = [str(dimension) for dimension in dimensions]

    if not factors:
        return "1"

    if all(not is_symbol(factor) for factor in factors):
        total = 1
        for factor in factors:
            total *= int(factor)

        return str(total)

    return " * ".join(factors)


def group_operands(graph, group) -> tuple:
    """Values the group reads that something outside it produced, in order."""
    index_of = graph.index_of
    inside = {index_of(node) for node in group.members}
    operands = []

    for node in group.members:
        for value in node.inputs:
            producer = graph.producer_of(value)

            if producer is not None and index_of(producer) in inside:
                continue
            if value not in operands:
                operands.append(value)

    return tuple(operands)


def tensor_operands(graph, values) -> tuple:
    """The operands that are tensors, in their original order."""
    return tuple(value for value in values if graph.values[value].shape)


def scalar_operands(graph, values) -> tuple:
    """The operands that are rank-0 inputs, in their original order.

    They are a different kind of ABI value from a tensor: the caller passes one
    number, the kernel reads it as a scalar, and neither a pointer nor layout
    metadata exists for it. Every backend filters on this rather than treating
    an empty shape as a tensor with no extent.
    """
    return tuple(value for value in values if not graph.values[value].shape)


def unique_formal(hint, used) -> str:
    """A generated formal name that no other formal already carries.

    Program-owned names keep their spelling (`a`), because that is what a
    reader of the generated kernel expects. They can still collide with a name
    a backend invented -- an input called `numel`, or `x_ptr` next to a tensor
    `x` -- and two formals with one name is a kernel that does not parse, so
    the collision is broken here instead of being discovered as a syntax error
    in generated text.
    """
    candidate = str(hint)
    suffix = 2

    while candidate in used:
        candidate = f"{hint}_{suffix}"
        suffix += 1

    used.add(candidate)

    return candidate


def group_results(graph, group) -> tuple:
    """Values the group produces that are not consumed inside it."""
    return tuple(
        node.output
        for node in group.members
        if node.output is not None and node.output not in group.fused_values
    )


def store_rotation_chain(graph, group) -> tuple:
    """Rotation nodes that only readdress the anchor's output, in chain order.

    A trailing ``move_first_last`` chain rooted at the reduction/row-wise
    anchor's output is a pure permutation of where results land; both sweep
    emitters fold it into the store address. The chain must be single-consumer
    and rooted exactly at the anchor so no other reader could observe an
    unrotated intermediate.
    """
    if group.anchor is None or group.anchor.output is None:
        return ()
    chain = []
    members = {node.inputs[0]: node for node in group.members
               if node.op == 'move_first_last'}
    root = group.anchor.output
    while root in members:
        node = members[root]
        consumers = graph.consumers_of(node.output)
        if len(consumers) != 1:
            break
        chain.append(node)
        root = node.output
    return tuple(chain)


def used_names(graph) -> set:
    """Every name a program already uses: its values, and its shape symbols.

    8.3 lets a shape be written with symbols, and a symbol is a NAME like any
    other, so a program may call a dimension `mask` or a value `slot0`.
    """
    used = {identifier(value) for value in graph.values}

    for tensor_type in graph.values.values():
        used |= {
            str(dimension)
            for dimension in tensor_type.shape
            if is_symbol(dimension)
        }

    return used


class Names:
    """The names one generated kernel may use.

    A backend invents names -- registers, loop counters, masks, dimension
    parameters -- in a namespace it does not own, and the program may already
    use any of them. Left alone the collision is silent and the generated code
    still parses; it just means something else. `bf16[N, mask]` with an
    rms_norm used to produce

        mask = offsets < mask
        base = row * mask

    where the second line multiplies the row index by the mask vector: a wrong
    kernel, from a valid program, with no diagnostic. So a backend reserves
    the program's own names first and invents everything else around them.
    This is alpha-renaming, which is the oldest thing a backend does, and both
    backends need the same answer.
    """

    def __init__(self, graph):
        self.used = used_names(graph)

    def reserve(self, name) -> str:
        """A name the program itself uses, kept exactly as it is."""
        self.used.add(name)

        return name

    def fresh(self, hint) -> str:
        """An invented name, renamed only if the program got there first."""
        candidate = hint
        counter = 2

        while candidate in self.used:
            candidate = f"{hint}_{counter}"
            counter += 1

        self.used.add(candidate)

        return candidate

    def numbered(self, prefix) -> str:
        """The next free `t0`, `t1`, ... -- how Triton names registers."""
        counter = 0

        while f"{prefix}{counter}" in self.used:
            counter += 1

        return self.fresh(f"{prefix}{counter}")


def dimension_symbols(graph) -> set:
    """Every symbolic dimension name the program uses.

    Kernel formals share one namespace, so an invented scalar formal must not
    take a name the dimension formals will need -- the collision produced a
    kernel whose `M` was both the row count and the scalar operand value
    (review 2026-10-26).
    """
    return {
        str(dimension)
        for value in graph.values.values()
        for dimension in value.shape
        if is_symbol(dimension)
    }


def is_pointwise(op: str) -> bool:
    """The per-operator fact this analysis needs; see frontend/operator.py.

    A reserved operator never reaches a graph -- it is refused while types are
    checked -- so it does not have to answer this.  Imported lazily: the
    registry lives in `frontend`, and a module-level import would make every
    layer above `frontend` pay for loading it before a graph exists.
    """
    from flowgpu.compiler.core.operators import OPERATORS

    spec = OPERATORS.get(op)

    return spec is not None and spec.pointwise is True


def row_key(shape):
    return tuple(str(d) for d in shape if str(d) != '1')


def full_key(shape):
    return row_key(shape[:-1]) + (str(shape[-1]),)


def compiler_fingerprint():
    """Digest of every implementation-layer module plus the grammar.

    Moved here from `runtime/cache.py` (review P2, layering): the `optimization`
    modules need the same digest to anchor their caches, and they may not import
    `runtime`.
    """
    root = Path(__file__).resolve().parent          # flowgpu/compiler/
    digest = hashlib.sha256()
    paths = [*root.glob('*.py'), root / 'grammar.lark']
    paths += [path for layer in ('frontend', 'ir', 'analysis', 'optimization', 'backend', 'runtime')
              for path in (root / layer).glob('*.py')]
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
