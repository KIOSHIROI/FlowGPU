"""Typed operation intent before scheduling and storage decisions.

Static reduction axes and logical permutations survive frontend construction.
This IR contains no tile, grid, slot, device pointer, or materialization policy.
The normalizer is the single boundary to the established executable dataflow.
"""
from dataclasses import dataclass

from compiler.core.literals import IntLiteral
from compiler.ir.dataflow import DataflowGraph, Node, INPUT, OUTPUT
from compiler.core.diagnostic import CompileError, DiagnosticKind, RuleError
from compiler.core.operators import AXIS_OPERATORS, OPERATORS
from compiler.ir.index_maps import canonical_permutation


@dataclass(frozen=True)
class OperationAttributes:
    axis: int | None = None
    permutation: tuple = ()
    target_shape: tuple = ()
    result_group: str | None = None
    result_index: int | None = None
    result_role: str | None = None
    page_size: int | None = None
    block_size: int | None = None
    # pad_cat's six F.pad-order literals: (w_l, w_r, h_t, h_b, t_front, t_back).
    # They are part of the operation ABI rather than operands, exactly like
    # page_size, and every one of them must be a literal because the output
    # extents are computed from them.
    pads: tuple = ()
    # segment_rank's two ABI literals: (num_experts, m_max). The output extents
    # are computed from the first and the destination space from the second, so
    # neither can be a symbolic operand -- the same reason pages and pads are
    # attributes.
    segment: tuple = ()
    # A rank-0 program-uniform value: one integer scalar read per program from
    # a 1-D device table (`ir/program_scalar_ir.ProgramScalar`). It is not a
    # tensor: every lane of a program sees the same number, its load address is
    # uniform, and an address that uses it as a base is `scalar_base +
    # affine(lane)`. The attribute is the IR contract; the lowering emits the
    # one load and folds the value into a base, and `analysis/write_address`
    # checks the emitted store addresses.
    program_scalar: object = None


@dataclass(frozen=True)
class SemanticNode:
    op: str
    inputs: tuple
    output: str | None
    part: int | None = None
    attributes: OperationAttributes = OperationAttributes()


@dataclass
class SemanticProgram:
    pipeline: str
    values: dict
    nodes: tuple
    aliases: dict
    shape_constraints: tuple = ()

    def __post_init__(self):
        self.validate()

    def validate(self):
        # Share structural validation rather than maintaining a second SSA rule.
        DataflowGraph(self.pipeline, self.values,
                      [Node(n.op, list(n.inputs), n.output, n.part, n.attributes) for n in self.nodes],
                      self.aliases)
        def fail(message):
            raise CompileError(DiagnosticKind.INTERNAL, 'invalid semantic IR: '+message,
                               pipeline=self.pipeline)
        for node in self.nodes:
            attrs = node.attributes
            if not isinstance(attrs, OperationAttributes):
                fail(f'{node.op} has untyped attributes')
            if node.op in AXIS_OPERATORS:
                if len(node.inputs) != OPERATORS[node.op].arity-1:
                    fail(f'{node.op} has incorrect tensor operand count')
                rank = len(self.values[node.inputs[0]].shape)
                if type(attrs.axis) is not int or not 0 <= attrs.axis < rank:
                    fail(f'{node.op} needs a normalized static axis')
                try:
                    result = OPERATORS[node.op].infer(
                        [self.values[x] for x in node.inputs]+[IntLiteral(attrs.axis)])
                except RuleError as error:
                    fail(str(error))
                if result != self.values[node.output]:
                    fail(f'{node.op} result type disagrees with its axis')
            elif attrs.axis is not None:
                fail(f'{node.op} has an unexpected axis')
            if node.op in {'move_first_last', 'transpose'}:
                shape = self.values[node.inputs[0]].shape
                expected = canonical_permutation(node.op,len(shape))
                if (attrs.permutation != expected or
                        any(type(a) is not int for a in attrs.permutation)):
                    fail(f'{node.op} has an invalid permutation')
                if self.values[node.output].shape != [shape[a] for a in expected]:
                    fail(f'{node.op} result shape disagrees with its permutation')
            elif attrs.permutation:
                fail(f'{node.op} has an unexpected permutation')
            if node.op == 'reshape':
                if attrs.target_shape != tuple(self.values[node.output].shape):
                    fail('reshape target shape disagrees with result type')
            elif attrs.target_shape:
                fail(f'{node.op} has an unexpected target shape')

    def as_dict(self):
        return dict(stage='semantic', pipeline=self.pipeline,
                    shape_constraints=[dict(operator=c.operator, left=c.left, right=c.right)
                                       for c in self.shape_constraints],
                    aliases=dict(self.aliases),
                    nodes=[dict(op=n.op, inputs=list(n.inputs), output=n.output, part=n.part,
                                axis=n.attributes.axis,
                                permutation=list(n.attributes.permutation),
                                target_shape=list(n.attributes.target_shape),
                                result_type=(dict(dtype=self.values[n.output].dtype,
                                                  shape=list(self.values[n.output].shape))
                                             if n.output is not None else None)) for n in self.nodes])


@dataclass(frozen=True)
class NormalizationDecision:
    value: str
    action: str
    reason: str


def normalize(program, *, fold_views=True):
    """Expand axes late and cancel proven identity permutations.

    Composition uses axis identities, never equal dimension sizes (a square
    transpose remains a transpose). Reshapes are not canceled: they may carry
    runtime element-count obligations. Input dtype boundaries remain operations.
    """
    if type(fold_views) is not bool:
        raise ValueError('fold_views must be a bool')
    program.validate()
    values = dict(program.values)
    aliases = dict(program.aliases)
    nodes, decisions, views = [], [], {}
    temporary = 0
    from compiler.ir.dataflow import resolve_alias
    exported = {resolve_alias(program.aliases, n.inputs[0])
                for n in program.nodes if n.op == OUTPUT}

    def resolve(name):
        from compiler.ir.dataflow import resolve_alias
        return resolve_alias(aliases, name)

    def fresh():
        nonlocal temporary
        while True:
            name = f'%axis{temporary}'
            temporary += 1
            if name not in values:
                return name

    def emit(op, inputs, output, part=None):
        inputs = [resolve(x) for x in inputs]
        if fold_views and op in {'move_first_last', 'transpose'}:
            rank = len(values[inputs[0]].shape)
            permutation = canonical_permutation(op,rank)
            root, previous = views.get(inputs[0], (inputs[0], tuple(range(rank))))
            composed = tuple(previous[a] for a in permutation)
            if composed == tuple(range(rank)) and output not in exported:
                if values[output] != values[root]:
                    raise AssertionError('identity permutation changed tensor type')
                aliases[output] = root
                decisions.append(NormalizationDecision(output, 'cancel_identity_view',
                    'composed axis permutation is identity; no rounding or reshape crossed'))
                return
            views[output] = (root, composed)
        attrs = next((n.attributes for n in program.nodes
                      if n.output == output and n.op == op), None)
        if attrs is None and op in {'move_first_last','transpose'}:
            attrs=OperationAttributes(permutation=canonical_permutation(op,len(values[inputs[0]].shape)))
        nodes.append(Node(op, inputs, output, part, attrs))

    def rotate(value, output):
        from compiler.core.types import TensorType
        t = values[value]
        values[output] = TensorType(t.dtype, t.shape[1:]+t.shape[:1])
        emit('move_first_last', [value], output)
        return resolve(output)

    for node in program.nodes:
        if node.op not in AXIS_OPERATORS:
            emit(node.op, node.inputs, node.output, node.part)
            continue
        x = resolve(node.inputs[0])
        rank = len(values[x].shape)
        count = (node.attributes.axis+1) % rank
        for _ in range(count):
            x = rotate(x, fresh())
        op = AXIS_OPERATORS[node.op]
        inputs = [x]+[resolve(v) for v in node.inputs[1:]]
        inverse = (rank-count) % rank
        result = fresh() if inverse else node.output
        values[result] = OPERATORS[op].infer([values[v] for v in inputs])
        emit(op, inputs, result)
        for step in range(inverse):
            result = rotate(result, node.output if step == inverse-1 else fresh())
        decisions.append(NormalizationDecision(node.output, 'lower_static_axis',
            f'{node.op} axis={node.attributes.axis}; canonical {op}; logical layout retained until normalization'))

    # Only orphaned permutations are removed. Keep computations, input contracts,
    # reshape checks and alias targets, including separately observable views.
    while fold_views:
        consumed = {v for n in nodes for v in n.inputs} | {resolve(v) for v in aliases}
        unused = [n for n in nodes if n.op in {'move_first_last', 'transpose'}
                  and n.output not in consumed]
        if not unused:
            break
        for n in unused:
            nodes.remove(n)
            values.pop(n.output)
            decisions.append(NormalizationDecision(n.output, 'remove_unused_view',
                'view has no consumer after normalization'))
    aliases = {name: resolve(name) for name in aliases}
    return DataflowGraph(program.pipeline, values, nodes, aliases), tuple(decisions)
