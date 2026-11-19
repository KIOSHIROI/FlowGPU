"""Lowering: a type-checked pipeline becomes a Dataflow IR (Design.md 13.4).

The step exists because the tree has *expressions* and the graph has
*values*. Every operand of an IR node is a name (7.1), so a nested operator
call cannot stay nested:

    y = add(mul(r, s), x)      becomes      %0 = mul(r, s)
                                           y  = add(%0, x)

That is the ordinary name-everything lowering, and it is why 13.4 is a real
stage rather than a second reading of the tree.

The graph constructor validates structural invariants as INTERNAL diagnostics.
The pipeline handed to `build` has already been parsed, name-checked and
typed. Unexpected lowering failures are compiler defects rather than bad
programs; unsupported AST shapes still raise assertions.
"""

from compiler.core.ast import Assignment, Call, InputDecl, Name, OutputDecl
from compiler.ir.dataflow import (
    INPUT,
    OUTPUT,
    DataflowGraph,
    resolve_alias,
)
from compiler.core.type_inference import TypeInference
from compiler.ir.semantic_ir import SemanticNode, OperationAttributes, SemanticProgram, normalize
from compiler.ir.index_maps import canonical_permutation


class DataflowBuilder:
    """Turn one checked pipeline into a graph."""

    def build(self, pipeline, symbols) -> DataflowGraph:
        return normalize(self.build_semantic(pipeline, symbols))[0]

    def __init__(self, *, allow_symbolic_split=False):
        self._allow_symbolic_split = allow_symbolic_split

    def build_semantic(self, pipeline, symbols, *, shape_constraints=()):
        """Preserve checked operation attributes before executable lowering."""
        self._symbols = symbols
        self._types = TypeInference(
            symbols, allow_symbolic_split=self._allow_symbolic_split)
        self._values = {}
        self._nodes = []
        self._aliases = {}
        self._temporaries = 0

        for statement in pipeline.statements:
            self._lower_statement(statement)

        return SemanticProgram(
            pipeline=pipeline.name,
            values=self._values,
            nodes=tuple(self._nodes),
            aliases=self._aliases,
            shape_constraints=tuple(shape_constraints),
        )

    # --- statements -----------------------------------------------------

    def _lower_statement(self, statement):
        if isinstance(statement, InputDecl):
            self._values[statement.name] = statement.type
            self._append(INPUT, [], statement.name)
            return

        if isinstance(statement, Assignment):
            self._lower_assignment(statement)
            return

        if isinstance(statement, OutputDecl):
            # The output node reads the value the name denotes, not the name:
            # an alias resolved here is what makes 13.7 see one live value
            # rather than two.
            resolved = self._resolve(statement.name)
            self._append(OUTPUT, [resolved], None)
            return

        raise AssertionError(f"cannot lower {type(statement).__name__}")

    def _lower_assignment(self, statement):
        value = statement.value

        if isinstance(value, Name):
            # `y = x` computes nothing and materializes nothing, so no node is
            # emitted: it is a second name for a value that already exists.
            # Emitting a copy node here would invent a buffer, and the whole
            # point of 7.3 is not to invent buffers.
            if len(statement.names) != 1:
                raise AssertionError(
                    "an alias assignment cannot bind several names"
                )
            self._values[statement.names[0]] = self._declared_type(statement.names[0])
            self._aliases[statement.names[0]] = self._resolve(value.name)
            return

        if isinstance(value, Call):
            # v0.2: a multi-name assignment lowers to one node per name, all
            # sharing the same op and inputs -- every value still has
            # exactly one producer, and the graph keeps "one node produces
            # one value" as its representation rather than widening Node.
            # The part index is the value's position in the assignment's
            # name list, which is the language's only result selector; it
            # rides on the node as metadata the backends read.
            if len(statement.names) > 1:
                count = len(statement.names)
                operands = [self._lower_expression(arg) for arg in value.args if isinstance(arg, (Name, Call))]
                if value.name == 'linear_decode_forward':
                    if count != 2:
                        raise AssertionError('linear_decode_forward requires exactly two outputs')
                    operands = [self._lower_expression(arg) for arg in value.args if isinstance(arg, (Name, Call))]
                    for index, name in enumerate(statement.names):
                        self._values[name] = self._declared_type(name)
                        self._append_part(value.name, operands, name, index,
                                          attributes=OperationAttributes(block_size=value.args[6].value,
                                                                         result_group=f'%linear_decode_{self._temporaries}',
                                                                         result_index=index,
                                                                         result_role=('output', 'cache')[index]))
                    return
                if value.name == 'segment_rank':
                    if count != 2:
                        raise AssertionError('segment_rank requires exactly two outputs')
                    group_id = f'%segment_rank_{self._temporaries}'
                    for index, name in enumerate(statement.names):
                        self._values[name] = self._declared_type(name)
                        self._append_part(
                            value.name, operands, name, index,
                            attributes=OperationAttributes(
                                segment=tuple(a.value for a in value.args[1:]),
                                result_group=group_id, result_index=index,
                                result_role=('counts', 'positions')[index]))
                    return
                if value.name == 'store_cache_4d':
                    if count != 2:
                        raise AssertionError('store_cache_4d requires exactly two outputs')
                    group_id = f'%store_cache_{self._temporaries}'
                    for index, name in enumerate(statement.names):
                        self._values[name] = self._declared_type(name)
                        self._append_part(
                            value.name, operands, name, index,
                            attributes=OperationAttributes(
                                result_group=group_id, result_index=index,
                                result_role=('key', 'value')[index],
                                page_size=value.args[5].value))
                    return
                for index, name in enumerate(statement.names):
                    self._values[name] = self._declared_type(name)
                    self._append_part(value.name, operands, name, index)
                return

            self._lower_call(
                value,
                output=statement.names[0],
                tensor_type=self._declared_type(statement.names[0]),
            )
            return

        raise AssertionError(
            f"cannot lower assignment of {type(value).__name__}"
        )

    # --- expressions ----------------------------------------------------

    def _lower_expression(self, expression) -> str:
        """Reduce an expression to one value name, emitting nodes if needed."""
        if isinstance(expression, Name):
            return self._resolve(expression.name)

        if isinstance(expression, Call):
            temporary = self._new_temporary()

            # The type is inferred rather than looked up: a temporary is not
            # a declared name, so the symbol table has never seen it. The
            # rules that infer it are the same ones that already checked this
            # expression, so this cannot fail differently here.
            self._lower_call(
                expression,
                output=temporary,
                tensor_type=self._types.infer(expression),
            )
            return temporary

        raise AssertionError(
            f"cannot lower expression of type {type(expression).__name__}"
        )

    def _lower_call(self, call, output, tensor_type):
        from compiler.core.operators import AXIS_OPERATORS, normalize_axis
        attributes = OperationAttributes()
        from compiler.ir.program_scalar_ir import (
            PROGRAM_SCALAR_INDEX,
            STAGED_PROGRAM_SCALAR_OP,
        )
        if call.name == STAGED_PROGRAM_SCALAR_OP:
            # The staged program-uniform scalar primitive: rank-0 result, one
            # device load per program, integer table. The spec rides on the
            # node so every emitter and the analysis read one representation;
            # the user-visible operator is registered on top of this later.
            from compiler.ir.program_scalar_ir import ProgramScalar
            if len(call.args) != 1 or not isinstance(call.args[0], Name):
                raise AssertionError(
                    'the staged program_scalar primitive takes one table operand'
                )
            table = self._lower_expression(call.args[0])
            attributes = OperationAttributes(program_scalar=ProgramScalar(
                table=table, index=PROGRAM_SCALAR_INDEX[STAGED_PROGRAM_SCALAR_OP],
                # the staged spec's dtype is the rule's result dtype; the
                # ProgramScalar default (int64) disagreed with an int32 table
                # (review P2, 2026-10-26)
                dtype=tensor_type.dtype))
            operands = [table]
            self._values[output] = tensor_type
            self._append(call.name, operands, output, attributes=attributes)
            return
        if call.name in AXIS_OPERATORS:
            xtype = self._types.infer(call.args[0])
            axis = normalize_axis(xtype, call.args[-1])
            attributes = OperationAttributes(axis=axis)
        elif call.name in {'move_first_last', 'transpose'}:
            rank = len(self._types.infer(call.args[0]).shape)
            axes = canonical_permutation(call.name,rank)
            attributes = OperationAttributes(permutation=axes)
        elif call.name == 'reshape':
            attributes = OperationAttributes(target_shape=tuple(call.args[1].shape))
        elif call.name == 'store_cache_4d':
            page_size = call.args[5]
            attributes = OperationAttributes(page_size=page_size.value)
        elif call.name == 'linear_decode_forward':
            attributes = OperationAttributes(block_size=call.args[6].value)
        elif call.name == 'pad_cat':
            attributes = OperationAttributes(pads=tuple(a.value for a in call.args[2:]))
        elif call.name == 'patchify':
            attributes = OperationAttributes(pads=tuple(a.value for a in call.args[1:]))
        elif call.name == 'segment_rank':
            attributes = OperationAttributes(segment=tuple(a.value for a in call.args[1:]))
        # Operands first, so a node is always created after the nodes that
        # produce its inputs. That is what makes construction order a
        # topological order. A shape literal is not an operand: it names
        # reshape's target shape and never denotes a value, so it is routed
        # to the rule rather than lowered.
        operands = [
            self._lower_expression(argument)
            for argument in call.args
            if isinstance(argument, (Name, Call))
        ]

        self._values[output] = tensor_type
        self._append(call.name, operands, output, attributes=attributes)

    # --- helpers --------------------------------------------------------

    def _append(self, op, inputs, output, *, attributes=None):
        self._nodes.append(SemanticNode(op=op, inputs=tuple(inputs), output=output,
                                      attributes=attributes or OperationAttributes()))

    def _append_part(self, op, inputs, output, part, *, attributes=None):
        self._nodes.append(
            SemanticNode(op=op, inputs=tuple(inputs), output=output, part=part,
                         attributes=attributes or OperationAttributes())
        )

    def _new_temporary(self) -> str:
        # `NAME` is /[a-zA-Z_][a-zA-Z0-9_]*/ in the grammar, so a `%` name
        # cannot be written by a user and cannot collide with one.
        temporary = f"%{self._temporaries}"
        self._temporaries += 1
        return temporary

    def _declared_type(self, name):
        tensor_type = self._symbols.lookup(name).type

        if tensor_type is None:
            # The symbol table writes a type back for everything it defines,
            # including inputs, so this cannot happen.
            raise AssertionError(f"'{name}' reached lowering without a type")

        return tensor_type

    def _resolve(self, name: str) -> str:
        return resolve_alias(self._aliases, name)
