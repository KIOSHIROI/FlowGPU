from compiler.core.ast import (
    Assignment,
    Call,
    PipelineDecl,
    InputDecl,
    OutputDecl,
    PoisonType,
    Symbol,
    is_poisoned,
)
from compiler.core.diagnostic import CompileError, RuleError, DiagnosticKind
from compiler.core.types import is_symbol
from compiler.core.symbol_table import SymbolTable
from compiler.core.type_inference import TypeInference


class SymbolTableBuilder:
    """Build the symbol table in one source-ordered pass.

    This pass is also the type checker. Every assignment infers the type of
    its right-hand side *before* the name is bound, and the inferred type is
    stored on the symbol -- so later statements can use it.

    Consequence (and language rule): the body must be ordered
    def-before-use. A forward reference is reported as an undefined symbol,
    which is the natural reading of a dataflow program: a value exists only
    after its producer.

    Every diagnostic leaves here as a CompileError carrying both a category
    and a position: the category comes from the layer that detected the
    failure, the position from the statement or expression node it happened
    in, and the pipeline name from the loop below.
    """

    def __init__(self, *, allow_symbolic_split=False):
        self._allow_symbolic_split = allow_symbolic_split

    def build(self, pipeline: PipelineDecl) -> SymbolTable:
        try:
            table = self._build(pipeline)
            self._reject_dimension_name_collisions(table)
            return table
        except CompileError as error:
            raise error.with_pipeline(pipeline.name) from error
        except RuleError as error:
            # A rejection that never met the node it came from. Keep the
            # category it declared rather than inventing one from the class.
            raise CompileError(
                error.kind, error.message, pipeline=pipeline.name
            ) from error

    def _build(self, pipeline: PipelineDecl) -> SymbolTable:
        table = SymbolTable()
        inferencer = TypeInference(
            table, allow_symbolic_split=self._allow_symbolic_split)

        found = []

        for statement in pipeline.statements:
            try:
                if isinstance(statement, InputDecl):
                    self._define_input(table, statement)

                elif isinstance(statement, Assignment):
                    self._define_assignment(
                        table, inferencer, statement
                    )

                elif isinstance(statement, OutputDecl):
                    self._check_output(table, statement)
            except CompileError as error:
                # Keep walking. When an assignment fails, its target is poisoned
                # so downstream expressions reading this value do not cascade.
                # Genuinely independent subsequent errors continue to be collected.
                found.append(error.with_pipeline(pipeline.name))
                if isinstance(statement, Assignment):
                    # v0.2: poison every name a multi-assignment binds, or a
                    # later reader of the second name cascades as undefined.
                    for name in statement.names:
                        self._poison(table, name)
                elif isinstance(statement, InputDecl):
                    self._poison(table, statement.name)

        if found:
            raise CompileError(
                found[0].kind,
                found[0].message,
                position=found[0].position,
                pipeline=found[0].pipeline,
                others=tuple(found[1:]),
            )

        return table

    def _define_input(self, table: SymbolTable, statement: InputDecl):
        if any(d.isdigit() and int(d) == 0 for d in statement.type.shape):
            raise CompileError(DiagnosticKind.SHAPE, "tensor dimensions must be positive", position=statement.position)
        if (
            statement.type.is_scalar
            and statement.type.dtype not in {"fp32", "int32", "int64"}
        ):
            # The host carries a scalar as a Python number, and a Python
            # number reaches the device as fp32 or as an integer. A bf16/fp16
            # declaration could not be honoured without a per-call rounding
            # step, so it is refused rather than silently ignored -- the exact
            # precision lie that made Task160's bf16-declared scalars drift.
            raise CompileError(
                DiagnosticKind.TYPE,
                f"scalar input '{statement.name}' must be declared fp32, "
                f"int32 or int64; a {statement.type.dtype} scalar would be "
                f"carried at fp32 and its declared precision would not hold",
                position=statement.position,
            )
        self._define(
            table,
            Symbol(
                name=statement.name,
                kind="input",
                type=statement.type,
            ),
            statement,
        )

    def _define_assignment(
        self,
        table: SymbolTable,
        inferencer: TypeInference,
        statement: Assignment,
    ):
        # Infer first, bind second. The inferred type is the only source of
        # truth for this name -- the user never re-declares it.
        count = len(statement.names)

        # v0.2: a multi-value operator under a single name, or several
        # names under any call, both take the multi path. The count is the
        # operator's arity selector, and a single-name call to a multi
        # operator is rejected here rather than in the rule -- the
        # mismatch is a property of the call site.
        spec = None
        if isinstance(statement.value, Call):
            from compiler.core.operators import OPERATORS  # deferred: avoids a cycle
            spec = OPERATORS.get(statement.value.name)

        is_multi_call = bool(getattr(spec, "multi", False))

        if count > 1 or is_multi_call:
            if not isinstance(statement.value, Call):
                raise CompileError(
                    DiagnosticKind.OPERATOR,
                    f"a {type(statement.value).__name__} cannot be "
                    f"assigned to {count} names",
                    position=statement.position,
                )
            if not is_multi_call:
                raise CompileError(
                    DiagnosticKind.OPERATOR,
                    f"operator '{statement.value.name}' produces one value "
                    f"and cannot be assigned to {count} names",
                    position=statement.position,
                )
            if count == 1:
                # The call site mismatch: a multi-value operator under a
                # single name has no result to select, so the statement is
                # told what it needs rather than silently selecting part 0.
                # The position is the operator, not the assignment: v0.1's
                # reserved-operator error pointed at the call, and a repair
                # loop reads the column to find it.
                raise CompileError(
                    DiagnosticKind.OPERATOR,
                    f"operator '{statement.value.name}' produces several "
                    f"values; assign several names, like "
                    f"a, b = {statement.value.name}(...)",
                    position=statement.value.position or statement.position,
                )
            inferred_list = inferencer.infer_multi(statement.value, count)
            for name, inferred in zip(statement.names, inferred_list):
                if is_poisoned(inferred):
                    self._poison(table, name)
                    continue
                self._define(
                    table,
                    Symbol(name=name, kind="value", type=inferred),
                    statement,
                )
            return

        inferred = inferencer.infer(statement.value)

        # A reshape's symbolic target dims must be dimensions an input
        # declares: the host binds each symbol off an input tensor's shape
        # (8.3), so a symbol no input carries has no value at run time and
        # no host can allocate the slot. An input NAMED like the symbol
        # does not count -- the name is a tensor, not a dimension.
        if (
            isinstance(statement.value, Call)
            and statement.value.name == "reshape"
            and not is_poisoned(inferred)
        ):
            declared = {
                str(dimension)
                for symbol in table.symbols.values()
                if symbol.kind == "input" and symbol.type is not None
                for dimension in symbol.type.shape
                if is_symbol(dimension)
            }
            for dimension in inferred.shape:
                if is_symbol(dimension) and str(dimension) not in declared:
                    raise CompileError(
                        DiagnosticKind.DEPENDENCY,
                        f"reshape dimension '{dimension}' is not declared "
                        f"by any input; a symbol is bound by the input "
                        f"tensor that carries it",
                        position=statement.value.position or statement.position,
                    )

        if is_poisoned(inferred):
            self._poison(table, statement.names[0])
            return

        self._define(
            table,
            Symbol(
                name=statement.names[0],
                kind="value",
                type=inferred,
            ),
            statement,
        )

    def _reject_dimension_name_collisions(self, table: SymbolTable) -> None:
        """A value name may not also be a dimension symbol (review P1-5).

        The two live in one namespace in every emitted target: the reference
        host would generate `def run_p(x, N, N)` (a SyntaxError), a kernel
        would shadow the dimension formal with the value, and the host would
        read the wrong one.  Rejecting the collision at the symbol table is one
        rule for all three lowerings instead of three separate renames.
        """
        dimensions = {
            str(dimension)
            for symbol in table.symbols.values()
            if symbol.type is not None
            for dimension in symbol.type.shape
            if is_symbol(dimension)
        }
        for symbol in table.symbols.values():
            if symbol.kind == 'value' and symbol.name in dimensions:
                raise CompileError(
                    DiagnosticKind.DEPENDENCY,
                    f"the name '{symbol.name}' is used both as a {symbol.kind} "
                    "and as a dimension symbol; rename one of them",
                )

    def _define(self, table, symbol, statement):
        try:
            table.define(symbol)
        except RuleError as error:
            raise CompileError(
                error.kind, error.message, position=statement.position
            ) from error

    def _check_output(self, table: SymbolTable, statement: OutputDecl):
        try:
            symbol = table.lookup(statement.name)
        except RuleError as error:
            raise CompileError(
                error.kind, error.message, position=statement.position
            ) from error

        if is_poisoned(symbol.type):
            # The producer of this output was already rejected. Do not report
            # an additional cascading failure.
            return

        if symbol.type is None:
            # Every path that defines a symbol supplies its type, so this is
            # an invariant of this pass rather than something a program can
            # break.
            raise AssertionError(
                f"symbol '{statement.name}' was defined without a type"
            )

        if symbol.type.is_scalar:
            # A scalar is a host number, not a buffer: there is nothing for
            # the host to return and no slot to allocate. Refuse it here, where
            # the program can still be repaired, rather than at allocation.
            raise CompileError(
                DiagnosticKind.TYPE,
                f"output '{statement.name}' is a scalar "
                f"({symbol.type.dtype}); a pipeline output must be a tensor",
                position=statement.position,
            )

    def _poison(self, table: SymbolTable, name: str):
        if table.contains(name):
            return
        table.define(
            Symbol(
                name=name,
                kind="poison",
                type=PoisonType(),
            )
        )
