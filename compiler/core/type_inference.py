from compiler.core.ast import (
    Call,
    Name,
    PoisonType,
    ShapeLiteral,
    TensorType,
    IntLiteral,
    DTypeLiteral,
    is_poisoned,
)
from compiler.core.diagnostic import CompileError, DiagnosticKind, RuleError

from compiler.core.dtype_rules import check_accumulation
from compiler.core.operators import (OPERATORS, SCALAR_ARGUMENT_OPERATORS,
                                    SCALAR_BROADCAST_OPERATORS)
from compiler.core.program_scalar import PROGRAM_UNIFORM_OPERATORS
from compiler.core.symbol_table import SymbolTable

# Operators whose final argument may name a static axis.
AXIS_LITERAL_OPERATORS = {'sum_axis', 'max_axis', 'softmax_axis', 'rms_norm_axis',
                          'cumsum_axis'}
# Operators whose static ABI ends in that many integer literals.
#   pad_cat(x, cache, w_l, w_r, h_t, h_b, t_front, t_back)
# The pads decide the output extents (T + t_front + t_back etc.), so they are
# part of the operation rather than operands -- the same reason `reshape`
# carries a shape literal.
STATIC_INT_TAIL = {'pad_cat': 6, 'patchify': 2, 'topk_last': 1,
                   'segment_rank': 2,
                   # the multi-result cache operators end in a static
                   # ABI literal too (page_size / BLOCK_SIZE)
                   'store_cache_4d': 1, 'linear_decode_forward': 1}



class TypeInference:
    """Type an expression, and say where it failed.

    The operator rules below this layer are pure: they take types and reject
    them, and never see the program (operator.py). This layer holds the AST
    node, so this is where a rejection acquires the position of the operator
    the user wrote.

    Operand-count and unknown-operator failures are decided here rather than
    in a rule, because they are properties of the call site, not of the
    operand types.
    """

    def __init__(self, table: SymbolTable, *, allow_symbolic_split=False):
        self.table = table
        # Opt-in language capability: a symbolic last axis is admitted by
        # `split` only when the caller enabled the split-view lowering.
        self.allow_symbolic_split = allow_symbolic_split

    def infer(self, expression):
        if isinstance(expression, Name):
            return self._infer_name(expression)

        if isinstance(expression, Call):
            return self._infer_call(expression)

        if isinstance(expression, ShapeLiteral):
            # A shape is not a value. The one place it means anything is
            # reshape's second argument, and _infer_call routes that
            # raw to the rule rather than through this method.
            raise CompileError(
                DiagnosticKind.TYPE,
                "a shape literal only names reshape's target shape; it is "
                "not a tensor",
                position=expression.position,
            )

        if isinstance(expression, IntLiteral):
            raise CompileError(DiagnosticKind.TYPE, 'an integer literal only names a static axis; it is not a tensor', position=expression.position)

        if isinstance(expression, DTypeLiteral):
            # Like a shape literal: a dtype names a cast target, not a value.
            raise CompileError(
                DiagnosticKind.TYPE,
                "a dtype literal only names cast_to's target dtype; it is not a tensor",
                position=expression.position,
            )

        # Not a user error: reaching this means the AST grew a node kind this
        # pass was not taught about.
        raise TypeError(
            f"Unsupported expression: "
            f"{type(expression).__name__}"
        )

    def _infer_name(self, expression):
        try:
            symbol = self.table.lookup(expression.name)
        except RuleError as error:
            raise self._at(error, expression) from error

        if symbol.type is None:
            raise ValueError(
                f"Symbol '{expression.name}' has no type"
            )

        return symbol.type

    def _infer_call(self, expr):
        spec = OPERATORS.get(expr.name)

        if spec is None:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"Unknown operator '{expr.name}'",
                position=expr.position,
            )

        if getattr(spec, "multi", False):
            # `z = add(split(x), y)`: a multi-result operator has no single
            # value to use as an operand, and the rule's signature expects the
            # assignment's name count.  Report it by name instead of letting the
            # rule raise a TypeError (review P1-6, 2026-10-26).
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"operator '{expr.name}' produces several values; assign "
                f"several names, like a, b = {expr.name}(...), and use those "
                "names as operands",
                position=expr.position,
            )

        if len(expr.args) != spec.arity:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"Operator '{expr.name}' expects "
                f"{spec.arity} arguments, "
                f"got {len(expr.args)}",
                position=expr.position,
            )

        # A shape literal reaches the rule raw: reshape is v0.2's only
        # shape-consuming operator, and its rule wants the literal, not an
        # inferred type. Anywhere else the literal is an error by name --
        # at the argument it was written, not in the rule's internals.
        args = []
        tail = STATIC_INT_TAIL.get(expr.name, 0)
        for index, arg in enumerate(expr.args):
            if isinstance(arg, IntLiteral):
                # Two ways an integer literal may appear: as the final static
                # axis of a reduction, or as part of a documented static-ABI
                # tail (pad_cat's six F.pad-order pads, whose values decide the
                # output extents and therefore cannot be operands).
                axis_position = (expr.name in AXIS_LITERAL_OPERATORS
                                 and arg is expr.args[-1])
                tail_position = tail and index >= len(expr.args) - tail
                if not (axis_position or tail_position):
                    raise CompileError(DiagnosticKind.TYPE, 'an integer literal only names the final static axis argument', position=arg.position)
                args.append(arg)
            elif isinstance(arg, ShapeLiteral):
                # Only reshape's *second* argument: a literal in the value slot
                # would reach the rule and crash on `.dtype`/`.shape`
                # (review P1-6, 2026-10-26).
                if expr.name != "reshape" or index != 1:
                    raise CompileError(
                        DiagnosticKind.TYPE,
                        f"a shape literal cannot be an argument of "
                        f"'{expr.name}'; it only names reshape's target shape",
                        position=arg.position,
                    )
                args.append(arg)
            elif isinstance(arg, DTypeLiteral):
                # Same routing as the shape literal: cast_to's rule wants the
                # written dtype in its second position, never in the value slot.
                if expr.name != "cast_to" or index != 1:
                    raise CompileError(
                        DiagnosticKind.TYPE,
                        f"a dtype literal cannot be an argument of "
                        f"'{expr.name}'; it only names cast_to's target dtype",
                        position=arg.position,
                    )
                args.append(arg)
            else:
                args.append(self.infer(arg))

        # Cascading error suppression: if any argument already failed in an
        # earlier statement, this expression is poisoned rather than wrong.
        if any(is_poisoned(arg) for arg in args):
            return PoisonType()

        self._reject_scalar_operands(expr, args)

        try:
            check_accumulation(expr.name, spec, args)

            inferred = spec.infer(args)

            return self._require_tensor_result(expr, inferred)
        except RuleError as error:
            raise self._at(error, expr) from error

    def infer_multi(self, expression, count):
        """Type a multi-value call for a known target count.

        The same shape as `_infer_call`: unknown operator, arity, poisoning
        and rule rejections are handled identically, so the only thing that
        differs is the signature of the rule itself -- a multi operator
        receives the count, because the language expresses it only in the
        assignment's left-hand names.
        """
        spec = OPERATORS.get(expression.name)

        if spec is None:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"Unknown operator '{expression.name}'",
                position=expression.position,
            )

        if len(expression.args) != spec.arity:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"Operator '{expression.name}' expects "
                f"{spec.arity} arguments, "
                f"got {len(expression.args)}",
                position=expression.position,
            )

        # Same literal routing as `_infer_call`: a literal is only legal where
        # an operator's static ABI says so, and the rule wants the literal
        # itself there.  Passing a literal through unconditionally surfaced as
        # `AttributeError: 'IntLiteral' has no attribute 'dtype'` (review P1-6).
        args = []
        tail = STATIC_INT_TAIL.get(expression.name, 0)
        for index, arg in enumerate(expression.args):
            if isinstance(arg, IntLiteral):
                axis_position = (expression.name in AXIS_LITERAL_OPERATORS
                                 and arg is expression.args[-1])
                tail_position = tail and index >= len(expression.args) - tail
                if not (axis_position or tail_position):
                    raise CompileError(
                        DiagnosticKind.TYPE,
                        "an integer literal only names the final static axis "
                        "argument",
                        position=arg.position,
                    )
                args.append(arg)
            elif isinstance(arg, ShapeLiteral):
                raise CompileError(
                    DiagnosticKind.TYPE,
                    f"a shape literal cannot be an argument of "
                    f"'{expression.name}'",
                    position=arg.position,
                )
            elif isinstance(arg, DTypeLiteral):
                raise CompileError(
                    DiagnosticKind.TYPE,
                    f"a dtype literal cannot be an argument of "
                    f"'{expression.name}'",
                    position=arg.position,
                )
            else:
                args.append(self.infer(arg))

        if any(is_poisoned(arg) for arg in args):
            return [PoisonType() for _ in range(count)]

        self._reject_scalar_operands(expression, args)

        try:
            check_accumulation(expression.name, spec, args)
            if expression.name == 'split':
                # The only multi operator with a policy-controlled rule: the
                # symbolic-axis capability is opt-in with the lowering.
                types = spec.infer(args, count,
                                   allow_symbolic=self.allow_symbolic_split)
            else:
                types = spec.infer(args, count)
        except RuleError as error:
            raise self._at(error, expression) from error

        if len(types) != count:
            # The rule defined a different number of results than the statement
            # binds names for.  For the fixed-arity multi operators this is the
            # user's assignment, not a compiler defect, so it is diagnosable
            # (review P1-6, 2026-10-26).
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"operator '{expression.name}' defines {len(types)} results, "
                f"but {count} names are assigned",
                position=expression.position,
            )
        return list(types)

    def _at(self, error: RuleError, node) -> CompileError:
        """The same rejection, now pointing at the node it came from."""
        return CompileError(
            kind=error.kind,
            message=error.message,
            position=node.position,
        )

    def _reject_scalar_operands(self, expression, args):
        """A rank-0 input is only a legal operand of the elementwise ops.

        Every other rule reads a shape (`x.shape[-1]`, `len(shape) == 2`,
        `shape[0]`), so a scalar there is not a case the rule forgot -- it is
        the wrong kind of value. Refusing it here, at the argument the program
        wrote, keeps the diagnostic about the program instead of a crash
        between the two layers.
        """
        if (expression.name in SCALAR_BROADCAST_OPERATORS
                or expression.name in SCALAR_ARGUMENT_OPERATORS):
            return

        for node, inferred in zip(expression.args, args):
            if not (isinstance(inferred, TensorType) and inferred.is_scalar):
                continue

            subject = (
                f"'{node.name}'" if isinstance(node, Name) else "an operand"
            )
            raise CompileError(
                DiagnosticKind.TYPE,
                f"operator '{expression.name}' needs tensor operands, but "
                f"{subject} is a scalar ({inferred.dtype}); a scalar input "
                f"can only mix with a tensor in elementwise arithmetic "
                f"(add, sub, mul, maximum, minimum)",
                position=getattr(node, "position", None) or expression.position,
            )

    def _require_tensor_result(self, expression, inferred):
        """A pipeline value is a tensor; an expression cannot produce a scalar.

        add/sub/mul already refuse two scalars, and every other operator
        refuses a scalar operand, so this holds for today's registry. It is
        checked rather than assumed because the alternative -- a rank-0 value
        reaching the buffer plan -- has no valid allocation.
        """
        if (isinstance(inferred, TensorType) and inferred.is_scalar
                and expression.name not in PROGRAM_UNIFORM_OPERATORS):
            # A rank-0 result is normally a scalar that would need a buffer it
            # cannot have. The program-uniform producer is the exception: its
            # value is one register per program, consumed inside the kernel that
            # emits it (see ir/program_scalar_ir).
            raise CompileError(
                DiagnosticKind.TYPE,
                f"operator '{expression.name}' produces a scalar; a pipeline "
                f"value must be a tensor",
                position=expression.position,
            )

        return inferred
