from lark import Transformer

from flowgpu.compiler.frontend.ast_node import (
    Assignment,
    Call,
    PipelineDecl,
    InputDecl,
    Name,
    OutputDecl,
    ShapeLiteral,
    TensorType,
    IntLiteral,
    DTypeLiteral,
)
from flowgpu.compiler.frontend.position import Position

# Infix arithmetic is surface sugar for an operator call: `x - y` means
# `sub(x, y)`. Both spellings must build the same node, so that one
# computation has exactly one representation in the AST -- which is the
# shape the IR uses too, a node named by its operator (Design.md 7.1).
INFIX_OPERATORS = {
    "PLUS": "add",
    "MINUS": "sub",
    "STAR": "mul",
}

class ASTBuilder(Transformer):
    """Build the AST, recording where each name and operator was written.

    Positions come from the tokens, never from the rules. A name's position
    is the position of the name the user wrote, which is exactly what a
    diagnostic wants to point at -- and it makes this builder independent of
    any grammar-level position propagation.
    """

    def pipeline_decl(self, items):
        name = str(items[1])
        
        statements = [
            item for item in items[3:-1]
            ]
        
        return PipelineDecl(
            name=name,
            statements=statements,
            position=Position.from_token(items[1]),
        )

    def integer_atom(self, items):
        return IntLiteral(int(str(items[-1])) * (-1 if len(items) == 2 else 1),
                          position=Position.from_token(items[0]))
    
    def input_decl(self, items):
        name = str(items[1])
        
        tensor_type = items[3]
        
        return InputDecl(
            name=name,
            type=tensor_type,
            position=Position.from_token(items[1]),
        )
    
    def output_decl(self, items):
        return OutputDecl(
            name=str(items[1]),
            position=Position.from_token(items[1]),
        )
    
    def assignment(self, items):
        # items: [NAME, (COMMA NAME)*, EQUAL, expr]. The names keep their
        # written order, which is the operator's output order.
        # COMMA and EQUAL arrive as plain strings, but lark's Token IS a
        # str subclass -- so the filter tests the exact type, not
        # isinstance, or every name would be discarded with the commas.
        names = [
            str(item)
            for item in items[:-1]
            if type(item) is not str
        ]
        value = items[-1]

        return Assignment(
            names=names,
            value=value,
            position=Position.from_token(items[0]),
        )
    
    def tensor_type(self, items):
        dtype = str(items[0])
        shape = items[2]
        
        return TensorType(dtype=dtype, shape=shape)

    def scalar_type(self, items):
        # A dtype with no brackets. Rank 0: one number, not a tensor with an
        # extent, which is why `fp16[1]` stays a tensor here.
        return TensorType(dtype=str(items[0]), shape=[])
    
    def shape(self, items):
        return [
            str(item)
            for item in items
            if str(item) != ","
        ]
        
    def _binary(self, items):
        # `?addition` / `?multiplication` inline the single-child case, so an
        # operator reduction always arrives as exactly [left, op, right].
        #
        # The operator is still a Token here -- there is deliberately no
        # PLUS/MINUS/STAR handler converting it to a str -- so it is
        # dispatched on its type. An operator the table does not know raises
        # KeyError immediately instead of being mistaken for an operand,
        # which is how the old `str(item) == "+"` comparison failed.
        left, operator, right = items

        # The position of a desugared call is the position of its operator:
        # for `x - y` that is the `-`, for `sub(x, y)` the `sub`. Both point
        # at the thing a diagnostic is about.
        return Call(
            name=INFIX_OPERATORS[operator.type],
            args=[left, right],
            position=Position.from_token(operator),
        )

    def addition(self, items):
        return self._binary(items)

    def multiplication(self, items):
        return self._binary(items)
    
    def call(self, items):
        name = str(items[0])

        # `call: NAME LPAR arguments? RPAR` always yields the two parenthesis
        # tokens, so "no arguments" is a 3-item list, not a 1-item one. The
        # old `len(items) == 1` test never fired and `f()` ended up with the
        # RPAR string as its single argument -- which made the arity error
        # report "got 1" for a call that passed nothing.
        args = items[2] if len(items) == 4 else []

        return Call(
            name=name,
            args=args,
            position=Position.from_token(items[0]),
        )
    
    def arguments(self, items):
        return [
            item for item in items
            if str(item) != ","
        ]
    
    def atom(self, items):
        item =  items[0]

        if isinstance(item, (Call, ShapeLiteral, IntLiteral, DTypeLiteral)):
            return item

        return Name(str(item), position=Position.from_token(item))

    def dtype_atom(self, items):
        # One DTYPE token. The position is kept so a dtype literal written
        # where it cannot be an operand is reported where it was written.
        token = items[0]
        return DTypeLiteral(
            dtype=str(token),
            position=Position.from_token(token),
        )

    def shape_atom(self, items):
        # items: [LBRACK, shape-list, RBRACK]; the shape rule returns a
        # plain list of dimension strings, so the literal is exactly that.
        dims = items[1]
        # The bracket token is the literal's own position.  The old search for
        # a non-str item found nothing because lark's Token subclasses str, so
        # every shape literal reached diagnostics with position=None (review
        # 2026-10-26).
        token = items[0]
        if not (hasattr(token, "line") and hasattr(token, "column")) and dims:
            # the grammar literal "[" arrives as a plain string, so fall back
            # to the first dimension token (a real Token with a position)
            token = dims[0]
        position = (Position.from_token(token)
                    if hasattr(token, "line") and hasattr(token, "column") else None)
        return ShapeLiteral(shape=dims, position=position)
    
    def start(self, items):
        # program: pipeline+ -- a file is one or more pipelines
        return list(items)
    
    def NAME(self, token):
        return token
    
    def DTYPE(self, token):
        # A Token is a str subclass, so declaration rules can keep writing
        # str(items[0]) and the literal keeps its position for diagnostics.
        return token
    
    def PIPELINE(self, token):
        return str(token)
    
    def INPUT(self, token):
        return str(token)
    
    def OUTPUT(self, token):
        return str(token)
    
    def LBRACE(self, token):
        return str(token)
    
    def RBRACE(self, token):
        return str(token)
    
    def LBRACK(self, token):
        # keep the token: its line/column is the shape literal's own position
        # (the old str(token) dropped it and every shape diagnostic lost its
        # position; re-review 2026-10-26)
        return token
    
    def RBRACK(self, token):
        return str(token)
    
    def LPAR(self, token):
        return str(token)
    
    def RPAR(self, token):
        return str(token)
    
    def COLON(self, token):
        return str(token)
    
    def COMMA(self, token):
        return str(token)
    
    def EQUAL(self, token):
        return str(token)
