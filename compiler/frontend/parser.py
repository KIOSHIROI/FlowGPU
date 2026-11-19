from pathlib import Path
from lark import Lark 
from compiler.frontend.ast_builder import ASTBuilder
    
parser = Lark.open(
    str(Path(__file__).resolve().parents[1] / "grammar.lark"),
    parser="lalr",
)

def parse(source: str):
    return parser.parse(source)

def parse_ast(source: str):
    tree = parser.parse(source)
    return ASTBuilder().transform(tree)

if __name__ == "__main__":
    source = """
    pipeline residual_norm {
        input x        : bf16[N,D]
        input residual : bf16[N,D]
        input weight   : bf16[D]
        y = x + residual
        z = rms_norm(y, weight)
        output z
    }
    """
    
    tree = parse(source)
    
    print("=== Parse Tree ===")
    print(tree.pretty())
    
    ast = parse_ast(source)
    
    print("=== AST ===")
    print(ast)