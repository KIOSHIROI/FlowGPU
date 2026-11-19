from flowgpu.compiler.core.ast import Symbol
from flowgpu.compiler.core.diagnostic import DiagnosticKind, RuleError


class SymbolTable:
    """Names bound to their single producer.

    Both failures below are dependency errors, in the sense of Design.md
    5.2: `h` used before it is defined is a dependency with no producer,
    and defining `h` twice is a dependency with two of them, which is why
    one name binds exactly one value.
    """

    def __init__(self):
        self.symbols = {}
    
    def define(self, symbol: Symbol):
        if symbol.name in self.symbols:
            raise RuleError(
                DiagnosticKind.DEPENDENCY,
                f"Symbol '{symbol.name}' is already defined",
            )
        
        self.symbols[symbol.name] = symbol
        
    def lookup(self, name: str) -> Symbol:
        if name not in self.symbols:
            raise RuleError(
                DiagnosticKind.DEPENDENCY,
                f"Undefined symbol '{name}'",
            )
        
        return self.symbols[name]
    
    def contains(self, name: str) -> bool:
        return name in self.symbols
