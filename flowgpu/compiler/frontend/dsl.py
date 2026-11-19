"""FlowGPU's text DSL as the first ``flowgpu.compiler.core.frontend.Frontend``.

This adapter is intentionally thin: it reuses the existing lexer/parser and
does not duplicate grammar or AST construction.  It is the reference frontend
for the course deliverable and the correctness baseline for future frontends.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DslFrontend:
    name: str = "flowgpu-dsl"
    description: str = "FlowGPU's declarative pipeline DSL (lexer + LALR parser)"
    source_extension: str = ".fgpu"

    def parse(self, source: str):
        from flowgpu.compiler.frontend.parser import parse_ast
        return parse_ast(source)
