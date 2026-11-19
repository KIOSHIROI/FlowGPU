"""Compatibility re-export of the frontend-neutral program AST.

The classes moved to ``flowgpu.compiler.core.ast`` in Phase 1b (de-DSL).  The DSL
parser and AST builder import them from here, and they are the same objects as
the core ones, so isinstance/equality are unchanged.
"""
from flowgpu.compiler.core.ast import *  # noqa: F401,F403
from flowgpu.compiler.core.ast import (  # noqa: F401
    Assignment,
    Call,
    InputDecl,
    OutputDecl,
    PipelineDecl,
    PoisonType,
    Symbol,
    is_poisoned,
)
from flowgpu.compiler.core.literals import DTypeLiteral, IntLiteral, Name, ShapeLiteral  # noqa: F401
from flowgpu.compiler.core.scalars import ScalarType
from flowgpu.compiler.core.types import TensorType, format_shape, is_symbol  # noqa: F401
