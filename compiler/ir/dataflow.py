"""Compatibility re-export of the frontend-neutral Dataflow IR.

The graph model moved to ``compiler.core.ir`` in Phase 1a of the de-DSL work.
Existing imports (``from compiler.ir.dataflow import DataflowGraph``) keep
working and refer to the same classes, so pass identity is unchanged.
"""
from compiler.core.ir import (  # noqa: F401
    INPUT,
    OUTPUT,
    DataflowGraph,
    Edge,
    Node,
    resolve_alias,
)

__all__ = ["INPUT", "OUTPUT", "DataflowGraph", "Edge", "Node", "resolve_alias"]
