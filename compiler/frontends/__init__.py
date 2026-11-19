"""Frontends other than the FlowGPU text DSL.

Phase 1a adds a programmatic frontend that constructs the frontend-neutral
Dataflow IR directly (no lexer, no parser).  The DSL stays the reference
frontend; this package is where a Triton importer or a Python eDSL will live.
"""
from compiler.frontends.programmatic import ProgrammaticFrontend

__all__ = ["ProgrammaticFrontend"]
