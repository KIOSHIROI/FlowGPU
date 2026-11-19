"""Compatibility re-export of the frontend-neutral operator registry.

The registry moved to ``flowgpu.compiler.core.operators`` in Phase 1b.  Same objects,
same dict identity, so registrations made through either module are visible.
"""
from flowgpu.compiler.core.operators import *  # noqa: F401,F403
from flowgpu.compiler.core.operators import OPERATORS, OperatorSpec, register_operator  # noqa: F401
