"""Compatibility re-export of the frontend-neutral operator registry.

The registry moved to ``compiler.core.operators`` in Phase 1b.  Same objects,
same dict identity, so registrations made through either module are visible.
"""
from compiler.core.operators import *  # noqa: F401,F403
from compiler.core.operators import OPERATORS, OperatorSpec, register_operator  # noqa: F401
