"""Staged registrar for `store_at` / `store_at_trusted`.

The type rule lives in ``compiler/frontend/store_at_rule.py`` (a leaf, so the
registry commit can import it without a cycle). This module exists only to add
the two registrations *without touching* ``compiler/frontend/operator.py``,
which the in-flight `segmented_cumsum` work owns; the coordinated merge is one
import line plus two `register_operator` calls there (see
``docs/REGISTRY_MERGE_STORE_AT_20261022.md``).

`register()` is idempotent and `unregister()` only removes the operators *this
module added*, so the same test fixture is safe before and after the merge
commit: after it, the canonical entries stay in place across tests.
"""

from compiler.frontend.operator import (
    OPERATORS,
    OperatorSpec,
    register_operator,
)
from compiler.frontend.store_at_rule import (
    STORE_AT_OPERATOR,
    STORE_AT_OPERATORS,
    STORE_AT_TRUSTED_OPERATOR,
    infer_store_at,
)

#: Back-compat aliases for the staging-era test imports.
STAGED_OPERATOR = STORE_AT_OPERATOR
TRUSTED_OPERATOR = STORE_AT_TRUSTED_OPERATOR
STAGED_OPERATORS = STORE_AT_OPERATORS

_ADDED = set()


def registration():
    """The two OperatorSpecs the coordinated registry commit adds."""
    return {
        name: OperatorSpec(name=name, arity=3, infer=infer_store_at,
                           pointwise=False, acc=None)
        for name in STORE_AT_OPERATORS
    }


def register():
    """Add the staged operators if the canonical registry does not have them."""
    for name, spec in registration().items():
        if name not in OPERATORS:
            register_operator(name, spec.arity, spec.infer,
                              pointwise=False, acc=None)
            _ADDED.add(name)
    return OPERATORS[STORE_AT_OPERATOR]


def unregister():
    """Undo only this module's staging; canonical entries survive the merge."""
    for name in tuple(_ADDED):
        OPERATORS.pop(name, None)
        _ADDED.discard(name)
