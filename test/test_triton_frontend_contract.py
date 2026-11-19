"""Generated kernels must be parseable by the Triton front end, not just Python.

The CPU-only checks in this repository run the reference backend and compare
parsed statements, so a construct that Python accepts but Triton's own AST
walker refuses passes every CPU gate and fails on the first device launch. That
happened on 2026-10-18: an emitted `if A and B and C and D:` guard is valid
Python and was rejected by Triton 3.3.1 with

    chained boolean operators (A or B or C) are not supported;
    use parentheses to split the chain.

`triton/compiler/code_generator.py` accepts exactly two values per `BoolOp` and
exactly one operator per `Compare`. Those are the two rules encoded here, so a
future code generator is caught on CPU instead of on a GPU.
"""
import ast
from pathlib import Path

import pytest

from compiler.backend.platform_configs import resolve_platform
from compiler.compile import compile_source

EXAMPLES = sorted(Path(__file__).resolve().parents[1].glob('examples/*.fgpu'))

CONFIGS = (
    ('default', None),
    ('block256', resolve_platform('cuda', block_size=256)),
    ('grid_stride', resolve_platform('cuda', mapping_strategy='grid_stride_loop',
                                     block_size=512, device_cores=108)),
)

# The restriction the front end enforces, stated once for both the check and
# its negative control.
FRONTEND_RULES = (
    'triton 3.3.1 code_generator.visit_BoolOp: a BoolOp must have exactly two values',
    'triton 3.3.1 code_generator.visit_Compare: a Compare must have one op and one comparator',
)


def frontend_violations(source):
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BoolOp) and len(node.values) != 2:
            violations.append(('BoolOp', node.lineno, len(node.values)))
        if isinstance(node, ast.Compare) and not (
                len(node.ops) == 1 and len(node.comparators) == 1):
            violations.append(('Compare', node.lineno, len(node.ops)))
    return violations


def generated_programs():
    """Every example that lowers, under each configuration, deduplicated."""
    seen = {}
    for path in EXAMPLES:
        source = path.read_text()
        for label, config in CONFIGS:
            try:
                compiled = compile_source(source, platform_config=config)
            except Exception:  # not every example lowers under every profile
                continue
            for pipeline in compiled.pipelines:
                for kernel in pipeline.triton.kernels:
                    if kernel.source:
                        seen[(path.name, label, kernel.name)] = kernel.source
    return seen


PROGRAMS = generated_programs()


def test_the_corpus_actually_covers_many_generated_kernels():
    assert len(PROGRAMS) >= 40, sorted(PROGRAMS)


@pytest.mark.parametrize('key', sorted(PROGRAMS))
def test_generated_kernel_uses_only_frontend_supported_predicates(key):
    violations = frontend_violations(PROGRAMS[key])
    assert not violations, (key, violations, FRONTEND_RULES)


def test_the_rule_rejects_what_the_backend_rejects():
    """Negative control: without this, the test above could be vacuous."""
    assert frontend_violations('if A and B and C:\n    pass\n')
    assert frontend_violations('if A < B < C:\n    pass\n')
    assert frontend_violations('if (A and B) and (C or D):\n    pass\n') == []
    assert frontend_violations('if A <= 2147483647:\n    pass\n') == []
