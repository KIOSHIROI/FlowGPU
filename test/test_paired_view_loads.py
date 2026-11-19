"""Alias proof and conservative domain admission for external paired reads."""
import ast
from dataclasses import replace
import itertools
import math
import random
from types import SimpleNamespace

import pytest

from compiler.backend.paired_views import adjacent_view_condition
from compiler.backend.platform_configs import resolve_platform
from compiler.compile import compile_source
from compiler.runtime.cache import cache_key


class Metadata:
    def __init__(self, strides, offset=0, storage=101):
        self.strides, self.offset, self.storage = tuple(strides), offset, storage

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def storage_offset(self):
        return self.offset

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage)


@pytest.mark.parametrize('rank', [1, 2, 3, 4])
def test_guard_proves_the_paired_read_set_for_random_row_gaps(rank):
    rng = random.Random(181810 + rank)
    expr = compile(adjacent_view_condition('left', 'right'), '<metadata>', 'eval')
    for _ in range(20):
        shape = [rng.randint(1, 5) for _ in range(rank)]
        strides = [2]
        for dim in reversed(shape[1:]):
            strides.insert(0, dim * strides[0] + rng.randint(0, 3))
        left = Metadata(strides, offset=7)
        right = Metadata(strides, offset=8)
        assert eval(expr, dict(left=left, right=right))
        addresses = [sum(c * s for c, s in zip(coords, strides))
                     for coords in itertools.product(*(range(dim) for dim in shape))]
        independent = [(left.offset + a, right.offset + a) for a in addresses]
        paired = [(left.offset + a, left.offset + a + 1) for a in addresses]
        assert paired == independent
        assert len(paired) == math.prod(shape)


@pytest.mark.parametrize('left,right', [
    (Metadata((34, 2)), Metadata((34, 2), 1, storage=202)),
    (Metadata((34, 2)), Metadata((34, 2), 17)),
    (Metadata((34, 2), 1), Metadata((34, 2), 0)),
    (Metadata((34, 2)), Metadata((35, 2), 1)),
    (Metadata((34, 1)), Metadata((34, 1), 1)),
    (Metadata((34, 0)), Metadata((34, 0), 1)),
])
def test_distinct_storage_split_reversed_and_mismatched_strides_fall_back(left, right):
    assert not eval(adjacent_view_condition('left', 'right'), dict(left=left, right=right))


def test_default_cache_identity_cli_and_boolean_validation(tmp_path, capsys):
    source = 'pipeline p { input x:fp32[B,D] input y:fp32[B,D] z=mul(x,y) output z }'
    base = resolve_platform()
    paired = resolve_platform(paired_view_loads=True)
    assert not base.paired_view_loads
    assert cache_key(source, {'platform_config': base}) != cache_key(source, {'platform_config': paired})
    for value in (1, 'yes', None):
        with pytest.raises(ValueError, match='boolean'):
            replace(base, paired_view_loads=value)
    from compiler.cli import main
    path = tmp_path / 'pair.fgpu'
    path.write_text(source)
    assert main([str(path), '--paired-view-loads', '--emit', 'json']) == 0
    assert '"paired_view_loads": true' in capsys.readouterr().out
    for mode in ('strategies', 'gather-strategies'):
        assert main([str(path), '--paired-view-loads', '--emit', mode]) == 2
        capsys.readouterr()


@pytest.mark.parametrize('primitive', [False, True])
def test_two_emitters_guard_paired_reads_without_changing_numeric_expressions(primitive):
    source = ('pipeline p { input x:fp16[B,D] input y:fp16[B,D] '
              'a=to_fp32(x) b=to_fp32(y) z=mul(a,b) output z }' if primitive else
              'pipeline p { input x:fp32[B,D] input y:fp32[B,D] a=silu(x) z=mul(a,y) output z }')
    base = compile_source(source).pipelines[0].triton
    candidate = compile_source(source, platform_config=resolve_platform(paired_view_loads=True),
                               prune_kernel_abi=True).pipelines[0].triton
    assert len(base.kernels) == len(candidate.kernels) == 1
    assert 'tl.split(' not in base.kernels[0].source
    assert 'tl.split(' in candidate.kernels[0].source
    assert 'untyped_storage().data_ptr()' in candidate.host
    tree = ast.parse(candidate.kernels[0].source)
    guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id.startswith('adjacent_views'))
    assert len(guard.orelse) >= 3  # both original reads, including a strided right address
    # The paired-read condition is passed as a launch argument, so it is
    # built from binary conjunctions only (restricted frontends read it).
    # Unrelated guards elsewhere in the host are not constrained here: their
    # shape belongs to the validation block's contract, not to this one.
    paired_operators = [n for n in ast.walk(ast.parse(candidate.host))
                        if isinstance(n, ast.BoolOp)
                        and 'untyped_storage' in ast.unparse(n)]
    assert paired_operators
    assert all(len(n.values) == 2 for n in paired_operators)
    assert candidate.kernels[0].grid == base.kernels[0].grid


@pytest.mark.parametrize('source', [
    'pipeline p { input x:fp16[B,D] input y:fp32[B,D] a=to_fp32(x) z=mul(a,y) output z }',
    'pipeline p { input x:fp32[B,D] input y:fp32[B,1] w=broadcast_like(y,x) z=mul(x,w) output z }',
    'pipeline p { input x:fp32[B,D] input y:fp32[B,D] z=mul(x,y) s=sum_last(z) output s }',
])
def test_different_dtype_broadcast_and_reduction_domains_do_not_nominate_pairs(source):
    compiled = compile_source(source, platform_config=resolve_platform(paired_view_loads=True)).pipelines[0]
    assert all('tl.split(' not in k.source for k in compiled.triton.kernels)
