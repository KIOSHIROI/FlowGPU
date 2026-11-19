"""Dispatch/cache and malformed-IR regressions from the design audit."""
import ast
import pytest
from compiler.ast_node import TensorType
from compiler.backends import library_dispatch
from compiler.cache import cached_module
from compiler.cli import main
from compiler.compile import compile_source
from compiler.dataflow import DataflowGraph, Node, INPUT, OUTPUT
from compiler.diagnostic import CompileError, DiagnosticKind
from compiler.fusion import fuse
from compiler.liveness import plan
from compiler.reference import lower as lower_reference
from compiler.triton_backend import lower as lower_triton

GEMM = ('pipeline p { input a : fp16[M,K] input b : fp16[K,N] '
        'y=matmul(a,b) z=silu(y) output z }')


def test_library_policy_cache_keys_and_actual_artifacts_do_not_collide(tmp_path, monkeypatch):
    selective = cached_module(GEMM, cache_dir=tmp_path, gpu_backend='hybrid', library_matmul='unfused')
    forced = cached_module(GEMM, cache_dir=tmp_path, gpu_backend='hybrid', library_matmul='all')
    assert selective[2] != forced[2] and selective[0] != forced[0]
    assert 'torch.mm(' not in selective[0]
    assert 'torch.mm(' in forced[0]
    ast.parse(selective[0]); ast.parse(forced[0])
    def forbidden(*args, **kwargs):
        raise AssertionError('cache hit must not recompile')
    monkeypatch.setattr('compiler.compile.compile_source', forbidden)
    for policy, expected in [('unfused', selective), ('all', forced)]:
        assert cached_module(GEMM, cache_dir=tmp_path, gpu_backend='hybrid',
                             library_matmul=policy) == (expected[0], True, expected[2])


def test_cli_policy_reaches_both_direct_and_cached_paths(tmp_path, capsys):
    src = tmp_path / 'p.fgpu'; src.write_text(GEMM)
    for cached in [False, True]:
        codes = []
        for policy in ['unfused', 'all']:
            args = [str(src), '--gpu-backend', 'hybrid', '--library-matmul', policy, '--emit', 'triton']
            if cached: args += ['--cache-dir', str(tmp_path / 'cache')]
            assert main(args) == 0
            code = capsys.readouterr().out; ast.parse(code); codes.append(code)
        assert 'torch.mm(' not in codes[0] and 'torch.mm(' in codes[1]


@pytest.mark.parametrize('backend,policy,expected', [
    ('triton', 'unfused', ['matmul']),
    ('hybrid', 'unfused', ['matmul']),
    ('hybrid', 'all', ['library_matmul', 'pointwise']),
])
def test_policy_opacity_and_emission_agree(backend, policy, expected):
    p = compile_source(GEMM, gpu_backend=backend, library_matmul=policy).pipelines[0]
    dispatch = library_dispatch(backend, policy)
    assert [k.kind for k in p.triton.kernels] == expected
    for g in p.fusion.groups:
        if g.anchor and g.anchor.op in dispatch.opaque_anchors:
            assert not g.fused
    bare = compile_source(GEMM.replace('z=silu(y)', 'z=y'), gpu_backend=backend,
                          library_matmul=policy).pipelines[0]
    assert bare.triton.kernels[0].kind == ('library_matmul' if backend == 'hybrid' else 'matmul')


def test_forced_library_rejects_a_fused_plan_instead_of_dropping_its_epilogue():
    p = compile_source(GEMM, gpu_backend='hybrid').pipelines[0]
    with pytest.raises(CompileError, match='contains a fused region') as err:
        lower_triton(p.pipeline, p.dataflow, p.fusion, p.memory,
                     gpu_backend='hybrid', library_matmul='all')
    assert err.value.kind == DiagnosticKind.INTERNAL and err.value.pipeline == 'p'


def tensor():
    return TensorType('fp32', ['2'])


@pytest.mark.parametrize('values,nodes,aliases,reason', [
    ({'x': tensor()}, [Node(INPUT, [], 'x'), Node(OUTPUT, ['lost'], None)], {}, 'missing value'),
    ({'x': tensor()}, [Node(INPUT, [], 'x'), Node('relu', ['x'], 'x')], {}, 'multiple producers'),
    ({'x': tensor(), 'y': tensor()}, [Node('relu', ['x'], 'y'), Node(INPUT, [], 'x')], {}, 'before its producer'),
    ({'x': tensor(), 'y': tensor()}, [Node(INPUT, [], 'x')], {}, 'without producers'),
    ({'x': tensor()}, [Node(INPUT, [], 'missing')], {}, 'untyped value'),
    ({'x': tensor(), 'a': tensor()}, [Node(INPUT, [], 'x')], {'a': 'lost'}, 'missing value'),
    ({'a': tensor(), 'b': tensor()}, [], {'a': 'b', 'b': 'lost'}, 'missing value'),
    ({'a': tensor(), 'b': tensor()}, [], {'a': 'b', 'b': 'a'}, 'alias cycle'),
    ({'x': tensor(), 'a': TensorType('fp16', ['2'])}, [Node(INPUT, [], 'x')], {'a': 'x'}, 'different type'),
    ({'x': tensor(), 'a': tensor()}, [Node(INPUT, [], 'x'), Node(OUTPUT, ['a'], None)], {'a': 'x'}, 'noncanonical alias'),
    ({'x': tensor()}, [Node(INPUT, [], 'x'), Node(OUTPUT, ['x'], 'x')], {}, 'produce none'),
    ({'x': tensor()}, [Node(INPUT, [], 'x'), Node('relu', ['x'], None)], {}, 'no result'),
    ({'x': tensor()}, [Node(INPUT, [], 'x'), Node('unknown', ['x'], 'x')], {}, 'unknown operator'),
])
def test_broken_graph_reports_internal_diagnostic(values, nodes, aliases, reason):
    with pytest.raises(CompileError, match=reason) as caught:
        DataflowGraph('broken', values, nodes, aliases)
    assert caught.value.as_dict() == dict(pipeline='broken', kind='internal',
                                         message=caught.value.message, line=None, column=None)


@pytest.mark.parametrize('entry', ['fusion', 'liveness', 'reference', 'triton', 'edges'])
def test_pass_boundaries_recheck_graph_after_external_mutation(entry):
    p = compile_source(GEMM).pipelines[0]
    # Node is frozen, but its input list is mutable: the graph can be damaged
    # after the constructor's check, including by future transformation passes.
    p.dataflow.nodes[-1].inputs[0] = 'missing_after_construction'
    with pytest.raises(CompileError, match='missing value'):
        if entry == 'fusion': fuse(p.dataflow)
        elif entry == 'liveness': plan(p.dataflow, p.fusion)
        elif entry == 'reference': lower_reference(p.pipeline, p.dataflow, p.fusion, p.memory)
        elif entry == 'triton': lower_triton(p.pipeline, p.dataflow, p.fusion, p.memory)
        else: p.dataflow.edges()


def test_alias_chain_and_split_remain_valid():
    for src in ['pipeline p { input x : fp32[2,3] a=x b=a output b }',
                'pipeline p { input x : fp32[2,6] a,b=split(x) output a output b }']:
        p = compile_source(src).pipelines[0]
        p.dataflow.validate()
        assert p.reference.complete and p.triton.complete


def test_reassignment_is_a_frontend_dependency_error_not_internal():
    with pytest.raises(CompileError, match="olumn 32: Symbol 'x' is already defined") as caught:
        compile_source('pipeline p { input x : fp32[2] x=relu(x) output x }')
    assert caught.value.kind == DiagnosticKind.DEPENDENCY
    assert caught.value.line is not None
