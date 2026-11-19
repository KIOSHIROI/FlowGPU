import ast
import pytest
from compiler.compile import compile_source
from compiler.platform_configs import resolve_platform
from compiler.triton_backend import module

SOURCE = '''pipeline tx {
    input x : fp16[M,N]
    input a : fp16[M,N]
    input b : fp16[N,M]
    z = add(x,a)
    t = transpose(z)
    y = sub(t,b)
    output y
}'''


@pytest.mark.parametrize('mapping', ['tiled', 'output_gather'])
def test_fused_transpose_never_reads_unmaterialized_prologue(mapping):
    prog = compile_source(SOURCE, platform_config=resolve_platform(transpose_mapping=mapping)).pipelines[0].triton
    assert prog.complete and len(prog.kernels) == 1
    assert prog.kernels[0].ops == ('add', 'transpose', 'sub')
    assert 'z_ptr' not in prog.kernels[0].source
    ast.parse(module([prog]))


def test_gather_carries_source_and_epilogue_view_contracts():
    prog = compile_source(SOURCE, platform_config=resolve_platform(transpose_mapping='output_gather')).pipelines[0].triton
    assert {a.text for a in prog.kernels[0].launch if a.kind == 'stride'} == {
        'x:0', 'x:1', 'a:0', 'a:1', 'b:0', 'b:1'}
    assert 'must be contiguous' not in prog.host


def test_tiled_transpose_still_rejects_views():
    prog = compile_source(SOURCE).pipelines[0].triton
    assert 'must be contiguous' in prog.host


def test_kunlun_layout_preference_does_not_grant_transpose_support():
    config = resolve_platform('kunlunxin')
    assert config.transpose_mapping == 'output_gather'
    prog = compile_source(SOURCE, platform_config=config).pipelines[0].triton
    assert not prog.complete and not prog.kernels
    assert 'unvalidated' in prog.refusals[0].reason


def test_unknown_transpose_mapping_is_rejected():
    with pytest.raises(ValueError, match='transpose_mapping'):
        resolve_platform(transpose_mapping='unknown')
