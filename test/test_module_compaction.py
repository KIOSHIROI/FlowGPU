import ast
from types import SimpleNamespace
import pytest
from compiler.module_compaction import compact_source
from compiler.compile import compile_source
from compiler.triton_backend import module


SOURCE = '''
@jit
def first(x):
    return x + 1
@jit
def second(x):
    return x + 1
def run_a(x):
    """a"""
    if x.dtype != torch.float16:
        raise ValueError('dtype')
    return first(x.value)
def run_b(x):
    """b"""
    if x.dtype != torch.float32:
        raise ValueError('dtype')
    return second(x.value)
'''


def test_shared_hosts_preserve_each_typed_contract_and_keyword_calls():
    torch = SimpleNamespace(float16=object(), float32=object())
    scope = {'torch': torch, 'jit': lambda f:f}
    code = compact_source(SOURCE)
    exec(code, scope)
    assert 'second' not in scope
    assert scope['run_a'](x=SimpleNamespace(dtype=torch.float16, value=3)) == 4
    assert scope['run_b'](SimpleNamespace(dtype=torch.float32, value=8)) == 9
    with pytest.raises(ValueError, match='(?:dtype)'):
        scope['run_a'](SimpleNamespace(dtype=torch.float32, value=3))
    with pytest.raises(ValueError, match='(?:dtype)'):
        scope['run_b'](SimpleNamespace(dtype=torch.float16, value=3))
    assert len([n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name.startswith('_fg_host_')]) == 1


def test_different_kernel_bodies_are_not_merged():
    code=compact_source(SOURCE.replace('return x + 1\ndef run_a','return x * 2\ndef run_a'))
    assert 'def second(' in code


def test_compaction_refuses_non_gather_and_is_opt_in():
    p=compile_source('pipeline p { input x : fp32[N] y=relu(x) output y }').pipelines[0].triton
    assert 'def run_p' in module([p])
    with pytest.raises(ValueError,match='row-gather'):
        module([p],compact_gathers=True)
