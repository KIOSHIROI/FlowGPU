"""Keep arithmetic and typed host contracts while sharing identical bodies."""
import ast
from dataclasses import replace
from types import SimpleNamespace
import pytest
from compiler.compile import compile_source
from compiler.backend.triton_backend import module
from compiler.analysis.module_compaction import compact_source


def program(name, dtype, out='fp32'):
    src = f'''pipeline {name} {{
      input gate:{dtype}[M,N] input up:{dtype}[M,N] input like:{out}[1]
      g=to_fp32(gate) u=to_fp32(up) a=silu(g) p=mul(a,u)
      out=cast_like(p,like) output out
    }}'''
    return compile_source(src, prune_kernel_abi=True).pipelines[0].triton


def kernels(source):
    return [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.decorator_list]


def test_identical_widened_arithmetic_shared_but_output_casts_retained():
    ps=[program('half','fp16'),program('single','fp32'),program('bfloat','bf16'),program('rounded','fp32','fp16')]
    original=module(ps); compact=module(ps,compact_fixed=True)
    assert len(kernels(original))==4 and len(kernels(compact))==2
    def key(fn):
        fn.name='_kernel'
        return ast.dump(fn,include_attributes=False)
    assert set(map(key,kernels(original)))==set(map(key,kernels(compact)))
    assert all('run_'+n in compact for n in ('half','single','bfloat','rounded'))
    assert len(compact)<len(original)


def test_fixed_profile_rejects_unsupported_or_stateful_emission():
    p=program('p','fp32')
    for bad in (replace(p,autotune=True),replace(p,custom_ops='custom'),replace(p,gpu_backend='hybrid'),replace(p,refusals=(object(),))):
        with pytest.raises(ValueError,match='fixed compaction'):module([bad],compact_fixed=True)
    with pytest.raises(ValueError,match='one module'):module([p],compact_fixed=True,compact_gathers=True)
    gemm=compile_source('pipeline p { input a:fp32[M,K] input b:fp32[K,N] y=matmul(a,b) output y }').pipelines[0].triton
    with pytest.raises(ValueError,match='fixed compaction'):module([gemm],compact_fixed=True)


def test_different_decorators_and_launch_options_are_preserved():
    source='''
@jit
def a(x): return x+1
@jit_other
def b(x): return x+1
def run_a(x): return a[x.grid](x, num_warps=4)
def run_b(x): return b[x.grid](x, num_warps=8)
'''
    compact=compact_source(source,profile='fixed-native')
    assert len(kernels(compact))==2
    assert 'num_warps=4' in compact and 'num_warps=8' in compact


@pytest.mark.parametrize('source',[
    'def a(): pass\ndef a(): pass','def _fg_bind(): pass','def _fg_host_0(): pass','def run_a(_fg_dtype_0): pass',
])
def test_ambiguous_or_colliding_names_are_rejected(source):
    with pytest.raises(ValueError, match='compaction'):compact_source(source,profile='fixed-native')


def test_keyword_calls_and_dtype_guards_survive_fixed_profile():
    source='''
@jit
def a(x): return x.value
@jit
def b(x): return x.value
def run_a(x):
    if x.dtype != torch.float16: raise ValueError('dtype')
    return a(x)
def run_b(x):
    if x.dtype != torch.float32: raise ValueError('dtype')
    return b(x)
'''
    fake=SimpleNamespace(float16=object(),float32=object())
    ns={'torch':fake,'jit':lambda f:f}
    exec(compact_source(source,profile='fixed-native'),ns)
    assert ns['run_a'](x=SimpleNamespace(dtype=fake.float16,value=9))==9
    assert ns['run_b'](SimpleNamespace(dtype=fake.float32,value=7))==7
    with pytest.raises(ValueError, match='(?:dtype)'):ns['run_a'](SimpleNamespace(dtype=fake.float32))
