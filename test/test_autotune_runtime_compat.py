"""Optional persistence must not prevent autotuning on older Triton runtimes."""
from types import SimpleNamespace

import pytest

from flowgpu.compiler.backend.autotune import cache_options_source
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source


@pytest.mark.parametrize('persistent', [False, True])
def test_timing_cache_keyword_is_only_sent_when_the_runtime_declares_it(persistent):
    seen={}
    def legacy(configs, key):
        seen.update(configs=configs,key=key)
    def modern(configs, key, cache_results=False):
        seen.update(configs=configs,key=key,cache_results=cache_results)
    runtime=modern if persistent else legacy
    ns={'triton':SimpleNamespace(autotune=runtime)}
    exec(cache_options_source(),ns)
    options=ns['_FG_AUTOTUNE_CACHE_KWARGS']
    runtime(configs=[],key=['D'],**options)
    assert options == ({'cache_results':True} if persistent else {})
    assert seen == dict(configs=[],key=['D'],**options)


@pytest.mark.parametrize('mode', ['gemm','fold'])
def test_generated_autotune_modules_include_one_compatibility_binding(mode):
    source = ('pipeline p { input a : fp16[M,K] input b : fp16[K,N] y=matmul(a,b) output y }'
              if mode=='gemm' else 'pipeline p { input x : fp32[B,D] y=softmax(x) output y }')
    options={'autotune':True} if mode=='gemm' else {'autotune_fold':True}
    p=compile_source(source,**options).pipelines[0].triton
    code=module([p])
    assert code.count('_FG_AUTOTUNE_CACHE_KWARGS =') == 1
    assert '**_FG_AUTOTUNE_CACHE_KWARGS' in code
    plain=module([compile_source(source).pipelines[0].triton])
    assert '_fg_inspect' not in plain
