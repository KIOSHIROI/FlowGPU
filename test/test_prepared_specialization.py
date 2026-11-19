import ast
import copy
import json

import pytest

from compiler.backend.specialization import freeze_autotune
from compiler.compile import compile_source
from compiler.triton_backend import module


POOL = '''pipeline pool { input kv : fp32[B,K,D] input scores : fp32[B,K,D]
v=to_fp32(kv) s=to_fp32(scores) w=softmax_axis(s,1)
p=mul(v,w) a=sum_axis(p,1) out=cast_like(a,kv) output out }'''


def emitted(source=POOL, **options):
    return module([compile_source(source, **options).pipelines[0].triton])


def first_configs(code):
    from compiler.backend.specialization import _configs
    tree = ast.parse(code)
    assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)}
    return {n.name: _configs(next(k.value for k in d.keywords if k.arg == 'configs'), assignments)[0]
            for n in tree.body if isinstance(n, ast.FunctionDef) for d in n.decorator_list
            if isinstance(d, ast.Call) and ast.unparse(d.func) == 'triton.autotune'}


def freeze(code, selections=None):
    return freeze_autotune(code, first_configs(code) if selections is None else selections,
                           pipeline='pool', device_key='device-proof',
                           context=[dict(shape=[3,129,31], stride=[7998,62,1], dtype='torch.float32',
                                         device='cuda:0', alignment_mod128=44)] * 2)


def test_freeze_owns_config_grid_and_preserves_numerical_kernel_bodies():
    code = emitted(autotune_fold=True)
    selected = first_configs(code)
    fixed = freeze(code, selected)
    assert fixed == freeze(code, json.loads(json.dumps(selected, sort_keys=True)))
    original = ast.parse(code)
    tree = ast.parse(fixed)
    assert '@triton.autotune' not in fixed
    assert 'lambda META' not in fixed
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fn in original.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in selected:
            assert ast.dump(ast.Module(body=fn.body, type_ignores=[])) == ast.dump(ast.Module(body=functions[fn.name].body, type_ignores=[]))
    launches = [n for n in ast.walk(functions['run_pool']) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                and n.func.value.id in selected]
    assert len(launches) == len(selected)
    for call in launches:
        assert {k.arg:ast.literal_eval(k.value) for k in call.keywords if k.arg in selected[call.func.value.id]} == selected[call.func.value.id]
    assert 'prepare again' in fixed
    assert '7998, 62, 1' in fixed and '% 128' in fixed


def test_freeze_gemm_resolves_global_config_table_and_meta_grid():
    code = emitted('pipeline pool { input x : bf16[M,K] input w : bf16[K,N] a=matmul(x,w) y=silu(a) output y }', autotune=True)
    fixed = freeze_autotune(code,first_configs(code),pipeline='pool',device_key='device-proof',
        context=[dict(shape=shape,stride=stride,dtype='torch.bfloat16',device='cuda:0',alignment_mod128=0)
                 for shape,stride in [([3,129],[129,1]),([129,31],[31,1])]])
    assert '@triton.autotune' not in fixed
    assert 'pool_k0[1, 1]' in fixed
    assert 'BLOCK_M=16' in fixed and 'num_stages=2' in fixed


def test_host_metadata_is_constant_but_context_guards_and_numerics_are_live():
    code=emitted(autotune_fold=True)
    fixed=freeze(code)
    tree=ast.parse(fixed)
    host=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_pool')
    launch=next(n for n in ast.walk(host) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript))
    assert not any(isinstance(n,ast.Attribute) and n.attr in {'shape','stride','is_contiguous'} for n in ast.walk(launch))
    assert 'prepared input context changed' in fixed
    assert fixed.index('kv.layout != torch.strided') < fixed.index('tuple(kv.stride())')
    assert 'torch.empty' in fixed and 'device=kv.device' in fixed
    # A conflicting declared type is never erased as an unproven guard.
    bad=freeze(code.replace('kv.dtype != torch.float32','kv.dtype != torch.float16'))
    assert 'torch.float32 != torch.float16' in bad


def test_freeze_rejects_stale_or_out_of_space_configs():
    code = emitted(autotune_fold=True)
    selected = first_configs(code)
    bad = copy.deepcopy(selected)
    bad[next(iter(bad))]['ROW'] = 123
    with pytest.raises(ValueError, match='outside emitted search space'):
        freeze(code, bad)
    with pytest.raises(ValueError, match='required per'):
        freeze(code, {})
    bad = dict(selected, missing_kernel={})
    with pytest.raises(ValueError, match='names do not match'):
        freeze(code, bad)
    with pytest.raises(ValueError, match='exactly one host launch'):
        freeze(code.replace(next(iter(selected))+'[lambda', 'absent[lambda'), selected)


def test_hooks_cannot_be_silently_dropped_and_defaults_remain_fixed():
    code = emitted(autotune_fold=True)
    with pytest.raises(ValueError, match='hooks/reset/restore'):
        freeze(code.replace('**_FG_AUTOTUNE_CACHE_KWARGS', "reset_to_zero=['out'], **_FG_AUTOTUNE_CACHE_KWARGS"))
    assert '@triton.autotune' not in emitted()


@pytest.mark.parametrize('change', [
    lambda code:code.replace("{'cache_results': True}", "{'cache_results': True, 'reset_to_zero': ['out']}"),
    lambda code:code+"\n_FG_AUTOTUNE_CACHE_KWARGS['reset_to_zero'] = ['out']\n",
    lambda code:code.replace('**_FG_AUTOTUNE_CACHE_KWARGS','**{}'),
])
def test_only_the_exact_unmodified_cache_compatibility_dictionary_can_be_frozen(change):
    code=emitted(autotune_fold=True)
    with pytest.raises(ValueError,match='hooks/reset/restore'):
        freeze(change(code),first_configs(code))


def test_fixed_cross_domain_can_prepare_metadata_without_claiming_autotune():
    from pathlib import Path
    source=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    code=emitted(source,autotune_fold=True,cross_domain_strategy='recompute')
    assert not first_configs(code)
    fixed=freeze_autotune(code,{},pipeline='lse',device_key='proof',context=[
        dict(shape=[7,3,2,33],stride=[198,66,33,1],dtype='torch.float16',device='cuda:0',alignment_mod128=0),
        dict(shape=[7,3,2],stride=[6,2,1],dtype='torch.float32',device='cuda:0',alignment_mod128=0)])
    assert '@triton.autotune' not in fixed and 'lse_k0[6, 2]' in fixed
    assert fixed.count('recv.stride(')==1  # Exact guard remains; launch queries are folded.
