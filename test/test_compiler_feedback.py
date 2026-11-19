import ast
import math
from pathlib import Path
import pytest
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler import reference
from flowgpu.compiler.triton_backend import module
from flowgpu.compiler.cache import cached_module
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler.traffic import plan_traffic, ratio_at
from flowgpu.compiler.cli import main

GEMM='pipeline mm { input a : fp16[M,K] input b : fp16[K,N] y = silu(matmul(a,b)) output y }'

def run_ref(src, tensors, dims):
    p=compile_source(src).pipelines[0].reference
    ns={};exec(reference.module([p]),ns)
    return ns['run_'+p.pipeline](*tensors,*[dims[d] for d in p.dimensions])

@pytest.mark.parametrize('group',[1,3,8])
def test_autotune_uses_selected_meta_for_grid_and_owns_launch_parameters(group):
    p=compile_source(GEMM,autotune=True,platform_config=resolve_platform(group_m=group)).pipelines[0].triton
    code=module([p]);ast.parse(code)
    assert "'cache_results': True" in code and '**_FG_AUTOTUNE_CACHE_KWARGS' in code
    assert "META['BLOCK_M']" in p.host and "META['BLOCK_N']" in p.host
    k=p.kernels[0]
    assert not any(arg.text in {'BLOCK_M','BLOCK_N','BLOCK_K'} for arg in k.launch)
    assert any(arg.kind == 'device_key' for arg in k.launch)
    assert '_flowgpu_device_key(a.device.index)' in p.host
    assert not k.launch_options
    assert 'num_stages=' not in p.host
    assert "key=['FG_M', 'FG_K', 'FG_N', 'FG_DEVICE_KEY']" in code


def test_autotune_platform_gate_and_pruning_remain_explicit():
    from flowgpu.compiler.autotune import configs_for
    configs=configs_for('kunlunxin')
    assert configs and all(m<=64 and n<=64 and k<=32 and stages==2 for m,n,k,stages,warps in configs)
    with pytest.raises(ValueError,match='explicitly enabling'):
        compile_source(GEMM,autotune=True,platform_config=resolve_platform('kunlunxin'))
    p=resolve_platform('kunlunxin',allowed_kernel_kinds=('matmul',))
    assert compile_source(GEMM,autotune=True,platform_config=p).pipelines[0].triton.complete
    assert compile_source(GEMM,autotune=True,gpu_backend='hybrid').pipelines[0].triton.complete
    with pytest.raises(ValueError, match='autotune unavailable for ascend'):compile_source(GEMM,autotune=True,platform_config=resolve_platform('ascend',mapping_strategy='direct_grid'))


def test_module_cache_hit_invalidation_and_corrupt_entry(tmp_path,monkeypatch):
    code,hit,key=cached_module(GEMM,cache_dir=tmp_path)
    assert not hit
    def forbidden(*args,**kwargs):raise AssertionError('cache hit recompiled')
    with monkeypatch.context() as m:
        m.setattr('flowgpu.compiler.compile.compile_source',forbidden)
        assert cached_module(GEMM,cache_dir=tmp_path)==(code,True,key)
    assert cached_module(GEMM,cache_dir=tmp_path,autotune=True)[2]!=key
    assert cached_module(GEMM.replace('silu','relu'),cache_dir=tmp_path)[2]!=key
    (tmp_path/(key+'.json')).write_text('{broken')
    assert not cached_module(GEMM,cache_dir=tmp_path)[1]
    with monkeypatch.context() as m:
        m.setattr('flowgpu.compiler.cache.compiler_fingerprint',lambda:'changed')
        assert cached_module(GEMM,cache_dir=tmp_path)[2]!=key


def test_cli_autotune_with_cache(tmp_path,capsys):
    source=tmp_path/'x.fgpu';source.write_text(GEMM)
    assert main([str(source),'--autotune','--cache-dir',str(tmp_path/'cache'),'--emit','triton'])==0
    ast.parse(capsys.readouterr().out)
    assert main([str(source),'--autotune','--gpu-backend','hybrid','--emit','triton'])==0


def test_traffic_selects_materialization_when_a_prologue_repeats_many_times():
    src='pipeline p { input a : fp16[64,32] input b : fp16[32,4096] input c : fp16[64,32] x = add(a,c) y = matmul(x,b) output y }'
    rule=compile_source(src,fusion_policy='rule').pipelines[0]
    cost=compile_source(src,fusion_policy='traffic').pipelines[0]
    assert len(rule.fusion.groups)==1 and len(cost.fusion.groups)==2
    rt=plan_traffic(rule.dataflow,rule.fusion,rule.memory,tiled=True)
    ct=plan_traffic(cost.dataflow,cost.fusion,cost.memory,tiled=True)
    assert ratio_at(ct,{})<ratio_at(rt,{})
    assert any('placement-aware' in c.reason for c in rt.charges)
    # Width 64 has no operand replication: fuse rather than materialize.
    small=compile_source(src.replace('4096','64')).pipelines[0]
    assert len(small.fusion.groups)==1


def test_gather_is_an_indexed_language_operation_with_integer_cast_and_flat_wrap():
    src=Path('examples/batch8/fill_bonus_tokens.fgpu').read_text().replace('bonus_template : int64','bonus_template : int32')
    got=run_ref(src,[[10,11,12,20,21,22,30,31,32],[3,1],[0,0]],{'T':3,'D':3,'N':2})
    assert got['out']==[12,20]
    g='pipeline g { input x : int64[2,3] input i : int64[2] y = gather_row_offset(x,i) output y }'
    assert run_ref(g,[[0,1,2,3,4,5],[-1,0]],{})['y']==[5,3]
    assert run_ref(g,[[0,1,2,3,4,5],[-100,100]],{})['y']==[0,0]


@pytest.mark.parametrize('base_e',[False,True])
def test_lse_language_program_matches_independent_math_and_all_invalid(base_e):
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text().replace('recv : fp16','recv : fp32')
    if not base_e:src=src.replace('weights = exp(','weights = exp2(').replace('logdenom = log(','logdenom = log2(')
    vals=[1,3,5,7,9,11] # R=3,B=H=1,D=2
    logs=[-20000.,float('nan'),-20001.]
    got=run_ref(src,[vals,logs],{'R':3,'B':1,'H':1,'D':2})
    weight=math.exp(-1) if base_e else .5
    assert got['out']==pytest.approx([(1+9*weight)/(1+weight),(3+11*weight)/(1+weight)],abs=1e-5)
    assert got['out_lse'][0]==pytest.approx(-20000+(math.log(1+weight) if base_e else math.log2(1+weight)),abs=.003)
    invalid=run_ref(src,[vals,[-math.inf]*3],{'R':3,'B':1,'H':1,'D':2})
    assert all(math.isnan(x) for x in invalid['out']) and invalid['out_lse']==[-math.inf]


def test_invalid_broadcast_dtype_and_reduction_are_diagnosed():
    from flowgpu.compiler.diagnostic import CompileError
    for src in ['pipeline p { input x : fp32[2,3] input z : fp32[2,4] y=broadcast_like(x,z) output y }',
                'pipeline p { input x : int32[2,3] y=sum_last(x) output y }',
                'pipeline p { input x : fp32[2,3] y=where(x,x,x) output y }']:
        with pytest.raises(CompileError, match='pe error in pipeline'):compile_source(src)


def test_rank_guard_precedes_symbolic_dimension_extraction():
    host=compile_source(GEMM).pipelines[0].triton.host
    assert host.index('.ndim')<host.index('.shape[0]')


def test_math_primitives_follow_ieee_zero_and_overflow():
    src='pipeline p { input a : fp32[3] input b : fp32[3] y=div(a,b) z=exp(a) output y output z }'
    got=run_ref(src,[[0,1,1000],[0,-0.0,1]],{})
    assert math.isnan(got['y'][0]) and got['y'][1]==-math.inf
    assert got['z'][2]==math.inf


def test_experimental_cli_can_emit_pruned_kunlun_source_without_claiming_validation(tmp_path,capsys):
    src=tmp_path/'x.fgpu';src.write_text(GEMM)
    assert main([str(src),'--emit','triton','--platform','kunlunxin','--autotune',
                 '--allow-experimental-kernels','matmul'])==0
    code=capsys.readouterr().out
    assert "'BLOCK_N': 256" not in code
    assert "'autotune_status': 'experimental'" in code


def test_shape_specialization_drives_traffic_and_cache_and_guards(tmp_path):
    src='pipeline p { input a : fp16[M,K] input b : fp16[K,N] input c : fp16[M,K] x=add(a,c) y=matmul(x,b) output y }'
    options={'shape_bindings':{'M':64,'K':32,'N':4096}}
    p=compile_source(src,**options).pipelines[0]
    assert len(p.fusion.groups)==2
    assert p.dataflow.values['y'].shape==['64','4096']
    assert 'shape != (64, 32,)' in p.triton.host
    assert 'M = ' not in p.triton.host
    wide=cached_module(src,cache_dir=tmp_path,**options)
    small=cached_module(src,cache_dir=tmp_path,shape_bindings={'M':64,'K':32,'N':64})
    assert wide[2]!=small[2]
    for bad in [{'typo':1},{'M':0},{'M':True}]:
        with pytest.raises(ValueError, match='(?:unknown shape symbols: typo|shape_bindings must map symbol names to positive integers)'):compile_source(src,shape_bindings=bad)


def test_cli_shape_specialization(tmp_path,capsys):
    src=tmp_path/'x.fgpu';src.write_text(GEMM)
    assert main([str(src),'--emit','triton','--shape','M=17','--shape','K=32','--shape','N=33'])==0
    text=capsys.readouterr().out
    assert 'shape != (17, 32,)' in text
