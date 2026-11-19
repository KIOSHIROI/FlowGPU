import ast
import math
from dataclasses import replace
from pathlib import Path
import pytest
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler import reference
from flowgpu.compiler.triton_backend import module
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler.traffic import indexed_read_count,at_unit
from flowgpu.compiler.cache import cached_module
from flowgpu.compiler.cli import main

SOURCE=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()


def cpu(source,x,l,bindings=None):
    p=compile_source(source,cross_domain_strategy='recompute',shape_bindings=bindings).pipelines[0]
    ns={};exec(reference.module([p.reference]),ns)
    dims={'R':3,'B':1,'H':2,'D':3}
    return ns['run_lse'](x,l,*[dims[d] for d in p.reference.dimensions]),p


@pytest.mark.parametrize('bindings',[None,{'R':1,'B':1,'H':2,'D':1},{'R':7,'B':2,'H':3,'D':33}])
def test_recompute_is_generic_closed_partition_and_preserves_exports(bindings):
    p=compile_source(SOURCE,cross_domain_strategy='recompute',shape_bindings=bindings).pipelines[0]
    assert len(p.triton.kernels)==1 and len(p.memory.slots)==2
    assert p.triton.kernels[0].results==('out','out_lse')
    assert p.fusion.groups[0].cross_domain
    members=p.fusion.groups[0].members
    assert len({p.dataflow.index_of(n) for n in members})==len(members)
    for v in p.fusion.fused_values:
        assert all(c in members for c in p.dataflow.consumers_of(v))
    src=module([p.triton]);ast.parse(src);assert 'if tl.program_id(1) == 0:' in src
    assert src.count('    for ')==3
    renamed=SOURCE.replace('pipeline lse','pipeline arbitrary').replace('weights','importance')
    assert len(compile_source(renamed,cross_domain_strategy='recompute').pipelines[0].triton.kernels)==1


def test_reference_against_independent_math_and_node_rounding():
    x=list(range(18));l=[0,1,2,3,-1,0]
    got,p=cpu(SOURCE,x,l)
    expected=[];out_lse=[]
    for h in range(2):
        vals=l[h::2];m=max(vals);w=[math.exp(a-m) for a in vals];s=sum(w)
        out_lse.append(math.log(s)+m)
        for d in range(3):expected.append(sum(x[(r*2+h)*3+d]*w[r] for r in range(3))/s)
    assert got['out']==pytest.approx(expected,rel=.001)
    assert got['out_lse']==pytest.approx(out_lse,rel=1e-6)
    invalid,_=cpu(SOURCE,x,[-math.inf]*6)
    assert all(math.isnan(v) for v in invalid['out'])
    assert invalid['out_lse']==[-math.inf]*2
    one,_=cpu(SOURCE,[1,2],[5,-math.inf],{'R':1,'B':1,'H':2,'D':1})
    assert one['out'][0]==1 and math.isnan(one['out'][1])


def test_exported_weight_or_extra_reader_preserves_materialized_boundary():
    for extra in ('output weights','z=exp(weights) output z'):
        src=SOURCE.replace('output out\n',extra+'\noutput out\n')
        p=compile_source(src,cross_domain_strategy='recompute').pipelines[0]
        assert not any(g.cross_domain for g in p.fusion.groups)
        assert 'weights' not in p.fusion.fused_values


def test_tile_override_cache_key_bounds_and_device_gate(tmp_path,capsys):
    config=resolve_platform(indexed_columns=512,indexed_rows=8)
    p=compile_source(SOURCE,platform_config=config,cross_domain_strategy='recompute').pipelines[0]
    assert p.fusion.groups[0].cross_tile==(512,8)
    keys=[cached_module(SOURCE,cache_dir=tmp_path,cross_domain_strategy=mode,platform_config=config)[2] for mode in ('materialize','recompute')]
    assert keys[0]!=keys[1]
    for kwargs in ({'indexed_columns':3},{'indexed_rows':0},{'indexed_columns':512,'indexed_rows':16}):
        with pytest.raises(ValueError, match='(?:indexed_columns must be a positive power of two|indexed_rows must be a positive power of two|indexed R/D tile exceeds reduction_block_limit)'):resolve_platform(**kwargs)
    for kwargs in ({'cross_domain_strategy':'bad'},{'cross_domain_strategy':'recompute','index_fusion':False},
                   {'cross_domain_strategy':'recompute','platform_config':resolve_platform('kunlunxin')}):
        with pytest.raises(ValueError, match='recompute'):compile_source(SOURCE,**kwargs)
    assert main(['examples/batch8/dcp_lse_combine.fgpu','--cross-domain-strategy','recompute','--indexed-columns','512','--indexed-rows','8','--emit','json'])==0
    assert '"cross_domain"' in capsys.readouterr().out


def test_traffic_charges_lse_recomputation_per_d_tile():
    p=compile_source(SOURCE,cross_domain_strategy='recompute',shape_bindings={'R':7,'B':2,'H':3,'D':129}).pipelines[0]
    g=p.fusion.groups[0]
    assert at_unit(indexed_read_count(p.dataflow,g,'lse_in'))==3*5
    assert at_unit(indexed_read_count(p.dataflow,g,'recv'))==1


def test_cross_domain_sums_share_scan_only_without_dependency_and_traffic_agrees():
    cfg=resolve_platform(coalesce_reduction_sweeps=True)
    p=compile_source(SOURCE,cross_domain_strategy='recompute',platform_config=cfg,
                     shape_bindings={'R':7,'B':2,'H':3,'D':129}).pipelines[0]
    g=p.fusion.groups[0]
    assert g.reduction_batches[-1]==('denom','numerator')
    assert p.triton.kernels[0].source.count('    for ')==2
    assert at_unit(indexed_read_count(p.dataflow,g,'lse_in'))==2*5
    # Normalizing weights before the wide reduction needs the completed sum.
    dependent=SOURCE.replace('weighted = mul(values, wide_w)',
        'early_d=reshape(denom,[B,H,1,1]) early_w=div(wide_w,broadcast_like(early_d,wide_w)) weighted=mul(values,early_w)')
    q=compile_source(dependent,cross_domain_strategy='recompute',platform_config=cfg).pipelines[0]
    for group,kernel in zip(q.fusion.groups,q.triton.kernels):
        if group.cross_domain:
            assert all(len(batch)==1 for batch in group.reduction_batches)
            assert kernel.source.count('    for ')==3


def test_same_domain_plan_handles_row_major_generic_weighted_mean():
    source='''pipeline mean {
 input score : fp32[G,R] input values : fp32[G,D,R]
 m=max_last(score) wm=broadcast_like(m,score)
 w=exp(sub(score,wm)) s=sum_last(w)
 view=reshape(w,[G,1,R]) wide=broadcast_like(view,values)
 n=sum_last(mul(values,wide)) sv=reshape(s,[G,1,1])
 out=reshape(div(n,broadcast_like(sv,n)),[G,D])
 stat=reshape(add(log(s),m),[G]) output out output stat
 }'''
    p=compile_source(source,cross_domain_strategy='recompute').pipelines[0]
    assert len(p.triton.kernels)==1 and p.fusion.groups[0].cross_domain==('G','R')
    ns={};exec(reference.module([p.reference]),ns)
    dims={'G':2,'D':2,'R':3};score=[0]*6;values=list(range(12))
    got=ns['run_mean'](score,values,*[dims[d] for d in p.reference.dimensions])
    assert got['out']==[1,4,7,10] and got['stat']==pytest.approx([math.log(3)]*2,rel=1e-6)


def test_permuting_computed_wide_values_retains_established_partition():
    # Rotating a computed tensor changes base-row identity, even when the
    # symbols happen to have equal concrete values at runtime.
    source=SOURCE.replace('values = move_first_last(recv32)','values = move_first_last(exp(recv32))')
    p=compile_source(source,cross_domain_strategy='recompute').pipelines[0]
    assert not any(g.cross_domain for g in p.fusion.groups)
