import ast
import math
from pathlib import Path
import pytest
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler import reference
from flowgpu.compiler.triton_backend import module
from flowgpu.compiler.cache import cached_module
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler.cli import main


def run(src, args, dims=None, **options):
    p = compile_source(src, **options).pipelines[0]
    ns = {}
    exec(reference.module([p.reference]), ns)
    return ns['run_' + p.pipeline.name](*args, *[(dims or {})[d] for d in p.reference.dimensions])


@pytest.mark.parametrize('filename,stages,slots', [('fill_bonus_tokens', 1, 1), ('dcp_lse_combine', 6, 6)])
def test_language_programs_eliminate_buffers_without_duplicating_nodes(filename, stages, slots):
    src = Path(f'examples/batch8/{filename}.fgpu').read_text()
    p = compile_source(src,reduction_fusion=False).pipelines[0]
    assert p.triton.complete and p.reference.complete
    assert len(p.triton.kernels) == stages and len(p.memory.slots) == slots
    members = [p.dataflow.index_of(n) for g in p.fusion.groups for n in g.members]
    assert len(members) == len(set(members))
    assert set(members) == {i for i,n in enumerate(p.dataflow.nodes) if n.op not in {'input','output'}}
    for g in p.fusion.groups:
        inside = {p.dataflow.index_of(n) for n in g.members}
        for v in g.fused_values:
            assert all(p.dataflow.index_of(n) in inside for n in p.dataflow.consumers_of(v))
        for n in g.members:
            for v in n.inputs:
                i = p.dataflow.index_of(p.dataflow.producer_of(v))
                assert i in inside or i < min(inside)
    ast.parse(module([p.triton])); ast.parse(reference.module([p.reference]))
    control = compile_source(src,index_fusion=False).pipelines[0]
    assert len(control.triton.kernels) == (4 if filename == 'fill_bonus_tokens' else 26)


def test_shared_values_and_declared_intermediates_keep_storage():
    src='pipeline p { input x : fp32[2,3] a=exp(x) y=sum_last(a) z=max_last(a) output a output y output z }'
    p=compile_source(src).pipelines[0]
    assert 'a' not in p.fusion.fused_values
    got=run(src,[[0,1,2,3,4,5]])
    assert got['a']==pytest.approx([math.exp(x) for x in range(6)],rel=1e-6)
    assert got['y']==pytest.approx([sum(math.exp(x) for x in range(3)),sum(math.exp(x) for x in range(3,6))],rel=1e-6)


def test_broadcast_reduce_does_not_repeat_the_inner_reduction_in_a_larger_output():
    src='pipeline p { input x : fp32[2,3] s=sum_last(x) b=broadcast_like(s,x) y=div(x,b) output y }'
    p=compile_source(src,reduction_fusion=False).pipelines[0]
    assert len(p.triton.kernels)==2
    assert 's' not in p.fusion.fused_values
    assert run(src,[[1,2,3,2,3,5]])['y']==pytest.approx([1/6,2/6,3/6,.2,.3,.5])


def test_composed_strides_views_and_broadcast_have_independent_scalar_semantics():
    src='pipeline p { input x : fp32[2,3] input w : fp32[3,1] v=move_first_last(x) b=broadcast_like(w,v) y=sum_last(mul(v,b)) output y }'
    p=compile_source(src).pipelines[0]
    assert len(p.triton.kernels)==1 and len(p.memory.slots)==1
    assert run(src,[[1,2,3,4,5,6],[2,3,4]])['y']==[10,21,36]
    assert 'FG_REDUCE_TILE: tl.constexpr' in p.triton.kernels[0].signature
    assert 'min(4096, triton.next_power_of_2(2))' in p.triton.host


def test_gather_masks_negative_wrap_and_preserves_int32_overflow():
    src=Path('examples/batch8/fill_bonus_tokens.fgpu').read_text().replace('lengths : int32','lengths : int64')
    assert run(src,[list(range(6)),[-100,100],[0,0]],{'T':2,'D':3,'N':2})['out']==[0,0]
    assert run(src,[list(range(6)),[0,1],[0,0]],{'T':2,'D':3,'N':2})['out']==[5,3]
    # Subtraction wraps at the original int32 node before int64 addressing.
    src=src.replace('lengths : int64','lengths : int32')
    assert run(src,[list(range(6)),[-2147483648,1],[0,0]],{'T':2,'D':3,'N':2})['out']==[0,3]


def test_masked_gather_of_a_fused_nonzero_expression_still_returns_zero():
    src='pipeline p { input x : fp32[2,3] input i : int32[2] a=exp(x) y=gather_row_offset(a,i) output y }'
    p=compile_source(src).pipelines[0]
    assert len(p.triton.kernels)==1
    assert run(src,[[0]*6,[-100,100]])['y']==[0,0]
    assert 'tl.where(gather_valid,' in p.triton.kernels[0].source


def test_rounding_and_reduction_nan_are_retained_inside_regions():
    src='pipeline p { input x : fp16[1,3] e=exp(x) y=sum_last(e) output y }'
    a=run(src,[[0.,.5,1.]])['y'];b=run(src,[[0.,.5,1.]],index_fusion=False)['y']
    assert a==b
    src='pipeline p { input x : fp32[1,3] e=to_fp32(x) y=max_last(e) output y }'
    assert math.isnan(run(src,[[1,math.nan,3]])['y'][0])
    assert 'float("nan")' in compile_source(src).pipelines[0].triton.kernels[0].source


def test_late_operand_cannot_be_read_before_its_kernel():
    src='pipeline p { input x : fp32[2,3] a=exp(x) s=sum_last(x) b=broadcast_like(s,x) y=add(a,b) output y }'
    p=compile_source(src,reduction_fusion=False).pipelines[0]
    # a's source position precedes the external s, so it stays materialized.
    assert 'a' not in p.fusion.fused_values
    assert run(src,[[0,0,0,0,0,0]])['y']==[1]*6


def test_cache_and_cli_control_keep_platform_validation_gate(tmp_path,capsys):
    src=Path('examples/batch8/fill_bonus_tokens.fgpu').read_text()
    assert cached_module(src,cache_dir=tmp_path,index_fusion=False)[2]!=cached_module(src,cache_dir=tmp_path,index_fusion=True)[2]
    path=tmp_path/'bonus.fgpu';path.write_text(src)
    assert main([str(path),'--no-index-fusion','--emit','triton','--cache-dir',str(tmp_path)])==0
    assert capsys.readouterr().out.count('@triton.jit')==4
    assert main([str(path),'--emit','triton'])==0
    assert capsys.readouterr().out.count('@triton.jit')==1
    p=compile_source(src,platform_config=resolve_platform('kunlunxin')).pipelines[0]
    assert not p.triton.complete and not any(g.indexed for g in p.fusion.groups)


def test_json_reports_reshape_regions_and_replicated_broadcast_reads(capsys):
    import json
    from flowgpu.compiler.traffic import indexed_read_count, at_unit
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    p=compile_source(src,shape_bindings={'R':7,'B':2,'H':3,'D':33},
                     cross_domain_strategy='materialize').pipelines[0]
    numerator_group=next(g for g in p.fusion.groups if any(n.output=='numerator' for n in g.members))
    assert at_unit(indexed_read_count(p.dataflow,numerator_group,'weights'))==33
    assert at_unit(indexed_read_count(p.dataflow,numerator_group,'denom'))==33
    assert main(['examples/batch8/dcp_lse_combine.fgpu','--emit','json'])==0
    report=json.loads(capsys.readouterr().out)
    assert any(g['indexed'] for g in report['pipelines'][0]['fusion']['kernels'])
