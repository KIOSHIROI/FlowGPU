import ast
from pathlib import Path
import pytest
from compiler.compile import compile_source
from compiler import reference, triton_backend
from compiler.diagnostic import CompileError
from compiler.platform_configs import resolve_platform
from compiler.cache import cached_module
from compiler.cli import main


def evaluate(src, args, **options):
    p = compile_source(src, **options).pipelines[0]
    ns = {}
    exec(reference.module([p.reference]), ns)
    return ns['run_' + p.pipeline.name](*args), p


@pytest.mark.parametrize('mapping', ['flat', 'row_tiled'])
@pytest.mark.parametrize('fuse', [False, True])
def test_unequal_heights_widths_types_shared_index_and_zero_fill(mapping, fuse):
    src = '''pipeline p { input x : fp32[3,2] input z : int64[2]
      input i : int64[6] y=gather_rows(x,i) w=gather_rows(z,i) output y output w }'''
    got, p = evaluate(src, [[1,2,3,4,5,6], [2**40,2**40+1], [0,-1,-3,2,2**32,-2**63]],
                      gather_fusion=fuse, platform_config=resolve_platform(gather_mapping=mapping))
    assert got['y'] == [1,2,5,6,1,2,5,6,0,0,0,0]
    assert got['w'] == [2**40,2**40+1,0,0,0,0]
    assert len(p.triton.kernels) == (1 if fuse else 2)
    assert len(p.memory.slots) == 2  # launch fusion doesn't remove output storage
    ast.parse(triton_backend.module([p.triton]))


@pytest.mark.parametrize('enabled', [False, True])
def test_gather_rows_composes_with_source_and_reduction_without_changing_oob(enabled):
    src = '''pipeline p { input x : fp32[2,2,3] input i : int64[3]
      e=exp(x) g=gather_rows(e,i) y=sum_last(g) output y }'''
    got, p = evaluate(src, [[0]*12, [-100,-1,2**40]], index_fusion=enabled)
    assert got['y'] == [0,0,3,3,0,0]
    assert len(p.triton.kernels) == (1 if enabled else 3)


def test_dtype_shape_errors_and_foreign_validation_gate():
    with pytest.raises(CompileError, match='integer indices'):
        compile_source('pipeline p { input x : fp32[2,3] input i : fp32[4] y=gather_rows(x,i) output y }')
    with pytest.raises(CompileError, match='1D indices'):
        compile_source('pipeline p { input x : fp32[2,3] input i : int32[2,2] y=gather_rows(x,i) output y }')
    src = 'pipeline p { input x : bool[3] input i : int32[2] y=gather_rows(x,i) output y }'
    assert evaluate(src, [[False,True,False], [1,3]])[0]['y'] == [True,False]
    p = compile_source(src, platform_config=resolve_platform('kunlunxin')).pipelines[0]
    assert not p.triton.complete and p.reference.complete
    with pytest.raises(ValueError, match='CUDA-only'):
        compile_source(src, platform_config=resolve_platform('kunlunxin'), gather_fusion=True)


def test_sibling_fusion_does_not_move_late_source_or_drop_other_readers():
    src = '''pipeline p { input x : fp32[2,3] input i : int32[1]
      a=gather_rows(x,i) z=exp(x) b=gather_rows(z,i) output a output b output z }'''
    got, p = evaluate(src, [[0]*6, [1]], gather_fusion=True)
    assert got == {'a':[0]*3,'b':[1]*3,'z':[1]*6}
    assert not any(g.row_gather for g in p.fusion.groups)
    src = '''pipeline p { input x : fp32[2,3] input i : int32[1] input j : int32[1]
      a=gather_rows(x,i) b=gather_rows(x,j) output a output b }'''
    assert len(compile_source(src,gather_fusion=True).pipelines[0].triton.kernels) == 2


def test_symbolic_shapes_cache_and_both_cli_paths(tmp_path, capsys):
    src = Path('examples/batch8/gather_spec_extras.fgpu').read_text()
    separate = compile_source(src).pipelines[0]
    fused = compile_source(src,gather_fusion=True).pipelines[0]
    assert len(separate.triton.kernels) == 4 and len(fused.triton.kernels) == 1
    assert tuple(fused.dataflow.values['out_hidden'].shape) == ('N','D')
    a=cached_module(src,cache_dir=tmp_path,gather_fusion=False)[2]
    b=cached_module(src,cache_dir=tmp_path,gather_fusion=True)[2]
    c=cached_module(src,cache_dir=tmp_path,gather_fusion=True,
                    platform_config=resolve_platform(gather_mapping='row_tiled'))[2]
    assert len({a,b,c}) == 3
    for cached in (False,True):
        args=['examples/batch8/gather_spec_extras.fgpu','--emit','triton',
              '--gather-fusion','--gather-mapping','row_tiled','--indexed-rows','8','--indexed-columns','64']
        if cached:args+=['--cache-dir',str(tmp_path)]
        assert main(args)==0
        assert capsys.readouterr().out.count('@triton.jit')==1


def test_invalid_mapping_and_tile_are_rejected():
    with pytest.raises(ValueError, match='gather_mapping'):
        resolve_platform(gather_mapping='guessed')
    src='pipeline p { input x : fp32[2,3] input i : int32[1] y=gather_rows(x,i) output y }'
    with pytest.raises(ValueError, match='tile bound'):
        compile_source(src,platform_config=resolve_platform(gather_mapping='row_tiled',indexed_rows=8192))


def test_explainable_unequal_width_candidate_counts_and_cli(capsys):
    import json
    from compiler.gather_strategy import compare_gather_strategies
    src = Path('examples/batch8/gather_spec_extras.fgpu').read_text()
    dims = dict(N=17,Tp=33,Ti=35,Tt=37,Th=39,Kp=7,Ki=11,D=129)
    comparison=compare_gather_strategies(src,dims,sm_count=108)
    items={i['name']:i for i in comparison['candidates']}
    flat=items['flat-fused'];tiled=items['tiled-fused-8x64']
    assert comparison['default']=='flat-separate' and not comparison['automatic_dispatch']
    assert items['flat-separate']['launches']==4 and flat['launches']==1
    assert tiled['kernels'][0]['grid']==[3,3]
    assert tiled['kernels'][0]['logical_index_loads']==17*3
    assert flat['kernels'][0]['logical_index_loads']==17*129
    assert tiled['kernels'][0]['resources'] is None
    assert all(i['evidence_status']=='structural' for i in comparison['candidates'])
    assert len(tiled['kernels'][0]['output_widths'])==4
    cli=['examples/batch8/gather_spec_extras.fgpu','--emit','gather-strategies','--device-cores','108']
    cli += [a for key,value in dims.items() for a in ['--shape',f'{key}={value}']]
    assert main(cli)==0
    assert json.loads(capsys.readouterr().out)['candidates'][0]['launches']==4
    assert main(cli+['--gather-fusion'])==2


def test_comparison_does_not_borrow_stale_device_evidence():
    from compiler.gather_strategy import compare_gather_strategies
    src='pipeline p { input x : fp32[3,2] input i : int32[2] y=gather_rows(x,i) output y }'
    evidence={'schema_version':1,'compiler_fingerprint':'stale','source_sha256':'stale',
              'device':{'sm_count':108},'records':[]}
    report=compare_gather_strategies(src,{},sm_count=108,evidence=evidence)
    assert all(i['evidence_status']=='structural' and i['evidence_rejections'] for i in report['candidates'])


def test_evidence_requires_correctness_bit_even_when_hashes_match():
    from compiler.gather_strategy import compare_gather_strategies
    src='pipeline p { input x : fp32[3,2] input i : int32[2] y=gather_rows(x,i) output y }'
    structural=compare_gather_strategies(src,{},sm_count=108)
    item=next(i for i in structural['candidates'] if i['name']=='flat-fused')
    record=dict(candidate='flat-fused',bindings={},module_sha256=item['module_sha256'],
                input_context={'x':dict(dtype='fp32',shape=[3,2],stride=[2,1]),
                               'i':dict(dtype='int32',shape=[2],stride=[1])},
                kernels=[dict(name=k['name']) for k in item['kernels']],
                samples=dict(graph_us=[1.0],wall_us=[2.0]))
    evidence=dict(schema_version=1,source_sha256=structural['source_sha256'],
                  compiler_fingerprint=structural['compiler_fingerprint'],
                  device=dict(name='synthetic-test-device',uuid='test-only',sm_count=108),records=[record])
    def observed():
        return next(i for i in compare_gather_strategies(src,{},sm_count=108,evidence=evidence)['candidates'] if i['name']=='flat-fused')
    assert observed()['evidence_rejections']==['input context or correctness evidence missing']
    record['correctness_passed']=True
    assert observed()['evidence_status']=='measured_for_recorded_context'
