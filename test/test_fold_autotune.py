import ast
import json

import pytest

from compiler.compile import compile_source
from compiler.backend.autotune import fold_configs
from compiler.platform_configs import resolve_platform
from compiler.triton_backend import module
from compiler.cache import cached_module
from compiler.cli import main


POOL = '''pipeline pool { input kv : fp32[B,K,D] input scores : fp32[B,K,D]
v=to_fp32(kv) s=to_fp32(scores) w=softmax_axis(s,1)
p=mul(v,w) a=sum_axis(p,1) out=cast_like(a,kv) output out }'''


def test_fold_tuning_owns_grid_and_signature_prefix_not_fixed_launch_tiles():
    p=compile_source(POOL,autotune_fold=True).pipelines[0]
    ast.parse(module([p.triton]))
    tuned=[k for k in p.triton.kernels if k.autotune_family]
    assert {k.autotune_family for k in tuned}=={'rowwise','indexed_reduction'}
    for k in tuned:
        tree=ast.parse(k.source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
        params=[n.arg for n in fn.args.args]
        count=len(k.launch)
        # Non-tuned arguments form the positional prefix. Tile parameters are
        # trailing kwargs supplied exclusively by triton.Config.
        assert params[:count]==[s.split(':')[0].strip() for s in k.parameters[:count]]
        assert params[count:]==(['ROW'] if k.autotune_family=='rowwise' else ['FG_COLUMNS','FG_ROWS'])
        assert len(params)==len(k.parameters)
        assert any(a.kind=='device_key' for a in k.launch)
        assert not {'num_warps','num_stages'} & dict(k.launch_options).keys()
        assert '@triton.autotune' in k.source and '**_FG_AUTOTUNE_CACHE_KWARGS' in k.source
    indexed=next(k for k in tuned if k.autotune_family=='indexed_reduction')
    assert "META['FG_ROWS']" in indexed.grid
    assert 'next_power_of_2' not in indexed.grid
    assert not any('next_power_of_2' in a.text for a in indexed.launch)


@pytest.mark.parametrize('op',['softmax','rms_norm'])
def test_literal_rows_still_stream_when_selected_tile_is_smaller(op):
    extra='input w : fp32[257]' if op=='rms_norm' else ''
    args='x,w' if extra else 'x'
    p=compile_source(f'pipeline p {{ input x : fp32[3,257] {extra} y={op}({args}) output y }}',autotune_fold=True).pipelines[0]
    k=p.triton.kernels[0]
    assert 'range(0, tl.cdiv(257, ROW))' in k.source
    assert 'triton.next_power_of_2(257)' not in p.triton.host


def test_unfused_and_indexed_reductions_both_tune_but_keep_rounding_controls():
    src='pipeline p { input x : fp32[B,K] y=sum_last(x) output y }'
    for indexed in [False,True]:
        p=compile_source(src,autotune_fold=True,index_fusion=indexed).pipelines[0]
        k=p.triton.kernels[0]
        assert k.autotune_family
        assert dict(k.launch_options)['enable_fp_fusion'] is False
        assert k.grid.startswith('lambda META:')


def test_resource_search_space_respects_explicit_lane_and_axis_caps():
    cfg=resolve_platform(reduction_block_limit=128,indexed_columns=32,indexed_rows=4)
    assert all(c[0]<=128 for c in fold_configs(cfg,'rowwise'))
    candidates=fold_configs(cfg,'indexed_reduction')
    assert candidates and all(c[0]<=32 and c[1]<=4 and c[0]*c[1]<=128 for c in candidates)
    with pytest.raises(ValueError, match='fold autotune currently requires CUDA device evidence'):fold_configs(resolve_platform('kunlunxin'),'rowwise')
    with pytest.raises(ValueError, match='fold autotune currently requires CUDA triton/hybrid'):compile_source(POOL,autotune_fold=True,platform_config=resolve_platform('kunlunxin'))


def test_cli_cache_policy_and_default_fixed_emission(tmp_path,capsys):
    path=tmp_path/'pool.fgpu';path.write_text(POOL)
    assert main([str(path),'--autotune-fold','--emit','json'])==0
    p=json.loads(capsys.readouterr().out)['pipelines'][0]
    assert p['triton']['autotune_fold']
    assert any(k['autotune_family']=='indexed_reduction' for k in p['triton']['kernels'])
    assert main([str(path),'--autotune-fold','--emit','triton','--cache-dir',str(tmp_path/'cache')])==0
    assert '@triton.autotune' in capsys.readouterr().out
    code,hit,key=cached_module(POOL,cache_dir=tmp_path/'cache')
    tuned,hit2,key2=cached_module(POOL,autotune_fold=True,cache_dir=tmp_path/'cache')
    assert key!=key2 and '@triton.autotune' not in code and code!=tuned


def test_default_other_paths_and_gemm_tuning_are_independent():
    p=compile_source(POOL).pipelines[0]
    assert not any(k.autotune_family for k in p.triton.kernels)
    with pytest.raises(ValueError, match='autotune_fold must be a bool'):compile_source(POOL,autotune_fold=1)
