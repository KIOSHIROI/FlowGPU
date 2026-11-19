"""Prove the linear view map independently of any tensor numerical kernel."""
import ast
import itertools
import json
import math
import random
import textwrap
from types import SimpleNamespace

import pytest

from compiler.backend.platform_configs import resolve_platform
from compiler.backend.triton_backend import _view_offset
from compiler.compile import compile_source
from compiler.runtime.cache import cache_key


def offset_code(rank, linearize, coordinates=None):
    dims=[f'd{i}' for i in range(rank)];strides=[f's{i}' for i in range(rank)]
    lines=[]
    address=_view_offset('x','idx',{'x':('contiguous',dims,strides)},
                         SimpleNamespace(fresh=lambda hint:hint),lines,
                         coordinates=coordinates,linearize=linearize)
    return textwrap.dedent('\n'.join(lines)),address


@pytest.mark.parametrize('rank',[1,2,3,4])
def test_coordinate_sum_equals_linear_or_fallback_for_random_layouts(rank):
    rng=random.Random(181571+rank)
    for case in range(30):
        shape=[rng.randint(1,4) for _ in range(rank)]
        step=rng.choice([-3,0,1,2,5])
        regular=[math.prod(shape[i+1:])*step for i in range(rank)]
        arbitrary=[rng.randint(-31,31) for _ in shape]
        for strides in (regular,arbitrary):
            for linearize in (False,True):
                code,address=offset_code(rank,linearize)
                parsed=compile(code,'generated_view','exec')
                metadata=dict(contiguous=False,**{f'd{i}':d for i,d in enumerate(shape)},
                              **{f's{i}':s for i,s in enumerate(strides)})
                for idx,coords in enumerate(itertools.product(*(range(d) for d in shape))):
                    ns=dict(metadata,idx=idx);exec(parsed,ns)
                    assert ns[address]==sum(c*s for c,s in zip(coords,strides))
                # The old first coordinate is intentionally not modulo dim0;
                # compare the same padded tail indices used by masked tiles.
                for idx in range(math.prod(shape),math.prod(shape)+17):
                    coords=[idx//math.prod(shape[i+1:]) for i in range(rank)]
                    coords=[c if i==0 else c%shape[i] for i,c in enumerate(coords)]
                    ns=dict(metadata,idx=idx);exec(parsed,ns)
                    assert ns[address]==sum(c*s for c,s in zip(coords,strides))


def test_coordinates_override_is_preserved_and_boolean_nodes_are_binary():
    for rank in (2,3,4):
        coordinates=[f'c{i}' for i in range(rank)]
        assert offset_code(rank,False,coordinates)==offset_code(rank,True,coordinates)
        code,_=offset_code(rank,True)
        assert all(len(n.values)==2 for n in ast.walk(ast.parse(code)) if isinstance(n,ast.BoolOp))


def test_split_and_row_gaps_do_not_satisfy_the_linear_guard():
    code,address=offset_code(2,True)
    for stride0,stride1 in [(34,1),(41,2),(2**31+7,2)]:
        ns=dict(idx=35,contiguous=False,d0=7,d1=17,s0=stride0,s1=stride1)
        exec(code,ns)
        assert ns[address]==(35//17)*stride0+(35%17)*stride1
        assert stride0!=17*stride1
    tree=ast.parse(code)
    linear=tree.body[1].body[0].body[0]
    assert not any(isinstance(n,(ast.Mod,ast.FloorDiv)) for n in ast.walk(linear))


@pytest.mark.parametrize('value',[1,'yes',None])
def test_bad_platform_flag_is_rejected(value):
    from dataclasses import replace
    with pytest.raises(ValueError,match='boolean'):
        replace(resolve_platform(),linear_view_indexing=value)


def test_opt_in_has_its_own_cache_identity_and_preserves_numerical_statements():
    source='pipeline p { input x : fp32[B,D] input y : fp32[B,D] a=to_fp32(x) z=mul(silu(a),y) output z }'
    a=resolve_platform();b=resolve_platform(linear_view_indexing=True)
    assert cache_key(source,{'platform_config':a})!=cache_key(source,{'platform_config':b})
    def body(config):
        (k,)=compile_source(source,platform_config=config).pipelines[0].triton.kernels
        nodes=ast.parse(k.source).body[0].body
        return [ast.dump(n) for n in nodes if not (
            isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_address')
            or isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Name)
            and n.test.operand.id.endswith('_contiguous'))]
    assert body(a)==body(b)


def test_cli_emits_the_opt_in_and_rejects_strategy_modes_that_own_their_configs(tmp_path,capsys):
    from compiler.cli import main
    path=tmp_path/'view.fgpu';path.write_text('pipeline p { input x : fp32[B,D] y=exp(x) output y }')
    assert main([str(path),'--linear-view-indexing','--emit','json'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['pipelines'][0]['triton']['platform_config']['linear_view_indexing']
    for mode in ('strategies','gather-strategies'):
        assert main([str(path),'--linear-view-indexing','--emit',mode])==2
        capsys.readouterr()
