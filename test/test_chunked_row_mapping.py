"""A row cap must cover the streaming branch, including fold autotuning."""
import ast

import pytest

from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.compile import compile_source


@pytest.mark.parametrize('operator', ['rmsnorm_hf', 'rms_norm', 'softmax'])
@pytest.mark.parametrize('width', ['D', '256'])
@pytest.mark.parametrize('autotune', [False, True])
def test_streaming_reductions_cap_rows_and_reset_the_accumulator(operator, width, autotune):
    expression = f'{operator}(x,w)' if operator != 'softmax' else 'softmax(x)'
    source = f'''pipeline rows {{
        input x : fp32[B,{width}]
        input w : fp32[{width}]
        y = {expression}
        output y
    }}'''
    config = resolve_platform(max_grid_dim=3, reduction_block_limit=128)
    p = compile_source(source, platform_config=config, autotune_fold=autotune).pipelines[0].triton
    (k,) = p.kernels
    assert k.kind == 'rowwise'
    loop = ast.parse(k.source).body[0].body[0]
    assert isinstance(loop, ast.For)
    assert getattr(loop.iter.args[-1].func, 'attr', '') == 'num_programs'
    assert 'else 3' in k.grid
    assert k.source.count('tl.program_id(0)') == 1
    bound = loop.iter.args[1]
    assert isinstance(bound, ast.Name)
    assert bound.id in {a.arg for a in ast.parse(k.source).body[0].args.args}
    index = next(i for i,parameter in enumerate(k.parameters)
                 if parameter.split(':')[0].strip() == bound.id)
    assert k.launch[index].text == 'B'
    # The first sweep's scalar initialization belongs to the outer row loop;
    # placing it outside accumulates unrelated rows when grid < row count.
    assigned = [n for n in loop.body if isinstance(n, ast.Assign)]
    assert any(isinstance(n.value, ast.Call) and
               getattr(n.value.func, 'attr', '') in {'zeros','full'} for n in assigned)


def test_an_ascend_profile_with_an_explicit_chunk_budget_is_also_capped():
    source = 'pipeline p { input x : fp32[B,D] input w : fp32[D] y=rmsnorm_hf(x,w) output y }'
    config = resolve_platform('ascend', mapping_strategy='direct_grid', reduction_block_limit=128,
                              allowed_kernel_kinds=('rowwise',))
    (k,) = compile_source(source, platform_config=config).pipelines[0].triton.kernels
    assert '65535' in k.grid
    assert 'tl.num_programs(0)' in k.source


@pytest.mark.parametrize('width', ['D','128'])
def test_capped_kernel_body_has_no_unbound_shape_symbols(width):
    source = f'pipeline p {{ input x : fp32[B,H,{width}] input w : fp32[{width}] y=rmsnorm_hf(x,w) output y }}'
    config = resolve_platform(max_grid_dim=3, reduction_block_limit=128)
    (k,) = compile_source(source,platform_config=config).pipelines[0].triton.kernels
    fn=ast.parse(k.source).body[0]
    parameters={a.arg for a in fn.args.args}
    stores={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    loads={n.id for statement in fn.body for n in ast.walk(statement)
           if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    assert not loads - parameters - stores - {'tl','range','float'}
    loop=fn.body[0]
    bound=loop.iter.args[1].id
    position=next(i for i,v in enumerate(k.parameters) if v.split(':')[0].strip()==bound)
    assert k.launch[position].text == 'B * H'
