"""RoPE must honor the same mapping contract as other flat kernels."""
import ast

import pytest

from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.backend.triton_backend import grid_ceiling, module
from flowgpu.compiler.compile import compile_source


SOURCE = '''pipeline rotary {
    input x : fp32[B,D]
    input rope_cos : fp32[B,D]
    input rope_sin : fp32[B,D]
    a = relu(x)
    r = rope(a,rope_cos,rope_sin)
    y = sub(r,x)
    output y
}'''


def program(**options):
    return compile_source(SOURCE, platform_config=resolve_platform(**options)).pipelines[0].triton


def test_rope_receives_block_and_warp_options():
    p = program(block_size=256, num_warps=8)
    (k,) = p.kernels
    assert k.kind == 'rope'
    assert k.grid == '(triton.cdiv(B * D, 256),)'
    assert dict(k.launch_options)['num_warps'] == 8
    assert k.launch[-1].text == '256'
    assert 'for ' not in k.source
    ast.parse(module([p]))


@pytest.mark.parametrize('options,limit', [
    ({'max_grid_dim': 7}, 7),
    ({'mapping_strategy': 'grid_stride_loop', 'device_cores': 3}, 3),
    ({'mapping_strategy': 'grid_stride_loop', 'device_cores': 9, 'max_grid_dim': 7}, 7),
])
def test_rope_loop_preserves_the_entire_fused_body(options, limit):
    plain = program()
    capped = program(**options)
    (base,), (actual,) = plain.kernels, capped.kernels
    base_body = ast.parse(base.source).body[0].body
    (loop,) = ast.parse(actual.source).body[0].body
    assert isinstance(loop, ast.For)
    assert f'else {limit}' in actual.grid
    assert actual.source.count('tl.program_id(0)') == 1
    # Map the loop tile back to the original program id: all loads, partner
    # indices, dtype boundaries and the final store must then be identical.
    class RestoreProgramId(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == loop.target.id and isinstance(node.ctx, ast.Load):
                return ast.parse('tl.program_id(0)', mode='eval').body
            return node
    restored = RestoreProgramId().visit(ast.Module(body=loop.body, type_ignores=[]))
    assert ast.dump(restored) == ast.dump(ast.Module(body=base_body, type_ignores=[]))
    ast.parse(module([capped]))


@pytest.mark.parametrize('cores,ceiling,expected', [(3,7,3),(9,7,7),(9,None,9)])
def test_loop_mapping_cannot_override_a_hardware_ceiling(cores, ceiling, expected):
    config = resolve_platform(mapping_strategy='grid_stride_loop', device_cores=cores,
                              max_grid_dim=ceiling)
    assert grid_ceiling(config) == expected


def test_rope_default_keeps_the_direct_grid_and_launch_contract():
    (k,) = program().kernels
    assert k.grid == '(triton.cdiv(B * D, BLOCK),)'
    assert k.launch_options == ()
    assert k.launch[-1].text == 'BLOCK'
    assert 'tl.num_programs(0)' not in k.source


def test_rope_loop_name_does_not_shadow_an_input():
    source = SOURCE.replace('input x :', 'input tile :').replace('relu(x)', 'relu(tile)').replace('sub(r,x)', 'sub(r,tile)')
    p = compile_source(source, platform_config=resolve_platform(max_grid_dim=3)).pipelines[0].triton
    (k,) = p.kernels
    loop = ast.parse(k.source).body[0].body[0]
    assert isinstance(loop, ast.For) and loop.target.id != 'tile'
    assert 'tile_ptr' in k.source
