from dataclasses import replace

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler.backend import reference
import pytest
from flowgpu.compiler.frontend.ast_node import TensorType
from flowgpu.compiler.frontend.operator import OPERATORS, register_operator
from flowgpu.compiler.frontend.diagnostic import CompileError


SOURCE='pipeline p { input x : fp32[2,3] y=relu(add(x,x)) output y }'


def test_anchorless_pointwise_groups_use_the_same_dispatch_and_platform_gate():
    p=compile_source(SOURCE).pipelines[0]
    assert p.fusion.groups[0].anchor is None
    assert p.triton.complete and p.triton.kernels[0].kind=='pointwise'
    p=compile_source(SOURCE,platform_config=resolve_platform(allowed_kernel_kinds=('rowwise',))).pipelines[0]
    assert not p.triton.complete
    assert p.triton.refusals[0].operator=='pointwise'
    assert p.reference.complete


def test_empty_reference_emitters_are_refused_instead_of_reported_complete(monkeypatch):
    original=reference.EMITTERS['pointwise']
    def empty(*args):return replace(original(*args),source='')
    monkeypatch.setitem(reference.EMITTERS,'pointwise',empty)
    p=compile_source(SOURCE).pipelines[0]
    assert not p.reference.complete
    assert not p.reference.kernels
    assert 'no executable numerical implementation' in p.reference.refusals[0].reason


def test_multi_result_reductions_must_honor_the_same_accumulator_floor(monkeypatch):
    saved=dict(OPERATORS)
    def pair(args,count):
        return [TensorType(args[0].dtype,args[0].shape.copy()) for _ in range(count)]
    try:
        register_operator('pair_reduce',1,pair,pointwise=False,acc='bf16',multi=True)
        with pytest.raises(CompileError,match='narrower than fp32'):
            compile_source('pipeline p { input x : bf16[2] a,b=pair_reduce(x) output a output b }')
    finally:
        OPERATORS.clear();OPERATORS.update(saved)
