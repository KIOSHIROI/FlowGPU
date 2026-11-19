import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from flowgpu.compiler.backend.kernel_abi import prune_unused_parameters
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.cache import cached_module
from main import main

SOURCE=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()


@pytest.mark.parametrize('strategy',['materialize','recompute'])
def test_opt_in_preserves_bodies_contracts_and_launch_bindings(strategy):
    old=compile_source(SOURCE,cross_domain_strategy=strategy).pipelines[0].triton
    new=compile_source(SOURCE,cross_domain_strategy=strategy,prune_kernel_abi=True).pipelines[0].triton
    assert not old.kernel_abi_audit
    assert any(a['removed'] for a in new.kernel_abi_audit)
    for before,after,audit in zip(old.kernels,new.kernels,new.kernel_abi_audit):
        a,b=ast.parse(before.source).body[0],ast.parse(after.source).body[0]
        assert [ast.dump(n) for n in a.body]==[ast.dump(n) for n in b.body]
        assert before.operands==after.operands and before.results==after.results
        assert before.grid==after.grid and before.launch_options==after.launch_options
        assert after.launch==tuple(before.launch[i] for i in audit['retained_indices'])
        assert after.parameters==tuple(before.parameters[i] for i in audit['retained_indices'])
    # All guards, allocation statements, grids and keyword options stay equal.
    hosts=[ast.parse(p.host) for p in (old,new)]
    for host in hosts:
        for call in ast.walk(host):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Subscript):call.args=[]
    assert ast.dump(hosts[0])==ast.dump(hosts[1])


def test_bad_signature_metadata_is_rejected_before_launch():
    kernel=compile_source(SOURCE,cross_domain_strategy='recompute').pipelines[0].triton.kernels[0]
    with pytest.raises(ValueError,match='contract mismatch'):
        prune_unused_parameters(replace(kernel,launch=()))
    with pytest.raises(ValueError,match='differ from parameter metadata'):
        prune_unused_parameters(replace(kernel,parameters=('other',)+kernel.parameters[1:]))


@pytest.mark.parametrize('options,reason',[
    ({'autotune':True},'autotune owns'),
    ({'gpu_backend':'hybrid'},'host/library ABI'),
])
def test_owned_autotune_and_library_abis_are_explicitly_skipped(options,reason):
    source='pipeline p { input a: fp16[M,K] input b: fp16[K,N] y=matmul(a,b) output y }'
    old=compile_source(source,**options).pipelines[0].triton
    new=compile_source(source,**options,prune_kernel_abi=True).pipelines[0].triton
    assert old.kernels==new.kernels
    assert reason in new.kernel_abi_audit[0]['skipped']


def test_cli_cached_and_uncached_use_the_same_option_and_separate_cache(tmp_path,capsys):
    source=tmp_path/'program.fgpu';source.write_text(SOURCE)
    flags=[str(source),'--cross-domain-strategy','recompute','--emit','triton']
    assert main(flags)==0;old=capsys.readouterr().out
    assert main(flags+['--prune-kernel-abi'])==0;new=capsys.readouterr().out
    assert old!=new
    cache=tmp_path/'cache'
    assert main(flags+['--cache-dir',str(cache)])==0
    assert capsys.readouterr().out.rstrip('\n')==old.rstrip('\n')
    assert main(flags+['--cache-dir',str(cache),'--prune-kernel-abi'])==0
    assert capsys.readouterr().out.rstrip('\n')==new.rstrip('\n')
    assert len(list(cache.glob('*.json')))==2
    assert main([str(source),'--prune-kernel-abi','--emit','json'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['pipelines'][0]['triton']['kernel_abi_audit']


def test_non_boolean_is_not_silently_enabled():
    with pytest.raises(ValueError,match='prune_kernel_abi must be a bool'):
        compile_source(SOURCE,prune_kernel_abi=1)
