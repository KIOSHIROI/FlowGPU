"""Run actual emitted host guards/bindings with CPU metadata tensors only.

Launch stubs check host wiring; they are not a device numerical oracle.
"""
import ast
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from compiler.compile import compile_source
from compiler.backend.triton_backend import module,device_key_source
from compiler.backend.specialization import freeze_autotune
from compiler.runtime.execution_plan import build_execution_plan,PreparedExecutionPlan


SOURCE='pipeline p { input x : fp32[N] y=relu(x) output y }'
CONTEXT=[dict(shape=[4],stride=[2],dtype='torch.float32',device='cuda:0',alignment_mod128=0)]


def prepared(source=SOURCE,pipeline='p',context=CONTEXT,**options):
    code=module([compile_source(source,**options).pipelines[0].triton])
    if 'def _flowgpu_device_key(' not in code:code+=device_key_source()
    return freeze_autotune(code,{},pipeline=pipeline,context=context,device_key='proof')


class Tensor:
    def __init__(self,shape=(4,),stride=(1,),ptr=4096,dtype='torch.float32',values=None):
        self.shape=shape;self._stride=stride;self.ptr=ptr;self.dtype=dtype
        self.device=Device()
        self.layout='strided';self.values=list(values or [0]*4)
    def stride(self,*axis):return self._stride if not axis else self._stride[axis[0]]
    def data_ptr(self):return self.ptr
    def element_size(self):return 4
    def is_contiguous(self):
        return self._stride==(1,)


class Device:
    index=0
    type='cuda'
    def __str__(self):return 'cuda:0'
    def __eq__(self,other):return str(self)==str(other)


def loaded(monkeypatch,descriptor):
    def no_allocate(*a,**kw):raise AssertionError('hot tensor allocation')
    torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='torch.float32',empty=no_allocate)
    monkeypatch.setitem(sys.modules,'torch',torch)
    calls=[]
    class Kernel:
        def __getitem__(self,grid):
            def run(x,out,*args,**kw):
                calls.append((grid,x,out))
                out.values[:]=[max(0,v) for v in x.values]
            return run
    host=next(n for n in ast.parse(descriptor.code).body if isinstance(n,ast.FunctionDef) and n.name=='run_into_p')
    import compiler.runtime.execution_plan as api
    namespace={'torch':torch,'p_k0':Kernel(),'_flowgpu_device_key':lambda index:'proof'}
    for n in ast.parse(descriptor.code).body:
        if isinstance(n,ast.ImportFrom) and n.module==api.__name__:
            for alias in n.names:namespace[alias.asname or alias.name]=getattr(api,alias.name)
    for n in ast.parse(descriptor.code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:namespace[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    exec(compile(ast.fix_missing_locations(ast.Module(body=[host],type_ignores=[])),'<generated-host>','exec'),namespace)
    return namespace['run_into_p'],calls


def test_new_inputs_and_distinct_buffers_preserve_old_output_without_hot_allocation(monkeypatch):
    code=prepared();plan=build_execution_plan(code,pipeline='p')
    run,calls=loaded(monkeypatch,plan)
    assert len(plan.buffers)==1 and plan.buffers[0].nbytes==16
    key=plan.buffers[0].name
    a=Tensor(stride=(2,),values=[-1,2,-3,4]);b=Tensor(ptr=8192)
    old=run(a,{key:b})['y']
    second=Tensor(ptr=12288)
    new=run(Tensor(stride=(2,),ptr=16384,values=[5,-6,7,-8]),{key:second})['y']
    assert old is b and old.values==[0,2,0,4]
    assert new is second and new.values==[5,0,7,0]
    assert len(calls)==2 and calls[0][0]==(1,)
    kernels=lambda s:{n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='p_k0'}
    assert kernels(code)==kernels(plan.code)


@pytest.mark.parametrize('bad',['stride','shape','dtype','alignment','output_shape','output_alias','output_stride','output_dtype'])
def test_bad_inputs_or_buffers_rejected_before_launch(monkeypatch,bad):
    plan=build_execution_plan(prepared(),pipeline='p');run,calls=loaded(monkeypatch,plan)
    x=Tensor(stride=(2,));out=Tensor(ptr=8192)
    if bad=='stride':x._stride=(1,)
    elif bad=='shape':x.shape=(5,)
    elif bad=='dtype':x.dtype='torch.int32'
    elif bad=='alignment':x.ptr+=4
    elif bad=='output_shape':out.shape=(3,)
    elif bad=='output_alias':out.ptr=x.ptr
    elif bad=='output_stride':out._stride=(2,)
    else:out.dtype='torch.int32'
    with pytest.raises(ValueError, match='(?:prepared input context changed; prepare again|execution plan buffer metadata/alignment mismatch: slot0|execution plan writable buffers must not overlap inputs or other buffers)'):run(x,{plan.buffers[0].name:out})
    assert not calls


def test_workspace_pairs_and_unknown_keys_are_rejected(monkeypatch):
    import compiler.runtime.execution_plan as api
    plan=build_execution_plan(prepared(),pipeline='p');loaded(monkeypatch,plan)
    x=Tensor(stride=(2,));a=Tensor(ptr=8192);b=Tensor(ptr=8192)
    with pytest.raises(ValueError,match='overlap'):
        api._check_buffers({'a':a,'b':b},(('a',(4,),'torch.float32'),('b',(4,),'torch.float32')),(x,))
    with pytest.raises(ValueError,match='exactly'):
        api._check_buffers({'unused':a},(),(x,))


def test_lse_materialize_and_recompute_plans_are_derived_from_real_slots():
    source=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    context=[dict(shape=[7,3,2,33],stride=[210,70,35,1],dtype='torch.float16',device='cuda:0',alignment_mod128=0),
             dict(shape=[7,3,2],stride=[12,4,1],dtype='torch.float32',device='cuda:0',alignment_mod128=0)]
    for mode in ('materialize','recompute'):
        fixed=prepared(source,'lse',context,cross_domain_strategy=mode)
        plan=build_execution_plan(fixed,pipeline='lse')
        assert len(plan.launches)==(2 if mode=='materialize' else 1)
        assert set(dict(plan.outputs))=={'out','out_lse'}
        assert all(b.nbytes>0 for b in plan.buffers)
        host=next(n for n in ast.parse(plan.code).body if isinstance(n,ast.FunctionDef) and n.name=='run_into_lse')
        assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.empty' for n in ast.walk(host))


def test_unprepared_or_library_host_is_not_silently_accepted():
    with pytest.raises(ValueError,match='prepared guards'):
        build_execution_plan(module([compile_source(SOURCE).pipelines[0].triton]),pipeline='p')
    fixed=prepared()
    fixed=fixed.replace('return {', 'torch.zeros(4)\n    return {')
    with pytest.raises(ValueError,match='unsupported host operation'):
        build_execution_plan(fixed,pipeline='p')


def test_span_specialization_checks_padding_and_refreshes_each_pointer(monkeypatch):
    plan=build_execution_plan(prepared(),pipeline='p');run,calls=loaded(monkeypatch,plan)
    x=Tensor(stride=(2,));key=plan.buffers[0].name
    with pytest.raises(ValueError,match='overlap'):run(x,{key:Tensor(ptr=x.ptr)})
    out=Tensor(ptr=8192);run(x,{key:out})
    # A different, still aligned input may overlap the same output on a later
    # call: a previously validated buffer set grants no permanent exemption.
    with pytest.raises(ValueError,match='overlap'):run(Tensor(stride=(2,),ptr=out.ptr),{key:out})
    assert len(calls)==1


def test_padded_span_overlap_beyond_dense_byte_count_is_rejected(monkeypatch):
    context=[dict(CONTEXT[0],shape=[65],stride=[2])]
    plan=build_execution_plan(prepared(context=context),pipeline='p');run,calls=loaded(monkeypatch,plan)
    # 384 > dense size 260 but < padded span 516; both pointers are aligned.
    with pytest.raises(ValueError,match='overlap'):
        run(Tensor(shape=(65,),stride=(2,)),{plan.buffers[0].name:Tensor(shape=(65,),ptr=4480)})
    assert not calls


@pytest.mark.parametrize('change',[
    ("tuple(x.stride()) != (2,)","tuple(x.stride()) == (2,)"),
    ("tuple(x.shape) != (4,)","tuple(x.shape) != (0,)"),
    ("tuple(x.stride()) != (2,)","tuple(x.stride()) != (-2,)"),
])
def test_static_spans_require_exact_guard_proof(change):
    code=prepared();assert change[0] in code
    with pytest.raises(ValueError,match='canonical exact'):
        build_execution_plan(code.replace(*change),pipeline='p')
