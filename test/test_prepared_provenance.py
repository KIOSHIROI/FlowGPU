"""Exercise real loading/specialization with CPU-only tensor metadata doubles."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

from compiler.runtime.preparation import prepare_pipeline


def test_cross_directory_module_reuse_reports_existing_actual_source(monkeypatch,tmp_path):
    import compiler.runtime.preparation as preparation
    class Device:
        type='cuda';index=0
        def __str__(self):return 'cuda:0'
    class Tensor:
        shape=(1,);layout='strided';dtype='torch.float32';device=Device()
        def __init__(self,value):self.value=value
        def stride(self):return (1,)
        def data_ptr(self):return 128
    torch=SimpleNamespace(Tensor=Tensor,strided='strided',float32='torch.float32',
        cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,current_device=lambda:0,synchronize=lambda device:None))
    monkeypatch.setitem(sys.modules,'torch',torch)
    code='''import torch
def _flowgpu_device_key(index): return 'cpu-metadata-proof'
def run_p(x): return {'out': x}
'''
    key=hashlib.sha256(code.encode()).hexdigest()
    monkeypatch.setattr(preparation,'cached_module',lambda *a,**kw:(code,False,key))
    before=set(sys.modules)
    try:
        first=Tensor(1)
        run_a,a=prepare_pipeline('unused',pipeline='p',args=(first,),cache_dir=tmp_path/'a',
            validator=lambda got:got['out'] is first)
        second=Tensor(2)
        run_b,b=prepare_pipeline('unused',pipeline='p',args=(second,),cache_dir=tmp_path/'b',
            validator=lambda got:got['out'] is second)
        assert run_a is run_b  # Real loader reused the content-addressed module.
        assert run_b(second)['out'].value==2
        assert a['source_sha256']==b['source_sha256']
        assert Path(b['source_path']).is_file()
        assert b['source_path']==run_b.__globals__['__file__']==a['source_path']
        assert hashlib.sha256(Path(b['source_path']).read_bytes()).hexdigest()==b['source_sha256']
    finally:
        for name in set(sys.modules)-before:
            if name.startswith('_flowgpu_'):sys.modules.pop(name,None)
