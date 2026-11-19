"""Load a symbolic generated pipeline once; guards live in its generated host.

The caller owns this trusted local source cache. This API does not bypass
shape/dtype/device/layout guards, and does not cache output tensors.
"""
import importlib.util
import sys
from pathlib import Path
from flowgpu.compiler.runtime.cache import cached_module


def prepare_pipeline(source, **kwargs):
    """Offline autotune and emit fixed-config guarded launches; CUDA required."""
    from flowgpu.compiler.runtime.preparation import prepare_pipeline as prepare
    return prepare(source, **kwargs)


def prepare_execution_plan(source, **kwargs):
    """Prepare guarded explicit-buffer execution; callers own buffer lifetime."""
    from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan as prepare
    return prepare(source, **kwargs)


def tune_strategy_pipeline(source, **kwargs):
    """Explicit exact-context reduction policy; default compilation stays fixed."""
    from flowgpu.compiler.optimization.strategy_tuning import tune_strategy_pipeline as tune
    return tune(source, **kwargs)


def compile_pipeline(source, *, pipeline, cache_dir, **options):
    code, hit, key = cached_module(source, cache_dir=cache_dir, **options)
    loaded = _load_generated_module(code, cache_dir=cache_dir, key=key)
    run = getattr(loaded, 'run_' + pipeline)
    return run


def _load_generated_module(code, *, cache_dir, key):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    name = '_flowgpu_' + key
    loaded = sys.modules.get(name)
    if loaded is None:
        # Keep a real source file for Triton JIT's inspect.getsource.
        path = Path(cache_dir) / (key + '.py')
        if not path.exists() or path.read_text() != code:
            from tempfile import NamedTemporaryFile
            import os
            with NamedTemporaryFile(mode='w', dir=cache_dir, suffix='.tmp', delete=False) as stream:
                stream.write(code)
                tmp = stream.name
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        try:
            spec.loader.exec_module(loaded)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return loaded


def bind_gather_launches(run, compiled, bindings, args):
    """Profiling-only fixed buffers from one guarded generated host invocation.

    Reads the emitted launch plan instead of rewriting kernels. The returned
    zero-argument call retains tensors; it does not accept unguarded new inputs.
    Index contents may be updated in place. Allocations/host guards are excluded,
    but direct Python/JIT dispatch remains part of this timing surface.
    """
    from flowgpu.compiler.optimization.strategy import launch_value
    from flowgpu.compiler.backend.triton_backend import module
    import inspect
    if any(n.op not in {'input','output','gather_rows'} for n in compiled.dataflow.nodes):
        raise ValueError('fixed binding currently requires only gather_rows')
    if Path(inspect.getsourcefile(run)).read_text() != module([compiled.triton]):
        raise ValueError('loaded module differs from emitted launch plan')
    outputs=run(*args)
    values={n.output:a for n,a in zip((n for n in compiled.dataflow.nodes if n.op=='input'),args)}
    values.update(outputs)
    launches=[]
    for kernel in compiled.triton.kernels:
        arguments=[]
        for arg in kernel.launch:
            if arg.kind=='tensor':arguments.append(values[arg.text])
            elif arg.kind=='value':arguments.append(launch_value(arg.text,bindings))
            elif arg.kind=='contiguous':arguments.append(values[arg.text].is_contiguous())
            elif arg.kind in {'shape','stride'}:
                value,axis=arg.text.rsplit(':',1)
                arguments.append((values[value].shape if arg.kind=='shape' else values[value].stride())[int(axis)])
            else:raise ValueError('unsupported fixed launch argument: '+arg.kind)
        launches.append((run.__globals__[kernel.name],launch_value(kernel.grid,bindings),arguments,dict(kernel.launch_options)))
    def call():
        for fn,grid,arguments,options in launches:fn[grid](*arguments,**options)
        return outputs
    return call,outputs
