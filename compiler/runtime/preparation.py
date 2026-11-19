"""Explicit offline tuning -> fixed-config generated deployment source.

Only CUDA, tensor-only, one-pipeline programs are prepared here. Preparation
must precede Graph capture and requires an independent validator. The cache
stores code/configurations, never outputs or input contents. Ordinary compile
defaults remain unchanged; freezing is not a claim that tuning beat the default.
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from compiler.backend.specialization import freeze_autotune
from compiler.runtime.cache import cached_module, compiler_fingerprint
from compiler.runtime.runtime import _load_generated_module
from compiler.optimization.preparation_cost import PhaseLedger


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path, record):
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _graph_microseconds(run, args, iters):
    """Device time of one call, from a capture when one is possible.

    A graph replay excludes host dispatch, which makes the comparison between
    arms stable; a straight loop includes allocation and is the fallback because
    an operator that cannot be captured is still worth selecting for.
    """
    import torch
    for _ in range(3):
        run(*args)
    torch.cuda.synchronize()
    try:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            run(*args)
        for _ in range(3):
            captured.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            captured.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters
    except Exception:
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            run(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters


def _measure_arms(source, *, pipeline, args, modules_dir, arms, iters, base_options):
    """Time every arm on the caller's own tensors and return the measurements.

    Each arm is compiled with the caller's remaining options merged in, so what is
    measured is the program that would ship, not a default-options stand-in.
    """
    measurements, best = [], None
    for label, arm_options, geometry in arms:
        try:
            merged = dict(base_options, **arm_options)
            code, _, key = cached_module(source, cache_dir=modules_dir, **merged)
            module = _load_generated_module(code, cache_dir=modules_dir, key=key)
            run = getattr(module, 'run_' + pipeline)
            us = _graph_microseconds(run, args, iters)
        except Exception as error:
            measurements.append(dict(arm=label, geometry=list(geometry) if geometry else None,
                                     error=type(error).__name__ + ': ' + str(error)[:200]))
            continue
        entry = dict(arm=label, geometry=list(geometry) if geometry else None, graph_us=us)
        measurements.append(entry)
        if best is None or us < best['graph_us']:
            best = entry
    return measurements, best


def _select_arm(source, *, pipeline, args, modules_dir, deployments, iters, context,
                base_options):
    """Pick an arm by measurement, caching the decision so a hit costs nothing.

    The decision key covers the compiler fingerprint, the source and the tensor
    metadata including address alignment, which is the same identity the prepared
    record uses. A hit therefore means the tensors are the ones this decision was
    measured on.
    """
    from compiler.runtime.tile_selection import (dimension_values, options_for_arm,
                                                 space_for_extents, tile_dimensions)
    # Which axis is reduced and how wide the other one is are properties of the
    # emitted program, not of argument order: read them out of the source the
    # compiler writes. 'auto' is safe on every target -- it resolves to
    # materialize where it cannot fuse -- so it is a usable probe.
    probe = dict(base_options, cross_domain_strategy='auto')
    code, _, _ = cached_module(source, cache_dir=modules_dir, **probe)
    try:
        row_dimension, wide_dimension = tile_dimensions(code)
        values = dimension_values(code, pipeline, args)
    except ValueError as error:
        # Nothing to choose between: keep the program as it is.
        return dict(base_options), dict(arm='unchanged', graph_us=None, geometry=None,
                                       cached=False, reason=str(error))
    arms = space_for_extents(values[row_dimension], values[wide_dimension])
    identity = dict(format=1, compiler_fingerprint=compiler_fingerprint(),
                    source_sha256=_digest(source), context=context, pipeline=pipeline,
                    row_dimension=row_dimension, wide_dimension=wide_dimension,
                    extents=[values[row_dimension], values[wide_dimension]],
                    space=[label for label, _, _ in arms])
    key = _digest(json.dumps(identity, sort_keys=True))
    path = deployments / ('selection-' + key + '.json')
    if path.exists():
        try:
            entry = json.loads(path.read_text())
            if entry['identity'] == identity:
                options = options_for_arm(entry['arm'],
                                         extents=(values[row_dimension], values[wide_dimension]))
                return options, dict(entry, cached=True)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    measurements, best = _measure_arms(source, pipeline=pipeline, args=args,
                                       modules_dir=modules_dir, arms=arms, iters=iters,
                                       base_options=base_options)
    if best is None:
        raise ValueError('no arm could be measured for this program')
    entry = dict(identity=identity, arm=best['arm'], graph_us=best['graph_us'],
                 geometry=best['geometry'], measurements=measurements, cached=False,
                 row_dimension=row_dimension, wide_dimension=wide_dimension)
    _write(path, entry)
    return options_for_arm(best['arm'],
                           extents=(values[row_dimension], values[wide_dimension])), entry


def prepare_pipeline(source, *, pipeline, args, cache_dir, validator,
                     select_tiles=False, selection_iters=10, **options):
    """Return (guarded generated function, provenance/preparation record).

    New tensors with identical metadata are accepted. Changed metadata/device
    requires explicit preparation; an unchanged cache hit performs no autotune.
    A cache record is reconstructed from the current emitted source before use.

    ``select_tiles`` measures the candidate arms of
    compiler/runtime/tile_selection.py on the caller's own tensors and prepares
    the fastest, instead of preparing whatever the default caps resolve to. It
    refuses a caller-supplied strategy or platform config, because the selection
    owns both. The decision is cached under the same identity as the prepared
    record, so a hit re-measures nothing.
    """
    # Checked before anything that needs a device: the option contract is not a
    # device question, so a misuse fails on its own terms.
    if select_tiles:
        if 'platform_config' in options or 'cross_domain_strategy' in options:
            raise ValueError('tile selection owns the cross-domain strategy and platform config')
        if type(selection_iters) is not int or selection_iters <= 0:
            raise ValueError('selection_iters must be a positive integer')
    import torch
    start = time.perf_counter()
    ledger = PhaseLedger()
    if not callable(validator):
        raise ValueError('an independent validator is required')
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('prepare outside CUDA Graph capture')
    if not args or any(not isinstance(a, torch.Tensor) or a.device.type != 'cuda' for a in args):
        raise ValueError('preparation requires CUDA tensor inputs')
    if any(a.layout != torch.strided or any(size<=0 for size in a.shape) for a in args):
        raise ValueError('preparation requires positive strided inputs')
    device = args[0].device
    if any(a.device != device for a in args) or device.index != torch.cuda.current_device():
        raise ValueError('prepare inside the single input CUDA device context')
    context = [dict(shape=list(a.shape), stride=list(a.stride()), dtype=str(a.dtype),
                    device=str(a.device), alignment_mod128=a.data_ptr() % 128) for a in args]
    root = Path(cache_dir)
    deployments = root / 'prepared'
    deployments.mkdir(parents=True, exist_ok=True)
    selection = None
    if select_tiles:
        options, selection = _select_arm(
            source, pipeline=pipeline, args=args,
            modules_dir=root/'modules', deployments=deployments, iters=selection_iters,
            context=context, base_options=options)
    code, _, module_key = ledger.call('source_cache_compile',cached_module,source,cache_dir=root/'modules',**options)
    original_sha = _digest(code)
    if selection is not None and selection.get('geometry'):
        # The chosen geometry must be the one in the source that is about to be
        # frozen. A dropped option produces a working kernel at some other tile,
        # and the only symptom would be a prepared module that quietly misses the
        # measurement it was chosen by.
        from compiler.runtime.tile_selection import assert_emitted_tiles, dimension_values
        assert_emitted_tiles(code, dimension_values(code, pipeline, args),
                             tuple(selection['geometry']))
    # The emitted table and source, not a global collection of old tuner keys,
    # are the authority for this preparation.
    import ast
    tree = ast.parse(code)
    hosts = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('run_')]
    if len(hosts) != 1 or hosts[0].name != 'run_' + pipeline or len(hosts[0].args.args) != len(args):
        raise ValueError('preparation requires exactly one tensor-only named pipeline')
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
             and any(isinstance(d, ast.Call) and ast.unparse(d.func) == 'triton.autotune'
                     for d in n.decorator_list)]
    load_key = module_key
    if not any(isinstance(n,ast.FunctionDef) and n.name=='_flowgpu_device_key' for n in tree.body):
        from compiler.backend.triton_backend import device_key_source
        code += device_key_source()
        load_key = _digest(code)
    loaded = ledger.call('module_load',_load_generated_module,code,cache_dir=root/'modules',key=load_key)
    device_key = ledger.call('device_identity',loaded._flowgpu_device_key,device.index)
    identity = dict(format=1, compiler_fingerprint=compiler_fingerprint(), module_key=module_key,
                    module_sha256=_digest(code), original_module_sha256=original_sha,
                    pipeline=pipeline, context=context, device_key=device_key)
    key = _digest(json.dumps(identity, sort_keys=True))
    path = deployments / (key + '.json')
    record = None
    if path.exists():
        try:
            entry = json.loads(path.read_text())
            if entry['identity'] == identity:
                rebuilt = ledger.call('cached_host_reconstruction',freeze_autotune,code,entry['selections'],
                                      pipeline=pipeline,context=context,device_key=device_key)
                if entry['code'] == rebuilt and entry['sha256'] == _digest(rebuilt):
                    record = entry
        except (OSError, ValueError, TypeError, KeyError):
            pass
    hit = record is not None
    if not hit:
        run = getattr(loaded, 'run_' + pipeline)
        def tuned_check():
            result=run(*args)
            torch.cuda.synchronize(device)
            if validator(result) is not True:raise ValueError('independent tuned correctness check failed')
        ledger.call('tuned_execution_validation',tuned_check)
        selections = {}
        for name in names:
            tuner = getattr(loaded, name)
            if getattr(tuner, 'best_config', None) is None:
                raise ValueError('autotuned kernel was not executed: ' + name)
            selections[name] = tuner.best_config.all_kwargs()
        fixed = ledger.call('host_specialization',freeze_autotune,code,selections,pipeline=pipeline,context=context,device_key=device_key)
        record = dict(identity=identity, selections=selections, code=fixed, sha256=_digest(fixed))
        if selection is not None:
            record['tile_selection'] = dict(arm=selection['arm'], graph_us=selection['graph_us'],
                                            geometry=selection['geometry'], cached=selection['cached'])
    prepared = ledger.call('prepared_module_load',_load_generated_module,record['code'],cache_dir=deployments,key=record['sha256'])
    run = getattr(prepared, 'run_' + pipeline)
    def prepared_check():
        result=run(*args)
        torch.cuda.synchronize(device)
        if validator(result) is not True:raise ValueError('independent prepared correctness check failed')
    ledger.call('prepared_execution_validation',prepared_check)
    if compiler_fingerprint() != identity['compiler_fingerprint']:
        raise ValueError('compiler changed during preparation')
    if not hit:
        ledger.call('prepared_record_write',_write,path,record)
    total=(time.perf_counter()-start)*1000
    summary=dict(identity=identity, selections=record['selections'], source_sha256=record['sha256'],
                 source_path=str(Path(prepared.__file__)), policy_path=str(path),
                 cache_hit=hit, autotuned_kernels=len(names), preparation_ms=total,phase_costs=ledger.finish(total))
    if selection is not None:
        # The decision this call made, not the one stored at write time: on a hit
        # the stored copy describes a previous call's measurement.
        summary['tile_selection']=selection
    return run, summary
