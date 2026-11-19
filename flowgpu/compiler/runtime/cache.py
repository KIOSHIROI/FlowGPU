"""Content-addressed generated-source cache (does not serialize compiler objects).

A cache hit requires source, all compiler source/grammar bytes, options, and
artifact digest to match. Triton's binary/device/tuning cache remains its own
responsibility; generated autotuners request persistent timings explicitly.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from dataclasses import is_dataclass, asdict


from flowgpu.compiler.shared import compiler_fingerprint  # noqa: E402  (re-exported)


def cache_key(source, options, fingerprint=None):
    # `flowgpu.compiler.cache` passes its module-level `compiler_fingerprint` (a
    # function, so a test may monkeypatch it); the implementation accepts
    # either that or an already-computed digest.
    fingerprint = fingerprint() if callable(fingerprint) else fingerprint

    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f'unsupported cache option: {type(value).__name__}')
    payload = json.dumps({'source': source,
                          'compiler': fingerprint if fingerprint is not None
                          else compiler_fingerprint(),
                          'options': options, 'format': 1,
                          # The emitter reads this environment switch at import
                          # time, so two runs that differ only in it must not
                          # share a cache entry (review P2, 2026-10-26).  The raw
                          # value is used: a synonym for "on" only costs a miss.
                          'strict_emission': os.environ.get('FLOWGPU_STRICT_EMISSION', '1'),
                          }, sort_keys=True,
                         default=encode, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


class IncompleteModule(ValueError):
    """The cached lowering refused part of the program.

    A refusal is a compilation result (exit code 1), not a cache/usage failure
    (exit code 2); the CLI maps the two differently and the distinction was
    lost when this raised a bare ValueError (review P2, 2026-10-26).
    """


def cached_module(source, *, cache_dir=None, fingerprint=None, **options):
    """Return (generated source, cache hit, key). Corruption becomes a miss."""
    key = cache_key(source, options, fingerprint)
    root = Path(cache_dir) if cache_dir is not None else None
    target = root / (key + '.json') if root else None
    if target and target.exists():
        try:
            entry = json.loads(target.read_text())
            code = entry['code']
            if (isinstance(code, str) and entry['key'] == key and entry['format'] == 1
                    and hashlib.sha256(code.encode()).hexdigest() == entry['sha256']):
                return code, True, key
        except (ValueError, KeyError, TypeError, OSError):
            pass
    from flowgpu.compiler.compile import compile_source
    from flowgpu.compiler.backend.triton_backend import module
    result = compile_source(source, **options)
    for pipeline in result.pipelines:
        if not pipeline.triton.complete:
            raise IncompleteModule('; '.join(pipeline.triton.messages()))
    code = module([p.triton for p in result.pipelines])
    if target:
        root.mkdir(parents=True, exist_ok=True)
        entry = {'format': 1, 'key': key, 'sha256': hashlib.sha256(code.encode()).hexdigest(), 'code': code}
        descriptor, tmp = tempfile.mkstemp(prefix=key, suffix='.tmp', dir=root)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(entry, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return code, False, key
