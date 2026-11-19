"""Backward-compatible cache facade.

The implementation lives in ``flowgpu.compiler.runtime.cache``.  The wrapper passes the
module-level ``compiler_fingerprint`` value explicitly instead of temporarily
rewriting the implementation's global: the old form made two concurrent callers
see each other's fingerprint (review P2, 2026-10-26).
"""
from flowgpu.compiler.runtime import cache as _impl

compiler_fingerprint = _impl.compiler_fingerprint
cache_key = _impl.cache_key


def cached_module(source, *, cache_dir=None, **options):
    return _impl.cached_module(source, cache_dir=cache_dir,
                               fingerprint=compiler_fingerprint, **options)
