"""Cross-backend operator migration and evidence.

The compiler (``flowgpu/compiler/``) is the engine.  This package provides the migration tooling: a model's inference
operator inventory, its coverage on a set of backend profiles, and a validated
evidence bundle that a reviewer can reproduce.

Nothing here claims a device result.  Coverage is derived from the compiler's
platform tables; correctness/performance records are accepted only when they
carry a reference, an artifact hash and a reproduction command.
"""
__version__ = "0.25.0"
