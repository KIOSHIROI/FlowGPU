"""Compatibility re-export: index/permutation contracts moved to core in Phase 1a."""
from flowgpu.compiler.core.index_maps import (  # noqa: F401
    canonical_permutation,
    inverse_coordinates,
    permutation_of,
)

__all__ = ["canonical_permutation", "inverse_coordinates", "permutation_of"]
