"""Unified semantic backend policy.

This module is intentionally a small facade over the existing dispatch and
platform configuration. It is descriptive today: lowerers may continue to use
their existing arguments while callers can inspect one validated policy.
"""

from dataclasses import dataclass

from compiler.backend.backends import LibraryDispatch, library_dispatch
from compiler.backend.platform_configs import PlatformConfig, resolve_platform


NUMERIC_MODES = ("fp64", "fp32")
CONCURRENCY_MODES = ("sequential", "overlap")


@dataclass(frozen=True)
class BackendPolicy:
    gpu_backend: str
    library_matmul: str
    platform: str
    allowed_kernel_kinds: tuple
    reduction_block_limit: int | None
    dispatch: LibraryDispatch
    numeric_mode: str = "fp64"
    concurrency: str = "sequential"


def make_backend_policy(*, gpu_backend="triton", library_matmul="unfused",
                        platform_config=None, numeric_mode="fp64",
                        concurrency="sequential") -> BackendPolicy:
    """Validate and snapshot the backend decisions used for a compilation."""
    if numeric_mode not in NUMERIC_MODES:
        raise ValueError(f"numeric_mode must be one of {NUMERIC_MODES}")
    if concurrency not in CONCURRENCY_MODES:
        raise ValueError(f"concurrency must be one of {CONCURRENCY_MODES}")
    config = resolve_platform() if platform_config is None else platform_config
    if not isinstance(config, PlatformConfig):
        raise TypeError("platform_config must be a PlatformConfig")
    if concurrency == "overlap" and config.device_type != "cuda":
        raise ValueError("overlap scheduling currently requires a CUDA platform")
    dispatch = library_dispatch(gpu_backend, library_matmul)
    return BackendPolicy(
        gpu_backend=gpu_backend,
        library_matmul=library_matmul,
        platform=config.name,
        allowed_kernel_kinds=config.allowed_kernel_kinds,
        reduction_block_limit=config.reduction_block_limit,
        dispatch=dispatch,
        numeric_mode=numeric_mode,
        concurrency=concurrency,
    )
