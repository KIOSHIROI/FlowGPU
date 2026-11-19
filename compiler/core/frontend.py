"""Frontend protocol and registry: the input side of the de-DSL boundary.

Phase 0 makes the direction explicit: a frontend turns source text into
whatever AST/IR representation it owns, and the compiler core never imports a
concrete frontend.  The DSL is one implementation (``compiler.frontend.dsl``);
a future Triton importer or Python eDSL can implement the same protocol without
touching the DSL parser.

The registry is deliberately small and side-effect free: importing
``compiler.core`` must not instantiate the DSL.  A frontend registers itself
explicitly (``register_frontend``) where the application chooses its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Frontend(Protocol):
    """Anything that can turn source text into a program representation."""

    name: str

    def parse(self, source: str) -> object:
        """Parse ``source``; the return type is owned by the frontend."""
        ...


@dataclass(frozen=True)
class FrontendInfo:
    name: str
    parse: Callable[[str], object]
    description: str = ""
    #: Optional human-facing extension, used by CLIs that accept a frontend.
    source_extension: str = ".fgpu"


_REGISTRY: dict[str, FrontendInfo] = {}


def register_frontend(frontend: Frontend | FrontendInfo) -> FrontendInfo:
    """Register a frontend under its ``name``; returns the stored record."""
    info = frontend if isinstance(frontend, FrontendInfo) else FrontendInfo(
        name=frontend.name, parse=frontend.parse,
        description=getattr(frontend, "description", ""),
        source_extension=getattr(frontend, "source_extension", ".fgpu"))
    if not info.name:
        raise ValueError("a frontend needs a non-empty name")
    existing = _REGISTRY.get(info.name)
    if existing is not None and existing != info:
        raise ValueError(f"frontend {info.name!r} is already registered")
    _REGISTRY[info.name] = info
    return info


def get_frontend(name: str) -> FrontendInfo:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown frontend {name!r}; registered: {known}") from None


def available_frontends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def clear_frontends() -> None:
    """Test helper; applications normally register once at start-up."""
    _REGISTRY.clear()
