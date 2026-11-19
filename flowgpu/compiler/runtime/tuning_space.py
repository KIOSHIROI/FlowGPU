"""Declarative candidate generation for measured deployment selection.

The compiler has several hand-written candidate tables today (cross-domain
tiles, gather mappings, fold autotune).  This module gives the runtime side one
shape: a candidate is a label, compiler options and an optional geometry; a
space is generated from the shared strategy families and deduplicated by the
geometry it actually emits.

It does not measure anything and does not choose a default.  Measurement stays
with ``runtime.preparation`` / ``runtime.tile_selection``, where a device and a
validator are available.
"""
from __future__ import annotations

from dataclasses import dataclass

MATERIALIZE = "materialize"
SPACE = (
    "materialize",
    "recompute",
    "recompute-r4096-d8-budget32768",
    "recompute-r4096-d16-budget65536",
    "recompute-r2048-d16-budget65536",
)


@dataclass(frozen=True)
class TuningCandidate:
    label: str
    options: dict
    geometry: tuple | None = None

    def as_dict(self) -> dict:
        return {"label": self.label, "options": dict(self.options),
                "geometry": list(self.geometry) if self.geometry else None}


def _recompute_table():
    from flowgpu.compiler.optimization.strategy import candidates
    return {spec.name: spec for spec in candidates(recompute=True, large_tiles=True,
                                                   wide_columns=True)}


def _platform_allows(candidate: TuningCandidate, platform_config) -> bool:
    """A non-CUDA target gets the materialize arm until its indexed contract exists."""
    if platform_config is None or candidate.label == MATERIALIZE:
        return True
    return (getattr(platform_config, "name", "") == "cuda"
            or bool(getattr(platform_config, "experimental_region_fusion", False)))


def generate_cross_domain_space(row_extent: int, column_extent: int,
                                names=SPACE, platform_config=None
                                ) -> tuple[TuningCandidate, ...]:
    """Generate the cross-domain candidate space for two actual extents.

    The geometry formulas live in ``tile_selection`` because its tests recover
    the emitted tiles; this function only names families and merges duplicates.
    """
    from flowgpu.compiler.runtime.tile_selection import tile_from_caps
    from flowgpu.compiler.backend.platform_configs import resolve_platform

    table = _recompute_table()
    generated: list[TuningCandidate] = []
    for name in names:
        if name in ("default", MATERIALIZE):
            generated.append(TuningCandidate(
                MATERIALIZE, {"cross_domain_strategy": MATERIALIZE}, None))
        elif name == "recompute":
            generated.append(TuningCandidate(
                "recompute", {"cross_domain_strategy": "recompute"},
                tile_from_caps(None, None, None, None, row_extent, column_extent)))
        elif name in table:
            spec = table[name]
            geometry = tile_from_caps(spec.columns, spec.rows, spec.budget,
                                      spec.warps, row_extent, column_extent)
            generated.append(TuningCandidate(
                name, spec.options(), geometry))
        else:
            raise ValueError("unknown candidate: " + name)

    deduped: list[TuningCandidate] = []
    seen = set()
    for candidate in generated:
        if not _platform_allows(candidate, platform_config):
            continue
        identity = candidate.geometry if candidate.geometry is not None else ("opaque", candidate.label)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(candidate)
    return tuple(deduped)


def options_for_candidate(space: tuple[TuningCandidate, ...], label: str) -> dict:
    for candidate in space:
        if candidate.label == label:
            return dict(candidate.options)
    raise KeyError("no such arm for this program: " + label)


__all__ = [
    "MATERIALIZE", "SPACE", "TuningCandidate",
    "generate_cross_domain_space", "options_for_candidate",
]
