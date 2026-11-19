"""Conservative kernel-level dependency analysis.

The plan is descriptive and does not alter launch order. It uses emitted
kernel operands/results and the memory plan to expose a stable DAG. In
particular, slot reuse creates explicit hazard edges so a future overlap
scheduler cannot race a reused allocation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSchedule:
    mode: str
    dependencies: tuple
    levels: tuple
    order: tuple

    @property
    def concurrent(self) -> bool:
        return any(len(level) > 1 for level in self.levels)


def _storage(memory, value):
    if memory.stage_plan is not None and value in {s.name for s in memory.stage_plan.scratch}:
        return 'scratch:'+value
    for buffer in memory.buffers:
        if buffer.value == value:
            return buffer.storage
    return None


def analyze_schedule(kernels, memory, *, mode="sequential") -> KernelSchedule:
    """Return a stable dependency DAG for emitted kernels.

    `sequential` deliberately preserves current lowering behavior by adding a
    chain between adjacent kernels. `overlap` exposes only true data and slot
    hazards; it is analysis-only until a runtime launcher consumes the plan.
    """
    if mode not in {"sequential", "overlap"}:
        raise ValueError("schedule mode must be sequential or overlap")
    kernels = tuple(kernels)
    count = len(kernels)
    edges = [set() for _ in kernels]

    if mode == "sequential":
        for index in range(1, count):
            edges[index].add(index - 1)
    else:
        writers = {}
        readers = {}
        for index, kernel in enumerate(kernels):
            for value in kernel.results:
                writers.setdefault(_storage(memory, value), []).append(index)
            for value in kernel.operands:
                readers.setdefault(_storage(memory, value), []).append(index)

        # Producer/consumer and conservative read/write hazards. None denotes
        # an input or an unmaterialized value and must not create an edge.
        for value, consumers in readers.items():
            if value is None:
                continue
            for producer in writers.get(value, ()):
                for consumer in consumers:
                    if producer != consumer:
                        edges[consumer].add(producer)
        for storage, producers in writers.items():
            if storage is None:
                continue
            for earlier in producers:
                for later in producers:
                    if earlier < later:
                        edges[later].add(earlier)
            consumers = readers.get(storage, ())
            for producer in producers:
                for consumer in consumers:
                    if producer != consumer:
                        edges[consumer].add(producer)

    levels = []
    remaining = set(range(count))
    while remaining:
        ready = tuple(index for index in sorted(remaining)
                      if not (edges[index] & remaining))
        if not ready:
            raise ValueError("kernel dependency graph contains a cycle")
        levels.append(ready)
        remaining.difference_update(ready)

    dependencies = tuple(tuple(sorted(edge)) for edge in edges)
    order = tuple(index for level in levels for index in level)
    return KernelSchedule(mode, dependencies, tuple(levels), order)
