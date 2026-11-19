"""Memory dependences between the kernels of a lowered plan.

Why this exists and what it is not
----------------------------------
A *value*-level reaching-definition analysis would be redundant here: the
language binds every name exactly once (Design.md 5), so the dataflow graph's
producer->consumer edges already are the def-use chains. What is missing is the
*other* level: the bytes. Two kernels are ordered by a buffer being written and
then read, and whether two accesses can touch the same bytes decides fusion
legality, buffer reuse and the backend dependence contracts.

Certainty is three-valued, and the middle value is the interesting one:

    must-overlap   the same buffer, whole tensor: a true dependence
                   (a producer's result consumed by a later kernel);
    may-overlap    the same buffer is both read and written by one plan, which
                   only a data-dependent index makes possible;
    no-overlap     distinct buffers: the plan is dependence-free through memory,
                   which is what a fully fused plan looks like.

Deliberately absent: loop-carried dependences *inside* an emitted kernel. The
language has no loops (Design.md 2), so a loop exists only in generated device
code, and this module reads the plan, not the code. That judgement now lives in
`backend/contracts.py::loop_carried_stores`, structurally over the emitted
kernel's loop tree, and it borrows this module's `MUST`/`MAY` constants so the
two granularities cannot drift into two vocabularies for the same idea:
between kernels here, inside one kernel there.

What must not be read into either of them: **addressing is not a dependence.**
Task154 v3/v4 failed 昆仑芯 with no loop, no accumulation and identical load and
store masks -- a permuted *store address*. No dependence analysis can see it,
because nothing there is a dependence; `analysis/access_shape` classifies it
(`PERMUTED`) from the IR. `RESULTS.md` records the correction, and
`test_backend_contracts.py` pins the silence so a clean loop report is never
mistaken for a statement about addressing.
"""
from dataclasses import dataclass

from compiler.analysis.access_shape import plan_shape

MUST = "must-overlap"
MAY = "may-overlap"

READ = "read"
WRITE = "write"


@dataclass(frozen=True)
class AccessSummary:
    """One kernel's access to one buffer."""

    kernel: str
    buffer: str
    role: str
    kind: str
    certainty: str = MUST

    def __post_init__(self):
        if self.role not in (READ, WRITE):
            raise ValueError(f"unknown access role: {self.role}")
        if self.certainty not in (MUST, MAY):
            raise ValueError(f"unknown certainty: {self.certainty}")


@dataclass(frozen=True)
class Dependence:
    """A memory dependence between two kernels of a plan, in plan order."""

    earlier: str
    later: str
    buffer: str
    kind: str          # RAW | WAR | WAW
    certainty: str

    def __str__(self):
        return f"{self.earlier} -> {self.later} {self.kind} on {self.buffer} ({self.certainty})"


def accesses(kernels):
    """Every read and write each kernel performs, with its access shape."""
    summaries = []
    for kernel in kernels:
        kind = plan_shape([kernel]).kind
        for value in getattr(kernel, "operands", ()) or ():
            summaries.append(AccessSummary(kernel.name, value, READ, kind))
        for value in getattr(kernel, "results", ()) or ():
            summaries.append(AccessSummary(kernel.name, value, WRITE, kind))
    return summaries


def conflicts(kernels):
    """The dependences a linear plan carries through memory.

    A plan is a sequence, so three orders are possible and all three are real:
    read-after-write through a materialised result (RAW), write-after-read when a
    kernel's output reuses a buffer an earlier kernel read (WAR), and
    write-after-write when two kernels write the same buffer (WAW). Only the
    first is a true dependence; the other two are hazards the slot allocator has
    to keep apart.
    """
    order = [kernel.name for kernel in kernels]
    written = {}
    read_by = {}
    found = []
    for index, kernel in enumerate(kernels):
        for value in getattr(kernel, "operands", ()) or ():
            if value in written and written[value] != kernel.name:
                found.append(Dependence(written[value], kernel.name, value, "RAW", MUST))
            read_by.setdefault(value, set()).add(kernel.name)
        for value in getattr(kernel, "results", ()) or ():
            for reader in sorted(read_by.get(value, ())):
                if reader != kernel.name:
                    found.append(Dependence(reader, kernel.name, value, "WAR", MAY))
            if value in written and written[value] != kernel.name:
                found.append(Dependence(written[value], kernel.name, value, "WAW", MAY))
            written[value] = kernel.name
    return found


def report(kernels):
    """A serialisable form for diagnostics and for the strategy layer."""
    return {
        "kernels": [kernel.name for kernel in kernels],
        "accesses": [
            {"kernel": item.kernel, "buffer": item.buffer, "role": item.role,
             "kind": item.kind, "certainty": item.certainty}
            for item in accesses(kernels)
        ],
        "conflicts": [
            {"earlier": item.earlier, "later": item.later, "buffer": item.buffer,
             "kind": item.kind, "certainty": item.certainty}
            for item in conflicts(kernels)
        ],
    }
