"""21.7 数值稳定性, as a check rather than a definition.

Design.md's criterion is a pattern match on the graph:

    A pipeline that spells out, from basic operators, a primitive the language
    already owns -- exp and sum for softmax, mul / sum / rsqrt for rms_norm --
    is a **stability defect candidate**. Not because it is wrong, but because
    it gives up the compiler's guarantee of a stable implementation. The
    primitives belong to the compiler, so subtracting the maximum, carrying the
    eps and reducing in a tree are the compiler's properties; a hand-unrolled
    equivalent makes them the agent's properties, and 21.5 says the agent
    cannot measure those.

v0.3 can express exp -> sum_last -> broadcast_like -> div. Such paths now
produce candidates on real programs; a candidate calls for inspection, not
proof of instability (the explicit max subtraction may already be present).
Legacy `sum` remains unavailable.

2026-10-20 (v0.5's value layer) added `sqrt`, `rsqrt`, `tanh`, `sigmoid` and the
rest of the elementary functions, so **the rms_norm expansion is now writable**
(`mul -> sum_last -> rsqrt`) and the check fires on it with no mock registration.
The softmax expansion still is not: it needs `sum`, which the language does not
have under that name (`sum_last` is the closest, and its pattern is present).
So the boundary is exactly one name now, and `test_stability.py` pins it.

A witness has to be *connected*: a search step may not reuse a node, but that
guard was removed rather than kept: both patterns have distinct steps, so the
sets of nodes per step are disjoint and one node can never fill two steps. A
future pattern that repeats an operator would need that guard back, which is
what the mutation that deleted it proved -- it changed no behaviour at all.

A witness has to be *connected*: three unrelated uses of `mul`, `sum` and
`rsqrt` are not a hand-written rms_norm, so each step must consume what the
previous step produced. What the check cannot decide is stated in 21.7 and not
attempted here: condition numbers, cancellation, overflow.
"""

from dataclasses import dataclass

from flowgpu.compiler.core.operators import OPERATORS


@dataclass(frozen=True)
class Pattern:
    """A primitive, and the chain of basic operators that would expand it."""

    primitive: str
    steps: tuple

    @property
    def expansion(self) -> str:
        return " -> ".join(self.steps)


# 21.7's own two examples and nothing else. A pattern belongs here when the
# language both owns the primitive and has every operator needed to spell out
# an equivalent by hand, so this table grows exactly when 19 grows.
#
# 2026-10-20: `rsqrt`/`sqrt` entered the language with v0.5's value layer, so
# the rms_norm expansion 21.7 names ("mul / sum / rsqrt") is now writable with
# the `sum_last` the language actually has. The pattern is added rather than the
# docstring left quietly claiming the expansion is inexpressible: a check whose
# stated value is "it holds today only because the language cannot express the
# expansions" has to start firing the moment that stops being true.
PATTERNS = (
    Pattern("rms_norm", ("mul", "sum", "rsqrt")),
    Pattern("rms_norm", ("mul", "sum_last", "rsqrt")),
    Pattern("softmax", ("exp", "sum", "div")),
    Pattern("softmax", ("exp", "sum_last", "div")),
)


@dataclass(frozen=True)
class Candidate:
    """A pipeline that looks like it expanded a primitive by hand."""

    primitive: str
    witness: tuple
    values: tuple

    def message(self) -> str:
        return (
            f"{self.primitive} looks hand-unrolled: "
            f"{' -> '.join(self.witness)} over {', '.join(self.values)}. "
            f"Use {self.primitive}: its numerical behaviour is then the "
            f"compiler's, and 21.5 says an agent cannot measure what it "
            f"gives up"
        )


def _consumes(graph, node, value) -> bool:
    """Whether `value` reaches one of `node`'s operands."""
    seen = set()
    stack = list(node.inputs)

    while stack:
        current = stack.pop()

        if current == value:
            return True

        if current in seen:
            continue

        seen.add(current)
        producer = graph.producer_of(current)

        if producer is not None:
            stack.extend(producer.inputs)

    return False


def _chains(graph, pattern) -> tuple:
    """Every way this pattern's operators occur in order and connected."""
    by_operator = {}

    for node in graph.nodes:
        by_operator.setdefault(node.op, []).append(node)

    if any(step not in by_operator for step in pattern.steps):
        return ()

    def extend(index, chosen):
        if index == len(pattern.steps):
            return [tuple(chosen)]

        found = []

        for node in by_operator[pattern.steps[index]]:
            if chosen and not _consumes(graph, node, chosen[-1].output):
                continue

            found.extend(extend(index + 1, chosen + [node]))

        return found

    return tuple(extend(0, []))


def check(graph) -> tuple:
    """21.7's match. An empty result is the expected answer today."""
    candidates = []

    for pattern in PATTERNS:
        for chain in _chains(graph, pattern):
            candidates.append(
                Candidate(
                    primitive=pattern.primitive,
                    witness=tuple(node.op for node in chain),
                    values=tuple(node.output for node in chain),
                )
            )

    return tuple(candidates)


def boundary(registered=None) -> tuple:
    """The operators that would start making this check fire.

    This is the check's whole present value: it holds today only because the
    language cannot express the expansions, so what is worth stating is *what
    would change that* rather than a report that passes. The returned names are
    sorted, so that a language which gains one operator changes this by exactly
    one name.
    """
    names = OPERATORS if registered is None else registered

    return tuple(
        sorted(
            {
                step
                for pattern in PATTERNS
                for step in pattern.steps
                if step not in names
            }
        )
    )
