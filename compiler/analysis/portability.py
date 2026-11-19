"""21.8 跨后端可移植性, as a measurement.

Design.md's definition: a good operator is one whose pipeline reaches N targets
from the same source, unchanged, and without A getting worse. This module turns
that sentence into numbers and one structural claim.

The numbers are per target: how many kernels it emits, and which operators it
cannot reach at all. The structural claim is why the rest is cheap. 13.6 and
13.7 run *before* any target exists and produce a plan both targets then follow,
so A cannot differ between them -- not "was measured equal" but "is the same
decision, computed once". A target could only degrade A by ignoring that plan,
which is exactly what the tests check it does not do.

What portability is *not*: the two targets do not share a calling convention.
Triton's host reads every dimension off the tensors it is handed; the reference's
host takes the dimensions explicitly, because a Python list carries no shape.
That difference is legitimate and documented -- it belongs to the interface, not
to the program -- and reporting it as a portability cost would be a category
error.
"""

from dataclasses import dataclass

from compiler.analysis.traffic import plan_traffic


@dataclass(frozen=True)
class Reach:
    """What one target does with a program."""

    target: str
    kernels: int
    skipped: int
    unreachable: tuple
    reasons: tuple

    @property
    def reaches_everything(self) -> bool:
        return not self.unreachable


@dataclass(frozen=True)
class Portability:
    """The shared plan, and what each target makes of it."""

    kernels: int
    slots: int
    a: float
    reaches: tuple

    @property
    def unreachable(self) -> tuple:
        """Operators no target can reach.

        21.8's "不可达算子数". The number worth reporting is not "this target
        cannot lower an operator" but "the operator reaches neither target": the first is a
        backend's limit, the second is the language's hole, and only the second
        says anything about the program. `target_limits` is the other set.
        """
        if not self.reaches:
            return ()

        return tuple(
            sorted(
                set.intersection(
                    *(set(reach.unreachable) for reach in self.reaches)
                )
            )
        )

    @property
    def target_limits(self) -> tuple:
        """Operators some target can reach and another cannot."""
        every = set().union(*(set(reach.unreachable) for reach in self.reaches))

        return tuple(sorted(every - set(self.unreachable)))

    @property
    def portable(self) -> bool:
        return all(reach.reaches_everything for reach in self.reaches)


def measure(pipeline_result) -> Portability:
    """Price a program's reachability, from the plan both targets follow."""
    traffic = plan_traffic(
        pipeline_result.dataflow, pipeline_result.fusion, pipeline_result.memory
    )

    reaches = []

    for target, program in (
        ("triton", pipeline_result.triton),
        ("reference", pipeline_result.reference),
    ):
        refusals = tuple(program.refusals)

        reaches.append(
            Reach(
                target=target,
                kernels=len(program.kernels),
                skipped=len(program.skipped) if program.skipped else 0,
                unreachable=tuple(
                    sorted({refusal.operator for refusal in refusals})
                ),
                reasons=tuple(
                    (refusal.operator, refusal.reason) for refusal in refusals
                ),
            )
        )

    return Portability(
        kernels=len(pipeline_result.fusion.groups),
        slots=len(pipeline_result.memory.slots),
        a=traffic.a,
        reaches=tuple(reaches),
    )
