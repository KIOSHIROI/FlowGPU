"""Which reduction arrangements are legal, and on what contract.

A program that owns only part of a reduction domain cannot finish the reduction
alone: the row maximum and the normalising denominator have to be shared across
the owners. FlowGPU currently emits every reduction so that **one program owns
the whole domain and sweeps it in bounded chunks**. Chunks are a sweep, not a
split, which is why no cross-program merge is needed. That is a contract, and
this module makes it a value instead of a paragraph in a docstring.

Three arrangements are defined:

``single_program``
    One program owns the whole domain and walks it in bounded chunks.
``two_stage``
    Several programs own disjoint parts; a merge stage combines per-part
    statistics before the epilogue.
``online``
    Several programs own disjoint parts; each keeps a running maximum and a
    scaled running sum, and the merge is folded into the sweep.

The latter two both need an explicit merge contract. The shifted-exponential
proof in ``flowgpu.compiler.ir.reduction_algebra`` is one such contract; a plain
max/sum merge would be another.

The decision is about legality, never profitability. A legal two-stage split can
still be slower than materialising the intermediate.
"""
from dataclasses import dataclass

SINGLE_PROGRAM = 'single_program'
TWO_STAGE = 'two_stage'
ONLINE = 'online'
ILLEGAL = 'illegal'


@dataclass(frozen=True)
class Coverage:
    """How a described schedule covers one reduction domain."""
    strategy: str
    legal: bool
    reason: str
    programs_along_domain: int

    def as_dict(self):
        return dict(strategy=self.strategy, legal=self.legal, reason=self.reason,
                    programs_along_domain=self.programs_along_domain)


def classify(*, programs_along_domain, merge_proven=False, online=False):
    """Decide the legal arrangement for a described reduction schedule.

    ``programs_along_domain`` is the number of programs that own a non-empty
    part of the reduction axis. One is the current emitter contract; anything
    larger is a split and needs a merge.
    """
    if type(programs_along_domain) is not int or programs_along_domain < 1:
        raise ValueError('programs_along_domain must be an integer of at least one')
    if programs_along_domain == 1:
        return Coverage(SINGLE_PROGRAM, True,
                        'one program owns the whole domain; chunks are a sweep, not a split',
                        programs_along_domain)
    if merge_proven:
        return Coverage(TWO_STAGE, True,
                        'disjoint owners plus a proven merge contract',
                        programs_along_domain)
    if online:
        return Coverage(ONLINE, True,
                        'disjoint owners with a running statistic carried in the sweep',
                        programs_along_domain)
    return Coverage(ILLEGAL, False,
                    'the reduction axis is owned by {} programs, so the row maximum and the '
                    'normalising denominator are incomplete in every one of them; '
                    'provide a merge contract or keep one program per domain'.format(
                        programs_along_domain),
                    programs_along_domain)


def require_legal(coverage):
    """Raise with the arrangement's own reason when it is not legal."""
    if not isinstance(coverage, Coverage):
        raise TypeError('require_legal expects a Coverage')
    if not coverage.legal:
        raise ValueError(coverage.reason)
    return coverage


def group_coverage(group, *, programs_along_domain=1, merge_proven=False, online=False):
    """Coverage records for a fusion group, one per declared reduction domain.

    A group without a reduction domain returns an empty tuple: there is nothing
    to share, so no arrangement applies. The domain width is not needed to
    decide legality, only to describe the schedule, so a symbolic width is fine.
    """
    domain = tuple(getattr(group, 'reduction_domain', ()) or ())
    if not domain:
        return ()
    return (classify(programs_along_domain=programs_along_domain,
                     merge_proven=merge_proven, online=online),)
