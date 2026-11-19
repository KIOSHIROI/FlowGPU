"""Analytic ranking of candidate configurations, and the limits it has.

Selecting a configuration normally means measuring every candidate. This module
scores them from structure alone, so the candidate set can be pruned before any
device time is spent. The score is small and interpretable, and it switches
between two regimes:

**Bandwidth regime** -- enough traffic per launch that occupancy can matter:

    cost = traffic / min(1, programs_per_sm / SATURATION) + LAUNCH_BYTES * launches

A kernel that puts fewer than ``SATURATION`` programs on each SM cannot saturate
memory throughput, so its effective bandwidth is proportionally lower. This is
the Little's-law argument from the survey, not a fitted black box.

**Latency regime** -- traffic per launch below ``LATENCY_SWITCH_BYTES``:

    cost = traffic + LAUNCH_BYTES * launches

A shape that small cannot fill the device whatever the grid says, so discounting
a candidate for low ``programs_per_sm`` inverts the true order. Before the switch
existed, every holdout failure had the same shape: the measured winner had a
larger tile and fewer programs, and the bandwidth model preferred the opposite.

Calibration and what it is worth
--------------------------------
The constants were chosen on the *screen* shapes of one frozen corpus
(an earlier internal calibration) and then applied unchanged to shapes the
search never saw. Across twenty-four such shapes, pruning seven candidates to
two never removed the measured winner; pruning to one removed it four times.
Pruning to two therefore halves the candidate set at no observed cost.

The same constants failed on gather-partial kernels: three of four holdouts lost
their winner, with a mean regret of about 20% and a worst case of 46%. Gather
cost is driven by index traffic that ``logical_traffic_bytes`` does not
represent, so ``FAMILIES`` restricts the pruner to reduction family groups. A
family outside that allowlist is refused rather than guessed.

This is a pruner, not a predictor. It never claims a candidate is fast; it
claims a small set is worth measuring. Top-1 agreement with measurement is 50%,
which is exactly why the kept set is still measured.
"""

from dataclasses import dataclass

#: Programs per SM at which memory throughput is treated as saturated.
SATURATION = 4.0
#: Fixed cost of one kernel launch, expressed in bytes of equivalent traffic.
LAUNCH_BYTES = 500.0
#: Tie-break weight for serialised chunks. Deliberately small: the model has no
#: basis for a magnitude here, it only needs to order candidates whose traffic
#: and launch count are equal.
SERIAL_TIEBREAK = 0.05
#: Traffic per launch below which occupancy is not a constraint. A shape this
#: small cannot fill the device whatever the grid says, so discounting a
#: candidate for low ``programs_per_sm`` inverts the true order. The switch is
#: the plateau midpoint of a threshold sweep on screen shapes; the plateau spans
#: 24000-48000 bytes per launch and the holdout result does not change across it.
LATENCY_SWITCH_BYTES = 30000.0

#: Kernel families the calibration covers. Anything else is refused.
FAMILIES = frozenset({'reduction'})


@dataclass(frozen=True)
class Ranked:
    name: str
    cost: float
    features: dict

    def as_dict(self):
        return dict(name=self.name, cost=self.cost, features=self.features)


def features(record):
    """Extract the structural inputs from one candidate record.

    Accepts the shape the compiler already produces for a candidate: traffic,
    launch count, and per-kernel grid/tile descriptors. Missing fields are an
    error rather than a default, because a zero traffic value would make the
    candidate look free.
    """
    if not isinstance(record, dict):
        raise TypeError('candidate record must be a mapping')
    for key in ('logical_traffic_bytes', 'launches', 'kernels'):
        if key not in record:
            raise ValueError('candidate record needs ' + key)
    kernels = record['kernels']
    if not isinstance(kernels, (list, tuple)) or not kernels:
        raise ValueError('candidate record needs at least one kernel descriptor')
    programs_per_sm = [kernel.get('programs_per_sm') or 0.0 for kernel in kernels]
    serial = [kernel.get('chunk_iterations_per_program') or 0 for kernel in kernels]
    return dict(traffic=float(record['logical_traffic_bytes']),
                launches=int(record['launches']),
                programs_per_sm=max(programs_per_sm),
                serial_chunks=sum(serial))


def bandwidth_fraction(programs_per_sm, *, saturation=SATURATION):
    """Share of peak bandwidth a launch can reach at this occupancy."""
    if not isinstance(programs_per_sm, (int, float)) or programs_per_sm < 0:
        raise ValueError('programs_per_sm must be a non-negative number')
    return min(1.0, programs_per_sm / saturation)


def cost(extracted, *, saturation=SATURATION, launch_bytes=LAUNCH_BYTES,
         serial_tiebreak=SERIAL_TIEBREAK, latency_switch_bytes=LATENCY_SWITCH_BYTES):
    """Byte-equivalent cost of one candidate. Lower is expected to be faster.

    Two regimes. When there is enough traffic per launch for occupancy to
    matter, low occupancy divides the effective bandwidth. Below the switch the
    shape is too small to fill the device at all, so the same discount would
    reward spreading the work thinner -- exactly backwards, and the cause of
    every holdout failure before the switch existed.
    """
    tiny = extracted['traffic'] <= latency_switch_bytes * extracted['launches']
    if tiny:
        share = 1.0
    else:
        share = bandwidth_fraction(extracted['programs_per_sm'], saturation=saturation)
    if share <= 0:
        # No program can reach memory at all: rank it last rather than dividing
        # by zero and pretending the cost is infinite.
        return float('inf')
    return (extracted['traffic'] / share
            + launch_bytes * extracted['launches']
            + serial_tiebreak * extracted['serial_chunks'])


def rank(records, *, family, saturation=SATURATION):
    """Order candidate records by analytic cost, cheapest first.

    ``family`` gates the calibration. A family outside ``FAMILIES`` raises: the
    constants above were never validated there, and silently applying them would
    turn an untested guess into a decision.
    """
    if family not in FAMILIES:
        raise ValueError('analytic ranking is calibrated for {} only, not {!r}'.format(
            ', '.join(sorted(FAMILIES)), family))
    if not isinstance(records, (list, tuple)) or len(records) < 2:
        raise ValueError('ranking needs at least two candidate records')
    names = [record.get('name') for record in records]
    if any(name is None for name in names) or len(set(names)) != len(names):
        raise ValueError('every candidate record needs a distinct name')
    ranked = [Ranked(record['name'],
                     cost(features(record), saturation=saturation),
                     features(record)) for record in records]
    return tuple(sorted(ranked, key=lambda item: (item.cost, item.name)))


def prune(records, *, keep, family, saturation=SATURATION, always_keep=()):
    """Keep the ``keep`` cheapest candidates and say why the rest were dropped.

    The returned reason names the dominating candidate, so a report can answer
    "why was this configuration not measured" without re-deriving the score.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError('keep must be a positive integer')
    ordered = rank(records, family=family, saturation=saturation)
    forced = frozenset(always_keep)
    unknown = forced - {item.name for item in ordered}
    if unknown:
        raise ValueError('always_keep names unknown candidates: ' + ', '.join(sorted(unknown)))
    if keep >= len(ordered):
        return ordered, ()
    # A tie at the boundary is kept together. Breaking it by name would decide
    # a measurement budget on alphabetical order: on one frozen holdout the
    # third and fourth cheapest scores were equal, and the measured winner was
    # the one the alphabetical tie-break dropped.
    boundary = ordered[keep - 1].cost
    # A comparison needs its reference: callers name the baseline in
    # ``always_keep`` so that pruning can never remove what everything else is
    # measured against.
    kept = tuple(item for item in ordered
                 if item.cost <= boundary or item.name in forced)
    best = ordered[0]
    dropped = tuple(dict(name=item.name, cost=item.cost,
                         reason='cost {:.1f} exceeds {:.1f} of {}'.format(
                             item.cost, best.cost, best.name))
                    for item in ordered if item.cost > boundary and item.name not in forced)
    return kept, dropped
