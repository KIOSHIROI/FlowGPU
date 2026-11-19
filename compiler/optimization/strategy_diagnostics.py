"""Separate empirical selection loss from candidate/handwritten coverage gaps."""
import math
import statistics


def diagnose(samples, selections, *, candidates, hand='hand'):
    """Consume fresh evaluation samples after choices have been frozen.

    The lowest observed candidate is a diagnostic comparator, not a deployable
    oracle or a claim that its sample minimum predicts future performance.
    """
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError('distinct evaluated candidates required')
    def median(name, metric):
        values=samples[name][metric]
        if (not isinstance(values,list) or len(values)<4 or any(type(v) not in {int,float}
                or not math.isfinite(v) or v<=0 for v in values)):
            raise ValueError('four finite positive evaluation rounds required')
        return statistics.median(values)
    result={}
    for metric in ('graph_us','wall_us'):
        costs={n:median(n,metric) for n in candidates}
        best=min(costs,key=lambda n:(costs[n],n))
        chosen={label:dict(candidate=n, median_us=costs[n],
                    ratio_to_lowest_observed=costs[n]/costs[best]) for label,n in selections[metric].items()}
        hand_cost=median(hand,metric) if hand in samples else None
        result[metric]=dict(lowest_observed_candidate=best, lowest_observed_us=costs[best],
            selections=chosen, hand_us=hand_cost,
            candidate_to_hand_ratio=costs[best]/hand_cost if hand_cost is not None else None)
    return result
