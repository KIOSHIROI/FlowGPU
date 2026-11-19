"""Statistical power for paired performance comparisons.

A selection rule is only meaningful when the difference between candidates
exceeds the measurement noise. This module makes that threshold computable: the
minimum detectable effect of a paired design, the rounds needed to resolve a
target effect, and the optimism of picking the best of k noisy candidates.

Everything here is pure arithmetic: no timing, no device, no module state. The
caller supplies a noise estimate; this module never invents one.
"""
import math

# Two-sided alpha = 0.05 (z = 1.959964) plus power 0.8 (z = 0.841621).
Z_ALPHA_2_005 = 1.959963984540054
Z_POWER_0_8 = 0.8416212335729143

# E[max of k iid standard normals] for k = 2..20, from quadrature of
# integral x * k * phi(x) * Phi(x)**(k-1) dx over [-12, 12] with Simpson.
# Known checks: k=2 -> 0.5642, k=10 -> 1.5388.
_EXPECTED_MAX = {
    2: 0.5642, 3: 0.8463, 4: 1.0294, 5: 1.1630, 6: 1.2672, 7: 1.3522,
    8: 1.4236, 9: 1.4850, 10: 1.5388, 11: 1.5864, 12: 1.6292, 13: 1.6680,
    14: 1.7034, 15: 1.7359, 16: 1.7660, 17: 1.7939, 18: 1.8200, 19: 1.8445,
    20: 1.8675,
}


def critical_z(alpha=0.05, power=0.8):
    """The multiplier in front of the standard error for a two-sided test.

    Only the two values used by the protocols are tabulated; anything else is
    rejected rather than approximated silently.
    """
    if alpha == 0.05 and power == 0.8:
        return Z_ALPHA_2_005 + Z_POWER_0_8
    raise ValueError('only alpha=0.05 and power=0.8 are tabulated')


def minimum_detectable_effect(sigma, rounds, *, alpha=0.05, power=0.8):
    """Relative effect a paired design can resolve, as a fraction.

    ``sigma`` is the standard deviation of the per-round log ratio (a paired
    design cancels common-mode drift, so this is much smaller than the
    independent-measurement spread). Returns e.g. 0.05 for 5%.
    """
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be a finite non-negative number')
    if type(rounds) is not int or rounds < 2:
        raise ValueError('at least two paired rounds are needed to estimate a mean')
    return math.expm1(critical_z(alpha, power) * sigma / math.sqrt(rounds))


def detectable(sigma, rounds, effect, *, alpha=0.05, power=0.8):
    """Whether ``effect`` exceeds the design's minimum detectable effect."""
    return effect >= minimum_detectable_effect(sigma, rounds, alpha=alpha, power=power)


def required_rounds(sigma, effect, *, alpha=0.05, power=0.8):
    """Rounds needed to resolve ``effect`` at the given paired noise."""
    if not isinstance(effect, (int, float)) or not math.isfinite(effect) or effect <= 0:
        raise ValueError('effect must be a finite positive fraction')
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be a finite non-negative number')
    if sigma == 0:
        return 2
    needed = (critical_z(alpha, power) * sigma / math.log1p(effect)) ** 2
    return max(2, math.ceil(needed))


def expected_max_of_normals(candidates):
    """Expected maximum of ``candidates`` iid standard normals."""
    if type(candidates) is not int or candidates < 2:
        raise ValueError('selection needs at least two candidates')
    if candidates in _EXPECTED_MAX:
        return _EXPECTED_MAX[candidates]
    if candidates > 20:
        # Asymptotic form, used only above the tabulated range.
        log_k = math.log(candidates)
        return math.sqrt(2 * log_k) - (math.log(log_k) + math.log(4 * math.pi)) / (
            2 * math.sqrt(2 * log_k))
    raise ValueError('candidate count must be at least two')


def selection_optimism(candidates, sigma):
    """Relative optimism of reporting the best of k noisy candidates.

    Returns a fraction: 0.08 means the observed winner is about 8% better than
    the true winner on average, purely from noise.
    """
    if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be a finite non-negative number')
    return math.expm1(expected_max_of_normals(candidates) * sigma)


def pooled_log_ratio_sigma(series):
    """Pooled standard deviation of per-round log ratios.

    ``series`` is an iterable of ``(candidate_values, control_values)`` pairs.
    Pairs with fewer than two rounds carry no variance information and are
    skipped; if nothing remains the result is ``None`` rather than zero, so a
    caller cannot mistake "unknown" for "no noise".
    """
    total = 0.0
    degrees = 0
    for values, control in series:
        if len(values) != len(control) or len(values) < 2:
            continue
        logs = [math.log(v / b) for v, b in zip(values, control) if v > 0 and b > 0]
        if len(logs) < 2:
            continue
        mean = sum(logs) / len(logs)
        total += sum((x - mean) ** 2 for x in logs)
        degrees += len(logs) - 1
    if degrees == 0:
        return None
    return math.sqrt(total / degrees)


def pooled_log_ratio_robust_sigma(series, *, trim=0.1):
    """Trimmed standard deviation of the pooled log ratios.

    Plain standard deviation is dominated by a single slow round: on the H100
    probe one 6.7x round among twenty moved the estimate from 0.9% to 42.5%.
    Trimming the extreme tenth of the ordered ratios keeps the scale of the
    bulk, which is what a gate should be calibrated on. MAD was rejected
    because it collapses to zero when most rounds agree exactly.
    """
    logs = _pooled_logs(series)
    if logs is None:
        return None
    return _trimmed_sd(logs, trim)


def _trimmed_sd(values, trim=0.1):
    ordered = sorted(values)
    n = len(ordered)
    k = int(n * trim)
    kept = ordered[k:n - k] if n - 2 * k >= 2 else ordered
    mean = sum(kept) / len(kept)
    return math.sqrt(sum((x - mean) ** 2 for x in kept) / (len(kept) - 1))


def paired_noise_scales(series, *, trim=0.1, outlier_sigmas=3.5):
    """Both scales plus an outlier count, for reports that must stay honest.

    ``outliers`` counts paired rounds further than ``outlier_sigmas`` robust
    sigma from the median. A non-zero count means the plain standard deviation
    is not a usable gate width and the trimmed scale should be used instead.
    """
    logs = _pooled_logs(series)
    if logs is None:
        return None
    middle = _median(logs)
    robust = _trimmed_sd(logs, trim)
    mean = sum(logs) / len(logs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
    deviations = [abs(x - middle) for x in logs]
    if robust > 0:
        outliers = sum(1 for x in deviations if x > outlier_sigmas * robust)
    else:
        outliers = sum(1 for x in deviations if x > 0)
    return dict(n=len(logs), sd=sd, robust=robust, outliers=outliers,
                median=middle, max_abs_deviation=max(deviations))


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _pooled_logs(series):
    logs = []
    for values, control in series:
        if len(values) != len(control) or len(values) < 2:
            continue
        logs.extend(math.log(v / b) for v, b in zip(values, control) if v > 0 and b > 0)
    return logs if len(logs) >= 2 else None


def report(sigma, rounds, effect=None, *, candidates=None, alpha=0.05, power=0.8,
           robust_sigma=None, outliers=None):
    """A JSON-safe summary for experiment reports."""
    summary = dict(sigma_log_ratio=sigma, rounds=rounds,
                   mde_relative=minimum_detectable_effect(sigma, rounds, alpha=alpha, power=power),
                   alpha=alpha, power=power)
    if robust_sigma is not None:
        summary['sigma_robust_log_ratio'] = robust_sigma
        summary['mde_robust_relative'] = minimum_detectable_effect(
            robust_sigma, rounds, alpha=alpha, power=power)
    if outliers is not None:
        summary['outlier_rounds'] = outliers
    if effect is not None:
        summary['expected_effect'] = effect
        summary['detectable'] = detectable(sigma, rounds, effect, alpha=alpha, power=power)
        if not summary['detectable']:
            summary['required_rounds'] = required_rounds(sigma, effect, alpha=alpha, power=power)
    if candidates is not None:
        summary['candidates'] = candidates
        summary['selection_optimism_relative'] = selection_optimism(candidates, sigma)
    return summary
