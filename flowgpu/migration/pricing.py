"""Pricing model for the flowgpu/migration/acceptance service (internal).

The numbers are hypotheses, not quotes.  They are built bottom-up so an
interview can test them: delivery cost = person-days x loaded day cost + GPU /
infrastructure cost + overhead + risk buffer.  List prices are package-level so
a customer can accept/reject a number in a 20-minute conversation.
"""
from __future__ import annotations

import dataclasses

#: Loaded internal cost of one engineer-day (salary + overhead baseline).
DEFAULT_DAY_COST = 1800
#: Billing rate used for custom scopes (specialised flowgpu/compiler/AI-infra work).
DEFAULT_BILLING_DAY = 2600


@dataclasses.dataclass(frozen=True)
class Package:
    name: str
    person_days: int
    price: int
    scope: str
    deliverables: tuple


#: Fixed-price ladder.  `price` is what we would quote before discount; the
#: calculator checks it clears cost with the configured margin.
PACKAGES = {
    "diagnostic": Package(
        "diagnostic", 10, 50000,
        "one model x one chip, 5-8 operators, 1 week",
        ("coverage matrix", "per-operator correctness evidence", "gap list with reasons",
         "one-page report")),
    "pilot": Package(
        "pilot", 30, 150000,
        "one model x two chips, 10-15 operators, 2-3 weeks",
        ("coverage matrix for both chips", "correctness evidence", "performance vs same-machine baseline",
         "generated/extern/hand attribution", "reproducible evidence bundle", "review meeting")),
    "acceptance": Package(
        "acceptance", 80, 400000,
        "one model x three chips, 20+ operators, 4-6 weeks",
        ("full acceptance package", "evidence bundle per chip", "regression script",
         "handover and training", "one maintenance window")),
}

#: Maintenance is priced as a percentage of the project fee per year.
MAINTENANCE_PERCENT = 0.18


def package_cost(package: Package, *, day_cost=DEFAULT_DAY_COST,
                 gpu_cost=0, overhead_rate=0.25, risk_rate=0.15) -> int:
    base = package.person_days * day_cost + gpu_cost
    return int(round(base * (1 + overhead_rate) * (1 + risk_rate)))


def quote(package_name, *, day_cost=DEFAULT_DAY_COST, gpu_cost=0,
          overhead_rate=0.25, risk_rate=0.15) -> dict:
    if package_name not in PACKAGES:
        raise ValueError(f"unknown package {package_name!r}; "
                         f"choose from {', '.join(sorted(PACKAGES))}")
    package = PACKAGES[package_name]
    cost = package_cost(package, day_cost=day_cost, gpu_cost=gpu_cost,
                        overhead_rate=overhead_rate, risk_rate=risk_rate)
    margin = (package.price - cost) / package.price
    return {
        "package": package.name,
        "person_days": package.person_days,
        "price": package.price,
        "cost_floor": cost,
        "gross_margin": round(margin, 4),
        "profitable": margin >= 0,
        "maintenance_price": int(round(package.price * MAINTENANCE_PERCENT)),
        "scope": package.scope,
        "deliverables": list(package.deliverables),
    }


def custom_quote(person_days, *, billing_day=DEFAULT_BILLING_DAY,
                 day_cost=DEFAULT_DAY_COST, gpu_cost=0,
                 overhead_rate=0.25, risk_rate=0.15) -> dict:
    if person_days <= 0:
        raise ValueError("person_days must be positive")
    cost = int(round((person_days * day_cost + gpu_cost)
                     * (1 + overhead_rate) * (1 + risk_rate)))
    price = int(round(person_days * billing_day))
    return {"package": "custom", "person_days": person_days, "price": price,
            "cost_floor": cost,
            "gross_margin": round((price - cost) / price, 4),
            "profitable": price >= cost}


def price_menu() -> list:
    return [quote(name) for name in ("diagnostic", "pilot", "acceptance")]
