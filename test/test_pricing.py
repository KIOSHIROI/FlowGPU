"""Pricing model: the price ladder must clear its cost floor."""
from __future__ import annotations

import pytest

from migration import cli, pricing


def test_every_package_clears_its_cost_floor_with_the_default_assumptions():
    for name in pricing.PACKAGES:
        record = pricing.quote(name)
        assert record["price"] > record["cost_floor"], name
        assert record["profitable"]
        assert 0.3 < record["gross_margin"] < 0.7, (name, record["gross_margin"])


def test_hiring_market_day_cost_compresses_margin_and_the_model_shows_it():
    founder = pricing.quote("pilot")
    hired = pricing.quote("pilot", day_cost=4000)
    assert hired["cost_floor"] > founder["cost_floor"]
    assert hired["gross_margin"] < founder["gross_margin"]
    assert hired["gross_margin"] < 0.35


def test_custom_quote_and_validation():
    record = pricing.custom_quote(25)
    assert record["person_days"] == 25 and record["price"] == 25 * pricing.DEFAULT_BILLING_DAY
    assert record["profitable"]
    with pytest.raises(ValueError):
        pricing.custom_quote(0)
    with pytest.raises(ValueError, match="unknown package"):
        pricing.quote("free")


def test_maintenance_is_a_percentage_of_the_project_price():
    record = pricing.quote("acceptance")
    assert record["maintenance_price"] == round(record["price"] * pricing.MAINTENANCE_PERCENT)


def test_price_cli_prints_the_menu(capsys):
    assert cli.main(["price"]) == 0
    output = capsys.readouterr().out
    for name in ("diagnostic", "pilot", "acceptance"):
        assert name in output
    assert cli.main(["price", "--package", "pilot"]) == 0
