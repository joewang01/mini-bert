"""Tests for the pre-open expected-move calculator."""

from __future__ import annotations

import math
import pathlib

import pytest

from premarket.expected_move import (
    BREAK_EVEN_VOL_RATIO,
    TRADING_DAYS,
    daily_vol_from_vix,
    summarise,
    total_var_from_straddle,
    variance_fraction_remaining,
)
from zerodte_sim.market import MarketConfig
from zerodte_sim.pricing import bs_delta, bs_price


def test_straddle_inverts_black_scholes():
    """Pricing an ATM straddle and inverting it recovers the variance."""
    spot, total_var = 7707.0, 0.0000450
    straddle = bs_price(spot, spot, total_var, True) + bs_price(
        spot, spot, total_var, False
    )
    assert total_var_from_straddle(straddle, spot) == pytest.approx(total_var, rel=1e-3)


def test_vix_scaling_is_square_root_of_time():
    """A VIX of sqrt(252) is exactly 1% a day, and the haircut scales linearly."""
    assert daily_vol_from_vix(math.sqrt(TRADING_DAYS), 1.0) == pytest.approx(0.01)
    full = daily_vol_from_vix(16.0, 1.0)
    assert daily_vol_from_vix(16.0, 0.70) == pytest.approx(0.70 * full)


def test_variance_remaining_spans_the_session():
    cfg = MarketConfig()
    assert variance_fraction_remaining(0, cfg) == pytest.approx(1.0)
    assert variance_fraction_remaining(390, cfg) == pytest.approx(0.0, abs=1e-9)


def test_variance_remaining_is_monotone():
    cfg = MarketConfig()
    minutes = list(range(0, 391, 30))
    left = [variance_fraction_remaining(m, cfg) for m in minutes]
    assert all(a >= b for a, b in zip(left, left[1:]))


def test_open_is_front_loaded():
    """The U-shaped clock must burn variance faster than wall-clock early on.

    This is the whole quantitative case for entering late, so it is worth a
    guard: at minute 120 less than 270/390 of the variance can remain.
    """
    cfg = MarketConfig()
    assert variance_fraction_remaining(120, cfg) < 270.0 / 390.0


def test_credit_and_risk_sum_to_the_width():
    rows = summarise(7707.0, 4.5e-5, (0.10, 0.15, 0.20), width=25.0, contracts=3)
    for row in rows:
        assert row["credit"] + row["max_loss"] == pytest.approx(25.0 * 100.0 * 3)
        assert row["break_even_win_rate"] == pytest.approx(
            row["max_loss"] / (25.0 * 100.0 * 3)
        )


def test_lower_delta_sits_further_out_for_less_credit():
    rows = summarise(7707.0, 4.5e-5, (0.10, 0.15, 0.20), width=25.0, contracts=1)
    assert rows[0]["distance_pct"] > rows[1]["distance_pct"] > rows[2]["distance_pct"]
    assert rows[0]["credit"] < rows[1]["credit"] < rows[2]["credit"]


def test_strikes_really_carry_the_requested_delta():
    """Guard the inversion itself, not just its monotonicity."""
    spot, total_var = 7707.0, 4.5e-5
    for row in summarise(spot, total_var, (0.10, 0.15, 0.20), 25.0, 1):
        actual = abs(bs_delta(spot, row["short_strike"], total_var, is_call=False))
        assert actual == pytest.approx(row["delta"], abs=1e-6)


def test_break_even_ratio_matches_the_simulator_readme():
    """The playbook's headline number must not drift from what was measured."""
    readme = pathlib.Path(__file__).resolve().parents[2] / "zerodte_sim" / "README.md"
    assert f"{BREAK_EVEN_VOL_RATIO}" in readme.read_text()
