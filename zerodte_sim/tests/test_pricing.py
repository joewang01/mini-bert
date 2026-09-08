import math

import pytest

from zerodte_sim.pricing import bs_delta, bs_price, intrinsic, strike_for_delta

VAR = 0.008 ** 2
SPOT = 6000.0


def test_put_call_parity():
    for strike in (5800.0, 6000.0, 6200.0):
        call = bs_price(SPOT, strike, VAR, True)
        put = bs_price(SPOT, strike, VAR, False)
        assert call - put == pytest.approx(SPOT - strike, abs=1e-8)


def test_zero_variance_is_intrinsic():
    for strike in (5900.0, 6100.0):
        for is_call in (True, False):
            assert bs_price(SPOT, strike, 0.0, is_call) == intrinsic(SPOT, strike, is_call)


def test_price_is_monotone_in_variance():
    # Non-decreasing everywhere, and strictly increasing once the strike is
    # within reach.  Far OTM at negligible variance the price underflows to
    # exactly zero, so strict monotonicity does not hold across that region.
    prices = [bs_price(SPOT, 5900.0, v, False) for v in (1e-9, 1e-6, 1e-5, 1e-4, 1e-3)]
    assert all(b >= a for a, b in zip(prices, prices[1:]))
    assert prices[-1] > prices[-2] > 0.0


def test_strike_for_delta_round_trips():
    for target in (0.05, 0.10, 0.25, 0.40):
        for is_call in (True, False):
            strike = strike_for_delta(SPOT, VAR, target, is_call)
            assert abs(bs_delta(SPOT, strike, VAR, is_call)) == pytest.approx(target, abs=1e-9)


def test_otm_side_is_correct():
    assert strike_for_delta(SPOT, VAR, 0.10, True) > SPOT
    assert strike_for_delta(SPOT, VAR, 0.10, False) < SPOT


def test_deep_expiry_does_not_blow_up():
    # The regime this simulator lives in: variance collapsing to zero.
    for v in (1e-12, 1e-15, 0.0):
        assert math.isfinite(bs_price(SPOT, 5990.0, v, False))
        assert math.isfinite(bs_delta(SPOT, 5990.0, v, False))
