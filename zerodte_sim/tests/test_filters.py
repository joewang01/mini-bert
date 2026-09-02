import numpy as np
import pytest

from zerodte_sim.config import AccountConfig, CostModel, FilterConfig, StrategyConfig
from zerodte_sim.engine import simulate_day
from zerodte_sim.filter_study import calibrate_threshold
from zerodte_sim.filters import should_skip, signal_value
from zerodte_sim.market import MarketConfig, MarketSimulator

EQUITY = 100_000.0


def make_days(n=4000, seed=0):
    cfg = MarketConfig()
    return cfg, MarketSimulator(cfg, np.random.default_rng(seed)).generate(n)


def test_no_filter_never_skips():
    _, days = make_days(500)
    assert not any(should_skip(d, 3, FilterConfig()) for d in days)


def test_perfect_rvol_is_the_vol_oracle():
    _, days = make_days(500)
    for d in days:
        assert signal_value(d, 3, FilterConfig(kind="rvol", corr=1.0)) == pytest.approx(
            signal_value(d, 3, FilterConfig(kind="oracle_vol"))
        )


def test_zero_correlation_rvol_carries_no_information():
    cfg, days = make_days(20_000)
    sig = np.array([signal_value(d, 3, FilterConfig(kind="rvol", corr=0.0)) for d in days])
    volz = np.array([d.vol_z for d in days])
    assert abs(np.corrcoef(sig, volz)[0, 1]) < 0.03


def test_rvol_signal_hits_its_requested_correlation():
    cfg, days = make_days(20_000)
    volz = np.array([d.vol_z for d in days])
    for corr in (0.3, 0.6, 0.9):
        sig = np.array([signal_value(d, 3, FilterConfig(kind="rvol", corr=corr)) for d in days])
        assert np.corrcoef(sig, volz)[0, 1] == pytest.approx(corr, abs=0.03)


def test_calibration_hits_the_target_skip_rate():
    market = MarketConfig()
    days = MarketSimulator(market, np.random.default_rng(555)).generate(20_000)
    for kind in ("opening_range", "opening_move", "oracle_vol"):
        base = FilterConfig(kind=kind)
        for target in (0.10, 0.20, 0.35):
            thr = calibrate_threshold(market, base, 3, target, n_days=20_000)
            filt = FilterConfig(kind=kind, threshold=thr)
            got = np.mean([should_skip(d, 3, filt) for d in days])
            assert got == pytest.approx(target, abs=0.02)


def test_skipping_is_monotone_in_threshold():
    _, days = make_days(3000)
    rates = [
        np.mean([should_skip(d, 3, FilterConfig(kind="opening_range", threshold=t)) for d in days])
        for t in (0.1, 0.2, 0.4, 0.8)
    ]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_filtered_sessions_take_no_risk_and_cost_nothing():
    _, days = make_days(2000)
    cfg = StrategyConfig(
        contracts=2,
        roll_policy="martingale",
        entry_filter=FilterConfig(kind="opening_range", threshold=0.2),
    )
    results = [simulate_day(d, cfg, CostModel(), AccountConfig(), EQUITY) for d in days]
    skipped = [r for r in results if r.exit_reason == "filtered"]
    assert skipped, "threshold should decline some sessions"
    for r in skipped:
        assert r.pnl == 0.0
        assert r.total_risk_opened == 0.0
        assert r.fees_paid == 0.0
        assert r.contracts_opened == 0


def test_filter_only_removes_days_it_does_not_alter_them():
    """A filtered run must equal the unfiltered run on every session it keeps."""
    _, days = make_days(2000)
    filt = FilterConfig(kind="opening_range", threshold=0.3)
    plain = StrategyConfig(contracts=2, roll_policy="martingale")
    gated = StrategyConfig(contracts=2, roll_policy="martingale", entry_filter=filt)
    for d in days:
        g = simulate_day(d, gated, CostModel(), AccountConfig(), EQUITY)
        if g.exit_reason == "filtered":
            continue
        p = simulate_day(d, plain, CostModel(), AccountConfig(), EQUITY)
        assert g.pnl == pytest.approx(p.pnl)
