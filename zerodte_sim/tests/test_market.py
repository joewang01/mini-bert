import numpy as np
import pytest

from zerodte_sim.market import MarketConfig, MarketSimulator


def terminal_variance_ratio(cfg, n_days=120_000, seed=3):
    days = MarketSimulator(cfg, np.random.default_rng(seed)).generate(n_days)
    r = np.log(np.array([d.spot(d.n_steps) for d in days]) / cfg.spot)
    return float((r ** 2).mean() / cfg.implied_daily_move ** 2)


def test_variance_clock_is_a_cdf():
    w = MarketConfig().variance_clock()
    assert w[0] == pytest.approx(0.0)
    assert w[-1] == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0)


def test_clock_is_u_shaped():
    cfg = MarketConfig()
    w = cfg.variance_clock()
    dw = np.diff(w)
    mid = len(dw) // 2
    assert dw[0] > dw[mid] and dw[-1] > dw[mid]


def test_autocorrelation_does_not_change_distance_travelled():
    # AR(1) cross-covariances silently inflate terminal variance unless they are
    # normalised out; at rho=0.05 the overshoot exceeds the whole variance premium.
    base = terminal_variance_ratio(MarketConfig(autocorr=0.0))
    for rho in (0.05, 0.25):
        assert terminal_variance_ratio(MarketConfig(autocorr=rho)) == pytest.approx(base, rel=0.02)


def test_realised_variance_matches_configured_premium():
    cfg = MarketConfig()
    assert terminal_variance_ratio(cfg) == pytest.approx(cfg.mean_variance_ratio, rel=0.05)


def test_mean_variance_ratio_includes_jumps():
    quiet = MarketConfig(jump_prob=0.0)
    jumpy = MarketConfig(jump_prob=0.05)
    assert jumpy.mean_variance_ratio > quiet.mean_variance_ratio


def test_skew_makes_otm_puts_richer_than_otm_calls():
    day = MarketSimulator(MarketConfig(), np.random.default_rng(0)).generate(1)[0]
    put_k = day.strike_for_delta(3, 0.10, False)
    call_k = day.strike_for_delta(3, 0.10, True)
    assert day.total_var(put_k, 3) > day.total_var(call_k, 3)


def test_strike_for_delta_is_skew_consistent():
    day = MarketSimulator(MarketConfig(), np.random.default_rng(0)).generate(1)[0]
    for target in (0.05, 0.10, 0.30):
        for is_call in (True, False):
            k = day.strike_for_delta(3, target, is_call)
            assert abs(day.delta(k, 3, is_call)) == pytest.approx(target, abs=1e-4)


def test_leverage_effect_lifts_vol_on_selloffs():
    cfg = MarketConfig()
    days = MarketSimulator(cfg, np.random.default_rng(5)).generate(4000)
    moves = np.array([d.spot(40) / cfg.spot - 1.0 for d in days])
    ivs = np.array([d.iv_level[40] for d in days])
    assert np.corrcoef(moves, ivs)[0, 1] < -0.8


def test_reversals_do_not_change_terminal_variance():
    """The excursion must lengthen the path without moving where the day ends.

    If it changed terminal variance it would silently re-calibrate the variance
    risk premium, which is the one number the whole study turns on.
    """
    base = terminal_variance_ratio(MarketConfig(reversal_amp=0.0), n_days=60_000)
    for amp in (0.6, 1.2):
        got = terminal_variance_ratio(MarketConfig(reversal_amp=amp), n_days=60_000)
        assert got == pytest.approx(base, rel=0.02)


def test_reversals_lengthen_the_intraday_path():
    def mean_range(amp):
        cfg = MarketConfig(reversal_amp=amp)
        days = MarketSimulator(cfg, np.random.default_rng(3)).generate(20_000)
        paths = np.array([d.spot_path for d in days])
        return float(np.mean(paths.max(axis=1) - paths.min(axis=1)))

    assert mean_range(1.2) > mean_range(0.6) > mean_range(0.0)


def test_v_shaped_days_actually_occur():
    """Sell the untested side after a selloff and this is what runs you over."""
    cfg = MarketConfig()
    days = MarketSimulator(cfg, np.random.default_rng(4)).generate(20_000)
    paths = np.array([d.spot_path for d in days]) / cfg.spot
    reversed_ = (paths.min(axis=1) < 1 - cfg.implied_daily_move) & (paths[:, -1] > 1 - 0.002)
    assert reversed_.mean() > 0.02
