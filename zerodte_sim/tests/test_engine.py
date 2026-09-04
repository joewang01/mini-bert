import numpy as np
import pytest

from zerodte_sim.config import AccountConfig, CostModel, StrategyConfig
from zerodte_sim.engine import simulate_day, simulate_path
from zerodte_sim.market import MarketConfig, MarketSimulator

EQUITY = 100_000.0


def make_days(n=500, seed=0, **market_kw):
    cfg = MarketConfig(**market_kw)
    return cfg, MarketSimulator(cfg, np.random.default_rng(seed)).generate(n)


def run(strategy, days, costs=None, account=None):
    costs = costs or CostModel()
    account = account or AccountConfig()
    return [simulate_day(d, strategy, costs, account, EQUITY) for d in days]


def test_loss_is_bounded_by_defined_risk():
    """A vertical cannot lose more than its width, whatever the roll policy."""
    _, days = make_days(1500, seed=1)
    account = AccountConfig()
    for policy in ("none", "same_size", "equal_risk", "martingale"):
        cfg = StrategyConfig(contracts=2, roll_policy=policy, max_rolls=3)
        for r in run(cfg, days, account=account):
            # Worst case is the cumulative risk of every spread opened, plus the
            # slippage paid to close each one -- not peak simultaneous margin.
            slip = 2 * CostModel().half_spread * r.contracts_opened * 100.0
            floor = -(r.total_risk_opened + r.fees_paid + slip + 1e-6)
            assert r.pnl >= floor, (policy, r.pnl, floor)


def test_margin_cap_is_respected():
    _, days = make_days(1500, seed=2)
    account = AccountConfig(max_margin_frac=0.25)
    cfg = StrategyConfig(contracts=2, roll_policy="martingale", max_rolls=5)
    cap = EQUITY * account.max_margin_frac
    for r in run(cfg, days, account=account):
        assert r.peak_margin <= cap + 1e-6


def test_equal_risk_roll_never_increases_exposure():
    """The one rule that turns a martingale into a legitimate adjustment."""
    _, days = make_days(2000, seed=3)
    base = StrategyConfig(contracts=2, roll_policy="none")
    equal = StrategyConfig(contracts=2, roll_policy="equal_risk", max_rolls=3)
    for d in days:
        b = simulate_day(d, base, CostModel(), AccountConfig(), EQUITY)
        e = simulate_day(d, equal, CostModel(), AccountConfig(), EQUITY)
        assert e.peak_margin <= b.peak_margin + 1e-6


def test_martingale_does_increase_exposure():
    """Guards the comparison: if this ever passed, the study would be vacuous."""
    _, days = make_days(2000, seed=3)
    base = StrategyConfig(contracts=2, roll_policy="none")
    mart = StrategyConfig(contracts=2, roll_policy="martingale", max_rolls=3)
    grew = [
        simulate_day(d, mart, CostModel(), AccountConfig(), EQUITY).peak_contracts
        > simulate_day(d, base, CostModel(), AccountConfig(), EQUITY).peak_contracts
        for d in days
    ]
    assert np.mean(grew) > 0.05


def test_max_rolls_is_honoured():
    _, days = make_days(1000, seed=4)
    for limit in (0, 1, 2):
        cfg = StrategyConfig(contracts=2, roll_policy="martingale", max_rolls=limit)
        assert max(r.rolls for r in run(cfg, days)) <= limit


def test_no_roll_policy_never_rolls():
    _, days = make_days(500, seed=5)
    assert all(r.rolls == 0 for r in run(StrategyConfig(roll_policy="none"), days))


def test_commissions_hurt_every_single_day():
    """Commissions do not move any trigger, so they must bite path-by-path."""
    _, days = make_days(800, seed=6)
    cfg = StrategyConfig(contracts=2)
    free = CostModel(commission=0.0, fees=0.0)
    paid = CostModel()
    for d in days:
        assert (
            simulate_day(d, cfg, free, AccountConfig(), EQUITY).pnl
            > simulate_day(d, cfg, paid, AccountConfig(), EQUITY).pnl
        )


def test_slippage_hurts_on_average_but_not_every_day():
    """Slippage lowers the credit, which lowers the stop trigger with it.

    That makes the comparison non-monotone on individual paths -- a wider
    spread occasionally stops a trade out earlier and cheaper.  The invariant
    that must hold is the aggregate one.
    """
    _, days = make_days(3000, seed=6)
    cfg = StrategyConfig(contracts=2)
    cheap = np.mean([r.pnl for r in run(cfg, days, costs=CostModel(half_spread=0.01))])
    dear = np.mean([r.pnl for r in run(cfg, days, costs=CostModel(half_spread=0.10))])
    assert cheap > dear


def test_credit_collected_matches_a_hand_calculation():
    _, days = make_days(1, seed=7)
    day = days[0]
    cfg = StrategyConfig(side="put", contracts=3, entry_minute=15, roll_policy="none")
    costs = CostModel()
    r = simulate_day(day, cfg, costs, AccountConfig(), EQUITY)
    step = day.cfg.minute_to_index(cfg.entry_minute)
    short_k = round(day.strike_for_delta(step, cfg.short_delta, False) / 5.0) * 5.0
    mid = day.price(short_k, step, False) - day.price(short_k - cfg.width, step, False)
    expected = (mid - 2 * costs.half_spread) * 3 * costs.contract_multiplier
    assert r.credit_collected == pytest.approx(expected)


def test_both_sides_opens_two_spreads():
    _, days = make_days(200, seed=8)
    cfg = StrategyConfig(side="both", roll_policy="none")
    assert all(r.spreads_opened == 2 for r in run(cfg, days))


def test_path_stops_at_ruin():
    _, days = make_days(252, seed=9)
    account = AccountConfig(start_equity=6_000.0, ruin_frac=0.5, max_margin_frac=1.0)
    cfg = StrategyConfig(contracts=2, roll_policy="martingale", max_rolls=5)
    path = simulate_path(days, cfg, CostModel(), account)
    if path.ruined:
        assert path.ruin_day is not None
        after = path.equity[path.ruin_day + 1:]
        assert np.allclose(after, after[0])  # frozen once ruined


def test_reproducible():
    _, days_a = make_days(300, seed=11)
    _, days_b = make_days(300, seed=11)
    cfg = StrategyConfig(contracts=2, roll_policy="martingale")
    a = [r.pnl for r in run(cfg, days_a)]
    b = [r.pnl for r in run(cfg, days_b)]
    assert a == b
