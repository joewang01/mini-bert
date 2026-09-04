"""Performance statistics chosen for a negatively-skewed strategy.

Mean and Sharpe flatter premium selling: a rule that converts many small losses
into a few enormous ones improves both the win rate and the *appearance* of the
equity curve while making the strategy worse.  The numbers that actually
discriminate are in the tail -- ``worst_day``, ``cvar_1``, ``days_to_erase``,
``p_ruin`` -- so they are reported alongside, never instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["DayStats", "PathStats", "Summary", "summarise"]

TRADING_DAYS = 252


@dataclass
class DayStats:
    """Pooled statistics over every simulated session."""

    n_days: int
    mean: float
    median: float
    std: float
    win_rate: float
    profit_factor: float
    skew: float
    excess_kurtosis: float
    worst_day: float
    best_day: float
    var_5: float
    cvar_5: float
    cvar_1: float
    days_to_erase: float
    """How many average winning days one worst-case day wipes out.  The single
    most legible measure of tail asymmetry for this kind of strategy."""
    traded_frac: float
    """Fraction of sessions actually traded.  Below 1.0 an entry filter is
    declining days, and the mean is spread over calendar days rather than
    traded ones -- skipping costs income as well as risk."""
    win_rate_traded: float
    """Wins as a fraction of *traded* sessions, so filters do not flatter it."""
    mean_rolls: float
    roll_day_frac: float
    max_peak_contracts: int
    mean_credit: float
    """Average credit collected per traded session.  Falls with a later entry --
    the economics of the trade change, not just the filter's information."""
    mean_peak_margin: float
    mean_total_risk: float
    """Average cumulative defined risk opened per day.  Diverges from
    ``mean_peak_margin`` exactly to the extent the policy rolls."""


@dataclass
class PathStats:
    """Distribution of outcomes across independent one-year accounts."""

    n_paths: int
    n_days: int
    median_return: float
    mean_return: float
    p05_return: float
    p95_return: float
    p_losing_year: float
    p_ruin: float
    median_max_drawdown: float
    worst_max_drawdown: float
    median_sharpe: float
    median_sortino: float


@dataclass
class Summary:
    label: str
    days: DayStats
    paths: PathStats

    def row(self) -> dict[str, object]:
        return {"strategy": self.label, **asdict(self.days), **asdict(self.paths)}


def _max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a positive fraction of the peak."""
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def summarise(label: str, paths: list, start_equity: float) -> Summary:
    """Build a :class:`Summary` from a list of :class:`~zerodte_sim.engine.PathResult`."""
    pnl = np.concatenate([p.daily_pnl for p in paths])
    rolls = np.concatenate([p.rolls for p in paths])
    peak_contracts = np.concatenate([p.peak_contracts for p in paths])
    peak_margin = np.concatenate([p.peak_margin for p in paths])
    total_risk = np.concatenate([p.total_risk for p in paths])
    credit = np.concatenate([p.credit for p in paths])


    traded = np.concatenate([p.total_risk for p in paths]) > 0
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = -losses.sum()
    mean_win = wins.mean() if wins.size else 0.0
    std = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
    centred = pnl - pnl.mean()
    skew = float((centred ** 3).mean() / std ** 3) if std > 0 else 0.0
    kurt = float((centred ** 4).mean() / std ** 4 - 3.0) if std > 0 else 0.0

    tail_5 = pnl[pnl <= np.percentile(pnl, 5)]
    tail_1 = pnl[pnl <= np.percentile(pnl, 1)]

    days = DayStats(
        n_days=int(pnl.size),
        mean=float(pnl.mean()),
        median=float(np.median(pnl)),
        std=std,
        win_rate=float((pnl > 0).mean()),
        profit_factor=float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        skew=skew,
        excess_kurtosis=kurt,
        worst_day=float(pnl.min()),
        best_day=float(pnl.max()),
        var_5=float(np.percentile(pnl, 5)),
        cvar_5=float(tail_5.mean()) if tail_5.size else 0.0,
        cvar_1=float(tail_1.mean()) if tail_1.size else 0.0,
        days_to_erase=float(-pnl.min() / mean_win) if mean_win > 0 else float("inf"),
        traded_frac=float(traded.mean()),
        win_rate_traded=float((pnl[traded] > 0).mean()) if traded.any() else 0.0,
        mean_rolls=float(rolls.mean()),
        roll_day_frac=float((rolls > 0).mean()),
        max_peak_contracts=int(peak_contracts.max()),
        mean_credit=float(credit[traded].mean()) if traded.any() else 0.0,
        mean_peak_margin=float(peak_margin.mean()),
        mean_total_risk=float(total_risk.mean()),
    )

    returns = np.array([p.equity[-1] / start_equity - 1.0 for p in paths])
    drawdowns = np.array([_max_drawdown(p.equity) for p in paths])
    sharpes, sortinos = [], []
    for p in paths:
        d = p.daily_pnl / start_equity
        sd = d.std(ddof=1)
        sharpes.append(d.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
        downside = d[d < 0]
        dsd = downside.std(ddof=1) if downside.size > 1 else 0.0
        sortinos.append(d.mean() / dsd * np.sqrt(TRADING_DAYS) if dsd > 0 else 0.0)

    path_stats = PathStats(
        n_paths=len(paths),
        n_days=int(paths[0].daily_pnl.size) if paths else 0,
        median_return=float(np.median(returns)),
        mean_return=float(returns.mean()),
        p05_return=float(np.percentile(returns, 5)),
        p95_return=float(np.percentile(returns, 95)),
        p_losing_year=float((returns < 0).mean()),
        p_ruin=float(np.mean([p.ruined for p in paths])),
        median_max_drawdown=float(np.median(drawdowns)),
        worst_max_drawdown=float(drawdowns.max()),
        median_sharpe=float(np.median(sharpes)),
        median_sortino=float(np.median(sortinos)),
    )
    return Summary(label, days, path_stats)
