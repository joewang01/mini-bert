"""Does an entry filter rescue the martingale?

The trap this module exists to avoid: a filter that works improves *every*
policy, so measuring a filtered martingale against an unfiltered baseline
credits the filter's gains to the roll rule.  Every comparison here therefore
holds the filter fixed and varies only the response to a losing side.

Thresholds are calibrated on a separate sample of days to a target skip rate,
so different signals are compared at equal selectivity rather than at whatever
threshold happens to flatter each one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import AccountConfig, CostModel, FilterConfig, StrategyConfig
from .experiment import Experiment, Variant
from .market import MarketConfig, MarketSimulator
from .metrics import Summary

__all__ = ["calibrate_threshold", "FilterSpec", "run_filter_study", "signal_power_sweep"]


def calibrate_threshold(
    market: MarketConfig,
    filt: FilterConfig,
    entry_step: int,
    target_skip: float,
    n_days: int = 40_000,
    seed: int = 987654,
) -> float:
    """Threshold at which ``filt`` declines ``target_skip`` of sessions.

    Calibrated on its own sample and seed, so the threshold is not fitted to
    the days it is later evaluated on.
    """
    from .filters import signal_value

    days = MarketSimulator(market, np.random.default_rng(seed)).generate(n_days)
    signals = np.array([signal_value(d, entry_step, filt) for d in days])
    return float(np.percentile(signals, 100.0 * (1.0 - target_skip)))


@dataclass(frozen=True)
class FilterSpec:
    label: str
    config: FilterConfig


def study_variants(base: StrategyConfig) -> list[Variant]:
    """The three responses worth separating once a filter is in play."""
    return [
        Variant("Hard stop, no roll", replace(base, roll_policy="none")),
        Variant("Equal-risk roll", replace(base, roll_policy="equal_risk", roll_side="same")),
        Variant("Martingale, same side", replace(base, roll_policy="martingale", roll_side="same")),
    ]


def run_filter_study(
    market: MarketConfig,
    costs: CostModel,
    account: AccountConfig,
    base: StrategyConfig,
    specs: list[FilterSpec],
    n_paths: int,
    days_per_path: int,
    seed: int,
    progress=None,
) -> dict[str, list[Summary]]:
    """Every policy under every filter, all on the same simulated markets."""
    out: dict[str, list[Summary]] = {}
    for spec in specs:
        filtered_base = replace(base, entry_filter=spec.config)
        exp = Experiment(market, costs, account, n_paths, days_per_path, seed)
        results = exp.run(study_variants(filtered_base), progress=progress)
        out[spec.label] = exp.summarise(results)
    return out


def signal_power_sweep(
    market: MarketConfig,
    costs: CostModel,
    account: AccountConfig,
    base: StrategyConfig,
    correlations: list[float],
    target_skip: float,
    entry_step: int,
    n_paths: int,
    days_per_path: int,
    seed: int,
    progress=None,
) -> list[tuple[float, list[Summary]]]:
    """How predictive must a volume-like signal be to matter?

    Returns ``(corr, summaries)`` per correlation, with every policy measured at
    the same skip rate on the same markets, so the answer is read as a gap
    between policies rather than as an improvement over doing nothing.
    """
    rows = []
    for corr in correlations:
        filt = FilterConfig(kind="rvol", corr=corr)
        filt = replace(filt, threshold=calibrate_threshold(market, filt, entry_step, target_skip))
        cfg = replace(base, entry_filter=filt)
        exp = Experiment(market, costs, account, n_paths, days_per_path, seed)
        rows.append((corr, exp.summarise(exp.run(study_variants(cfg), progress=progress))))
    return rows
