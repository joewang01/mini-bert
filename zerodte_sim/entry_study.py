"""Does entering later help, and if so, is it the filter or the trade?

Signal power rises sharply with how late in the session it is measured, which
suggests delaying entry.  But a later entry also changes the trade itself: less
premium, less time to expiry, a tighter absolute stop on a smaller credit.  Read
naively, a later-entry run conflates the two.

This module separates them by running each entry time twice -- once unfiltered,
once filtered -- on identical markets.  The unfiltered row is the pure economics
of entering later; the gap between the rows is what the filter contributes at
that hour.  Filter thresholds are recalibrated per entry time, since the signal's
distribution shifts as the session accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import AccountConfig, CostModel, FilterConfig, StrategyConfig
from .experiment import Experiment
from .filter_study import calibrate_threshold, study_variants
from .market import MarketConfig
from .metrics import Summary

__all__ = ["EntryRow", "run_entry_study"]


@dataclass(frozen=True)
class EntryRow:
    minute: int
    unfiltered: list[Summary]
    filtered: list[Summary]


def run_entry_study(
    market: MarketConfig,
    costs: CostModel,
    account: AccountConfig,
    base: StrategyConfig,
    entry_minutes: list[int],
    skip_rate: float,
    filter_kind: str,
    n_paths: int,
    days_per_path: int,
    seed: int,
    progress=None,
) -> list[EntryRow]:
    rows: list[EntryRow] = []
    for minute in entry_minutes:
        at_minute = replace(base, entry_minute=minute)
        step = market.minute_to_index(minute)

        plain = Experiment(market, costs, account, n_paths, days_per_path, seed)
        unfiltered = plain.summarise(plain.run(study_variants(at_minute), progress=progress))

        base_filter = FilterConfig(kind=filter_kind)
        tuned = replace(
            base_filter,
            threshold=calibrate_threshold(market, base_filter, step, skip_rate),
        )
        gated = Experiment(market, costs, account, n_paths, days_per_path, seed)
        filtered = gated.summarise(
            gated.run(study_variants(replace(at_minute, entry_filter=tuned)), progress=progress)
        )

        rows.append(EntryRow(minute, unfiltered, filtered))
    return rows
