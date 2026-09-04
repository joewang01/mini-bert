"""Run several strategies over the *same* simulated markets and compare them.

Common random numbers are the point of this module.  Each account path gets a
deterministic seed, every strategy sees that identical sequence of sessions, and
differences between strategies are therefore attributable to the rules rather
than to sampling luck.  Comparing policies on independently drawn markets would
need far more paths to say anything at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import AccountConfig, CostModel, StrategyConfig
from .engine import PathResult, simulate_path
from .market import MarketConfig, MarketSimulator
from .metrics import Summary, summarise

__all__ = ["Experiment", "Variant", "DEFAULT_VARIANTS"]


@dataclass(frozen=True)
class Variant:
    label: str
    config: StrategyConfig


def default_variants(base: StrategyConfig) -> list[Variant]:
    """The five policies the review turns on, all sharing one base trade."""
    return [
        Variant(
            "Hard stop, no roll",
            replace(base, roll_policy="none"),
        ),
        Variant(
            "No stop, hold to expiry",
            replace(base, roll_policy="none", stop_loss_mult=None),
        ),
        Variant(
            "Re-enter, same size",
            replace(base, roll_policy="same_size", roll_side="same"),
        ),
        Variant(
            "Equal-risk roll",
            replace(base, roll_policy="equal_risk", roll_side="same"),
        ),
        Variant(
            "Martingale, same side",
            replace(base, roll_policy="martingale", roll_side="same"),
        ),
        Variant(
            "Martingale, both sides",
            replace(base, roll_policy="martingale", roll_side="both"),
        ),
    ]


DEFAULT_VARIANTS = default_variants


class Experiment:
    """Compare strategy variants on a shared set of simulated market paths."""

    def __init__(
        self,
        market: MarketConfig,
        costs: CostModel,
        account: AccountConfig,
        n_paths: int = 300,
        days_per_path: int = 252,
        seed: int = 20240101,
    ) -> None:
        self.market = market
        self.costs = costs
        self.account = account
        self.n_paths = n_paths
        self.days_per_path = days_per_path
        self.seed = seed

    def _path_seed(self, index: int) -> int:
        return self.seed + index * 7919  # any fixed stride; primes avoid aliasing

    def run(self, variants: list[Variant], progress=None) -> dict[str, list[PathResult]]:
        """Simulate every variant over identical markets.

        Markets are regenerated per path rather than held in memory all at once;
        the seed makes that reproducible and keeps peak memory flat in
        ``n_paths``.
        """
        results: dict[str, list[PathResult]] = {v.label: [] for v in variants}
        for i in range(self.n_paths):
            rng = np.random.default_rng(self._path_seed(i))
            surfaces = MarketSimulator(self.market, rng).generate(self.days_per_path)
            for v in variants:
                results[v.label].append(
                    simulate_path(surfaces, v.config, self.costs, self.account)
                )
            if progress is not None:
                progress(i + 1, self.n_paths)
        return results

    def summarise(self, results: dict[str, list[PathResult]]) -> list[Summary]:
        return [
            summarise(label, paths, self.account.start_equity)
            for label, paths in results.items()
        ]
