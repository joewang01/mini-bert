"""Entry filters: rules that skip a session entirely before any trade is placed.

The motivating idea is that the worst days are visible in advance -- unusually
heavy early volume, an outsized opening move -- so you can decline to trade
them.  This module makes that testable, and separates two very different
classes of rule:

**Realisable.** ``opening_move`` and ``opening_range`` read only the price path
up to the entry time.  They have no free parameters: whatever predictive power
they have is whatever the market model actually contains, so their results are
an honest read of what price action alone buys you.

**Parameterised.** ``rvol`` stands in for a relative-volume signal.  Volume is
not modelled explicitly; instead the signal is a noisy observation of the day's
realised vol whose ``corr`` you set. That correlation is doing all the work, and
it is the number to measure from your own fills rather than assume. Sweep it.

**Oracles.** ``oracle_vol`` reads the day's realised vol directly. It cannot be
traded -- it is the ``corr = 1`` bound that says how much room any volume-like
signal could possibly have.

A filter that helps is not thereby an argument for the strategy it is bolted
onto: a good filter improves *every* policy, so a filtered martingale must be
compared against a filtered disciplined rule, never against an unfiltered one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .market import DaySurface

__all__ = ["FilterConfig", "signal_value", "should_skip"]

FilterKind = Literal["none", "opening_move", "opening_range", "rvol", "oracle_vol"]


@dataclass(frozen=True)
class FilterConfig:
    kind: FilterKind = "none"

    threshold: float = float("inf")
    """Skip the session when the signal exceeds this.

    Units follow ``kind``: ``opening_move`` and ``opening_range`` are in
    multiples of the implied daily move (so 0.5 means "already travelled half
    the day's expected move before we would have entered"); ``rvol`` and
    ``oracle_vol`` are in standard deviations of log realised vol.
    """

    corr: float = 0.6
    """``rvol`` only: correlation between the signal and the day's realised
    vol.  1.0 degenerates to ``oracle_vol``; 0.0 is pure noise."""

    def __post_init__(self) -> None:
        if not -1.0 <= self.corr <= 1.0:
            raise ValueError("corr must be in [-1, 1]")


def signal_value(surface: DaySurface, entry_step: int, cfg: FilterConfig) -> float:
    """The filter's signal for this session, as of ``entry_step``."""
    if cfg.kind == "none":
        return float("-inf")

    if cfg.kind in ("opening_move", "opening_range"):
        window = surface.spot_path[: entry_step + 1]
        open_price = float(window[0])
        if cfg.kind == "opening_move":
            travelled = abs(float(window[-1]) / open_price - 1.0)
        else:
            travelled = (float(window.max()) - float(window.min())) / open_price
        return travelled / surface.cfg.implied_daily_move

    if cfg.kind == "oracle_vol":
        return surface.vol_z

    if cfg.kind == "rvol":
        return (
            cfg.corr * surface.vol_z
            + math.sqrt(max(0.0, 1.0 - cfg.corr ** 2)) * surface.signal_noise
        )

    raise ValueError(f"unknown filter kind: {cfg.kind!r}")


def should_skip(surface: DaySurface, entry_step: int, cfg: FilterConfig) -> bool:
    if cfg.kind == "none":
        return False
    return signal_value(surface, entry_step, cfg) > cfg.threshold
