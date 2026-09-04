"""Entry filters: rules that skip a session entirely before any trade is placed.

The motivating idea is that the worst days are visible in advance -- unusually
heavy early volume, an outsized opening move -- so you can decline to trade
them.  This module makes that testable, and separates two very different
classes of rule:

**Realisable.** ``opening_move``, ``opening_range``, ``vwap_distance`` and
``vwap_stretch`` read only the price path up to the entry time.  They have no
free parameters: whatever predictive power they show is power the market model
actually contains, so their results are an honest read of what price action
alone buys you.

The two VWAP signals target a different failure mode from the rest.  A
volatility filter ranks days by how *far* they travel; the session that ruins a
rolled book is one that travels in a *straight line*, and a merely-above-average
vol day that walks one way slips through a vol filter untouched.  Distance from
the volume-weighted average price reads direction and path, so it sees exactly
that day.

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

import numpy as np

from .market import DaySurface

__all__ = ["FilterConfig", "signal_value", "should_skip", "vwap_series"]

FilterKind = Literal[
    "none",
    "opening_move",
    "opening_range",
    "vwap_distance",
    "vwap_stretch",
    "rvol",
    "oracle_vol",
]


def vwap_series(surface: DaySurface, upto: int) -> np.ndarray:
    """Running volume-weighted average price through step ``upto``.

    **This is a proxy, not VWAP.**  The simulator does not model volume, so the
    weights are the variance realised in each interval, taken from the variance
    clock.  Intraday volume and volatility are both U-shaped and move together,
    which makes that a reasonable stand-in -- but it is a stand-in, and a real
    VWAP on real tape will not match it exactly.

    ``vwap[0]`` is the open, since no interval has traded yet.
    """
    spot = surface.spot_path[: upto + 1]
    if upto < 1:
        return spot.copy()
    var_left = surface.var_left[: upto + 1]
    weight = np.maximum(var_left[:-1] - var_left[1:], 0.0)
    typical = 0.5 * (spot[:-1] + spot[1:])
    numerator = np.cumsum(typical * weight)
    denominator = np.cumsum(weight)
    out = np.empty(upto + 1)
    out[0] = spot[0]
    out[1:] = np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-300), spot[1:])
    return out


@dataclass(frozen=True)
class FilterConfig:
    kind: FilterKind = "none"

    threshold: float = float("inf")
    """Skip the session when the signal exceeds this.

    Units follow ``kind``: ``opening_move``, ``opening_range``,
    ``vwap_distance`` and ``vwap_stretch`` are in multiples of the implied daily
    move (so 0.5 means "already travelled half the day's expected move before we
    would have entered"); ``rvol`` and ``oracle_vol`` are in standard deviations
    of log realised vol.
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

    if cfg.kind in ("vwap_distance", "vwap_stretch"):
        spot = surface.spot_path[: entry_step + 1]
        vwap = vwap_series(surface, entry_step)
        scale = float(spot[0]) * surface.cfg.implied_daily_move
        if cfg.kind == "vwap_distance":
            # How stretched price is from the session average right now.
            return abs(float(spot[-1]) - float(vwap[-1])) / scale
        # Mean *signed* gap: large only when price has held one side of VWAP
        # all session.  A round trip cancels itself out, which is the point --
        # this separates a trend day from an equally violent whipsaw.
        return abs(float(np.mean(spot - vwap))) / scale

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
