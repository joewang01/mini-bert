"""Configuration objects for the strategy, its costs, and the account."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["CostModel", "StrategyConfig", "AccountConfig", "Side", "RollPolicy", "RollSide"]

Side = Literal["put", "call", "both"]
RollPolicy = Literal["none", "same_size", "martingale", "equal_risk"]
RollSide = Literal["same", "opposite", "both"]


@dataclass
class CostModel:
    """Execution costs.  These are not a footnote -- rolling multiplies them.

    Defaults are for SPX 0DTE at a retail broker and are, if anything, mildly
    optimistic: a two-leg spread costs ``2 * half_spread`` in slippage to open
    and the same to close, so a round trip on a $2.50 credit gives up about
    8% of the credit before commissions.
    """

    half_spread: float = 0.05
    """Half the bid/ask, per leg, in option points."""

    commission: float = 0.65
    fees: float = 0.35
    """Per contract, per leg."""

    min_close_value: float = 0.05
    """Spreads marked below this at the flatten time are left to expire rather
    than paying two legs of slippage to buy back a nickel."""

    contract_multiplier: float = 100.0

    def trade_fees(self, contracts: int, legs: int = 2) -> float:
        """Dollar commissions + fees for trading ``contracts`` of an N-leg spread."""
        return (self.commission + self.fees) * contracts * legs


@dataclass
class StrategyConfig:
    """A short-vertical 0DTE program, including how it reacts to a loser."""

    # --- the initial trade -------------------------------------------------
    side: Side = "put"
    entry_minute: int = 15
    short_delta: float = 0.10
    width: float = 25.0
    contracts: int = 1
    strike_increment: float = 5.0

    # --- management --------------------------------------------------------
    profit_target: float | None = 0.50
    """Close a spread once it has captured this fraction of its credit."""

    stop_loss_mult: float | None = 2.0
    """Close a spread once its cost to close reaches this multiple of the
    credit received.  ``None`` holds every spread to expiry."""

    flatten_minute: int | None = 375
    """Close anything still open at this minute (375 = 15:45 ET).  ``None``
    holds to cash settlement."""

    daily_stop_dollars: float | None = None
    """Hard per-day loss limit across all positions.  ``None`` disables it."""

    # --- what happens after a stop ----------------------------------------
    roll_policy: RollPolicy = "none"
    """How the program responds when a spread is stopped out.

    ``none``        stop trading that side for the day (the disciplined base case)
    ``same_size``   re-enter further out at the *original* contract count
    ``martingale``  size the new spreads to recover the day's realised loss
    ``equal_risk``  re-enter with a max loss no greater than what was just closed
    """

    roll_side: RollSide = "same"
    """Which side the replacement spreads are sold on: the tested side, the
    untested side (turning the position into a condor), or split across both."""

    roll_delta: float | None = None
    """Short delta for replacement spreads; defaults to ``short_delta``."""

    roll_hold_to_expiry: bool = True
    """Exempt replacement spreads from ``profit_target``.

    This matters for a fair comparison.  A roll is sized to recover the day's
    loss out of the *full* credit; if the replacement is then closed at 50% of
    that credit it recovers only half by construction, and the policy looks bad
    for a reason that is really a bookkeeping mismatch.  Traders who roll to
    "make the day back" hold the roll to settlement, so that is the default.
    The stop loss still applies -- that is what triggers the next roll."""

    max_rolls: int = 2
    min_credit: float = 0.20
    """Refuse to sell a spread for less than this.  Late in the day there is
    simply no premium left, and this is what ends the martingale."""

    max_total_contracts: int = 200
    """Cap on contracts opened *cumulatively* over a session, across all rolls.
    A backstop only -- ``AccountConfig.max_margin_frac`` is what normally
    binds."""

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if not 0.0 < self.short_delta < 0.5:
            raise ValueError("short_delta must be in (0, 0.5)")
        if self.contracts < 1:
            raise ValueError("contracts must be >= 1")
        if abs(self.width / self.strike_increment - round(self.width / self.strike_increment)) > 1e-9:
            raise ValueError("width must be a multiple of strike_increment")

    @property
    def effective_roll_delta(self) -> float:
        return self.short_delta if self.roll_delta is None else self.roll_delta

    @property
    def sides(self) -> tuple[bool, ...]:
        """Initial legs as ``is_call`` flags."""
        if self.side == "put":
            return (False,)
        if self.side == "call":
            return (True,)
        return (False, True)


@dataclass
class AccountConfig:
    start_equity: float = 100_000.0

    max_margin_frac: float = 0.50
    """Cap on total spread margin as a fraction of equity.  Reg-T margin on a
    short vertical equals its max loss, so this is literally the fraction of
    the account that can be lost in one session.  It is also what eventually
    stops a martingale -- and the reason a martingale that is *not* capped is a
    different strategy entirely."""

    ruin_frac: float = 0.50
    """Equity below this fraction of the starting balance ends the path."""

    scale_with_equity: bool = False
    """If set, contract counts scale with equity instead of staying fixed."""
