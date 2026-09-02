"""Day-level and path-level simulation of a short 0DTE vertical program.

The day loop walks the variance clock in ``step_minutes`` increments, marking
every open spread against the simulated surface and applying, in order: cash
settlement, the flatten time, the portfolio stop, then each spread's own profit
target and stop loss.  A stopped spread hands control to the roll policy, which
is where the interesting behaviour lives.

Accounting convention: ``cash`` is realised dollars including commissions, so a
day's P&L is ``cash`` once every position is closed or settled.  Option prices
are in index points and become dollars via ``CostModel.contract_multiplier``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import AccountConfig, CostModel, StrategyConfig
from .market import DaySurface

__all__ = ["OpenSpread", "DayResult", "PathResult", "simulate_day", "simulate_path"]


@dataclass
class OpenSpread:
    """A short vertical: sold at ``short_strike``, protected at ``long_strike``."""

    is_call: bool
    short_strike: float
    long_strike: float
    contracts: int
    credit: float
    """Points per spread, net of entry slippage but before commissions."""
    opened_step: int
    take_profit: bool = True
    """False for replacement spreads held to settlement -- see
    ``StrategyConfig.roll_hold_to_expiry``."""

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    @property
    def risk_points(self) -> float:
        """Max loss per spread in points -- and, at Reg-T, its margin."""
        return self.width - self.credit


@dataclass
class DayResult:
    pnl: float = 0.0
    credit_collected: float = 0.0
    fees_paid: float = 0.0
    rolls: int = 0
    spreads_opened: int = 0
    contracts_opened: int = 0
    peak_margin: float = 0.0
    total_risk_opened: float = 0.0
    """Sum of the defined risk of every spread opened today.

    This is the day's true worst case, and with rolls it exceeds
    ``peak_margin``: a spread stopped at a partial loss frees its margin, but
    the dollars are already gone, and the replacement can still lose its own
    full width.  Sizing off peak margin instead of this number is how a rolled
    book turns out to be carrying several times the risk its margin suggests."""
    peak_contracts: int = 0
    """Largest number of contracts open at once (``contracts_opened`` is the
    running total across the day, which is larger once rolls are counted)."""
    max_adverse: float = 0.0
    """Worst intraday mark-to-market, in dollars (<= 0)."""
    exit_reason: str = "expiry"


@dataclass
class PathResult:
    daily_pnl: np.ndarray
    equity: np.ndarray
    """Equity after each day; ``equity[0]`` is the starting balance."""
    rolls: np.ndarray
    contracts: np.ndarray
    peak_contracts: np.ndarray
    total_risk: np.ndarray
    peak_margin: np.ndarray
    max_adverse: np.ndarray
    exit_reasons: list[str]
    ruined: bool
    ruin_day: int | None


def _round_to(value: float, increment: float) -> float:
    return round(value / increment) * increment


def simulate_day(
    surface: DaySurface,
    cfg: StrategyConfig,
    costs: CostModel,
    account: AccountConfig,
    equity: float,
) -> DayResult:
    """Simulate one session and return the day's outcome."""
    mult = costs.contract_multiplier
    n_steps = surface.n_steps
    mcfg = surface.cfg
    entry = mcfg.minute_to_index(cfg.entry_minute)
    flatten = n_steps if cfg.flatten_minute is None else mcfg.minute_to_index(cfg.flatten_minute)
    roll_delta = cfg.effective_roll_delta
    margin_cap = max(equity, 0.0) * account.max_margin_frac

    open_spreads: list[OpenSpread] = []
    res = DayResult()
    cash = 0.0

    def margin() -> float:
        return sum(s.risk_points * s.contracts for s in open_spreads) * mult

    def mid(spread: OpenSpread, step: int) -> float:
        """Mid price of the spread -- the debit to close it, before slippage."""
        return surface.price(spread.short_strike, step, spread.is_call) - surface.price(
            spread.long_strike, step, spread.is_call
        )

    def quote(is_call: bool, delta: float, step: int) -> tuple[float, float, float] | None:
        """Strikes and achievable credit for a new spread, or None if unsellable."""
        short_k = _round_to(surface.strike_for_delta(step, delta, is_call), cfg.strike_increment)
        long_k = short_k + cfg.width if is_call else short_k - cfg.width
        if long_k <= 0.0:
            return None
        raw = surface.price(short_k, step, is_call) - surface.price(long_k, step, is_call)
        credit = raw - 2.0 * costs.half_spread
        if credit < cfg.min_credit or credit >= cfg.width:
            return None
        return short_k, long_k, credit

    def open_spread(
        is_call: bool, delta: float, contracts: int, step: int, take_profit: bool = True
    ) -> OpenSpread | None:
        """Sell ``contracts`` spreads, trimming the size to fit margin and limits."""
        nonlocal cash
        if contracts < 1:
            return None
        q = quote(is_call, delta, step)
        if q is None:
            return None
        short_k, long_k, credit = q

        risk_dollars = (cfg.width - credit) * mult
        room = margin_cap - margin()
        contracts = min(contracts, int(room // risk_dollars) if risk_dollars > 0 else 0)
        contracts = min(contracts, cfg.max_total_contracts - res.contracts_opened)
        if contracts < 1:
            return None

        spread = OpenSpread(is_call, short_k, long_k, contracts, credit, step, take_profit)
        open_spreads.append(spread)
        fees = costs.trade_fees(contracts)
        cash += credit * contracts * mult - fees
        res.credit_collected += credit * contracts * mult
        res.fees_paid += fees
        res.spreads_opened += 1
        res.contracts_opened += contracts
        res.total_risk_opened += spread.risk_points * contracts * mult
        res.peak_contracts = max(res.peak_contracts, sum(s.contracts for s in open_spreads))
        res.peak_margin = max(res.peak_margin, margin())
        return spread

    def day_pnl(step: int) -> float:
        """Realised cash plus the mark on whatever is still open.

        The martingale sizes against this rather than ``cash`` alone.  With a
        spread still open on the untested side, its credit is already in
        ``cash`` but has not been earned yet, so ``-cash`` understates how far
        down the day actually is and the roll would come up short.
        """
        unrealised = sum(
            (s.credit - mid(s, step)) * s.contracts for s in open_spreads
        ) * mult
        return cash + unrealised

    def close_spread(spread: OpenSpread, step: int, debit: float) -> None:
        nonlocal cash
        fees = costs.trade_fees(spread.contracts)
        cash -= debit * spread.contracts * mult + fees
        res.fees_paid += fees
        open_spreads.remove(spread)

    def handle_stop(stopped: OpenSpread, step: int) -> None:
        """Apply the roll policy after ``stopped`` was closed at a loss."""
        if cfg.roll_policy == "none" or res.rolls >= cfg.max_rolls:
            return
        if cfg.roll_side == "same":
            targets = (stopped.is_call,)
        elif cfg.roll_side == "opposite":
            targets = (not stopped.is_call,)
        else:
            targets = (False, True)

        opened_any = False
        for is_call in targets:
            q = quote(is_call, roll_delta, step)
            if q is None:
                continue
            credit = q[2]
            if cfg.roll_policy == "same_size":
                want = stopped.contracts
            elif cfg.roll_policy == "martingale":
                # Size so that, if the replacements expire worthless, the day
                # finishes flat.  This is the doubling-down rule.
                need = max(0.0, -day_pnl(step)) / len(targets)
                want = math.ceil(need / (credit * mult)) if credit > 0 else 0
            else:  # equal_risk -- never take on more risk than we just closed
                budget = stopped.risk_points * stopped.contracts / len(targets)
                risk_new = cfg.width - credit
                want = int(budget // risk_new) if risk_new > 0 else 0
            if open_spread(
                is_call, roll_delta, want, step, not cfg.roll_hold_to_expiry
            ) is not None:
                opened_any = True
        if opened_any:
            res.rolls += 1

    # --- entry ---------------------------------------------------------------
    size = cfg.contracts
    if account.scale_with_equity and account.start_equity > 0:
        size = max(1, int(round(size * equity / account.start_equity)))
    for is_call in cfg.sides:
        open_spread(is_call, cfg.short_delta, size, entry)
    if not open_spreads:
        res.exit_reason = "no_trade"
        return res

    # --- session loop --------------------------------------------------------
    for step in range(entry + 1, n_steps + 1):
        if not open_spreads:
            break

        marks = [mid(s, step) for s in open_spreads]
        open_pnl = sum(
            (s.credit - d) * s.contracts for s, d in zip(open_spreads, marks)
        ) * mult
        res.max_adverse = min(res.max_adverse, cash + open_pnl)

        if step >= n_steps:
            # Cash settlement: intrinsic value, no slippage, no closing commission.
            for spread, value in zip(list(open_spreads), marks):
                cash -= value * spread.contracts * mult
                open_spreads.remove(spread)
            res.exit_reason = "expiry"
            break

        if step >= flatten:
            for spread, value in zip(list(open_spreads), marks):
                if value <= costs.min_close_value:
                    open_spreads.remove(spread)  # leave it to expire worthless
                else:
                    close_spread(spread, step, value + 2.0 * costs.half_spread)
            res.exit_reason = "flatten"
            break

        if cfg.daily_stop_dollars is not None and cash + open_pnl <= -cfg.daily_stop_dollars:
            for spread, value in zip(list(open_spreads), marks):
                close_spread(spread, step, value + 2.0 * costs.half_spread)
            res.exit_reason = "daily_stop"
            break

        for spread, value in list(zip(open_spreads, marks)):
            if (
                spread.take_profit
                and cfg.profit_target is not None
                and value <= spread.credit * (1.0 - cfg.profit_target)
            ):
                close_spread(spread, step, value + 2.0 * costs.half_spread)
                res.exit_reason = "profit_target"
            elif cfg.stop_loss_mult is not None and value >= spread.credit * cfg.stop_loss_mult:
                close_spread(spread, step, value + 2.0 * costs.half_spread)
                res.exit_reason = "stopped"
                handle_stop(spread, step)

    res.pnl = cash
    return res


def simulate_path(
    surfaces: list[DaySurface],
    cfg: StrategyConfig,
    costs: CostModel,
    account: AccountConfig,
) -> PathResult:
    """Run one account through a sequence of sessions."""
    n = len(surfaces)
    pnl = np.zeros(n)
    equity = np.empty(n + 1)
    equity[0] = account.start_equity
    rolls = np.zeros(n, dtype=int)
    contracts = np.zeros(n, dtype=int)
    peak_contracts = np.zeros(n, dtype=int)
    total_risk = np.zeros(n)
    peak_margin = np.zeros(n)
    max_adverse = np.zeros(n)
    reasons: list[str] = []
    ruin_level = account.start_equity * account.ruin_frac
    ruined = False
    ruin_day: int | None = None

    for i, surface in enumerate(surfaces):
        if ruined:
            equity[i + 1] = equity[i]
            reasons.append("ruined")
            continue
        day = simulate_day(surface, cfg, costs, account, equity[i])
        pnl[i] = day.pnl
        equity[i + 1] = equity[i] + day.pnl
        rolls[i] = day.rolls
        contracts[i] = day.contracts_opened
        peak_contracts[i] = day.peak_contracts
        total_risk[i] = day.total_risk_opened
        peak_margin[i] = day.peak_margin
        max_adverse[i] = day.max_adverse
        reasons.append(day.exit_reason)
        if equity[i + 1] <= ruin_level:
            ruined = True
            ruin_day = i

    return PathResult(
        pnl, equity, rolls, contracts, peak_contracts, total_risk, peak_margin, max_adverse,
        reasons, ruined, ruin_day,
    )
