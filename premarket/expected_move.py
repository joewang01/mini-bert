"""Turn a pre-open quote into the numbers that decide a 0DTE short vertical.

The morning question is never "will the market go up or down".  It is whether
today's *realised* travel will come in under what the options are charging for
it.  :mod:`zerodte_sim` measured where that line sits: swept unfiltered across
variance-premium regimes, the strategy crosses zero at a median realised/implied
vol ratio of **0.787** (mean variance ratio 0.913).  Above that line costs eat
the credit and no roll rule in the study rescues it.

So this module takes what you can see at 09:15 -- spot, VIX or the 0DTE ATM
straddle -- and prints the travel budget that ratio implies, plus the strikes,
credit and touch probability that follow from it.

Usage::

    python -m premarket.expected_move --spot 7707 --vix 15.3
    python -m premarket.expected_move --spot 7707 --straddle 52 --entry-minute 120

Both vol inputs are supported because they answer different questions.  VIX is
a 30-day number and systematically overstates a quiet 0DTE session, so the
``--vix-to-0dte`` haircut (default 0.70) maps it onto the day.  If you can read
the actual ATM straddle, pass ``--straddle`` and the haircut is not used at all.
"""

from __future__ import annotations

import argparse
import math

from zerodte_sim.market import MarketConfig
from zerodte_sim.pricing import bs_price, strike_for_delta

# Measured in zerodte_sim: the median realised/implied vol ratio at which the
# strategy crosses zero, unfiltered, after costs.  See zerodte_sim/README.md.
BREAK_EVEN_VOL_RATIO = 0.787

TRADING_DAYS = 252
# ATM straddle = S * sqrt(2v/pi) at zero rates; invert for v.
_ATM_STRADDLE_COEF = math.sqrt(2.0 / math.pi)


def daily_vol_from_vix(vix: float, haircut: float) -> float:
    """VIX (annualised %, 30-day) -> a one-session vol fraction.

    The haircut is not a fudge: VIX prices a month of event risk, while a 0DTE
    session with no catalyst on the calendar routinely realises well under the
    square-root-of-time scaling.  0.70 is the middle of the usual 0.65-0.75
    band; check it against the live straddle whenever you can.
    """
    return haircut * (vix / 100.0) / math.sqrt(TRADING_DAYS)


def total_var_from_straddle(straddle: float, spot: float) -> float:
    """Total variance implied by an ATM straddle price."""
    return (straddle / (spot * _ATM_STRADDLE_COEF)) ** 2


def variance_fraction_remaining(entry_minute: int, cfg: MarketConfig) -> float:
    """Share of the session's variance still ahead at ``entry_minute``.

    Read off the same U-shaped variance clock the simulator walks, so a brief
    written here and a run of the engine agree about what a 10:30 entry means.
    Waiting out the open is worth more than the clock-time fraction suggests:
    the first half hour carries a disproportionate share of the day's variance.
    """
    clock = cfg.variance_clock()
    return float(1.0 - clock[cfg.minute_to_index(entry_minute)])


def summarise(
    spot: float,
    total_var: float,
    deltas: tuple[float, ...],
    width: float,
    contracts: int,
    multiplier: float = 100.0,
) -> list[dict[str, float]]:
    """Strike, credit and risk for a short vertical at each delta."""
    rows = []
    for delta in deltas:
        short_k = strike_for_delta(spot, total_var, delta, is_call=False)
        long_k = short_k - width
        credit = bs_price(spot, short_k, total_var, False) - bs_price(
            spot, long_k, total_var, False
        )
        risk = width - credit
        rows.append(
            {
                "delta": delta,
                "short_strike": short_k,
                "distance_pct": 100.0 * (spot - short_k) / spot,
                "credit": credit * multiplier * contracts,
                "max_loss": risk * multiplier * contracts,
                # Driftless reflection principle: touching is about twice as
                # likely as finishing through.  The number that matters for a
                # roll rule is the touch, not the expiry.
                "touch_prob": min(2.0 * delta, 1.0),
                "break_even_win_rate": risk / width,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spot", type=float, required=True, help="index level")
    vol = p.add_mutually_exclusive_group(required=True)
    vol.add_argument("--vix", type=float, help="VIX level, e.g. 15.3")
    vol.add_argument("--straddle", type=float, help="live 0DTE ATM straddle, in points")
    p.add_argument("--vix-to-0dte", type=float, default=0.70, help="VIX haircut (0.65-0.75)")
    p.add_argument("--entry-minute", type=int, default=0, help="minutes after 09:30")
    p.add_argument("--width", type=float, default=25.0, help="vertical width in points")
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument(
        "--deltas",
        default="0.10,0.15,0.20",
        help="comma-separated short-leg deltas",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MarketConfig()
    remaining = variance_fraction_remaining(args.entry_minute, cfg)

    if args.straddle is not None:
        # A live straddle already prices only the variance still to come.
        total_var = total_var_from_straddle(args.straddle, args.spot)
        source = f"ATM straddle {args.straddle:.1f}"
    else:
        full_day = daily_vol_from_vix(args.vix, args.vix_to_0dte) ** 2
        total_var = full_day * remaining
        source = f"VIX {args.vix:.2f} x {args.vix_to_0dte:.2f}"

    implied_move = args.spot * math.sqrt(total_var)
    # What a screen calls "the expected move" is usually the straddle price,
    # which is the mean absolute move -- sqrt(2/pi) = 0.80 of one sigma, not
    # one sigma.  Print both so the brief and the platform agree.
    mean_abs_move = _ATM_STRADDLE_COEF * implied_move
    budget = BREAK_EVEN_VOL_RATIO * implied_move

    print(f"spot {args.spot:,.0f}   entry +{args.entry_minute}m   vol from {source}")
    if args.entry_minute:
        print(f"variance still ahead: {100.0 * remaining:.0f}% of the session")
    print()
    print(f"  implied move (1 sigma)   {implied_move:7.1f} pts   {100.0 * implied_move / args.spot:5.2f}%")
    print(f"  ATM straddle / exp. move {mean_abs_move:7.1f} pts   {100.0 * mean_abs_move / args.spot:5.2f}%")
    print(f"  break-even travel budget {budget:7.1f} pts   {100.0 * budget / args.spot:5.2f}%")
    print(f"    -> realise more than that and the day is a loser on average")
    print()
    header = f"  {'delta':>6} {'short K':>9} {'dist':>7} {'credit':>9} {'max loss':>9} {'P(touch)':>9} {'BE win%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    deltas = tuple(float(d) for d in args.deltas.split(","))
    for row in summarise(args.spot, total_var, deltas, args.width, args.contracts):
        print(
            f"  {row['delta']:6.2f} {row['short_strike']:9,.0f} "
            f"{row['distance_pct']:6.2f}% {row['credit']:9,.0f} {row['max_loss']:9,.0f} "
            f"{100.0 * row['touch_prob']:8.0f}% {100.0 * row['break_even_win_rate']:7.0f}%"
        )
    print()
    print("  P(touch) is how often you get to make the roll decision.")
    print("  Answer it before the open, not while it is happening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
