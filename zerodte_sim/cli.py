"""Command line entry point: ``python -m zerodte_sim``."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace

import numpy as np

from .config import AccountConfig, CostModel, StrategyConfig
from .experiment import Experiment, default_variants
from .market import MarketConfig
from .metrics import summarise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zerodte_sim",
        description="Monte Carlo comparison of roll policies for short 0DTE verticals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    m = p.add_argument_group("market")
    m.add_argument("--spot", type=float, default=6000.0)
    m.add_argument("--implied-move", type=float, default=0.0080,
                   help="ATM implied one-day move, as a fraction of spot")
    m.add_argument("--vrp", type=float, default=0.78,
                   help="median realised/implied vol; 1.0 removes the seller's edge")
    m.add_argument("--vrp-sd", type=float, default=0.42)

    s = p.add_argument_group("strategy")
    s.add_argument("--side", choices=("put", "call", "both"), default="put")
    s.add_argument("--delta", type=float, default=0.10, help="short strike delta")
    s.add_argument("--width", type=float, default=25.0)
    s.add_argument("--contracts", type=int, default=2)
    s.add_argument("--entry-minute", type=int, default=15)
    s.add_argument("--flatten-minute", type=int, default=375)
    s.add_argument("--profit-target", type=float, default=0.50)
    s.add_argument("--stop-mult", type=float, default=2.0)
    s.add_argument("--max-rolls", type=int, default=2)

    c = p.add_argument_group("costs and account")
    c.add_argument("--half-spread", type=float, default=0.05, help="per leg, in option points")
    c.add_argument("--commission", type=float, default=0.65, help="per contract per leg")
    c.add_argument("--fees", type=float, default=0.35, help="per contract per leg")
    c.add_argument("--equity", type=float, default=100_000.0)
    c.add_argument("--max-margin-frac", type=float, default=0.50)

    r = p.add_argument_group("run")
    r.add_argument("--paths", type=int, default=300, help="independent accounts")
    r.add_argument("--days", type=int, default=252, help="sessions per account")
    r.add_argument("--seed", type=int, default=20240101)
    r.add_argument("--csv", default=None, help="write the summary table here")
    r.add_argument("--plot", default=None, help="write the four-panel dashboard here")
    r.add_argument("--sensitivity", action="store_true",
                   help="also sweep execution cost and variance premium")
    r.add_argument("--filter-study", action="store_true",
                   help="test whether an entry filter rescues the martingale")
    r.add_argument("--skip-rate", type=float, default=0.20,
                   help="fraction of sessions an entry filter declines")
    r.add_argument("--entry-study", action="store_true",
                   help="separate the economics of a later entry from the filter's gain")
    r.add_argument("--entry-minutes", default="15,30,60,90,120,180",
                   help="comma-separated entry times for --entry-study")
    r.add_argument("--entry-filter", default="opening_range",
                   help="filter kind used by --entry-study")
    return p


_COLUMNS = [
    # header, width, accessor, precision
    ("policy", 24, lambda s: s.label, None),
    ("$/day", 9, lambda s: s.days.mean, 2),
    ("win%", 6, lambda s: 100 * s.days.win_rate, 1),
    ("med yr%", 8, lambda s: 100 * s.paths.median_return, 1),
    ("p05 yr%", 8, lambda s: 100 * s.paths.p05_return, 1),
    ("worst day", 10, lambda s: s.days.worst_day, 0),
    ("CVaR1%", 9, lambda s: s.days.cvar_1, 0),
    ("erase", 7, lambda s: s.days.days_to_erase, 0),
    ("maxDD%", 7, lambda s: 100 * s.paths.median_max_drawdown, 1),
    ("ruin%", 6, lambda s: 100 * s.paths.p_ruin, 1),
    ("risk/day", 9, lambda s: s.days.mean_total_risk, 0),
]


def _fmt(summaries) -> str:
    header = "".join(
        f"{name:<{w}}" if prec is None else f"{name:>{w}}"
        for name, w, _, prec in _COLUMNS
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            "".join(
                f"{get(s):<{w}}" if prec is None else f"{get(s):>{w},.{prec}f}"
                for _, w, get, prec in _COLUMNS
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    market = MarketConfig(
        spot=args.spot,
        implied_daily_move=args.implied_move,
        vrp_median=args.vrp,
        vrp_log_sd=args.vrp_sd,
    )
    costs = CostModel(half_spread=args.half_spread, commission=args.commission, fees=args.fees)
    account = AccountConfig(start_equity=args.equity, max_margin_frac=args.max_margin_frac)
    base = StrategyConfig(
        side=args.side,
        short_delta=args.delta,
        width=args.width,
        contracts=args.contracts,
        entry_minute=args.entry_minute,
        flatten_minute=args.flatten_minute,
        profit_target=args.profit_target,
        stop_loss_mult=args.stop_mult,
        max_rolls=args.max_rolls,
    )

    print(f"market: implied daily move {100*market.implied_daily_move:.2f}%  "
          f"E[RV]/IV variance ratio {market.mean_variance_ratio:.3f}")
    print(f"trade:  {args.contracts}x {args.side} {args.delta:.2f}-delta {args.width:.0f}-wide, "
          f"stop {args.stop_mult}x credit, target {100*args.profit_target:.0f}%")
    print(f"costs:  {args.half_spread:.3f}/leg slippage, "
          f"${args.commission + args.fees:.2f}/contract/leg all-in\n")

    exp = Experiment(market, costs, account, args.paths, args.days, args.seed)
    variants = default_variants(base)

    def progress(done, total):
        pct = 100 * done / total
        sys.stderr.write(f"\r  simulating {done}/{total} accounts ({pct:.0f}%)")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")

    results = exp.run(variants, progress=progress)
    summaries = exp.summarise(results)
    print(_fmt(summaries))
    print("\n'erase' = worst day divided by the average winning day.")
    print("'risk/day' = mean cumulative defined risk opened per session.")

    baseline = "Hard stop, no roll"
    target = "Martingale, same side"
    base_pnl = np.concatenate([p.daily_pnl for p in results[baseline]])
    mart_pnl = np.concatenate([p.daily_pnl for p in results[target]])
    rolled = np.concatenate([p.rolls for p in results[target]]) > 0
    paired = mart_pnl[rolled] - base_pnl[rolled]
    print(f"\nOn the {rolled.sum():,} sessions where a roll fired "
          f"({100*rolled.mean():.1f}% of all sessions):")
    print(f"  median outcome  no-roll ${np.median(base_pnl[rolled]):>9,.0f}   "
          f"martingale ${np.median(mart_pnl[rolled]):>9,.0f}")
    print(f"  mean outcome    no-roll ${base_pnl[rolled].mean():>9,.0f}   "
          f"martingale ${mart_pnl[rolled].mean():>9,.0f}")
    print(f"  rolling rescued {100*(paired>100).mean():.1f}% of them, "
          f"worsened {100*(paired<-100).mean():.1f}%, worst case ${paired.min():,.0f}")

    if args.sensitivity:
        _sensitivity(market, costs, account, base, args)

    if args.filter_study:
        _filter_study(market, costs, account, base, args, progress)

    if args.entry_study:
        _entry_study(market, costs, account, base, args, progress)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            rows = [s.row() for s in summaries]
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.plot:
        from .plots import dashboard
        dashboard(summaries, results, account.start_equity, paired, args.plot)
        print(f"wrote {args.plot}")
    return 0


def _sensitivity(market, costs, account, base, args) -> None:
    """How much of the result is execution quality, and how much is real edge?"""
    print("\n" + "=" * 72)
    print("Sensitivity of the disciplined baseline (hard stop, no roll), $/day")
    print("=" * 72)
    exp = Experiment(market, costs, account, max(60, args.paths // 5), args.days, args.seed)
    stop = [v for v in default_variants(base) if v.label == "Hard stop, no roll"]

    print("\n  execution cost (half-spread per leg):")
    for hs in (0.00, 0.02, 0.05, 0.10):
        e = Experiment(market, replace(costs, half_spread=hs), account,
                       max(60, args.paths // 5), args.days, args.seed)
        s = e.summarise(e.run(stop))[0]
        print(f"    {hs:.2f}  ->  {s.days.mean:+8.2f}/day   median year {100*s.paths.median_return:+6.1f}%")

    print("\n  variance risk premium (median realised/implied vol):")
    for vrp in (0.70, 0.78, 0.90, 1.00):
        m = replace(market, vrp_median=vrp)
        e = Experiment(m, costs, account, max(60, args.paths // 5), args.days, args.seed)
        s = e.summarise(e.run(stop))[0]
        print(f"    {vrp:.2f}  ->  {s.days.mean:+8.2f}/day   "
              f"(E[RV]/IV variance ratio {m.mean_variance_ratio:.2f})")


if __name__ == "__main__":
    raise SystemExit(main())


_STUDY_COLUMNS = [
    ("policy", 24, lambda s: s.label, None),
    ("traded%", 9, lambda s: 100 * s.days.traded_frac, 1),
    ("$/day", 9, lambda s: s.days.mean, 2),
    ("win%(t)", 9, lambda s: 100 * s.days.win_rate_traded, 1),
    ("med yr%", 9, lambda s: 100 * s.paths.median_return, 1),
    ("p05 yr%", 9, lambda s: 100 * s.paths.p05_return, 1),
    ("worst day", 11, lambda s: s.days.worst_day, 0),
    ("CVaR1%", 10, lambda s: s.days.cvar_1, 0),
    ("maxDD%", 8, lambda s: 100 * s.paths.median_max_drawdown, 1),
]


def _study_table(summaries) -> str:
    header = "".join(
        f"{name:<{w}}" if prec is None else f"{name:>{w}}"
        for name, w, _, prec in _STUDY_COLUMNS
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            "".join(
                f"{get(s):<{w}}" if prec is None else f"{get(s):>{w},.{prec}f}"
                for _, w, get, prec in _STUDY_COLUMNS
            )
        )
    return "\n".join(lines)


def _filter_study(market, costs, account, base, args, progress) -> None:
    """Can skipping the days that look dangerous make a martingale safe?"""
    from dataclasses import replace

    from .config import FilterConfig
    from .filter_study import (
        FilterSpec,
        calibrate_threshold,
        run_filter_study,
        signal_power_sweep,
    )

    entry_step = market.minute_to_index(base.entry_minute)
    skip = args.skip_rate

    def spec(label, **kw):
        filt = FilterConfig(**kw)
        return FilterSpec(
            label,
            replace(filt, threshold=calibrate_threshold(market, filt, entry_step, skip)),
        )

    specs = [
        FilterSpec("no filter", FilterConfig()),
        spec("opening range (realisable)", kind="opening_range"),
        spec("VWAP distance (realisable)", kind="vwap_distance"),
        spec("VWAP stretch (realisable)", kind="vwap_stretch"),
        spec("rvol proxy, corr 0.60", kind="rvol", corr=0.60),
        spec("rvol proxy, corr 0.90", kind="rvol", corr=0.90),
        spec("perfect vol oracle", kind="oracle_vol"),
    ]

    print("\n" + "=" * 92)
    print(f"Entry filters, each declining {100*skip:.0f}% of sessions, applied to every policy")
    print("=" * 92)
    print("Thresholds calibrated on a separate sample. 'win%(t)' counts wins among *traded* days.")

    results = run_filter_study(
        market, costs, account, base, specs,
        max(80, args.paths // 3), args.days, args.seed, progress,
    )
    for label, summaries in results.items():
        print(f"\n  {label}")
        for line in _study_table(summaries).splitlines():
            print("    " + line)

    print("\n" + "=" * 92)
    print(f"How predictive must a volume signal be? (all at {100*skip:.0f}% skipped)")
    print("=" * 92)
    rows = signal_power_sweep(
        market, costs, account, base, [0.0, 0.3, 0.6, 0.9, 1.0], skip,
        entry_step, max(80, args.paths // 3), args.days, args.seed, progress,
    )
    labels = [s.label for s in rows[0][1]]
    header = f"{'corr':>6}" + "".join(f"{lab.split(',')[0]:>34}" for lab in labels)
    print(header)
    print(f"{'':>6}" + "".join(f"{'$/day':>12}{'worst day':>12}{'CVaR1%':>10}" for _ in labels))
    print("-" * len(header))
    for corr, summaries in rows:
        line = f"{corr:>6.2f}"
        for s in summaries:
            line += f"{s.days.mean:>12,.2f}{s.days.worst_day:>12,.0f}{s.days.cvar_1:>10,.0f}"
        print(line)


def _entry_study(market, costs, account, base, args, progress) -> None:
    """Is a later entry worth it, and is the gain the filter or the trade?"""
    from .entry_study import run_entry_study

    minutes = [int(m) for m in args.entry_minutes.split(",")]
    skip = args.skip_rate

    print("\n" + "=" * 100)
    print(f"Entry time study -- filter '{args.entry_filter}' declining {100*skip:.0f}% of sessions")
    print("=" * 100)
    print("'plain' enters at that time with no filter, so it isolates the economics of a later")
    print("entry. 'filtered' adds the filter. 'gain' is what the filter buys at that hour.")

    rows = run_entry_study(
        market, costs, account, base, minutes, skip, args.entry_filter,
        max(80, args.paths // 3), args.days, args.seed, progress,
    )

    labels = [s.label for s in rows[0].unfiltered]
    for i, label in enumerate(labels):
        print(f"\n  {label}")
        print(f"    {'entry':>7}{'credit':>9}{'plain $/day':>14}{'filtered $/day':>16}"
              f"{'gain':>9}{'filt med yr':>13}{'filt worst':>12}")
        print("    " + "-" * 80)
        for row in rows:
            plain, gated = row.unfiltered[i], row.filtered[i]
            print(f"    {row.minute:>5}m{plain.days.mean_credit:>9,.0f}"
                  f"{plain.days.mean:>14,.2f}{gated.days.mean:>16,.2f}"
                  f"{gated.days.mean - plain.days.mean:>9,.2f}"
                  f"{100*gated.paths.median_return:>12.1f}%{gated.days.worst_day:>12,.0f}")
