"""Charts for the roll-policy comparison.

Colour follows the *policy*, consistently across every panel, so a reader can
track one strategy from the equity curve to the tail chart without re-reading a
legend.  Every series is direct-labelled: three of the six hues sit below 3:1
against the surface, which obliges visible labels rather than colour alone.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

__all__ = ["SERIES_COLORS", "dashboard"]

# Categorical slots 1-6, in fixed order.  Never cycled, never reassigned by rank.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

_DOLLARS = FuncFormatter(lambda v, _: f"${v:,.0f}")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    return ax


def _declutter(values, min_gap):
    """Nudge nearly-coincident label positions apart, preserving their order.

    Several policies finish within a few dollars of each other, which stacks
    their direct labels into an unreadable pile.  Direct labels are not optional
    here -- three of the six hues sit below 3:1 against the surface -- so the
    labels move instead.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for rank in range(1, len(order)):
        lo, hi = order[rank - 1], order[rank]
        if out[hi] - out[lo] < min_gap:
            out[hi] = out[lo] + min_gap
    return out


def _short(label: str) -> str:
    return {
        "Hard stop, no roll": "Hard stop",
        "No stop, hold to expiry": "No stop",
        "Re-enter, same size": "Same size",
        "Equal-risk roll": "Equal risk",
        "Martingale, same side": "Martingale",
        "Martingale, both sides": "Mart. both",
    }.get(label, label)


def dashboard(summaries, results, start_equity, paired, out_path):
    """Render the four-panel comparison.

    ``paired`` is the per-day difference (martingale minus no-roll) restricted
    to days a roll actually happened -- the panel that makes the asymmetry
    legible rather than merely tabulated.
    """
    labels = [s.label for s in summaries]
    colors = {lab: SERIES_COLORS[i % len(SERIES_COLORS)] for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.83, top=0.875, bottom=0.07, hspace=0.36, wspace=0.32)

    # --- 1. median equity curve --------------------------------------------
    ax = _style(axes[0, 0])
    medians = {lab: np.median(np.array([p.equity for p in results[lab]]), axis=0) for lab in labels}
    for lab in labels:
        ax.plot(medians[lab], color=colors[lab], linewidth=2.0, solid_capstyle="round")
    ends = [medians[lab][-1] for lab in labels]
    span = max(max(m.max() for m in medians.values()) - min(m.min() for m in medians.values()), 1.0)
    for lab, y_label in zip(labels, _declutter(ends, 0.045 * span)):
        ax.annotate(
            _short(lab),
            xy=(len(medians[lab]) - 1, y_label),
            xytext=(8, 0),
            textcoords="offset points",
            color=colors[lab],
            fontsize=8.5,
            va="center",
            fontweight="medium",
        )
    ax.axhline(start_equity, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.yaxis.set_major_formatter(_DOLLARS)
    ax.set_xlabel("trading day", color=INK_2, fontsize=9)
    ax.set_title("Median equity path", color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=10)

    # --- 2. tail: worst day and 1% CVaR ------------------------------------
    ax = _style(axes[0, 1])
    order = np.argsort([s.days.worst_day for s in summaries])
    y = np.arange(len(order))
    for rank, idx in enumerate(order):
        s = summaries[idx]
        ax.barh(rank, s.days.worst_day, height=0.52, color=colors[s.label], zorder=3)
        ax.plot(
            [s.days.cvar_1], [rank], marker="D", markersize=6.0,
            color=SURFACE, markeredgecolor=colors[s.label], markeredgewidth=1.8, zorder=4,
        )
        ax.annotate(
            f"${s.days.worst_day:,.0f}",
            xy=(s.days.worst_day, rank), xytext=(-7, 0), textcoords="offset points",
            ha="right", va="center", fontsize=8.5, color=INK_2,
        )
    ax.set_yticks(y, [_short(summaries[i].label) for i in order], color=INK_2, fontsize=9)
    # Headroom so the longest bar's value label does not run into the tick labels.
    ax.set_xlim(min(s.days.worst_day for s in summaries) * 1.22, 0.0)
    ax.xaxis.set_major_formatter(_DOLLARS)
    ax.set_title(
        "Worst single day  (bar)  ·  mean of worst 1% of days  (diamond)",
        color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=10,
    )

    # --- 3. what the roll actually does ------------------------------------
    ax = _style(axes[1, 0])
    clip = np.clip(paired, np.percentile(paired, 0.5), np.percentile(paired, 99.5))
    ax.hist(clip, bins=70, color=SERIES_COLORS[0], alpha=0.85, zorder=3)
    ax.axvline(0.0, color=AXIS, linewidth=1.2)
    rescued = float((paired > 100).mean()) * 100
    worsened = float((paired < -100).mean()) * 100
    ax.annotate(
        f"rescued the day\n{rescued:.0f}% of rolls",
        xy=(0.72, 0.80), xycoords="axes fraction", fontsize=9, color=INK_2, ha="center",
    )
    ax.annotate(
        f"made it worse\n{worsened:.0f}% of rolls\nworst: ${paired.min():,.0f}",
        xy=(0.24, 0.80), xycoords="axes fraction", fontsize=9, color=INK_2, ha="center",
    )
    ax.xaxis.set_major_formatter(_DOLLARS)
    ax.set_xlabel("martingale P&L minus no-roll P&L, same day", color=INK_2, fontsize=9)
    ax.set_ylabel("days", color=INK_2, fontsize=9)
    ax.set_title(
        "Rolling on days it was triggered  (tails clipped at 0.5 / 99.5 pct)",
        color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=10,
    )

    # --- 4. the trade-off ---------------------------------------------------
    ax = _style(axes[1, 1])
    xs = [s.paths.median_return * 100 for s in summaries]
    ys = [s.days.cvar_1 for s in summaries]
    x_span = max(max(xs) - min(xs), 1e-9)
    y_span = max(max(ys) - min(ys), 1e-9)
    for i, s in enumerate(summaries):
        ax.scatter([xs[i]], [ys[i]], s=110, color=colors[s.label], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        # Drop the label below the marker when another point sits just above.
        crowded = any(
            j != i
            and abs(xs[j] - xs[i]) < 0.12 * x_span
            and 0 < ys[j] - ys[i] < 0.12 * y_span
            for j in range(len(summaries))
        )
        ax.annotate(
            _short(s.label), xy=(xs[i], ys[i]),
            xytext=(0, -18 if crowded else 11), textcoords="offset points",
            ha="center", fontsize=8.5, color=colors[s.label], fontweight="medium",
        )
    ax.axvline(0.0, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.yaxis.set_major_formatter(_DOLLARS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("median annual return", color=INK_2, fontsize=9)
    ax.set_ylabel("mean of worst 1% of days", color=INK_2, fontsize=9)
    ax.set_title(
        "Return against tail risk  (up and to the right is better)",
        color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=10,
    )

    fig.suptitle(
        "Short 0DTE verticals: what the roll rule does to the tail",
        x=0.07, y=0.965, ha="left", color=INK, fontsize=15, fontweight="semibold",
    )
    fig.text(
        0.07, 0.925,
        f"{summaries[0].paths.n_paths} independent accounts × {summaries[0].paths.n_days} sessions, "
        "identical simulated markets across every policy",
        ha="left", color=MUTED, fontsize=9.5,
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
