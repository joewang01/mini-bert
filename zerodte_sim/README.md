# 0DTE short-vertical roll simulator

A Monte Carlo model of selling 0DTE vertical credit spreads and, in particular,
of what happens when you respond to a losing side by opening **more** verticals.

The question it exists to answer: rolling a tested side raises your win rate and
smooths your equity curve, so it feels like risk management. Is it?

Requires `numpy` (and `matplotlib` for `--plot`).

```
python -m zerodte_sim --paths 400 --days 252 --sensitivity \
    --csv results/summary.csv --plot results/dashboard.png
```

`results/run.txt`, `results/summary.csv` and `results/dashboard.png` in this
repo are the output of exactly that command.

## What it models

**Price path.** Each session is 390 minutes walked on a **variance clock** --
the fraction of the day's variance realised by each moment -- which is U-shaped,
so the open and the last hour carry far more variance per minute than midday.
The same clock drives both the underlying's motion and the options' decay, so
they stay consistent with each other.

**Where the day's move comes from.** Realised daily vol is drawn as a multiple
of what was implied at the open. The multiplier's median sits below 1 -- that is
the seller's edge -- but it is lognormal, so it has a long right tail. A fraction
`theta` of the day's variance is then delivered as one-directional **drift** and
the rest as noise. Total variance is identical either way, but a trend day walks
through your short strike and keeps going while a chop day with the same
realised vol never touches it. That distinction is what makes rolling
path-dependent rather than merely size-dependent. Poisson intraday jumps supply
the rest of the tail.

**Vol surface.** A put skew (IV rising in OTM-put moneyness) plus a **leverage
effect**: implied vol expands when the market falls. This cuts both ways for a
roller and both directions are modelled -- a losing put spread costs more to
close than its realised move alone implies, *and* the replacement spreads sold
after the drop pay a richer credit.

**Execution.** Half-spread slippage per leg on entry and exit, plus commissions
and fees per contract per leg. Cash settlement at expiry costs nothing; spreads
marked below `min_close_value` at the flatten time are left to expire.

**Account.** Reg-T margin on a short vertical equals its max loss. Total margin
is capped at `max_margin_frac` of equity, which is what eventually stops a
martingale -- an uncapped one is a different strategy and a worse one.

## The policies it compares

All six share one base trade, so differences are attributable to the rules:

| policy | on a stop |
|---|---|
| `Hard stop, no roll` | stop trading that side for the day |
| `No stop, hold to expiry` | no stop at all |
| `Re-enter, same size` | re-enter further out at the original size |
| `Equal-risk roll` | re-enter with max loss no greater than what was closed |
| `Martingale, same side` | size replacements to recover the day's realised loss |
| `Martingale, both sides` | same, split across put and call side |

Every policy is run over **identical simulated markets** (common random numbers,
one deterministic seed per account path), so a comparison needs far fewer paths
to say anything than independently drawn markets would.

## Reading the output

`win%`, `$/day` and Sharpe flatter this family of strategies: a rule that turns
many small losses into a few enormous ones improves all three while making the
strategy worse. The columns that discriminate are the tail ones:

- **`worst day`** and **`CVaR1%`** -- the single worst session, and the mean of
  the worst 1% of them.
- **`erase`** -- how many average winning days one worst-case day wipes out.
- **`risk/day`** -- mean *cumulative* defined risk opened per session. This is
  the number to size against. It diverges from peak margin exactly to the extent
  a policy rolls: a spread stopped at a partial loss frees its margin, but the
  dollars are already gone and the replacement can still lose its own full
  width. Sizing off peak margin is how a rolled book turns out to be carrying
  several times the risk its margin suggests.
- **`ruin%`** -- fraction of accounts that broke `ruin_frac` of starting equity.

## Calibrating it to your own trading

The defaults describe a 2-lot 10-delta 25-wide SPX put spread on $100k. Points
worth re-pointing at your own numbers, roughly in order of how much they matter:

1. **`--vrp`** (median realised/implied vol). This *is* the edge. Check
   `MarketConfig.mean_variance_ratio` after changing it -- because the multiplier
   is lognormal, raising `--vrp-sd` inflates mean realised variance even with the
   median pinned, and the mean is what decides whether the strategy earns
   anything. Run it at `1.00` as a stress test.
2. **`--half-spread`**. Measure your actual fills against the mid; do not guess.
   The `--sensitivity` sweep shows why.
3. **`--contracts`**, **`--width`**, **`--delta`** -- your actual trade.
4. **`--max-margin-frac`** -- how far a martingale is allowed to run.

## What it does not model

Worth knowing before you lean on a number:

- **One underlying, one expiry.** No cross-asset or overnight risk, and no
  rolling into a later expiry.
- **European cash settlement**, i.e. SPX/XSP. There is no early assignment and
  no pin-and-gap exposure, so results are *optimistic* for anyone trading this
  on SPY or QQQ, where a short leg finishing ITM against an OTM long leg leaves
  real overnight exposure.
- **Fills at mid plus a fixed half-spread.** No queue position, no partial
  fills, no widening under stress -- so the modelled cost of a forced exit
  during a fast move is too kind.
- **A margin cap that always binds cleanly.** No broker auto-liquidation at the
  worst possible moment, which is a real tail this model omits.
- **No regime persistence across days.** Sessions are independent draws, so
  clustered volatility -- several bad days in a row -- is under-represented and
  drawdowns are, if anything, understated.
- **A parametric surface**, not real option chains. Skew and the leverage effect
  are stylised, not fitted.

Every one of those omissions points the same way: the real thing has a fatter
left tail than this simulator does.

## Layout

| file | contents |
|---|---|
| `pricing.py` | Black-Scholes in total variance; strike-from-delta inversion |
| `market.py` | variance clock, path generation, vol surface |
| `config.py` | strategy, cost and account configuration |
| `engine.py` | day loop, roll policies, path simulation |
| `metrics.py` | tail-aware performance statistics |
| `experiment.py` | common-random-number comparison harness |
| `plots.py` | four-panel dashboard |
| `cli.py` | `python -m zerodte_sim` |
| `tests/` | invariants: pricing identities, variance calibration, defined-risk bounds, margin caps, roll-policy semantics |

Run the tests with `python -m pytest zerodte_sim/tests -q`.
