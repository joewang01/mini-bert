# Pre-open playbook for 0DTE short verticals

Run this between 08:00 and 09:20 ET. It answers two questions and nothing else:

1. **Is a big move possible today, and what would cause it?**
2. **If something is already moving, what is it worth in SPX points?**

The point of writing it down is that both questions get answered *before* the
open, when a wrong answer costs nothing. Answering them at 11:40 with a tested
put spread on is how the −$24,206 day in `zerodte_sim` happens.

## The one number everything reduces to

`zerodte_sim` swept the strategy unfiltered across variance-premium regimes and
found it crosses zero at a **median realised/implied vol ratio of 0.787**. That
is the whole business in one line: the day is profitable if the index travels
less than about four-fifths of what the options charged for.

```
python -m premarket.expected_move --spot 7707 --vix 15.3 --width 25
```

prints the implied move, that 0.787 travel budget in index points, and the
strike / credit / touch probability for each candidate delta. Pass `--straddle`
instead of `--vix` when you can read the live ATM straddle — it is the better
input, since VIX prices a month of event risk rather than today.

Everything below is about one judgement: **is today's catalyst set likely to
push realised travel above or below that budget?**

## Step 1 — Scheduled catalysts

Check the calendar for today, and also for the rest of the week. A print two
days out matters, because premium stays bid into it while realised vol sits
still. Those are the best days to sell.

| Tier | Events | Typical SPX day range | Verdict |
|---|---|---|---|
| **1** | CPI, NFP, FOMC decision, PCE | 1.0–2.5% | Do not sell 0DTE premium into it. The credit never pays for the gap risk. |
| **2** | PPI, retail sales, ISM, JOLTS, CPI-adjacent revisions, Fed chair testimony | 0.6–1.2% | Trade only after the print, entering late, at reduced size. |
| **3** | Consumer confidence, housing, regional Feds, Treasury auctions, Fed speakers | 0.3–0.7% | Tradeable. Note the time and expect a pulse, not a trend. |
| **4** | NFIB, consumer credit, EIA inventories, weekly claims | noise | Ignore. |

Ranges are historical rules of thumb, not model output — use them to rank
danger, not to size positions.

Two calendar features are worth as much as any release:

- **FOMC quiet period** (the ~10 days before a meeting): no Fed speakers at all.
  This removes the most common source of unscheduled intraday repricing and is
  a genuine, checkable tailwind for a premium seller.
- **The day before a Tier 1 print.** Implied stays elevated, realised usually
  does not. This is the single most favourable recurring setup in the calendar.

## Step 2 — Unscheduled risk (the part that actually kills you)

Scheduled events are priced. What is not priced is the live geopolitical or
policy situation that can produce a headline at 11:20 with no warning. Score
each open situation:

- **Is it escalating or de-escalating?** A third consecutive day of oil gains on
  military exchange is escalation. A signed framework is not.
- **Is there a discrete trigger that could fire today?** Strait closure, a
  strike on a named target, a tariff deadline, a court ruling, a scheduled vote.
- **Which direction is the tail?** Nearly all of them are downside. This is why
  the put side is structurally more dangerous than the call side, and why an
  iron condor is not symmetric in risk even when it is symmetric in delta.

If a live situation has a discrete trigger that could fire during the session,
**that is a reason to size down or skip, no matter how quiet the calendar is.**
Realised vol on those days is bimodal, and the simulator's worst days are drawn
from exactly that distribution.

## Step 3 — If it is already moving pre-market

Do not read the percentage and stop there. Classify it, because the same gap
size means opposite things depending on type.

**Priced-in vs. new.** An event announced two weeks ago that takes legal effect
today (a tariff schedule, an index rebalance) has already been traded. It is a
sector event, not an index event. A weekend military strike is new information
and is still being absorbed — that one keeps moving after the open.

**Then size it with this:**

| Gap in futures | What it usually means for the session |
|---|---|
| < 0.3% | Noise. Trade normally. |
| 0.3–0.6% | Real but contained. Trade, but shift strikes away from the gap direction. |
| 0.6–1.0% | Elevated realised vol is likely to *persist* all session. Enter late, half size, or skip. |
| > 1.0% | Skip. Gap days trend, and trend days are what the roll rule cannot survive. |

The empirical basis for the last row is in the simulator: the losing days are
directional-travel days, not merely high-vol days, and `opening_range` — a pure
price-action filter — predicted the martingale's P&L better than a *perfect*
volatility oracle (−0.382 vs −0.265).

**Impact sizing for a live event.** Translate into index points before deciding:

| Shock | Historical SPX intraday response |
|---|---|
| Oil +5–10% on supply disruption | −1.0% to −2.0% |
| Major-economy tariff escalation, new | −0.5% to −1.5% |
| Tariff already announced, taking effect | −0.1% to −0.3%, sector-concentrated |
| Single mega-cap earnings shock (>8% move) | −0.3% to −0.8% index |
| Sovereign / credit event, new | −1.5% and up, with IV expansion |

Compare the estimate against the travel budget from step 1. If the plausible
shock is a multiple of the budget, the credit is not compensating you.

## Step 4 — Commit to the roll rule before the open

The decision that matters most is made when it costs nothing to make.

- **Never martingale into an unscheduled-headline day.** `equal_risk` has a
  worst day that does not move — −$4,122 across every filter, signal quality,
  skip rate and variance-premium regime tested. The martingale's worst day is a
  lottery: it rescues ~80% of losing sessions and turns the median rolled day
  from −$338 to +$37, while the worst goes from −$4,454 to −$24,206.
- **Write the kill switch down.** Name the headline that makes you close rather
  than roll. On a Middle East escalation day that is a strait-closure or
  tanker-strike story; the move will outrun your fills.
- **Later entry helps, but not for the reason it seems.** The measured benefit
  comes from exposure time, not from better information: stop-out rate falls
  28.1% → 21.9% between minute 15 and minute 180 while credit falls only 12%.
  Entering at minute 60–120 is well supported. Waiting for "clarity" is not.

## Known limits of this procedure

- Every magnitude table above is judgement calibrated on historical episodes,
  not simulator output. Treat them as a ranking, not a forecast.
- The simulator's filtered results are optimistic — its implied vol does not
  respond to realised range, so it under-rewards skipping. The unfiltered
  0.787 break-even is the trustworthy number and is what this playbook uses.
- Nothing here forecasts direction, and no part of it should be read as
  suggesting a directional view is available. It sizes and screens; that is all.
