# CME FedWatch daily change ledger

A scheduled Claude Code routine checks the
[CME FedWatch tool](https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html)
readings once a day and analyzes what moved since the previous reading.

- **Live ledger (updated daily):** https://claude.ai/code/artifact/02f414ce-fbfb-4788-af2f-8365d3bd0da7
- **Schedule:** every day at 21:30 UTC (5:30pm ET), after 30-day fed funds futures settle.
- **Routine:** "CME FedWatch daily change ledger — 5:30pm ET" in the account's Routines list.

## How it works

1. Reads the live ledger page and parses the `fedwatch-history` JSON block embedded in it.
2. Searches the web for same-day FedWatch readings for the next FOMC meeting
   (hike / hold / cut) and, when available, cumulative odds for the two meetings after it.
   `cmegroup.com` and most finance sites are blocked for direct fetching from the routine's
   environment, so figures come from same-day press reports and mirrors of the tool.
3. Computes the day-over-day change, classifies moves of 3 points or more as material,
   attributes them to a catalyst, and rewrites the page's analysis, chart, and table.
4. Republishes the page at the same URL and pushes a one-line summary to the phone.

## Data rules

- Only readings explicitly dated that day are logged; stale figures are never carried forward.
- Later meetings without a same-day reading are recorded as `null` and shown as "not pinned".
- After an FOMC decision the tracked meeting rolls forward and a new probability series starts.

`snapshots/` holds the baseline snapshot captured when the routine was set up. The running
history lives in the ledger page itself.
