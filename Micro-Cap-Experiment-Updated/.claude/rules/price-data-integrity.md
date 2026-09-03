# Price Data Integrity Rules

Hard rules for sourcing prices used in any analysis or order recommendation.
Origin: 2026-07-23 — a stale WebSearch quote ($35.59) was reported as ATRC's live
after-hours price when the real print was $31.47 (−4.64%), producing a stop
recommendation ($32.50) that sat *above* the market and would have force-executed
at the next open.

---

## Source hierarchy — browser is the DEFAULT, WebSearch is for discovery

**Revised 2026-08-20.** The previous version of this table assigned "guidance, PTs,
filings" to WebSearch. That was wrong, and it caused repeated misses (see *Why this
changed* below): WebSearch returns cached prose and does **not** reliably return
forward P/E, beta, 52-week range, current analyst PT, or TTM revenue/EPS — all of
which sit on a single quote page and all of which move recommendations.

| Data type | Authoritative source | Never use |
|-----------|---------------------|-----------|
| Settled daily close, volume, OHLC | `trading_script.py` daily run (yfinance) | WebSearch |
| Live / after-hours / pre-market / intraday price | **Browser** — live quote page | **WebSearch** |
| **Valuation + analyst data** (P/E, forward P/E, PT, rating, beta, 52-wk range, market cap, TTM revenue/EPS) | **Browser** — quote page | **WebSearch** |
| **Actual earnings results** (revenue/EPS/EBITDA/guidance) | **Browser** — live release (StockTitan/press-wire/IR) | WebSearch alone |
| **Company news feed / recency check** | **Browser** — ticker news page | WebSearch alone |
| Catalyst *dates*, historical guidance, discovering that something exists at all | WebSearch (≥2 sources), then browser-verify anything decision-relevant | — |

**Rule of thumb:** if the data lives at a URL you can construct from a ticker
(`stockanalysis.com/stocks/TICKER/`, `stocktitan.net/news/TICKER/`,
`marketwatch.com/investing/stock/TICKER`), **fetch it — do not search for it.**
WebSearch's job is to tell you what you don't know to look for; the browser's job is
to get the actual numbers.

**One quote-page fetch returns**: price, extended-hours price + timestamp, market cap,
TTM revenue and growth, net income, EPS, P/E, **forward P/E**, shares out, volume,
day range, **52-week range**, **beta**, **analyst rating**, **price target**, and
earnings date. That single call replaces several WebSearches and is strictly better.

### Why this changed — four misses, all the same shape

Each was caught only because the user intervened and asked for a browser check:

1. **ARDT exit (2026-08-20)** — recommended selling on "thesis spent." The live page
   showed **forward P/E 9.51, PT $12.68 (+21%), Buy, revenue +3.3%, beta 0.70**. The
   exit was withdrawn. None of that was in the daily data or my WebSearch.
2. **FOXF buy (2026-08-12)** — recommended on a "beat and raise." The live page showed
   **TTM EPS −$7.14** and a broken higher-low base. The buy was withdrawn.
3. **PAR entry (2026-08-17)** — priced off a pre-market print of +1.10%; the live page
   showed **332 shares of before-hours volume** — one trade, no information. The real
   session opened −3.25%.
4. **ARDT earnings night (2026-08-04)** — a +6.98% AH pop read as a clean beat; the live
   release showed a **mixed print** (EPS missed, EBITDA −32% YoY).

The common failure was not laziness about prices — it was **recommending an action
without pulling the one page that contains the deciding facts.** See the
*Pre-Recommendation Verification* gate in `analysis-workflow.md`.

**Never source a live or extended-hours price from WebSearch.** Search results are
undated cached snippets; the crawler's snapshot may be hours or weeks old, and the
snippet will still read like a current quote. This is not a reliability question —
it is a structural limitation of the tool.

To get a live quote: `preview_start` / `navigate` to a quote page
(e.g. `https://www.cnbc.com/quotes/TICKER`), then `get_page_text`.

## Unexplained move? A press wire cannot explain an analyst-driven one

When a holding moves materially and the cause is not obvious, **the press-release wire and
the quote page's own feed are not sufficient** — they will often show nothing, and "nothing
found" then gets mistaken for "no driver."

- `stocktitan.net` is a **company press-release wire**. Analyst upgrades, downgrades and
  price-target changes are *third-party* actions and never appear there. Searching it for the
  cause of an analyst-driven move returns a guaranteed false negative.
- `stockanalysis.com`'s feed does carry TheFly analyst items, but **lags by hours**.

**So for any unexplained move, check an analyst-action source explicitly** before concluding
the driver is unknown:
- `investing.com/news/analyst-ratings/` (search the ticker)
- `benzinga.com` movers columns — these name the analyst, firm, old and new target
- a dated web search for "TICKER price target" restricted to that day

**A same-day jump in the consensus price target is itself the tell.** If a quote page's
consensus PT moves materially in one session, an analyst acted — go find who and why, and
read the actual note.

- Reason: 2026-09-02 — ATRC closed **+7.50%** at a 52-week high on 2× volume. The StockTitan
  wire showed nothing since 8/25 and the quote-page feed's latest item was 6 days old, so the
  daily review recorded the driver as "unestablished" while noting the consensus PT had
  jumped $49.56 → $51.67 in a day. The actual cause was **Needham raising its target to $64
  from $45 (Buy, Mike Matson)** on the BoxX-NoAF trial. Reading the note changed the
  interpretation materially: the trial's **30-day data does not arrive until H1 2027**, long
  after the experiment ends, and Needham attributed the raise to **"peer multiple expansion"**
  rather than new company data — making the move a sentiment re-rating on a distant catalyst,
  not a fundamental step-change. **Knowing *why* a position moved changes what the stop should
  do about it.**

## Timestamp requirement

Before using any non-close price in analysis or an order, the quote must show
**both a session label and a timestamp** — e.g. `After Hours: Last | 5:46 PM EDT`.

- No visible timestamp → **do not use the number.** Say "after-hours pricing not
  verified" and either fetch it live or defer the recommendation.
- Never restate a price from a prior message as current; re-fetch it.

## Reconciliation test (catches stale quotes)

An extended-hours quote must cohere with the regular-session close:

```
AH price ≈ close + stated change,  and  stated % ≈ stated change / reference price
```

If the numbers do not reconcile, **discard the quote — do not explain it.**
Constructing a narrative that makes inconsistent data make sense (an "AH spike it
pulled back from", a "different reference price") is the failure mode itself, not
a resolution of it.

Also sanity-check direction: a large extended-hours move opposite to the day's
trend is possible but demands a live re-fetch before it is acted on.

## Order-side check (mandatory before recommending any stop or limit)

Verify every recommended level is on the executable side of the **last verified**
price:

- **Stop-loss** must be BELOW the current price. A stop above market executes
  immediately at the open.
- **Buy limit** at or below the intended entry; **sell limit** at or above.
- State the price the check was run against, with its timestamp, in the
  recommendation.

If the current price cannot be verified, do not name a stop level — recommend the
action ("raise the stop after the open") and compute the number in the next daily.

## Range check — a stop must clear the noise band, not just the last price

"Below the current price" is necessary but **not sufficient**. A stop executes on the
**intraday low**, not the close, so the order-side check must be run against the
**recent trading range and the stock's ATR** — never the close alone:

Before naming any stop level, state all three:

1. **Today's low** (and the 5–10 day lowest low). **A proposed stop above the most
   recent day's low is inside normal noise — reject it.**
2. **ATR(14)** in dollars and as a % of price. The stop must sit at least
   **1.5 × ATR** below the reference price, per `entry-discipline.md`
   (target 1.75 × ATR below entry, or the swing low / technical level, whichever is wider).
3. **The resulting max loss** — if the properly-wide stop implies more risk than
   intended, **reduce position size; do not tighten the stop.**

Compute ATR from price history (yfinance), don't estimate it by eye.

- Reason: 2026-08-04 — ARDT (ATR(14) $0.497 = **4.62%** of price, avg daily range
  4.71%) was recommended a $10.45 stop because it was "below the $10.75 close." But
  that same session's **low was $10.42** — the stop would have fired on a day the
  stock closed **unchanged**, pre-print. At 0.6 × ATR it was under half the rule
  minimum, while the existing $9.55 (~2.5 × ATR, below the 10-day low $10.03) was
  correctly calibrated. Tightening a stop into the noise band converts a *thesis*
  exit into a *coin-flip* exit — realizing the loss **and** forfeiting the recovery.

**Corollary — de-risk by size, not by stop tightness.** When a position's risk feels
too high (a mixed earnings print, a shaky thesis), the correct lever is selling
shares, not moving the stop inside the noise band. The broker's one-open-order-per-stock
constraint also means a position cannot carry two stop levels, so partial protection
is achieved by reducing share count.

**Post-event volatility:** volatility is *elevated* for several sessions after an
earnings print. Do not set a stop off an after-hours price — extended-hours prints
frequently do not survive the open. Wait for a regular-session close to re-anchor,
then trail.

## Correcting a bad price

If a price already given to the user turns out to be wrong, correct it plainly and
immediately, and **explicitly withdraw any order recommendation derived from it**
("do not place that order") before giving the revised level. A stale price that has
reached an order recommendation is a live financial risk, not a cosmetic error.
