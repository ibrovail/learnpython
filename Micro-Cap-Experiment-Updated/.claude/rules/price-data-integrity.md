# Price Data Integrity Rules

Hard rules for sourcing prices used in any analysis or order recommendation.
Origin: 2026-07-23 — a stale WebSearch quote ($35.59) was reported as ATRC's live
after-hours price when the real print was $31.47 (−4.64%), producing a stop
recommendation ($32.50) that sat *above* the market and would have force-executed
at the next open.

---

## Source hierarchy — which tool for which price

| Price type | Authoritative source | Never use |
|------------|---------------------|-----------|
| Settled daily close, volume, OHLC | `trading_script.py` daily run (yfinance) | WebSearch |
| After-hours / pre-market / intraday | **Browser tool** on a live quote page (`mcp__Claude_Browser__`) | **WebSearch** |
| Earnings dates, guidance, PTs, filings | WebSearch (≥2 sources) | — |

**Never source a live or extended-hours price from WebSearch.** Search results are
undated cached snippets; the crawler's snapshot may be hours or weeks old, and the
snippet will still read like a current quote. This is not a reliability question —
it is a structural limitation of the tool.

To get a live quote: `preview_start` / `navigate` to a quote page
(e.g. `https://www.cnbc.com/quotes/TICKER`), then `get_page_text`.

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
