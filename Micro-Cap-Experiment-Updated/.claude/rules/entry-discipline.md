# Entry Discipline Rules

Hard rules for new-position selection in the weekend deep research and daily analysis. Every recommended BUY must pass ALL of these checks before being proposed.

---

## Post-Earnings Cooldown

- **Never recommend a buy within 3 trading days of an earnings print** at a price more than +5% above the post-print close.
- Reason: ARLO Week 34 (entered $15.25 Monday, three trading days after Wednesday close of $13.60 pre-print, post-print closed $15.32 — bought at the post-print high, gave back -10.9% on entry day).
- Wait for ≥1 trading week of post-print consolidation showing a higher-low base.
- **"Enter pre-print (binary risk acknowledged)" is no longer an option.** `portfolio_rules.md`
  prohibits initiation within **10 trading sessions before a known earnings date**. This cooldown
  covers the window after a print; that guard covers the window before it. Together they close
  both sides. Holding an existing position through its print remains normal and permitted — the
  prohibition is on buying into one.

## Distance-from-Base Limits

For every screener candidate considered for entry, compute and report:
- Distance from 50-day SMA (must be ≤ 40% above for a momentum entry)
- Distance from 20-day SMA (must be ≤ 20% above)
- Days since 20-day breakout (avoid days 1-3 of a new breakout if the move is >+10% cumulative)

If a candidate is >50% above its 50-day SMA, it is **disqualified** for a fresh buy regardless of catalyst strength.

**Rationale, re-grounded 2026-09-17.** This rule previously justified itself by mean-reversion
risk "within a 5-trading-day window" — a horizon the book no longer holds to. The thresholds are
retained because Phase 2 validated them directly: every price-computable gate removed names that
subsequently did *worse* than the survivors (median 10-session excess — >40% above the 50-day
**−3.2pp**, >20% above the 20-day **−2.3pp**, fresh breakout **−2.1pp**). Buying extended is
costly regardless of how long you then hold.

**Known limitation:** that validation was measured at 10 sessions only. The book now holds
40–60. Phase 3.5 (`Experiment Details/Horizon Factor Study — Phase 3.5.md`) reports how the
gated groups performed at the adopted horizon. Until it does, treat these thresholds as
evidence-backed at 10 sessions and provisional at 40.

## ATR-Based Stop Sizing

Stops must be set at the wider of:
- 1.75 × 14-day ATR below entry, OR
- The most recent swing low on the daily chart, OR
- A technical level (50-day SMA, post-breakout VWAP)

A stop within 1.5 × ATR is too tight for normal daily noise and will be triggered by a routine down day.

**1.5×ATR is a floor; 1.75×ATR is the target.** These are deliberately different numbers. A stop
may never *sit* below 1.5×ATR; a stop is *placed* at 1.75×ATR. The anti-ratchet rule in
`portfolio_rules.md` depends on headroom existing between them — collapsing the two would break it.

If the required stop would create a max-loss exceeding **2% of equity** — the standing
risk-per-trade budget — **reduce position size**, do not tighten the stop. *(The 5% figure that
formerly appeared in `portfolio_rules.md` was a standing contradiction with this line; resolved in
favour of 2% on 2026-09-17.)*

## Pre-Open Verification

Before recommending an open-of-Monday buy at the close-of-Friday price:
1. Check Monday pre-market action via the **browser tool** on a live quote page
   (`mcp__Claude_Browser__` → a quote page such as `https://www.cnbc.com/quotes/TICKER`).
   **Never source the pre-market price from WebSearch** — it returns undated cached
   quotes (see `.claude/rules/price-data-integrity.md`). Require a session label +
   timestamp on the quote before using it.
2. If pre-market is down >2%, downgrade to a limit at the pre-market price or pass entirely.
3. If pre-market is flat-to-up, the Friday-close limit is acceptable.
4. If a live pre-market quote cannot be verified, do not name a fixed limit at the stale
   close — recommend a limit at/below the last verified price and state the timestamp.

## Thesis-Input Freshness (verify the driver, not the story about it)

Every BUY thesis rests on one or more **inputs**. Before recommending entry, list the
inputs the thesis depends on and confirm each is **current** — dated within the last
**10 trading days** — from a source that post-dates any summary you are tempted to rely on.

**Time-varying external drivers** — a commodity/spot price (fertilizer, oil, gas, metals,
freight/tanker rates, ethanol, power), an interest rate, an FX cross, a tariff/subsidy, or
a named supply/demand condition (e.g. "supply disruption keeping prices elevated") — must
be verified at their **current level AND direction** before entry:

- **Never source a live driver from the company's earnings call or any note that predates
  the current data.** A Q1 (April) call is not evidence about July spot prices. Pull the
  driver from a dated market source (price index, trade press) within the last 10 trading days.
- **Report the driver's latest dated value and its 4-week trend** (rising / falling / flat)
  in the candidate evaluation. A thesis that requires the driver to *stay elevated* is
  **INVALID if the driver has fallen for ≥3 consecutive weeks** — do not recommend the entry.
- **A reversing driver kills the thesis even with strong screener momentum**, because the
  screener's trailing 20-day momentum lags a driver that just turned.
- Reason: LXU Week 46 — entered on "sustained nitrogen pricing into 2027" taken from LSB's
  April Q1 call, without checking current fertilizer prices. UAN32 had fallen **5 straight
  weeks** (−15% MoM) as the Middle East supply disruption eased mid-June; the stock was
  rated Strong Sell and closed **−6.4% on entry day**. The thesis input was stale before the
  order was written.

**General principle — date every material input.** For any claim that moves the entry
decision (catalyst date, guidance, analyst PT/rating, the driver above), note its source
date. If the claim is time-sensitive and older than its natural refresh cycle, re-verify it
against a current source before relying on it. This is the same failure mode as a stale
price quote (`.claude/rules/price-data-integrity.md`): trusting a cached narrative over
current data. If sources disagree or the latest data is unavailable, mark it
**INSUFFICIENT CONFIRMATION** and do not lean on it.

## Screener Score is Sourcing, Not Conviction

> **Horizon note — updated 2026-09-17 after Phase 3.5 reported.** The figures below are Phase 2's
> **10-session** measurements. Phase 3.5 re-tested at the adopted 40–60 session horizon: **all six
> signals retain support**, and the practical edge rises sharply with holding period:
>
> | Top 50 minus survivor median | 5s | 10s | 20s | **40s** | **60s** |
> |---|---|---|---|---|---|
> | Excess | +0.50pp | +0.78pp | +2.85pp | **+5.57pp** | **+6.99pp** |
>
> Phase 2's central weakness also reverses: the quintile spread on **mean** returns — near zero or
> negative at 10 sessions — is **+4 to +5pp at 40** and higher at 60. So "under 1pp per 10
> sessions, mostly defensive" is accurate for 10 sessions and **understates the edge at the
> horizon actually used.**
>
> **Do not over-read it.** Phase 3.5's effective sample is **1.75 independent observations at 40
> sessions and 0.83 at 60** — its t-statistics (up to 14.8) count overlapping weekly windows as
> independent and are not trustworthy. The defensible reading is **"no evidence of breakdown at
> the adopted horizon,"** not confirmation. Screener rank remains sourcing, not conviction.

Screener composite score — since 2026-09-15 the equal-weighted ranks of low volatility, proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio and distance above the 50-day SMA (the six signals that passed the Phase 2 factor study; 20-day momentum is reported but no longer scored) — identifies *candidates* but does NOT confer fundamental conviction. **Its measured edge is small and mostly defensive:** the top 15 beat the surviving universe by under 1pp per 10 sessions, largely by avoiding volatile losers rather than by finding winners. Apply the full 5-step verification to every screener pick — **step 1 requires the browser quote page (PRV gate, `analysis-workflow.md`), not WebSearch**:
1. **Fundamental quality — from the live quote page**: TTM revenue **and growth %**, TTM EPS/net income, forward P/E vs trailing, analyst rating + price target, 52-week range position, beta. *Shrinking revenue is the single strongest disqualifier this book has found* (TDAY −8.3% YoY → exited; FOXF −4.5% with TTM EPS −$7.14 → withdrawn; PAR +18.8% → bought).
2. Catalyst durability over the chosen timing window
3. **Thesis-input freshness** — identify the thesis's time-varying driver(s) and verify each is current and not reversing (see *Thesis-Input Freshness* above). Mandatory for any commodity/rate/FX/subsidy-dependent name.
4. Distance-from-base and post-earnings cooldown checks
5. Liquidity (ADV >$1M for full sizing, $500K-$1M for half sizing)

Conviction rating starts at 2/5 for any screener pick and can only rise on the strength of independent web-research evidence, not the screener score itself.

**Name the primary thesis driver.** Every recommended entry must state the time-varying input its
thesis depends on (see *Thesis-Input Freshness* above). This is not documentation — it is the
input to the **driver cap** in `portfolio_rules.md`: at most 2 positions may share a primary
driver. The cap cannot be enforced in code, so an unnamed driver is a rule violation rather than
an omission.

*Terminology: these are **screener-sourced plays**. The former name, "momentum/technical plays",
was retired on 2026-09-17 — 20-day momentum showed no ranking skill and is no longer scored. The
six validated signals describe calm stocks near their highs with volume confirmation in an
established uptrend: low-volatility trend continuation, not momentum.*

**The screener pre-applies these gates as a backstop, not a substitute** (since 2026-09-14): prohibited businesses, deal-pinned, >40% above the 50-day / >20% above the 20-day SMA, days 1–3 of a >10% breakout, post-earnings jump, shrinking revenue (Finviz Sales Q/Q < 0) and liquidity — all *before* ranking, so the list is not filled with names that fail by hand. The gates run on Finviz/yfinance data, and a gate whose input is missing passes the name, so **every check above is still run on the quote page.** A `REVIEW` flag marks industries mixing prohibited and permitted businesses. The thresholds live in `screener.py` constants — change the rule here first, then the constant.

## Deal-Pinned Stocks Cannot Generate Short-Horizon Return

A stock trading under a pending acquisition is pinned near the deal price and cannot move
meaningfully inside a short window, whatever its screener score. **Reject any candidate whose
ATR(14) is below 0.75% of price.** That one test is sufficient. Confirm the cause on the quote
page's news feed (a cash deal price the stock trades just under is decisive), then drop it
without further research.

- **Why ATR alone (revised 2026-09-14, same day as written):** the first version required three
  conditions together — ATR below ~0.5%, 20-day momentum within ±1%, and price above the analyst
  target or a pennies-wide range. On that evening's gated screen it let two confirmed all-cash
  takeover targets through and ranked them **#1 and #2**: **DV** (Nielsen at $13.60; momentum
  +1.65%, just outside ±1%) and **PAYO** (Nuvei at $7.40; analyst target 3.6% above the price —
  the targets had reset to the deal price, not above it). Across the 1,191-stock universe the
  1st-percentile ATR was 0.35% and the 2nd percentile 1.45%; all 16 stocks under 0.8% had a
  pinned profile (the highest was 0.45%). A small cap that is free to move does not trade that
  quietly, so the extra conditions only created ways for a real deal target to slip through.

- Reason: 2026-09-14 — BZH, UTZ and DV all ranked in the top 15 with ATRs of **0.28–0.34%** and
  ~0% momentum. BZH — a **beta-2.18** homebuilder — closed **0.00%** in an **8-cent range**
  ($33.25–$33.33) while trading 17% above its $27.50 analyst target: pinned. A low-volatility
  factor in the screener actively *rewards* this pattern, so it will recur.

## Stop-Placement Surfacing

The **weekend deep research report** may include the reminder "Place this stop with your broker before the next market open" in a BUY recommendation, since at that stage the order has not yet been placed.

**Exception — stop specified in a `run daily` command:** When the user executes a buy via `run daily: buy N TICKER limit $X stop $A/$B`, the stop has **already been placed with the brokerage** — the `stop $A/$B` syntax confirms it. Do NOT tell the user to place or confirm the broker stop, and do NOT mark the position "STOP NOT LIVE." The CSV stop field is informational, but the user's broker stop is already live. No confirmation reminder is needed in the post-buy daily analysis.

---

## Day-1 Drawdown Rule

If a freshly opened position closes -8% or worse on entry day:
- **Default action: exit at next market open** unless an explicit positive-news catalyst surfaced after entry.
- Reason: a −8% same-day move signals thesis break, not noise. Continuing to hold rationalizes a
  bad entry. *(Re-grounded 2026-09-17: this previously read "on a momentum-screened name", which
  referenced a selection method the book no longer uses. The rule survives the rename because it
  is about the entry being wrong on day one, not about how the name was sourced — and it is
  independent of holding horizon: a thesis that breaks on day 1 does not improve over 60
  sessions.)*
- This rule overrides the stop-loss field — exit even if the stop was not breached intraday.
