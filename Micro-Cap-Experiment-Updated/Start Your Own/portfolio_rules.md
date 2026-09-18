# Portfolio Rules

**Standing operating rules for the indefinite live phase.** Read this file before beginning any
analysis session, daily or weekend.

This file states the rules **as they currently are**. It deliberately contains no dated amendment
blocks — the reasoning behind each rule, the failures that produced it, and every superseded
version live in `Experiment Details/Rules Amendment History.md`. Where a rule exists because
something went wrong, the origin is named in one line and the full account is in that file.

---

## Mandate and Horizon

- **No end date.** The 52-week experiment closed at the 2026-09-11 close. This is an ongoing live
  process. `experiment_config.json` → `end_date` and `total_weeks` are `null`.
- **Holding horizon: 40–60 trading sessions.** Positions are underwritten to be held for months,
  not days. This is the horizon every other rule in this file is calibrated to.
- **Universe ceiling: $5Bn, frozen until Phase 4 reports** (~December 2026). The ceiling is not
  to be changed in the interim: Phase 4 compares three composite scores across ~12 weekly screens,
  and changing the universe mid-collection destroys the comparison.
- **Scoreboard.** The benchmark gap is reported **both** since inception and since the
  2026-09-11 re-base (`benchmark_base_date`). Both are always shown. A re-base may never be the
  only figure reported — that is what would let a future bad stretch be quietly buried.

---

## Budget

- No new capital beyond what is shown in the portfolio snapshot unless explicitly approved.
- Track cash to the cent after every proposed trade.
- Out-of-cycle capital injections may be declared in `<capital_injection>`. When `planned=true`,
  add `<amount>` to the `Cash Balance` shown in `<portfolio_snapshot>` and use the combined total
  as available capital for all sizing calculations.
- **Cash floor: 15% of equity.** Standing rule. Capital below this line is not deployable.

---

## Execution Limits

- Long-only. Full shares only (no fractional).
- No options, shorting, leverage, margin, or derivatives.

---

## Universe

- U.S.-listed common stocks, nano-cap to small-cap, **market cap up to $5Bn**
  (`screener.py` enforces `MAX_MARKET_CAP = 5e9`).
- Allowed exchanges: NYSE, NASDAQ, NYSE American.
- Existing positions above **$5Bn** may be held or sold; no new shares may be added.

---

## Exclusions

**Security types**
- OTC / pink sheets
- ETFs, ETNs, closed-end funds, SPACs
- Rights, warrants, units, preferred shares, ADRs
- Bankrupt or halted issuers

**Prohibited businesses — never recommend, regardless of screen rank or fundamentals**
- **Prisons and detention centers** — private corrections and immigration-detention operators
- **Weapons, defence and firearms** — defence contractors, arms makers, gun and ammunition companies
- **Predatory lending** — payday lenders, pawn lenders, high-cost subprime installment credit
- **Israeli-affiliated companies**

*Not* prohibited: fossil fuels, tobacco, gambling, alcohol, cannabis, adult entertainment.

**Enforcement.** `screener.py` hard-excludes the **Aerospace & Defense** industry and a ticker
blocklist. Two industries mix prohibited and legitimate businesses — **Security & Protection
Services** and **Credit Services** — and are **not** auto-excluded; any candidate in them must be
checked by hand. **Israeli affiliation cannot be screened by industry** and must be checked per
name. **Every candidate is checked against this list at the PRV gate, before a recommendation is
written** (`.claude/rules/analysis-workflow.md`).

*Origin: 2026-09-14, CXW. See amendment history.*

---

## Risk Control

### Stops

- **Every long position carries a stop.** No exceptions.
- **Initial stop at entry:** the **wider** of
  - **1.75 × ATR(14) below entry**, or
  - the most recent swing low on the daily chart, or
  - a technical level (50-day SMA, post-breakout VWAP).

  If the resulting stop would breach the risk budget below, **reduce position size — never tighten
  the stop.**
- **Trailing stop floor:** `max(1.5 × ATR(14), 15% below the 20-day rolling high)`.
- **1.5×ATR is a floor, 1.75×ATR is a target.** These are two different numbers and the
  distinction is deliberate. A stop may never *sit* below 1.5×ATR; a stop is *placed* at 1.75×ATR.
  Collapsing them breaks the anti-ratchet test below, which requires headroom between the two.

### Raising a stop — anti-ratchet minimum

Do **not** raise a stop unless the raise is **≥ 0.5 × ATR(14)** *and* the new level still leaves
**≥ 1.5 × ATR** of room below the reference price. Both conditions, every time.

- Compute the candidate level at **1.75 × ATR** below the reference price, then apply the
  0.5×ATR size test to that level. Raising *to* the 1.5×ATR floor lands exactly on the boundary,
  where it either fails on floating-point or passes with zero margin.
- A raise failing either test is **declined, not reduced**. Wait until a qualifying raise exists.
- *Why:* small trailing raises bank trivial profit while measurably increasing stop-out
  probability, and a stop can never be lowered afterwards, so the cost is permanent.
  *Origin: ATRC, 2026-08-27 — a $1.35 raise left the book's best thesis 0.38×ATR from being
  stopped out.*

### Stop restoration — mechanical, once per entry

If a stop comes to sit **below 1.5 × ATR(14)** of the current price **through price movement
alone** — never through tightening — it may be reset **once per entry** to a level computed at
**1.75 × ATR** below the reference price.

- **"Once per entry," not once per ticker.** A position that is stopped out and later re-entered
  after the re-entry ban begins a new entry with a fresh allowance.
- This is the **sole exception** to "never lower a stop," and it is deliberately mechanical:
  it either drifted below the band or it did not. **Conviction, thesis strength, analyst targets
  and unrealised P&L are explicitly NOT inputs.**
- The restored level must still pass the **range check** (below the most recent session's low).
  Where the formula and the range check disagree, the range check wins.
- **Eligibility is mechanical; using it is not.** A qualifying stop *may* be reset, not *must* be.
  The test for whether to bother: can the thesis plausibly deliver **inside the position's
  intended hold period**? Room is worth paying for only when something can happen in that window
  to use it.
- *Honest record: used once (Week 51, as an ad-hoc suspension before this rule existed). Two of
  the three restorations were never needed; the third delayed rather than prevented a stop-out.
  Net −$3.44 realised against $14.20 of additional risk carried. A narrow safety valve, not a tool.*

### Position sizing

- **Risk per trade: 2% of portfolio equity.**
  ```
  shares = (portfolio_equity × 0.02) / (entry_price − stop_price)
  ```
- **Single-name ceiling: 30% of portfolio equity.**
- Actual size is the **minimum** of the risk-based figure, the 30% ceiling, and available cash
  above the floor.
- *Note on position count:* at 2% risk with 1.75×ATR stops, a position works out to roughly
  `0.02 / (1.75 × ATR%)` of equity — about 19–29% for a typical small cap. Against an 85%
  deployable book that fits roughly **four** positions. The 5–6 position figure in the Allocation
  Framework is a **ceiling, not a plan.**
- **No averaging down:** once a position falls >5% from entry, do not add shares unless a material
  new positive catalyst is confirmed by ≥2 independent sources.
- **Slippage guard:** if the intended order size exceeds 10% of the stock's average daily dollar
  volume, reduce the position to ≤5% of ADV.

### Drawdown circuit breaker

**Trigger metric: CURRENT drawdown from the running peak** — not the maximum drawdown ever
recorded. A breaker written on the historical minimum fires forever on a decline the book has
already recovered from. `trading_script.py` prints this as **Current Drawdown (from peak)** with
the breaker state beside it.

**Basis: the injection-neutral (time-weighted) index, never raw equity.** Raw equity steps up
whenever capital is added, so a deposit would reset the peak and move the trigger with no market
event behind it. Against $547.64 of injections into a book that started at $142.13, that is not a
rounding concern — measured on raw equity the max drawdown reads **−24.99%**, and on the
injection-neutral index **−37.26%**, a 12.3-point difference.

**Peak basis: the re-based series (2026-09-11 onward) governs.** The indefinite phase keeps its
own scoreboard, and the breaker follows that scoreboard. Since-inception drawdown is reported
alongside as context but does **not** trigger anything.
*Why:* on a since-inception peak the book currently sits **−18.81%** below its 2026-01-08 high
(index 1.4111 → 1.1457), which is 1.2 points from the de-risk line — permanently near-armed over
an April drawdown it has since recovered **+29.4%** from. A fresh peak is also the more sensitive
setting, not the weaker one: starting at the peak means any −20% decline from here fires at once.

| Current drawdown from re-based peak | Required response |
|---|---|
| **−20%** | No new initiations. Halve position sizes on anything already planned. Mandatory written review of every holding in that week's report. |
| **−30%** | Go to cash. Full re-read of this rules file before any re-entry, documented. |

- *Origin: the Week 30 pivot — 100% cash at a −20% gap — which was improvised and worked (alpha
  −12.70% → +14.82%, Sharpe −0.02 → 2.29, max drawdown −37.26% → −7.70%). This rule exists so the
  next one is not improvised.*
- **Deliberately not gap-based.** A gap trigger fires when the index rises, creating pressure to
  manufacture a recovery. That pressure has a name here: TYRA, −$29.04 on a binary readout the
  weekend report had recommended against, taken while trailing the benchmark with seven sessions
  left. Drawdown measures capital destroyed; the gap measures someone else's good year.

### Correlated-risk limits

- **Primary — driver cap: at most 2 positions may share the same primary thesis driver.**
  The driver is the time-varying input the thesis depends on, already required by
  *Thesis-Input Freshness* (`.claude/rules/entry-discipline.md`): a commodity or spot price, an
  interest rate, an FX cross, a tariff or subsidy, or a named supply/demand condition.
  **Every holding's primary driver must be named in the portfolio snapshot of every weekend
  report.** An unnamed driver is a rule violation, not an omission.
- **Backstop — GICS sector cap: at most 3 positions in any one sector.** Uniform across all
  sectors; no per-sector exception.
- *Why both:* GICS sector is a crude proxy for correlated risk — too tight (a device maker and a
  clinical-stage biotech are both "Health Care" and share nothing) and too loose (an oil-services
  name and a tanker operator sit in different sectors and share one crude price). The driver cap
  targets the real exposure; the sector cap is a coarse net underneath it.
- *Enforcement note:* **neither cap can be gated in code.** `screener.py`'s `--max-per-sector`
  governs watchlist composition only and has no knowledge of the portfolio. `trading_script.py`
  **surfaces** live sector counts and driver tags in the portfolio snapshot so neither can be
  overlooked, but the limits are applied at research time. Surfaced, not gated.

### Market regime filter

- **Regime test:** IWM below its 50-day SMA = RISK-OFF. Flag the status in every report.
- Regime is a **market condition, not a calendar event** — this filter has no expiry and is not
  relaxed by the passage of time.
- Capacity under each regime is set in the **Allocation Framework** below.
- Flag any stop breach or position sizing violation immediately.

---

## Position Management

### The trailing stop is the only exit

There are **no mechanical partial sells.** A position runs until its trailing stop takes it out,
or until it fails the re-underwrite below.

- *Why:* across 82 closed trades the book produced a 50% win rate and **−$3.84 of net realised
  P&L** — turnover paid nothing — while its one substantial winner was the position that was
  simply held. Mechanical trimming makes sense against a deadline; there is no longer a deadline.
- *Operational fit:* the broker permits one resting order per stock, and the protective stop
  occupies it. A partial already required cancelling the stop, selling, then re-placing it. The
  stop-only regime matches how the account actually works.
- **Consequence: the trailing stop floor is now the single most load-bearing line in this file.**
  Set it deliberately and maintain it every session.

### Mandatory re-underwrite at 60 sessions

At **60 trading sessions held**, a position must be **re-justified in writing** — current thesis,
current primary driver, current conviction, current catalyst if any — against the standard applied
to a fresh buy. If it would not be bought today, exit it.

- **This is not a time stop.** A winner passes trivially and keeps running; nothing is sold for
  being old. The rule exists because with partials removed, the stop is the only automatic exit,
  so a position that drifts sideways indefinitely would otherwise consume a slot forever.
- `trading_script.py` reports **sessions held** per position so the review triggers visibly.

### Post-catalyst reassessment

Within 1 trading day of any dated catalyst resolving: recalculate the stop under the normal
trailing rules, re-evaluate conviction with documented rationale, and — if the stock is trading
below where the normal trailing stop would sit — either document a specific time-bound reason to
hold or exit at market. Log the assessment in the daily analysis.

---

## Order Defaults

- Standard limit DAY orders for the next trading session unless otherwise specified.
- Limit orders preferred. Market orders require explicit reasoning.
- **Platform constraint — one open order per stock.** The broker allows only ONE resting order per
  position, and the protective **GTC stop-limit occupies that slot**.
  - The protective GTC stop-limit is the **default resting order** on every holding.
  - Any exit-into-strength target is a price **ALERT, not a resting order**. To act on one:
    **cancel the stop → place the DAY limit or market sell → if unfilled by close, re-place the
    stop.** There is a brief unprotected window, acceptable only while actively watching.
  - GTC limit sells are not supported at all; only GTC stop-limit sells.

---

## Research Safeguards

### Verification
- Do NOT hallucinate tickers. Every ticker must be a verified, currently listed U.S. security on
  an allowed exchange.
- Market cap, float, liquidity and catalyst data must come from reputable, current sources and be
  confirmed by at least two of them.
- Provide citations for every holding and new candidate: source name, URL, access timestamp.

### Catalyst Confirmation
- Any catalyst claim must be confirmed by at least two independent sources.
- If confirmation is insufficient, state **"INSUFFICIENT CONFIRMATION"** and do not rely on it.

### Liquidity Filters
- Price ≥ $1.00
- 3-month average daily dollar volume ≥ $500,000
- Bid-ask spread ≤ 2% (or ≤ $0.05 if price < $5)
- Float ≥ 5M shares (unless justified with reasoning)

### Trend Filters
- **Above the 50-day SMA at entry — hard gate.** Enforced in `screener.py`. `vs_sma50` passed the
  Phase 2 pre-registered test (t 2.21).
- **Distance from the 20-day SMA — reported, not disqualifying.** Compute and state it for every
  candidate. A candidate below its 20-day SMA requires **written justification**, but is not
  automatically blocked. Phase 2 rated `vs_sma20` **UNPROVEN** (t 1.50, below the 2.0 bar).
  *Corrected 2026-09-17, same day:* Phase 3.5 found it tests **better at the adopted 40-session
  horizon** (IC 0.103) than at 10 (0.042). The downgrade stands — below the 20-day SMA is an
  argument to make, not an automatic block — but it no longer rests on "the signal does not
  work." It rests on the gate being **absolute and unwaivable** once the binary-catalyst waiver
  was deleted, which is too rigid for a signal this marginal. Revisit at Phase 4.

### Entry Requirements

- **Catalyst window (catalyst plays only): 90 calendar days.** Matches the 40–60 session holding
  horizon. Screener-sourced plays do not require a dated catalyst.
- **Binary-thesis entries are prohibited.** Do not initiate a position whose **thesis is the
  outcome of a binary event** — an announced-date, pass/fail event expected to move the stock
  ≥20% either way: FDA PDUFA decisions, pass/fail clinical readouts, contract award deadlines,
  permit or patent rulings, M&A close/termination.
  - *Why:* a stop cannot bound an overnight gap, so position size becomes the entire risk control
    and the framework was never designed for that. *Origin: TYRA, 2026-09-08 — gapped through its
    stop at the open exactly as the report forecast, −$29.04.*
- **Holding through a scheduled event is not a binary-thesis entry.** Every stock reports earnings;
  a 40–60 session hold spans at least one print by definition. Owning a company through its
  earnings date is normal and permitted. The prohibition is on **buying the coin flip**, not on
  **continuing to own a business**.
- **No initiation within 10 trading sessions before a known earnings date.** This is the guard that
  stops an unintended binary entry — buying days before a print on a non-binary thesis produces
  the same gap exposure under a different label. With the post-earnings cooldown in
  `entry-discipline.md`, both sides of a print are covered.
- **No re-entry ban:** once a ticker is stopped out it is banned from re-entry for 10 trading
  sessions. Flag any proposed re-entry inside the blackout window.

### No Candidates Rule
If no candidates pass all filters, hold cash and explain why. Do not force trades.

---

## Allocation Framework

Two buckets. Since binary-thesis entries are prohibited, both carry the same risk profile and are
**sized identically** — by the 2% risk budget and the 30% ceiling. They differ only in provenance
and in their capacity under RISK-OFF.

### Catalyst plays
- Require a confirmed, non-binary catalyst within **90 calendar days**.
- Sized by the standard risk formula. **There is no separate size cap.**
  *(The former 15%-per-play cap existed solely because a stop cannot bound a binary gap. With
  binary-thesis entries prohibited, its justification no longer exists.)*

### Screener-sourced plays
*(formerly "momentum/technical plays" — renamed 2026-09-17)*

- Sourced from the quantitative screener watchlist (`screener.py`).
- **No catalyst date required.**
- **Entry basis: the six signals the Phase 2 study actually validated** — low return volatility,
  proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio, and
  distance above the 50-day SMA. Together these describe **calm stocks near their highs with
  volume confirmation in an established uptrend** — low-volatility trend continuation.
- **20-day momentum is not an entry criterion.** It showed no ranking skill in this universe
  (IC 0.031, t 1.24 at 10 sessions) and is no longer scored by the screener. It is reported only.
  *This resolves the contradiction between this file and the screener flagged at Phase 3.*

### Capacity by regime

| | RISK-ON | RISK-OFF |
|---|---|---|
| Catalyst plays | up to 2 positions | **up to 3 positions**, standard sizing |
| Screener-sourced plays | up to 4 positions | permitted at **half the risk budget (1%)**, defensive profile only |
| Total positions | 5–6 ceiling | 5–6 ceiling |

- **RISK-OFF defensive profile** (all required): top-decile `low_vol` among gate survivors, near
  the 60-day high, and positive volume confirmation.
- **⏳ SUNSET — the RISK-OFF screener allowance expires at Phase 4** unless the pre-registered
  regime test confirms it. It rests on suggestive but incomplete evidence: Phase 2 found the
  signals' skill concentrates in weeks the median stock fell (IC 0.169 vs 0.009), but
  "falling-median week" and "RISK-OFF week" are different measurements, and the only direct
  RISK-OFF evidence is **2 formation dates** pointing the other way. If Phase 4 does not confirm
  it, this row reverts to a full freeze automatically. The top-decile threshold is likewise
  provisional — it was chosen by judgment, not by the study.
- *Why raise catalyst capacity under RISK-OFF:* the previous combination — freezing screener
  entries while capping catalyst plays at 1 position / 15% — made high cash **arithmetically
  unavoidable** rather than a market judgment. Catalyst returns are idiosyncratic and largely
  tape-independent, which is the exposure worth holding when market beta is the thing doing damage.

### Correlated-risk limits (repeated here because they bind at allocation time)
- At most **2 positions sharing a primary thesis driver**.
- At most **3 positions in any one GICS sector**.
