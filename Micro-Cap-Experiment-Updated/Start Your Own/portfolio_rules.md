# Portfolio Rules

These rules govern all analysis — daily and weekend. Read this file before beginning any analysis session.

---

## ⚠️ Final-stretch amendments (authorized 2026-08-15, Week 49)

With ~4.5 weeks left, a −4.8% benchmark gap and **54% of the book in cash**, the constraint
set — not the market — had become the binding limit on deployment (Week 48: 8 of 15 screened
candidates were excluded by rule, and the field yielded nothing investable). The user
authorized three relaxations for the remainder of the experiment:

1. **Health Care sector cap raised 2 → 3 positions.** Other sectors remain capped at 2.
2. **Universe market-cap ceiling raised $2B → $5B.** `screener.py` now fetches Finviz's
   mid-and-under bucket and trims to `MAX_MARKET_CAP = 5e9`.
3. **Adding to existing winning positions is permitted** as a deployment route, subject to
   the unchanged 30%-per-name cap and the no-averaging-down rule (which still forbids adding
   to a position more than 5% below entry without a confirmed new catalyst).

**Unchanged at the time:** 5% risk-per-trade, 30% single-name cap, all excluded security
classes (ETFs, closed-end funds/BDCs, SPACs, ADRs, units/warrants), stop-loss requirements
and range checks, and every entry-discipline rule. *(The 15% cash floor was subsequently
amended — see below.)*

---

## ⚠️ Cash floor amendment (authorized 2026-08-28, Week 50)

**Cash floor lowered 15% → 8%.** At 15% the floor had become a dead constraint: $11 of $128
was deployable, which neither protected the book nor funded a position. Analysis at the time
established that the floor was *not* the real binding limit — the **30% single-name cap** was,
since four of five holdings were disqualified as add targets on other grounds and the fifth
(PAR) had only ~$77 of headroom regardless. Lowering to 8% therefore unlocked essentially all
available deployment; lowering further would have unlocked nothing.

---

## ⚠️ One-time stop restoration (authorized 2026-08-31, Week 51)

**The "never lower a stop" rule is suspended for a single, documented adjustment, then
resumes in full.**

By 8/31 all five positions sat **inside 1.0×ATR** of their stops — ATRC 0.64×, CADL 0.48×,
WWW 0.68×, TILE 1.01×, PAR 1.30×. None had been tightened into that state; the market walked
prices down to fixed lines. The result was that every stop had drifted **below the
`max(1.5×ATR, …)` band this rules file itself mandates**, and that the one-open-order-per-stock
constraint made every position ineligible to receive new capital, since new shares inherit the
existing stop.

The decisive argument was arithmetic, not sentiment: with a −1.6% to −2.5% benchmark gap and
13 sessions left, **being stopped into cash locks the deficit permanently** — cash has no
mechanism to recover a gap. Meanwhile the $69 of free cash could not close it either (it would
need +27.5%, or SPY +9.2% at 3× leverage). Only the ~$683 of holdings could deliver the
required +2.79%, so the sole question worth acting on was whether those positions survive.

**Scope and limits of this authorization:**
- Applies to **ATRC, PAR and TILE only**, restoring each to ~1.75×ATR. CADL and WWW were
  deliberately left untouched.
- Each restored level was checked against the live price, today's low, the 10-day low and the
  ATR band before being named.
- **Single use.** The no-lowering rule resumes immediately afterwards. "Restore the band" is
  infinitely reusable as prices fall — that ratchet, not the $14 of added risk, is the reason
  the rule exists, and it is why this is bounded to one adjustment rather than a standing
  permission.

**Superseded 2026-09-02.** This ad-hoc authorization has been replaced by two standing rules
under *Risk Control* — the **anti-ratchet minimum** on raises and the **mechanical, once-per-
position restoration**. A conviction-gated version was considered and **rejected**: conviction
peaks on losers, regenerates at every new low, and would be certified by the same party that
chose the position. The outcome of this Week 51 exception is graded honestly in that section.

---

## Budget

- No new capital beyond what is shown in the portfolio snapshot unless explicitly approved.
- Track cash to the cent after every proposed trade.
- Out-of-cycle capital injections may be declared in `<capital_injection>`. When `planned=true`, add `<amount>` to the `Cash Balance` shown in `<portfolio_snapshot>` and use the combined total as available capital for all sizing calculations.

---

## Execution Limits

- Long-only. Full shares only (no fractional).
- No options, shorting, leverage, margin, or derivatives.

---

## Universe

- U.S.-listed common stocks: nano-cap to small-cap (market cap up to $2Bn).
- Allow up to $2Bn market cap for plays.
- Allowed exchanges: NYSE, NASDAQ, NYSE American.
- Existing positions above $2Bn may be held or sold; no new shares may be added.

---

## Exclusions

- OTC / pink sheets
- ETFs, ETNs, closed-end funds, SPACs
- Rights, warrants, units, preferred shares, ADRs
- Bankrupt or halted issuers
- Defence companies
- Israeli-affiliated companies

---

## Risk Control

- Maintain or set stop-losses on ALL long positions (default: max(1.5×ATR(14), 10% below entry)).
- **Raising a stop — anti-ratchet minimum.** Do **not** raise a stop unless the raise is
  **≥0.5×ATR(14)** *and* the new level still leaves **≥1.5×ATR** of room below the reference
  price. Both conditions, every time.
  - *Why:* small trailing raises bank trivial profit while measurably increasing stop-out
    probability, and because a stop can never be lowered afterwards, the cost is permanent.
    2026-08-27: ATRC's stop was raised $45.85 → $46.30 — **$1.35** of extra locked profit on
    3 shares — and by 8/31 the position sat **0.38×ATR** from being stopped out of the book's
    best thesis. The same session's report called the raise poor value while making it.
  - A raise that fails either test is **declined, not reduced**. Wait for the price to advance
    enough that a qualifying raise exists.
  - **Target the 1.75×ATR level, not the trailing floor.** The trailing-stop floor is defined
    at `max(1.5×ATR, 15% below the rolling high)`, and this rule requires ≥1.5×ATR of room —
    the *same number*. So raising **to** the floor always lands exactly on the boundary, where
    it either fails on floating-point or passes with zero margin. Compute the candidate level
    at **1.75×ATR** below the reference price (the target in `entry-discipline.md`), then apply
    the 0.5×ATR size test to that level. The floor is the minimum a stop may sit at, not the
    level to raise it to.
  - *Worked example, 2026-09-03 — both raises correctly declined:*

    | | Price | ATR | Stop | 1.75×ATR level | Raise size | Verdict |
    |---|---|---|---|---|---|---|
    | ATRC | $52.46 | 2.010 | $48.50 | $48.94 | **0.22×ATR** | blocked |
    | PAR | $18.90 | 0.850 | $17.05 | $17.41 | **0.42×ATR** | blocked |

    Both would have been made under the old regime. The 8/27 ATRC raise that created the
    0.38×ATR trap was 0.26×ATR — the same size as these.

- **Stop restoration — mechanical, once per position.** If a stop comes to sit **below
  1.5×ATR(14)** of the current price **through price movement alone** — never through
  tightening — it may be reset **once per position, for the life of the experiment**, to a
  level computed at **1.75×ATR** below the reference price. Never lower than that formula,
  never a second time.
  - This is the **sole exception** to "never lower a stop," and it is deliberately mechanical:
    it either drifted below the band or it did not. **Conviction, thesis strength, analyst
    targets and unrealised P&L are explicitly NOT inputs.** The restored level comes from the
    formula, not from judgment about the position.
  - The restored level must still pass the standard **range check** (below the most recent
    session's low). Where the formula and the range check disagree, the range check wins and
    the level goes below the low.
  - *Why conviction is excluded:* conviction peaks on losers. WWW carried a Buy rating and a
    price target whose upside **widened from +16% to +23% as the stock fell** — it would have
    scored *higher* on any conviction test at $19.73 than at entry, and a conviction gate
    would have funded the entire slide to −8.6%. As price falls, both the ATR band and the
    valuation case regenerate, so a discretionary version has no stopping point.
  - **Honest record of the one time this was used** (Week 51, 2026-08-31, as an ad-hoc
    suspension before this rule existed): of the three restorations, **two were unnecessary**
    — ATRC's old $46.30 stop was never touched (min low $46.61) and PAR's $17.50 was never
    approached (min low $18.28), together carrying **$10.80** of extra risk for nothing. The
    third, TILE, did prevent a stop-out at $37.10 (+15.6%) — but only delayed it: TILE stopped
    out on **2026-09-03 at $36.24 (+12.9%)**, so the restoration **cost 2.7 percentage points,
    about $3.44**. Final tally: **−$3.44 realised against $14.20 of additional risk carried**,
    with two of the three restorations never needed at all. The reasoning was sound; the
    outcome was not. Treat this exception as a narrow safety valve, not a tool.
  - **Eligibility is mechanical; using it is not.** The rule is permissive — a qualifying stop
    *may* be reset, not *must* be. 2026-09-03: CADL qualified (stop set at 1.61×ATR on 8/27,
    drifted to 0.71×ATR on price alone, allowance unused) and the restoration was **declined**,
    because the formula level of $11.45 converts a locked **+8.0% into +0.2%** to buy room for
    a thesis whose only catalyst (the CAN-2409 BLA, Q4 2026) falls outside the experiment.
    Room is worth paying for only when something can happen inside the runway to use it.

- **Anti-ratchet, back-tested against every raise actually made** (Aug 24 – Sep 2). It blocks
  the churn and permits the substance, which is the whole intent:

  | Date | Raise | Size | Room left | Verdict |
  |---|---|---|---|---|
  | 08-24 | ATRC $44.20→$45.85 | 0.98×ATR | 1.86×ATR | allowed |
  | 08-24 | PAR $16.50→$17.50 | 0.98×ATR | 1.99×ATR | allowed |
  | 08-25 | CADL $12.00→$12.19 | 0.25×ATR | 1.88×ATR | **blocked** |
  | 08-27 | ATRC $45.85→$46.30 | 0.26×ATR | 1.75×ATR | **blocked** |
  | 08-27 | CADL $12.19→$12.35 | 0.18×ATR | 1.61×ATR | **blocked** |
  | 09-02 | ATRC $44.35→$48.50 | 2.29×ATR | 2.25×ATR | allowed |

  Had the rule been in force, ATRC would have entered 8/31 with a **$45.85** stop rather than
  $46.30 — **0.90×ATR** from the price instead of 0.38×ATR. Still inside the band, so the
  restoration would still have been available, but the position would never have come within
  31 cents of being stopped out of the book's best thesis.

- **Restoration used to date:** ATRC, PAR, TILE (all 2026-08-31). **None of the three is
  eligible again.** CADL and WWW never used theirs.

- **Binary event stop override:** for positions held through a date-certain binary catalyst (see definition in Entry Requirements), the stop-loss may be set at the nearest major technical support level (200-day SMA, prior selloff floor, key horizontal support) rather than the standard ATR/percentage formula, provided: (a) the wider stop still results in ≤5% portfolio risk (or ≤3.75% if the SMA waiver was used for entry), (b) the override rationale is documented in the weekly report, and (c) the override automatically expires when the event resolves — see post-catalyst reassessment.
- **Position sizing (risk-per-trade):** size so that hitting the stop costs no more than 5% of portfolio equity:
  ```
  shares = (portfolio_equity × 0.05) / (entry_price − stop_price)
  ```
  Absolute ceiling: no single name may exceed 30% of portfolio equity.
- **No averaging down:** once a position falls >5% from entry, do not add shares unless a material new positive catalyst is confirmed with ≥2 independent sources.
- **Partial profit-taking:** sell ~1/3 at +30% gain, ~1/3 at +60% gain; let the remaining third run with a trailing stop at max(1.5×ATR(14), 15% below 20-day rolling high).
- **Partial profit deferral:** The partial profit rule may be deferred at any stage when all of the following are true:
  1. **New catalyst:** A material new catalyst has emerged since entry that was not part of the original thesis (e.g., contract win, strategic partnership, regulatory approval, major customer announcement), confirmed by ≥2 independent sources.
  2. **Trailing stop protection by stage:**
     - Deferring at +30%: trailing stop must lock in at least **+15%** from entry
     - Deferring at +60%: trailing stop must lock in at least **+40%** from entry
  3. **Position cap:** Position must not exceed 30% of portfolio equity.
  4. **Conviction:** Position conviction must be ≥4/5, assessed on the underlying thesis and catalyst — not on macro or broad market conditions.
  - **Management:** The trailing stop replaces the partial sell as the primary risk control. Continue raising the stop per the standard trailing rule. Document deferral status and updated rationale in each weekend report. **Re-evaluate the deferral** (not forced execution — re-evaluate) if: the catalyst fails to deliver (contract canceled, partnership dissolved, regulatory setback); conviction drops below 4/5 on position-specific factors (not macro); two earnings cycles pass without the catalyst materially impacting revenue, guidance, or fundamentals; or the position exceeds 30% of equity.
  - **Multiple deferrals:** Multiple partials may be deferred simultaneously on the same position if each independently meets the criteria above. However, if both the +30% and +60% partials are deferred on the same position, the smaller of the two deferred partials (~1/3) must be executed if the position pulls back more than 15% from its post-catalyst high. This ensures at least partial profit is taken on a meaningful reversal, rather than riding a full round-trip on zero realized gains.
- **Pre-catalyst exit orders:** for any position held through a date-certain binary catalyst, set a price alert at +30% from entry at least 2 trading days before the event date. When the alert triggers, **cancel the protective stop, place a DAY limit sell** for ~1/3 of the position at the alert price, and re-place the stop if unfilled by close. This captures spike-and-reverse profit and reduces gap risk. If the sell fills before the event, do NOT replace the sold shares. *(See Order Defaults — one open order per stock: the exit-limit and the protective stop cannot be armed simultaneously.)*
- **Post-catalyst reassessment:** within 1 trading day of any date-certain binary catalyst resolving (approval/rejection, beat/miss, awarded/denied): (1) remove any binary event stop override and recalculate the stop using normal trailing stop rules, (2) re-evaluate conviction with documented rationale, (3) if the stock is trading below where the normal trailing stop would be, either document a specific time-bound reason to hold or exit at market, (4) log the assessment in the daily analysis.
- **Market regime filter:** if IWM is below its 50-day SMA, restrict new initiations to high-conviction catalyst-driven plays only. Freeze new momentum/technical initiations until the next weekend review. Existing momentum positions are held with current stops. Flag the regime status in every report.
- **Slippage guard:** if the intended order size exceeds 10% of the stock's average daily dollar volume, reduce the position to ≤5% of ADV.
- Flag any stop breach or position sizing violation immediately.

---

## Order Defaults

- Standard limit DAY orders placed for the next trading session unless otherwise specified.
- Limit orders preferred. Market orders require explicit reasoning.
- **Platform constraint — one open order per stock.** The broker allows only ONE resting order per position, and each holding's protective **GTC stop-limit occupies that slot**. Therefore a profit-take / partial / exit-into-strength limit **cannot coexist with the protective stop** — do NOT recommend "arming" a DAY limit sell alongside an active stop; it is not executable.
  - **The protective GTC stop-limit is the default resting order** on every holding.
  - **Profit-take / exit-into-strength targets are price ALERTS, not resting orders.** When an alert triggers and the user chooses to act, the workflow is: **cancel the stop → place the DAY limit (or market) sell → if unfilled by close, re-place the stop.** There is a brief unprotected window during the swap, acceptable only while actively watching.
  - **Practical consequence:** if a position is meant to be managed by its stop (e.g. a fading-catalyst runner), the stop IS the exit — don't also plan a resting exit-limit. Only pursue the alert-and-swap when actively watching for an intraday rally.
  - GTC limit sells are not supported at all; only GTC stop-limit sells.

---

## Research Safeguards

### Verification
- Do NOT hallucinate tickers. Every ticker must be a verified, currently listed U.S. security on an allowed exchange.
- All market cap, float, liquidity, and catalyst data must come from reputable, up-to-date sources and must be confirmed by at least two of the sources.
- Provide citations for every holding and new candidate: source name, URL, and access timestamp.

### Catalyst Confirmation
- Any claim about catalysts (earnings dates, contract awards, regulatory decisions, etc.) must be confirmed by at least two independent sources.
- If confirmation is insufficient, explicitly state "INSUFFICIENT CONFIRMATION" and do not rely on it.

### Liquidity Filters
- Price ≥ $1.00
- 3-month average daily dollar volume ≥ $500,000
- Bid-ask spread ≤ 2% (or ≤ $0.05 if price < $5)
- Float ≥ 5M shares (unless justified with reasoning)
- Relative strength: stock price must be above its 20-day SMA at the time of initiation, **UNLESS** all of the following are true: (1) the stock has a date-certain binary catalyst within 15 trading days, (2) the stock is within 5% of its 20-day SMA, and (3) the catalyst is confirmed by ≥2 independent sources. When this waiver is applied, reduce the risk budget from 5% to 3.75% of equity.

### Entry Requirements
- **Catalyst within 60 days (catalyst plays only):** catalyst-play initiations must have a confirmed near-term catalyst (earnings, FDA decision, contract award, etc.) within 60 days. Momentum/technical plays from the screener watchlist do not require a dated catalyst — see Allocation Framework below.
- **Date-certain binary catalyst (definition):** an event with a publicly announced date or regulatory deadline and a pass/fail outcome expected to move the stock ≥20% in either direction. Examples across sectors: FDA PDUFA date, earnings report date, government contract award deadline, permit ruling date, drill/assay result release date, patent ruling date, M&A close/termination date. Excluded: vague timelines ("H1 2026 data readout"), analyst day presentations, conference appearances.
- **No re-entry ban:** once a ticker is stopped out, it is banned from re-entry for 10 trading days. Flag any proposed re-entry that falls within the blackout window.

### No Candidates Rule
If no candidates pass all filters, hold cash and explain why. Do not force trades.

---

## Allocation Framework

The portfolio uses two complementary strategies:

### Catalyst Plays (existing rules above apply)
- Require confirmed catalyst within 60 days
- Binary event framework applies for date-certain events
- **Max allocation: 1 position, max 15% of equity per binary event play**
- All existing risk control rules apply

### Momentum/Technical Plays
- Sourced from the quantitative screener watchlist (`screener.py`)
- **No catalyst date required** — entry based on momentum, volume confirmation, and technical setup
- Must still pass all liquidity filters (price ≥$1, ADV ≥$500K, float ≥5M)
- Must be above 20-day SMA at entry (no SMA waiver for momentum plays)
- Standard stop-loss: max(1.5×ATR(14), 10% below entry)
- **Minimum hold: 5 trading days** unless stop is triggered (prevents overtrading)
- Max allocation: 3-4 positions

### Sector Diversification
- **No more than 2 of 5 positions may be in the same GICS sector.** If 2 positions are in the same sector, new candidates from that sector are blocked until one exits.
- The screener watchlist includes sector tags — check before initiating.
