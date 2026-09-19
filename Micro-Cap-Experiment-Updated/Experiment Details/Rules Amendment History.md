# Rules Amendment History

The standing rules live in `Start Your Own/portfolio_rules.md`. That file states the rules **as
they are**. This file holds **how they got that way** — every dated amendment, the failure that
prompted it, and the honest accounting of whether it worked.

Split out 2026-09-17. Before that date both lived in one file, which had grown to 24KB of which
roughly the first 40% was dated amendment notices — a changelog wearing the costume of an
operating manual. The full superseded text is preserved in git at `e838891`.

---

# 2026-09-17 — Graduation to an indefinite live system

The largest single revision in the project's history: **13 decisions, 6 rules deleted outright,
3 new rules, and 1 research study pulled forward by three months.**

## Why this review happened

The 52-week experiment closed 2026-09-11 and the horizon was set to `null` on 2026-09-14. But
only the *horizon* was changed. The rules themselves had been written against a deadline and
were inherited wholesale. A review found that a substantial number of them were still shaped by
a race that no longer had a finish line — and, in two cases, actively argued the wrong way.

## Decisions

### Strategy

| # | Decision | Replaces |
|---|---|---|
| 1 | **Holding horizon: 40–60 sessions** (months, not days) | Implicit 5–20 session horizon inherited from the catalyst-timing directive |
| 2 | **Universe ceiling $5Bn, frozen until Phase 4** | Open question; a proposal to raise it was declined |
| 3 | **Trailing stop is the only exit** | Mechanical partials at +30% / +60% |
| 4 | **Mandatory written re-underwrite at 60 sessions** | Nothing — new |
| 5 | **Binary-thesis entries prohibited**; catalyst window 60 → **90 calendar days** | Binary plays permitted at 1 position / 15% equity |

**On raising the market-cap ceiling (declined).** A proposal to raise the $5Bn ceiling was
considered and rejected on four grounds. (1) *Alpha requires dispersion, and dispersion falls as
market cap rises* — a narrower spread of outcomes means even perfect selection earns less.
(2) *Factor contamination*: `low_vol` and `squeeze` are strongly related to company size; inside a
$5Bn band "calm" carries stock-specific information, but across a $20Bn band it degenerates into
"largest company in the list," turning the book into a crowded low-volatility large-cap strategy
that was never what Phase 2 measured. (3) *Phase 4 integrity*: the universe already shifted twice
mid-Phase-2 (−25% in July, +55% on 2026-08-16); changing it again during Phase 4's 12-screen
collection window destroys the comparison. (4) *No operational pressure*: at a $724 book with
~$120 orders and a 10%-of-ADV slippage cap, capacity exceeds need by roughly 240×.

The deployment argument was examined separately and found to rest on a misdiagnosis: the
2026-09-14 screen produced **814 qualifying stocks** and the book still bought nothing. The
binding constraint was the regime freeze and the catalyst cap, not universe size — addressed by
decisions 11 and 12 instead.

### Risk

| # | Decision | Replaces |
|---|---|---|
| 6 | **Risk per trade: 2% of equity** | A standing contradiction: `portfolio_rules.md` said 5%, `entry-discipline.md` said 2% |
| 7 | **Drawdown circuit breaker** at −20% / −30%, on the injection-neutral series | Nothing — new |
| 8 | **Driver cap (2 per shared thesis driver) + uniform GICS sector cap of 3** | "2 of 5 positions" per sector, with Health Care specially raised to 3 |
| 9 | **Initial entry stop unified at 1.75×ATR** | `portfolio_rules.md` said 1.5×ATR at entry; `entry-discipline.md` said 1.75× |
| 10 | **Stop restoration: once per *entry*** | "once per position, for the life of the experiment" |

**On the 5% vs 2% risk contradiction.** Resolved in favour of 2% after checking what it actually
costs. On a real entry — $57.14 with a 1.75×ATR stop at $53.40, so $3.74/share of risk, against a
$724 book — 2% yields 3 shares while 5% yields 9 shares that the 30% single-name cap immediately
cuts back to 3. **Identical outcome at current size.** The 30% cap had been doing all the work;
adopting 2% costs nothing today and becomes the real constraint as the book grows, which is when
it matters. At 5% across 5 positions, 25% of equity would be at risk simultaneously.

**On the drawdown trigger being deliberately *not* gap-based.** A gap trigger fires when the index
rises, which creates pressure to manufacture a recovery. That pressure has a name in this book:
TYRA, −$29.04 on a binary readout the weekend report had explicitly recommended against, taken as
an aggressive override while trailing the benchmark with seven sessions left. Drawdown measures
capital destroyed; the gap measures someone else's good year.

**On why the sector cap is 3 and uniform.** The Health Care exception at 3 had been adopted in
Week 49 purely to unblock deployment under deadline pressure and was never re-justified on merit.
Rather than revert it to 2, the cap was levelled *up* — because the diagnosis was that GICS sector
is simply the wrong unit. It is **too tight** (a device maker and a clinical-stage biotech share
nothing but a label) and **too loose** (an oil-services name and a tanker operator sit in different
sectors and live on the same crude price). The two stop-outs of 2026-09-15/16 illustrate it: PAR
died in a restaurant-tech selloff and VTS on WTI −3.2%, in different sectors, satisfying the cap
entirely while the book lost two positions in 48 hours. The **driver cap** targets the real
exposure; the sector cap is a coarse net beneath it.

### Deployment

| # | Decision | Replaces |
|---|---|---|
| 11 | **RISK-OFF catalyst capacity: up to 3 positions, standard sizing** | 1 position, 15% of equity |
| 12 | **RISK-OFF screener entries permitted at half risk budget**, defensive profile, **sunsets at Phase 4** | Total freeze on screener-sourced initiations |
| 13 | **15% cash floor promoted to a standing rule** | Existed only inside a dated amendment block, absent from Budget and Risk Control |

**The design bug these fix.** Under RISK-OFF the rules froze screener-sourced plays — the only
bucket with capacity — while capping catalyst plays at 1 position / 15% of equity. Maximum
possible deployment was therefore ~15% plus existing holdings. The book's 76.3% cash position on
2026-09-16 was not a market judgment or a failure of nerve; it was **arithmetically unavoidable**.
No amount of research effort could have deployed that capital, because no permitted strategy had
room for it.

A **benchmark-parity floor** (parking excess cash in SPY/IWM above a threshold) was considered and
**not adopted**. It would have removed the drag outright, but it pulls the gap metric toward zero
rather than positive — it stops you losing to the index without helping you beat it — and it would
have required carving an exception into the ETF exclusion.

### Research

| # | Decision |
|---|---|
| 14 | **Phase 3.5 runs now**, pre-registered, testing 40 and 60 sessions on existing reconstructed data |
| 15 | **Phase 4 scope expanded**: 3-way score comparison, fundamentals, the unexplained watchlist gap, **and a pre-registered regime test** |

### Documentation

| # | Decision |
|---|---|
| 16 | Standing rules split from amendment history (this file) |
| 17 | Benchmark gap reported **both** since inception and since re-base, permanently |
| 18 | "Momentum/technical plays" renamed **screener-sourced plays**, entry basis rewritten to the six validated signals |
| 19 | "Experiment" framing retired from the operating docs |

**Decision 18 closes the open question left by Phase 3.** The rulebook had continued to define
these entries as based on "momentum, volume confirmation, and technical setup" after Phase 2 found
20-day momentum has no ranking skill and the screener stopped scoring it. The rulebook and the
screener had been describing two different strategies. The six surviving signals describe **calm
stocks near their highs with volume confirmation in an established uptrend** — low-volatility
trend continuation, not momentum.

## Rules deleted outright

| Rule | Why it no longer exists |
|---|---|
| **Partial profit-taking** (+30% / +60%) | Superseded by trailing-stop-only (decision 3) |
| **Partial profit deferral** — 4 conditions, stage-based stop floors, the multiple-deferral pullback clause | Machinery for a rule that no longer exists. ~20 lines, the most complex apparatus in the file |
| **Binary event stop override** | Nothing left to override (decision 5) |
| **Pre-catalyst exit orders** (+30% alert → cancel stop → limit sell) | Depended on both binary catalysts and partials |
| **20-day SMA waiver** | Its trigger condition was "a date-certain binary catalyst within 15 trading days" — now impossible. It would have sat permanently unreachable |
| **5-day minimum hold** | Subsumed by a 40–60 session horizon |

## Findings from the pre-implementation critique

The consolidated decision set was critiqued before implementation. Seven problems were found,
three of them blocking. Recording them because the errors are instructive:

1. **"Ban binary events" + "hold for months" would have broken the system outright.** The
   rulebook's definition of a date-certain binary catalyst explicitly listed **"earnings report
   date"** among its examples. A 40–60 session hold spans at least one print by definition, so the
   ban as drafted would have prohibited holding any stock at all. Resolved by separating a
   *binary-thesis entry* from *holding through a scheduled event*, plus a 10-session pre-earnings
   initiation guard.

2. **Unifying the stop at 1.75×ATR would have broken the anti-ratchet rule.** The 1.5× / 1.75×
   split is deliberate, not a contradiction: 1.5×ATR is the floor a stop may never sit below,
   1.75×ATR is where it is placed. The anti-ratchet test requires headroom between them and says
   so explicitly. The genuine contradiction was narrower — the *initial entry* stop only.

3. **The drawdown trigger would have been contaminated by capital injections.** `max_drawdown`
   (`trading_script.py`) runs on raw equity with no injection adjustment, unlike TWR and the
   S&P-equivalent which are explicitly filtered. Against $547.64 of injections into a book that
   started at $142.13, a deposit would reset the peak and move the trigger with no market event
   behind it. Resolved by specifying the injection-neutral series and naming both peaks.

4. **A claim made during the review was simply wrong.** The sector cap was described as "enforced
   in the screener as it is today." It is not: `--max-per-sector` governs watchlist composition
   and the screener has no knowledge of the portfolio. Both halves of decision 8 are research-time
   judgment — which weakened the argument that had been used against a driver-cap-only option.
   Resolved by surfacing sector counts and driver tags in the portfolio snapshot: **surfaced, not
   gated.**

5. **The catalyst bucket had kept a 15% size cap whose justification had just been deleted.** That
   cap existed *because* a stop cannot bound a binary gap. Carrying the number forward after
   removing its reason is the same error the review was convened to fix.

6. **Phase 4 could not have answered the horizon question in December.** A 60-session forward
   measurement needs ~60 sessions *after* each formation date; screens began 2026-09-14, yielding
   2–3 usable dates by December. The Phase 2 infrastructure already reconstructs universes back to
   2026-04-17, and April–June dates carry full forward data today — so the most load-bearing
   decision in the ledger was testable immediately. Became Phase 3.5.

7. **"Provisional pending Phase 4" is a note, not a mechanism.** Decision 12 was given an
   automatic sunset instead.

Two smaller items: the **Day-1 drawdown rule** still justified itself by reference to "a
momentum-screened name" after momentum ceased to be a selection criterion; and **"must be above
the 20-day SMA at entry"** had become unwaivable while resting on `vs_sma20`, which Phase 2 rated
UNPROVEN (t 1.50). The first was re-grounded, the second downgraded to a reported check requiring
written justification.

---

# Superseded amendment blocks

*Preserved verbatim from `portfolio_rules.md` as it stood at `e838891`, before the 2026-09-17
restructure. These describe the rules as they were, and are retained for the reasoning they
contain — not as current policy.*

## ⚠️ Indefinite phase begins (decided with the user 2026-09-14)

**The 52-week experiment is closed as of the 2026-09-11 close.** The portfolio continues as an ongoing live process with no end date. Decisions:

- **Horizon:** `experiment_config.json` → `end_date` and `total_weeks` set to `null`. With no runway to plan against, catalysts are judged against the full **60-day** window in the Allocation Framework, not an experiment end date.
- **Benchmark gap re-based at the 2026-09-11 close.** The 52-week result — equity **$733.51** vs S&P-equivalent **$768.52**, **gap −4.56%**, TWR alpha **+1.11%** (first printed as +0.55%: the S&P leg of TWR started one session early, on 2025-09-18; fixed 2026-09-14 — the gap was unaffected) — is recorded in the final readout and does not carry forward. The indefinite phase keeps its own scoreboard from that date.
- **Cash floor reverted 8% → 15%.** The Week 50 cut existed only because the runway was short.
- **Week 49 final-stretch amendments retained** — healthcare cap of 3, $5Bn ceiling, adding to winners. The deadline prompted them, but none adds risk over a longer horizon.
- **Retired:** the hold-through-the-close endgame directive, and the pre-committed ATRC 1-share partial at $54.88, whose rationale (BoxX-NoAF data falling outside the runway) no longer holds. ATRC's partial is governed by the standard deferral criteria, re-checked each weekend.
- **Unchanged — the market regime filter.** It depends on market conditions, not the calendar. Removing the deadline removes any reason to force momentum entries into a RISK-OFF tape. **An earnings date alone does not make a name a catalyst play**: the catalyst must be the thesis, not merely a date on the calendar that happens to fall inside 60 days.

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



---

# Superseded Risk Control section

*Also preserved verbatim from `e838891`. The new standing rules carry the conclusions of this
section in condensed form; the evidence tables — the anti-ratchet back-test, the declined-raise
worked example, and the honest accounting of the one stop restoration ever used — are kept here
in full, because they are the reason those rules are written the way they are.*

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

