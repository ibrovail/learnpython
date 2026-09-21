# Week 54 — Research Report (2026-09-20)

First report of the indefinite phase run on a fired trigger rather than the calendar, and the
first written to the six-section format adopted 2026-09-19.

## 1. Scoreboard

| Metric | Value | Note |
|---|---|---|
| Gap vs S&P-equivalent, since re-base (2026-09-11) | **−0.76%** (−$5.57: $727.32 vs $732.89) | the scoreboard |
| Gap vs S&P-equivalent, since inception (2025-09-19) | −5.28% (−$40.55: $727.32 vs $767.87) | context only; TWR alpha since inception +0.23% |
| Current drawdown from peak | −1.28% | clear — DE-RISK at −20%, CASH at −30% |
| Regime | **RISK-OFF**, IWM −3.74% vs its 50-day ($284.10 vs $295.14) | from `<market_regime>`; RISK-OFF since 2026-08-31 (14 sessions) |
| Equity · cash · deployable above the 15% floor | $727.32 · $553.02 (76.0%) · **$443.92** | floor = $109.10 |
| Buys sought · stage-1 checks | 3 · 15 | from `<research_trigger>` |

The re-based gap is −0.76% after nine sessions, so the indefinite phase starts close to level and
nothing about this report is a catch-up exercise. The binding constraint is not conviction but
arithmetic: RISK-OFF halves the risk budget on screener-sourced entries to 1%, which — against
stops 1.75×ATR wide on low-volatility names — produces positions of roughly $200 each, and
$443.92 of deployable cash funds **two**, not the three the trigger asks for. The drawdown line is
nowhere near the breaker, so the circuit breaker does not shape anything here.

## 2. Deployment — the research funnel

Capacity under RISK-OFF: catalyst plays up to 3 positions at standard sizing; screener-sourced
plays permitted only at **half the risk budget (1% of equity)** and only on the defensive profile
(`rank_low_vol` ≥ 0.90, `near_high` ≥ −5%, `vol_5_50` > 1.0). Buys sought: 3. Only **7 of the 100
screened names** clear that defensive profile while also trading above their 50-day SMA — 5 in the
top 50 (CON, CFFN, CVBF, HOPE, LTC) and 2 in the extension (CLBK, HAE).

**A caveat on this week's volume signals.** Friday 2026-09-18 was quarterly triple-witching *and*
the trade date for the S&P index rebalance that takes effect before Monday's open. Every name on
the watchlist shows an inflated one-day volume ratio (2×–8.5×), and the two most extreme —
SHEN 8.51× (#45) and BXDC 8.40× (#18) — are index events, not information: SHEN is being **deleted**
from the S&P SmallCap 600 on 2026-09-21. Two of the six scored signals (`volume_ratio`, `vol_5_50`)
are therefore contaminated this week, and rank was treated as sourcing only, as the rules require.

### Stage 1 — quick checks

| # | Ticker | Source | Sector | Mkt cap | Rev growth | Next earnings | Result |
|---|---|---|---|---|---|---|---|
| 1 | CON | screener #1 | Health Care | $4.52B | +14.0% | 2026-11-05 | → stage 2 |
| 2 | CFFN | screener #3 | Financials | $1.10B | +17.2% | 2026-10-28 | → stage 2 |
| 3 | CVBF | screener #5 | Financials | $4.00B | +12.7% | 2026-10-21 | → stage 2 |
| 4 | OPCH | screener #7 | Health Care | $3.64B | +5.9% | 2026-10-29 | → stage 2 |
| 5 | EXPO | screener #10 | Industrials | $3.25B | +9.2% | 2026-10-29 | → stage 2 |
| 6 | HOPE | screener #12 | Financials | $1.78B | +31.1% | 2026-10-27 | → stage 2 |
| 7 | HRMY | screener #13 | Health Care | $2.45B | +24.3% | 2026-11-03 | → stage 2 |
| 8 | LTC | screener #15 | Real Estate | $2.29B | +59.4% | 2026-11-03 | PASS · valuation |
| 9 | TALO | screener #26 | Energy | $2.84B | +2.5% | 2026-11-04 | PASS · negative-earnings |
| 10 | KODK | screener #36 | Industrials | $935M | +9.5% | 2026-11-05 | PASS · negative-earnings |
| 11 | DNOW | screener #37 | Industrials | $2.82B | +69.8% | 2026-11-04 | PASS · negative-earnings |
| 12 | AVO | screener #39 | Consumer Staples | $1.13B | **−6.2%** | 2026-12 (est.) | PASS · shrinking-revenue |
| 13 | PUBM | screener #50 | Technology | $799M | **−1.0%** | 2026-11-09 | PASS · shrinking-revenue |
| 14 | CLBK | extended #51 | Financials | $3.12B | +42.7% | 2026-10-19 | → stage 2 |
| 15 | HAE | extended #78 | Health Care | $4.93B | +0.4% | 2026-11-05 | PASS · valuation |

**Spread:** ranks 1–15 → 8 names (CON, CFFN, CVBF, OPCH, EXPO, HOPE, HRMY, LTC); ranks 16+ → 7
(TALO, KODK, DNOW, AVO, PUBM, CLBK, HAE); GICS sectors → 7 (Health Care, Financials, Real Estate,
Industrials, Energy, Consumer Staples, Technology); below $2Bn → 5 (CFFN, HOPE, KODK, AVO, PUBM).
All four quotas met. No name was killed by the earnings guard: the whole list last reported between
2026-07-22 and 2026-09-08, so next prints cluster in late October and early November, 20+ sessions
out. Eighteen of the top 50 sat below their 50-day SMA and were excluded from the draw before
stage 1, per the trend rule.

### Stage 2 — full research

**CON (Concentra Group Holdings) — BUY · conviction 3/5**
- **Thesis:** the largest U.S. occupational-health network — work-injury care, DOT and pre-placement
  physicals, employer drug screening — growing revenue +14.0% and EPS +29.4% at 20.6× forward
  earnings with a 0.63 beta, and the only name on the list rated Strong Buy with a double-digit
  target gap. **Primary driver:** U.S. hiring, which sets pre-placement and workers-comp visit
  volumes — August payrolls **+162,000** against a +53,000 consensus, unemployment steady at 4.1%,
  with June and July both revised up (BLS via CNBC, released 2026-09-04). Four-week direction:
  **improving** (July −23,000 as first printed → +21,000 revised → August +162,000).
- **Catalyst:** none dated, and none required — this is a screener-sourced entry, not a catalyst
  play. The Q3 print on 2026-11-05 is the next scheduled event, 33 sessions out.
- **Quote page** (stockanalysis.com/stocks/CON/, accessed 2026-09-20; quote at close 2026-09-18
  4:00 PM EDT, after-hours $35.47 +0.01% at 7:30 PM EDT): price **$35.47**, TTM revenue $2.29B
  **+14.0%**, TTM EPS **$1.53 (+29.4%)**, net income $195.03M (+33.1%), PE 23.22, **forward PE
  20.57**, analysts **Strong Buy** (9), target **$41.00 (+15.59%)**, 52-week range $18.55–$36.71
  (price 3.4% below the high), **beta 0.63**, market cap $4.52B, volume 4.87M (rebalance-inflated).
- **Entry checks:** +2.02% above the 20-day SMA and +6.16% above the 50-day — inside the ≤20% and
  0–40% limits; not in days 1–3 of a >10% breakout (screener gate applied pre-ranking, and the
  20-day advance is 2%); next earnings 2026-11-05, far outside the 10-session guard; last print
  2026-08-06, so the 3-day post-earnings cooldown does not apply; not a binary thesis — no
  pass/fail event anywhere in the story; **prohibited-business check: clear** — occupational health
  services (Medical Care Facilities), not prisons/detention, weapons, predatory lending, and not
  Israeli-affiliated. Defensive profile: `rank_low_vol` 0.906 ✓, `near_high` −3.38% ✓,
  `vol_5_50` 1.926 ✓.
- **Bear case:** hiring is the whole driver, and a labour market that rolls over takes visit volumes
  with it — a single weak payrolls print can reprice this name faster than the stop can be adjusted;
  a 5.19× one-day volume ratio also means Friday's tape was mechanical, not a real demand signal.
- **Decision:** BUY 4 shares, limit $35.47, stop $33.92 / $33.77.

**HOPE (Hope Bancorp) — BUY · conviction 3/5**
- **Thesis:** a $1.78B Korean-American commercial bank two years into a balance-sheet repair that is
  now showing up in the numbers — TTM revenue +31.1%, TTM EPS $1.00 (+173.8%) — trading at **10.1×
  forward earnings** on a 4.0% dividend with the cheapest forward multiple on the list.
  **Primary driver:** the U.S. short-rate path through net interest margin. NIM was **2.96%** in
  Q2 2026, **+6bp** sequentially and **+27bp** year over year, with loan yields at 5.73% (+4bp) and
  interest-bearing deposit costs at 3.31% (**−6bp**); management guides to further, slower expansion
  in H2 as the CD book reprices. Four-week direction: **rising, with a new offset** — the FOMC
  **raised** rates 25bp to 3.75%–4.00% on **2026-09-16**, its first hike since 2023, with a majority
  of officials open to another this year. That lifts asset yields immediately but re-opens deposit
  competition, so the margin tailwind continues at a slower rate rather than reversing.
- **Catalyst:** acquisition of the **Commercial Banking Unit of SMBC MANUBANK**, which received
  **all** required regulatory approvals (FDIC and the California DFPI) on **2026-09-02** and is
  expected to close in **early Q4 2026** — inside the 90-day window and, with approvals already
  granted, **not a binary event**: the pass/fail step has already passed. Confirmed by the company
  release (2026-09-02) and by the earlier announcement trail on the same feed.
- **Quote page** (stockanalysis.com/stocks/HOPE/, accessed 2026-09-20; quote at close 2026-09-18
  4:00 PM EDT, after-hours $13.96 0.00% at 7:30 PM EDT): price **$13.96**, TTM revenue $534.03M
  **+31.1%**, TTM EPS **$1.00 (+173.8%)**, net income $127.81M (+185.0%), PE 13.99, **forward PE
  10.13**, analysts **Buy** (4), target **$15.50 (+11.03%)**, 52-week range $9.80–$14.59 (price 4.3%
  below the high), **beta 0.81**, dividend $0.56 (4.01%), market cap $1.78B.
- **Entry checks:** **+0.15% above both the 20-day and the 50-day SMA** — above the 50-day as the
  hard rule requires, but only just, and that thin margin is stated rather than glossed; no breakout
  to be early in; next earnings 2026-10-27, 25 sessions out; last print 2026-07-27, so no
  post-earnings cooldown; non-binary as set out above; **prohibited-business check: clear** — a
  regional commercial bank (Banks – Regional), not a payday or high-cost subprime lender, no
  weapons, prisons or Israeli affiliation. Defensive profile: `rank_low_vol` 0.971 ✓, `near_high`
  −4.32% ✓, `vol_5_50` 1.317 ✓.
- **Bear case:** the +174% EPS growth is measured against a 2025 that fell −38%, so the headline
  flatters a recovery rather than describing a franchise; and a Fed that has started hiking is a
  headwind for small-cap banks generally — deposit costs reprice faster than loan books when the
  curve moves against them.
- **Decision:** BUY 15 shares, limit $13.96, stop $13.48 / $13.38.

**CVBF (CVB Financial) — PASS · conviction 2/5**
- **Thesis:** clears every gate — defensive profile, +15.29% to a $26.17 target, forward PE 12.0,
  beta 0.65. **Primary driver:** the same rate path as HOPE. **Catalyst:** none dated.
- **Quote page** (accessed 2026-09-20; close 2026-09-18): $22.70, TTM revenue $572.48M +12.7%,
  **TTM EPS $1.43, −1.8%**, net income +1.5%, PE 15.84, forward PE 12.04, Buy, target $26.17,
  52-week $17.95–$23.41, beta 0.65.
- **Decision:** PASS on **capacity**, not on merit. Earnings are flat where HOPE's are inflecting,
  and a second regional bank would put two positions on one driver with no room left for a third —
  cash funds two buys. It is the first name back if either order is not filled.

**CFFN (Capitol Federal Financial) — PASS · conviction 2/5**
- The calmest name on the entire list (`rank_low_vol` 0.995) and the cheapest at 11.7× forward, but
  the **target is $9.50 against a $8.90 price — +6.74%** — and Piper Sandler's raise to that level
  came with a **Neutral** rating. A liability-sensitive thrift whose book is long-duration
  residential mortgage is also the wrong shape for a Fed that just hiked.
  **Decision:** PASS · valuation — 6.7% of upside does not pay for 2% of risk.

**CLBK (Columbia Financial) — PASS · conviction 2/5**
- The largest target gap on the list (+22.37%) and the lowest beta (0.26), fresh from a **second-step
  conversion completed in July 2026** that left it heavily capitalised. Against that: **PE 44.7**,
  a **2.44% NIM** (up 25bp YoY but still thin) that a hiking Fed pressures from the funding side,
  a price **−0.18% below its 20-day SMA**, only 3 analysts covering, and an **unexplained Friday
  after-hours move of −3.11% on 17.6M shares** that the news feed does not account for (last release
  2026-07-30). **Decision:** PASS · valuation — and an unexplained move is a reason to wait, not to
  size into.

**OPCH (Option Care Health) — PASS · conviction 2/5**
- Home-infusion services, +17.61% to target and 15.0× forward, but **`rank_low_vol` 0.849 fails the
  RISK-OFF defensive profile**, and with no dated catalyst it cannot come in as a catalyst play
  either. The chart is the tell: $24.30 against a 52-week high of $36.80, **−34%**, even while
  +4.57% above its 50-day. **Decision:** PASS · weak-catalyst.

**EXPO (Exponent) — PASS · conviction 2/5**
- A high-quality engineering-consulting franchise (+22.99% to target, 68% gross margins implied by a
  30.7 PE the market has long paid) that misses the defensive profile on **`near_high` −5.03%
  against a −5.00% limit** — by three basis points. The rule is a threshold, not a suggestion, and
  there is no confirmed catalyst to route it in the other way. **Decision:** PASS · weak-catalyst.
  Worth re-checking next weekend: a 0.03pp miss will not survive an ordinary up day.

**HRMY (Harmony Biosciences) — PASS · conviction 2/5**
- Genuinely cheap for the growth — $960M of TTM revenue (+24.3%) from WAKIX, 10.5× forward — but
  `rank_low_vol` 0.796 fails the defensive profile, and the only catalyst in view is the **TEMPO
  Phase 3 readout of pitolisant in Prader-Willi syndrome**, whose date could not be confirmed from
  any dated source: **INSUFFICIENT CONFIRMATION**. A pass/fail registrational readout is precisely
  the binary-thesis entry the rules prohibit. **Decision:** PASS · binary-thesis.

### Research log

All 15 funnel names logged with `log_research.py` (batch, 2026-09-20, week 54): **7 stage-1 kills**
— negative-earnings 3 (TALO, KODK, DNOW), shrinking-revenue 2 (AVO, PUBM), valuation 2 (LTC, HAE) —
and **8 stage-2 names**: **2 BUY** (CON, HOPE) and **6 PASS** — valuation 3 (CFFN, CLBK, and CVBF
logged under capacity), weak-catalyst 2 (OPCH, EXPO), binary-thesis 1 (HRMY). `research_log.csv`
holds 15 rows for the week.

## 3. Exact orders

- **Action:** BUY
- **Ticker:** CON (Concentra Group Holdings Parent, NYSE)
- **Shares:** 4
- **Order type:** limit
- **Limit price:** $35.47
- **Time in force:** DAY
- **Intended execution:** 2026-09-21 — run the pre-open check in `entry-discipline.md` first: if
  pre-market is down more than 2%, drop the limit to the pre-market price or pass; Friday's
  after-hours print was $35.47, unchanged, at 7:30 PM EDT.
- **Stop loss / stop limit:** **$33.92 / $33.77** — 1.75×ATR below entry (ATR(14) = 2.504% =
  $0.888; 1.75× = $1.554), below Friday's low of $34.78 and below the 20-day SMA ($34.77), above the
  50-day ($33.41). Place this stop with your broker before the next market open.
- **Sizing:** risk budget = **1% of equity under RISK-OFF** (half the standard 2%) = $7.27; shares =
  $7.27 ÷ $1.55 = 4.69 → **4 shares**; cost $141.88 = **19.5% of equity**; max loss at the stop
  $6.20 = **0.85% of equity**; order size is a rounding error against average daily dollar volume
  (well under 0.01%).
- **Rationale:** the highest-ranked name on the screen also happens to be the only Strong Buy with
  growing revenue *and* growing EPS, a 0.63 beta and a verified, improving driver.

- **Action:** BUY
- **Ticker:** HOPE (Hope Bancorp, NASDAQ)
- **Shares:** 15
- **Order type:** limit
- **Limit price:** $13.96
- **Time in force:** DAY
- **Intended execution:** 2026-09-21 — same pre-open check; Friday's after-hours print was $13.96,
  unchanged, at 7:30 PM EDT.
- **Stop loss / stop limit:** **$13.48 / $13.38** — 1.75×ATR below entry (ATR(14) = 1.965% =
  $0.274; 1.75× = $0.480), below Friday's low of $13.76 and below both the 20-day and 50-day SMA
  ($13.94). Place this stop with your broker before the next market open.
- **Sizing:** risk budget 1% of equity = $7.27; shares = $7.27 ÷ $0.48 = 15.1 → **15 shares**; cost
  $209.40 = **28.8% of equity**, inside the 30% ceiling ($218.20); max loss at the stop $7.20 =
  **0.99% of equity**; order size far below 10% of average daily dollar volume.
- **Rationale:** the cheapest forward multiple on the list, an inflecting margin, and a confirmed
  non-binary catalyst closing inside the holding horizon.

**No third order.** The trigger sought three buys and the funnel produced only two that clear the
RISK-OFF rules; after these orders $201.74 of cash remains, of which **$92.64** sits above the 15%
floor — below a viable position at the stop distances these names require. Per the No Candidates
Rule, the residue is held as cash rather than spent on the next-best name.

**Cash to the cent:** $553.02 − $141.88 − $209.40 = **$201.74** (27.7% of equity).

## 4. Holdings — by exception

| Ticker | Shares | Price | P&L | Stop (room in ATR) | Sessions held | Primary driver | Status |
|---|---|---|---|---|---|---|---|
| ATRC | 3 | $58.10 | **+69.4%** (+$71.40) | $55.27 / $55.12 (**1.28×ATR**) | 49 | U.S. Afib procedure volumes and AtriClip/cryo device adoption | **FULL** — index add completes at Monday's open; hold, no trim |
| CON | 4 | $35.47 (limit) | new | $33.92 / $33.77 (1.75×ATR) | 0 | U.S. hiring / occupational-health visit volumes | New position |
| HOPE | 15 | $13.96 (limit) | new | $13.48 / $13.38 (1.75×ATR) | 0 | U.S. short-rate path via NIM; MANUBANK CBU close | New position |

**ATRC — flagged FULL on volume (8.7× its 20-day average), and the subject of this weekend's
question.** The index event is confirmed from the primary source: S&P Dow Jones Indices' release of
**2026-09-04** lists "S&P SmallCap 600 · Addition · AtriCure · ATRC · Health Care", effective
**before the open on Monday 2026-09-21**. It is a **straight addition** — ATRC appears in no
corresponding deletion row, so this is not a MidCap 400 migration and the flow is one-directional:
SmallCap 600 trackers must buy. That is what Friday's 8.67M shares were, and what most of the
+12.8% since the announcement was.

**The hold-or-trim call, run as the thesis-exit test — would it be bought today at $58.10?**
As a *new* position it would not be: consensus sits at **$51.67, 11.07% below the price**, the
trailing PE is 275 and the forward 244, and the structural bid disappears at Monday's open. But
that test is not what the rules ask. A trim needs the same written, evidence-based case as a full
exit — **new, verified information that breaks or materially weakens the thesis** — and there is
none. The business printed 12.8% revenue growth in Q2 and raised FY26 guidance on both revenue and
adjusted EBITDA; Piper Sandler raised its target to **$60** on 2026-08-27 and BTIG to **$55** on
2026-08-24, both *above* the stale consensus average that UBS's $50 (7 weeks old) drags down.
Selling because a position is up 69% and looks extended is specifically named in the rules as *not*
a thesis exit — it is the stop's job, and the record behind that rule is 13 discretionary exits
netting −$0.30 against 14 stop exits netting +$61.00. **Decision: hold all 3 shares. No trim.**

**The honest risk, stated plainly:** the stop at $55.27 sits **1.28×ATR** below the price, inside
the 1.5×ATR placement band, and it cannot be lowered — the once-per-entry restoration was used on
2026-08-31. A single ordinary down day of 1.3×ATR takes this position out with +61.1% locked. No
raise qualifies either: the 2.0×ATR raise target is $53.68, which is *below* the existing stop, and
stops never move down outside restoration. So the position is carried as-is, and the honest
description is that ATRC is now a stop-managed position whose exit timing is largely out of our
hands. **Post-catalyst reassessment (the index add resolves Monday):** the normal trailing stop
would sit at $53.68 and the price is far above it, so there is no exit trigger; conviction stays
**4/5**, down from 5/5 — not on any thesis damage, but because the two things that drove the last
month's move (index demand and target raises) are now behind it, and 49 sessions in, the
**60-session re-underwrite falls due around 2026-10-05**.

## 5. Risk checks after proposed trades

| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | **PASS** — HOPE 28.8%, ATRC 24.0%, CON 19.5% |
| Risk per trade | ≤2% of equity at the stop | **PASS** — CON 0.85%, HOPE 0.99%, ATRC 1.17% (3.01% aggregate) |
| Cash floor | ≥15% of equity after trades | **PASS** — $201.74 = 27.7% against a $109.10 floor |
| Driver cap | ≤2 positions per primary driver | **PASS** — three distinct drivers: Afib procedure volumes (ATRC), U.S. hiring (CON), the short-rate path (HOPE) |
| Sector cap | ≤3 positions per GICS sector | **PASS** — Health Care 2 (ATRC, CON), Financials 1 (HOPE) |
| Position count | ceiling 5–6 | **PASS** — 3 positions |
| Slippage | each order ≤10% of average daily dollar volume | **PASS** — $141.88 and $209.40 against multi-million-dollar daily turnover |
| Regime capacity | RISK-OFF: ≤3 catalyst; screener at half risk, defensive profile only | **PASS** — both entries sized at the 1% half-budget; both clear all three defensive-profile thresholds |
| Circuit breaker | current drawdown vs −20% / −30% | **PASS** — −1.28% from the re-based peak |
| Exclusions | no prohibited business, binary thesis, or earnings inside 10 sessions | **PASS** — both checked at the PRV gate; next prints 33 and 25 sessions out |

Every row reads PASS. No order was withdrawn.

## 6. Thesis summary

**ATRC (AtriCure) — HOLD | Conviction 4/5 (from 5/5)**
+69.4% at $58.10 and 49 sessions held. Joins the **S&P SmallCap 600 before Monday 2026-09-21's
open** as a straight addition — confirmed from the S&P DJI release of 2026-09-04 — which is what
Friday's 8.67M shares were; the structural bid ends there. **Driver:** U.S. Afib procedure volumes
and AtriClip/cryo adoption, supported by Q2 revenue +12.8% and raised FY26 guidance.
**Stop $55.27 / $55.12, 1.28×ATR of room, and it cannot be lowered** — restoration was spent on
8/31 and the 2.0×ATR raise target ($53.68) sits below it, so no raise qualifies either. No trim:
the case for one rests on price level and a completed flow event, not on new negative information,
and the rules reserve discretionary selling for a verified thesis break. **What would change the
view:** a guidance cut, a competitive read-out against AtriClip, or the 60-session re-underwrite
due around 2026-10-05 — whichever comes first.

**CON (Concentra Group Holdings) — INITIATE | Conviction 3/5**
4 shares at $35.47, stop $33.92 / $33.77 (1.75×ATR), $141.88 = 19.5% of equity, risk 0.85%.
Occupational health with revenue +14.0%, EPS +29.4%, 20.6× forward, beta 0.63 and a Strong Buy with
a $41.00 target. **Driver:** U.S. hiring — August payrolls +162,000 against +53,000 expected
(2026-09-04), with June and July revised up; improving over four weeks. **What would change the
view:** two consecutive weak payroll prints, or any sign that employer visit volumes are lagging
headline hiring.

**HOPE (Hope Bancorp) — INITIATE | Conviction 3/5**
15 shares at $13.96, stop $13.48 / $13.38 (1.75×ATR), $209.40 = 28.8% of equity, risk 0.99%.
Revenue +31.1%, EPS +173.8% off a depressed 2025 base, 10.1× forward, 4.0% yield, an $15.50 target.
**Driver:** the short-rate path through NIM — 2.96% in Q2, +6bp sequentially, deposit costs
−6bp — now partly offset by the FOMC's 25bp **hike** on 2026-09-16 to 3.75%–4.00%. **Catalyst:**
the SMBC MANUBANK Commercial Banking Unit acquisition, fully approved on 2026-09-02, closing early
Q4. **What would change the view:** a second hike with deposit costs turning back up, or the
MANUBANK close slipping out of Q4.

**Portfolio.** Three positions and 27.7% cash after the orders, which is the RISK-OFF rules working
rather than a market call: the half risk budget plus 1.75×ATR stops sizes positions at roughly
$200, and $443.92 of deployable cash bought two of the three buys the trigger asked for. Both
entries are deliberately low-beta (0.63 and 0.81) and both clear the defensive profile on all three
thresholds — the book is not chasing the −0.76% gap, and after nine sessions there is nothing to
chase. **Before the next report:** the pre-open check on both limits Monday morning; ATRC's first
week without an index bid, and its 60-session re-underwrite around 2026-10-05; HOPE's MANUBANK
close in early Q4; and the October FOMC, which is now a live risk to the second position's driver.
Re-entry bans: PAR until ~2026-09-29, VTS until ~2026-09-30.

## Sources

- S&P Dow Jones Indices press release — https://press.spglobal.com/2026-09-04-Bloom-Energy,-Illumina,-and-Everpure-Set-to-Join-S-P-500-Others-to-Join-S-P-100,-S-P-MidCap-400,-and-S-P-SmallCap-600 — confirmed ATRC is a **straight addition** to the S&P SmallCap 600 effective before the open on 2026-09-21, with no offsetting MidCap 400 deletion, and SHEN's deletion from the same index — accessed 2026-09-20
- StockTitan, ATRC news feed — https://www.stocktitan.net/news/ATRC/ — live news check: no material release since the 2026-09-04 index announcement; latest items are conference transcripts — accessed 2026-09-20
- stockanalysis.com — https://stockanalysis.com/stocks/ATRC/ — ATRC quote page: $58.10, market cap $2.96B, TTM revenue $569.62M +13.9%, TTM EPS $0.21, forward PE 244, Buy with a $51.67 target (−11.07%), 52-week $25.36–$59.88, beta 1.26, earnings 2026-10-28; Piper $60 (2026-08-27) and BTIG $55 (2026-08-24) — accessed 2026-09-20
- stockanalysis.com — https://stockanalysis.com/stocks/CON/ — CON quote page and every figure in its stage-2 block — accessed 2026-09-20
- stockanalysis.com — https://stockanalysis.com/stocks/HOPE/ — HOPE quote page and every figure in its stage-2 block — accessed 2026-09-20
- StockTitan, HOPE news feed — https://www.stocktitan.net/news/HOPE/ — confirmed Bank of Hope received all required regulatory approvals (FDIC, California DFPI) for the SMBC MANUBANK Commercial Banking Unit acquisition on 2026-09-02, closing expected early Q4 2026 — accessed 2026-09-20
- Hope Bancorp Q2 2026 results (via Investing.com transcript and the company release) — https://www.investing.com/news/transcripts/earnings-call-transcript-hope-bancorp-tops-revenue-view-in-q2-2026-shares-rise-93CH-4814834 — NIM 2.96% (+6bp QoQ, +27bp YoY), loan yield 5.73%, interest-bearing deposit cost 3.31% (−6bp), guidance for continued but slower NIM expansion — accessed 2026-09-20
- CNBC — https://www.cnbc.com/2026/09/16/fed-rate-decision-september-2026.html — FOMC raised the target range 25bp to 3.75%–4.00% on 2026-09-16, first hike since 2023, with a majority open to another this year — accessed 2026-09-20
- CNBC — https://www.cnbc.com/2026/09/04/jobs-report-august-2026.html — August payrolls +162,000 vs +53,000 expected, unemployment 4.1%, June and July revised up — accessed 2026-09-20
- StockTitan, CON news feed — https://www.stocktitan.net/news/CON/ — live news check: nothing material since the 2026-09-14 Daytona Beach centre opening — accessed 2026-09-20
- StockTitan, CLBK news feed — https://www.stocktitan.net/news/CLBK/ — Q2 2026 results, NIM 2.44% (+25bp YoY), second-step conversion completed July 2026; no release explaining Friday's after-hours move — accessed 2026-09-20
- stockanalysis.com quote pages for the remaining funnel names — CFFN, CVBF, CLBK, OPCH, EXPO, HRMY, LTC, HAE, TALO, KODK, DNOW, AVO, PUBM (https://stockanalysis.com/stocks/TICKER/) — the revenue, EPS, rating, target, 52-week and earnings-date figures behind every stage-1 and stage-2 decision above — accessed 2026-09-20
- Harmony Biosciences investor relations — https://ir.harmonybiosciences.com/news-releases/news-release-details/harmony-biosciences-initiates-global-phase-3-registrational — TEMPO Phase 3 registrational trial of pitolisant in Prader-Willi syndrome is under way; no readout date disclosed (INSUFFICIENT CONFIRMATION) — accessed 2026-09-20

---

*Week 54 Research Report. Generated 2026-09-20 by Claude Code.*
