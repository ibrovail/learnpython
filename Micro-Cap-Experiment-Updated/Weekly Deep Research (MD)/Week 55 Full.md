# Week 55 — Research Report (2026-09-28)

Run Monday pre-market rather than over the weekend, so the screen's formation date is 2026-09-28
and every quote below carries Friday's close with a Monday pre-market print beside it. The trigger
fired on the free slot; the funnel produced **no buy**.

## 1. Scoreboard

| Metric | Value | Note |
|---|---|---|
| Gap vs S&P-equivalent, since re-base (2026-09-11) | **−2.27%** (−$16.84: $724.95 vs $741.79) | the scoreboard |
| Gap vs S&P-equivalent, since inception (2025-09-19) | −6.72% (−$52.25: $724.95 vs $777.20) | context only; TWR alpha −1.54% |
| Current drawdown from peak | −1.60% | clear — DE-RISK at −20%, CASH at −30% |
| Regime | **RISK-OFF**, IWM −4.09% vs its 50-day ($281.97 vs $294.00) | 19 sessions, since 2026-08-31 |
| Equity · cash · deployable above the 15% floor | $724.95 · $237.33 (32.7%) · **$128.59** | floor $108.74 |
| Buys sought · stage-1 checks | 1 · 10 | from `<research_trigger>` |

The gap widened 1.51pp over the week, and the whole move is ATRC: it was −1.18% on Thursday with
ATRC at $60.77, and −2.27% on Friday after it closed $58.38. Nothing else changed materially — CON
and HOPE both finished the week above where they traded on Wednesday. The week's lesson is
concentration: a 24% position with a 3.9% ATR moves the scoreboard more than the other two
holdings and a third of the book in cash combined.

## 2. Deployment — the research funnel

Capacity under RISK-OFF: up to 3 catalyst plays at standard 2% sizing, plus screener-sourced
entries at **half the risk budget (1%)** and only on the defensive profile (`rank_low_vol` ≥ 0.90,
`near_high` ≥ −5%, `vol_5_50` > 1.0). Buys sought: 1.

**The screen could not supply a candidate, and the reason is worth stating precisely.** Of the top
50, only **23 trade above their 50-day SMA** — the hard trend gate removes 27 before research
starts. Of all 100 names across both lists, exactly **one** clears the RISK-OFF defensive profile:
**LTC**, at rank 14. That is the same name this book passed a week ago, and its consensus upside has
since fallen from 5.7% to **4.8%** against a Hold rating. So the defensive-profile route offered one
name, and it was the one name already rejected on its merits. Every other candidate had to qualify
as a catalyst play, and none had a dated, confirmed, non-binary catalyst inside 90 days.

### Stage 1 — quick checks

| # | Ticker | Source | Sector | Mkt cap | Rev growth | Next earnings | Result |
|---|---|---|---|---|---|---|---|
| 1 | VFF | screener #8 | Consumer Defensive | $388M | +15.3% | 2026-11-09 | PASS · risk-off profile |
| 2 | SIBN | screener #9 | Health Care | $833M | +15.3% | 2026-11-09 | PASS · negative-earnings |
| 3 | LTC | screener #14 | Real Estate | $2.31B | +59.4% | 2026-11-03 | PASS · valuation |
| 4 | MQ | screener #15 | Technology | $1.77B | +22.4% | 2026-11-03 | PASS · Hold, profile |
| 5 | CYRX | screener #16 | Industrials | $882M | +12.1% | 2026-11-03 | PASS · negative-earnings |
| 6 | CNK | screener #19 | Communication Services | $4.33B | +4.5% | 2026-11-04 | PASS · valuation |
| 7 | ADMA | screener #22 | Health Care | $2.10B | +8.0% | 2026-11-04 | PASS · profile, no catalyst |
| 8 | HNI | screener #29 | Consumer Cyclical | $3.41B | +70.1% | 2026-10-27 | PASS · TTM EPS −97.8% |
| 9 | GCT | screener #31 | Technology | $1.93B | +22.9% | 2026-11-05 | PASS · valuation |
| 10 | EGY | screener #48 | Energy | $612M | **−25.5%** | 2026-11-09 | PASS · shrinking-revenue |

**Spread:** ranks 1–15 → 4 names (VFF, SIBN, LTC, MQ); ranks 16+ → 6 (CYRX, CNK, ADMA, HNI, GCT,
EGY); GICS sectors → 8; below $2Bn → 5 (VFF, SIBN, MQ, CYRX, GCT at $1.93B). All four quotas met.
No name was killed by the earnings guard — the whole list reports between 2026-10-27 and 2026-11-09,
21 sessions out or more. **Prohibited-business check: all ten clear.** VFF is a cannabis and produce
grower, which `portfolio_rules.md` explicitly permits; no name sits in Security & Protection
Services or Credit Services.

**Why each one failed, in one line:**
- **VFF** — beta 1.38 and `rank_low_vol` 0.78 fail the defensive profile; a Strong Buy with +83.9%
  to target, but a $3.17 stock at 1.4 beta is the opposite of what RISK-OFF permits, and there is
  no dated catalyst to route it in as a catalyst play.
- **SIBN** — TTM net income −$14.63M, EPS −$0.33, and 8.4% below its 60-day high, so it fails both
  the earnings quality test and the profile's `near_high` gate.
- **LTC** — the only profile qualifier, and a Hold with **+4.8%** to target. 4.8% of upside does not
  pay for 2% of risk.
- **MQ** — Hold consensus, beta 1.29, forward PE 49, and 9.3% below its 60-day high.
- **CYRX** — TTM net income −$45.39M, EPS −$0.91, beta **1.90**.
- **CNK** — PT +5.4% at the 52-week high, with TTM EPS −4.0% and net income −26.3%.
- **ADMA** — profitable and cheap (PE 13.4) but fails the profile, has no dated catalyst, TTM EPS is
  **−17%**, and the price sits **54% below** its 52-week high. A +81% target gap on a chart like
  that is analysts disagreeing with the market, not an edge.
- **HNI** — revenue +70.1% is acquisition arithmetic; TTM EPS collapsed to **$0.07, −97.8%**.
- **GCT** — genuinely cheap and growing (PE 12.8, revenue +22.9%, EPS +27.0%) but only **+5.6%** to
  target and beta 1.65. The closest call on the list; it fails on upside, not quality.
- **EGY** — TTM revenue **−25.5%** with net income −$109.44M. The strongest disqualifier this book
  has found, applied without hesitation.

### Stage 2 — full research

**None.** No stage-1 name survived, so there is nothing to research in depth. This is the
**No Candidates Rule** operating exactly as written: *if no candidates pass all filters, hold cash
and explain why; do not force trades.*

For completeness, the extension (ranks 51–100) was scanned from the watchlist columns rather than
the quote page, because the top 50 supplied the full stage-1 count. It offers nothing: of six names
above their 50-day with `rank_low_vol` ≥ 0.80, four have consensus upside under 7% (APLE +6.1%,
HAE +5.9%, PK +3.5%, AMN +4.6%), BLFS trades **above** its target (−19.8%), and TCPC is a business
development company — a closed-end fund structure, which is an **excluded security type**.

### Research log

All 10 stage-1 names logged with `log_research.py` (batch, 2026-09-28, week 55): **10 PASS, 0 BUY**
— valuation 4 (LTC, CNK, GCT, and EGY logged under shrinking-revenue), negative-earnings 2 (SIBN,
CYRX), weak-catalyst 3 (VFF, MQ, ADMA), other 1 (HNI). `research_log.csv` now holds 25 rows across
weeks 54–55: 2 buys and 23 passes.

## 3. Exact orders

**No orders — no candidate cleared the RISK-OFF rules.** One name qualified on the defensive
profile and it offers 4.8% of consensus upside against a Hold rating. Cash stays at **$237.33
(32.7% of equity)**, of which $128.59 is deployable above the floor. The next screen is 2026-10-03.

## 4. Holdings — by exception

| Ticker | Shares | Price | P&L | Stop (room in ATR) | Sessions held | Primary driver | Status |
|---|---|---|---|---|---|---|---|
| ATRC | 3 | $58.38 | **+70.2%** (+$72.24) | $55.27 / $55.12 (**1.35×ATR**) | 54 | U.S. Afib procedure volumes; AtriClip and cryo adoption | Full review below — re-underwrite due ~2026-10-05 |
| CON | 3 | $35.01 | −0.8% (−$0.90) | $33.19 / $33.04 (1.94×ATR) | 5 | U.S. hiring → occupational-health visit volumes | No change — thesis intact |
| HOPE | 15 | $13.83 | −0.9% (−$1.95) | $13.48 / $13.38 (1.38×ATR) | 5 | U.S. short-rate path via NIM; MANUBANK close | No change — driver under watch, see below |

### ATRC — Friday's reversal, and the re-underwrite

**The move:** it opened Friday at $60.85, printed $61.32, and closed **$58.38, −3.93%** on 1.59M
shares (1.1× average) — a $3.42 intraday round trip two sessions after a 52-week high of $61.70.
That is **−1.11×ATR**, below the 1.5×ATR bar that forces a full review, which is why the script
marked it LINE. It gets one anyway, because it is 24% of equity and the week's entire scoreboard
move.

**All five unexplained-move categories, checked:**

| Category | Result |
|---|---|
| Company announcement | Nothing on the wire since the 2026-09-04 index release |
| Analyst action | **None.** Consensus PT unchanged at $53.33 across the move; newest note is Stifel, 2026-09-17 |
| SEC filings | **Found — and it does not explain the move.** See below |
| Index changes | The S&P SmallCap 600 addition took effect before the open on 2026-09-21; complete |
| Sector / macro | Does not cover it: XBI −0.65% and SPY **+0.54%** on the day ATRC fell 3.93% |

**The filing, weighed rather than assumed.** A Form 4 filed 2026-09-24 at 17:00 shows CTO
**Salvatore Privitera sold 4,519 shares at $60.00 on 2026-09-23**, under a 10b5-1 plan **adopted
2026-02-20**, leaving him 142,745 shares. A Form 144 the same day notes it was his third September
sale. Now the arithmetic the rules require: 4,519 shares is **0.28% of Friday's 1,591,611-share
session**, and $271,140 is **0.009%** of a $2.97B market cap, executed under a plan set seven months
ago. It cannot move this stock 3.9%. Finding it was correct; blaming the move on it would not be —
the same conclusion, for the same reason, as PAR's Form 4 on 2026-09-08.

**So the move stays unexplained, with one leading hypothesis:** a **post-inclusion giveback**. The
stock ran +12.8% into the 9/21 index effective date, made its high two sessions after it, and then
reversed on ordinary volume with nothing else to point at. This book's own price-integrity rule
anticipates exactly that — *forced buying ends at the effective date, and additions often give back
part of the pre-inclusion run.* Stated as the leading explanation, not a verified cause.

**Re-underwrite prep — due around 2026-10-05, six sessions out.** Doing the work now rather than
meeting it cold:

- **Thesis, current:** the Afib and left-atrial-appendage device franchise is compounding. TTM
  revenue $569.62M **+13.9%**; consensus FY26 revenue $606.23M (+13.4%) and FY27 $679.71M (+12.1%);
  FY26 EPS **$0.28 against −$0.11 in 2025**, FY27 $0.43 (+51.9%); FY26 gross margin 76.1%. The Q2
  print raised both revenue and adjusted-EBITDA guidance. Nothing here has weakened.
- **Driver, current:** U.S. Afib procedure volumes and AtriClip/cryo adoption. Unchanged, and
  Stifel's 9/17 note confirms LeAAPS enrollment complete with BoxX-NoAF still enrolling.
- **Dispersion, because the average misleads here.** 10 analysts: **5 Strong Buy, 3 Buy, 2 Hold, 0
  Sell**. Consensus $53.33 (−8.65%), **median $55**, low **$36 (−38.3%)**, high **$65 (+11.3%)**.
  The four most recent notes all sit above the price — Stifel **$65** (9/17), Needham **$64** (9/02),
  Piper Sandler **$60** (8/27), BTIG **$55** (8/24) — while a **$36** outlier drags the average. The
  honest read is not "analysts see 9% downside"; it is "current-vintage targets are $55–$65 and one
  stale target is doing the work in the average."
- **Valuation, honestly:** 277× trailing and 245× forward earnings. This is priced on revenue growth
  and margin expansion, not earnings, and that is the position's real risk.
- **Conviction: 4/5**, unchanged from last week.

**One thing to settle before 10/5, and it is a rules question, not a market one.** The re-underwrite
asks whether the position "would be bought today." Read literally against every entry gate, ATRC
fails two that exist to govern *new* capital under RISK-OFF: beta 1.26 keeps it out of the
defensive profile, and it has no dated catalyst inside 90 days (BoxX-NoAF data is H1 2027). Read
that way, the rule would force the sale of the book's best position for failing gates about
deployment. The same rule also says *"a winner passes trivially and keeps running; nothing is sold
for being old,"* which resolves it the other way. **My reading: the fresh-buy standard means the
thesis and quality tests — growing revenue, non-binary, driver intact, distance from base, the
earnings guard — not the regime capacity gates, which govern deploying new capital rather than
holding existing exposure.** On that reading ATRC passes the re-underwrite comfortably. The rules
file should say so explicitly before 10/5, so the decision is not made by an ambiguity.

**Stop:** $55.27 / $55.12, **1.35×ATR** below the close (ATR $2.297). No raise qualifies — the
2.0×ATR candidate is $53.79, *below* the existing stop, and stops never move down outside the
once-per-entry restoration, which this entry spent on 2026-08-31. If it fills, the position banks
**+$62.91, +61.1%**.

### HOPE — the driver is the open question

No change to the position, but the thesis input moved against it this week and that belongs on the
record. The buy case was net interest margin expanding as CDs reprice down: 2.96% in Q2, +6bp
sequentially, interest-bearing deposit costs −6bp. On 2026-09-23 the 10-year yield spiked above
**5.12%**, its highest since 2007, on a flash composite PMI of 58.4 and hawkish Fed commentary,
with the market pricing **more** hikes rather than fewer. Rising funding costs are precisely what
stalls the mechanism the thesis rests on. That is one week of repricing, not the three consecutive
weeks that would invalidate a driver under *Thesis-Input Freshness*, so the position stands — but
if the October FOMC confirms the hawkish path, this thesis needs re-argument rather than patience.
Against that, the stock spent Wednesday within three cents of its stop and finished the week 2.5%
above it, and it fell less than its index on the worst day. Stop unchanged at $13.48 / $13.38,
1.38×ATR, restoration still unused.

### CON — no change

$35.01, five sessions in, 1.94×ATR of stop room, and the best-positioned stop in the book after
last week's correction. Strong Buy with a $41.00 target; nothing on the wire since 9/14; earnings
2026-11-05. The hiring driver got a supportive datapoint in the 58.4 PMI. No raise qualifies (the
2.0×ATR candidate $33.13 sits below the current stop).

## 5. Risk checks after proposed trades

| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | **PASS** — HOPE 28.6%, ATRC 24.2%, CON 14.5% |
| Risk per trade | ≤2% of equity at the stop | **PASS** — CON $6.36 (0.88%), HOPE $7.20 (0.99%); ATRC locks a gain |
| Cash floor | ≥15% of equity | **PASS** — $237.33 = 32.7% against a $108.74 floor |
| Driver cap | ≤2 positions per primary driver | **PASS** — three distinct drivers: Afib procedure volumes, U.S. hiring, the short-rate path |
| Sector cap | ≤3 positions per GICS sector | **PASS** — Health Care 2, Financials 1 |
| Position count | ceiling 5–6 | **PASS** — 3 positions |
| Slippage | ≤10% of average daily dollar volume | **N/A** — no orders |
| Regime capacity | RISK-OFF: ≤3 catalyst; screener at half risk, defensive profile only | **PASS** — no new entries; both screener-sourced holdings sized at the 1% half-budget |
| Circuit breaker | current drawdown vs −20% / −30% | **PASS** — −1.60% from the re-based peak |
| Exclusions | no prohibited business, binary thesis, or earnings inside 10 sessions | **PASS** — all ten stage-1 names checked; nearest print 2026-10-27 |

Every row reads PASS. No order was withdrawn because none was written.

## 6. Thesis summary

**ATRC (AtriCure) — HOLD | Conviction 4/5**
+70.2% at $58.38, **54 sessions** held, re-underwrite due ~2026-10-05. Friday's −3.93% reversal from
a 52-week high survived all five unexplained-move checks: no wire news, no analyst action, a CTO
Form 4 that is 0.28% of the session's volume and cannot explain it, the index add complete on 9/21,
and a sector that rose. **Leading explanation: post-inclusion giveback**, which this book's own rule
predicts. **Driver:** U.S. Afib procedure volumes, unchanged; FY26 revenue consensus $606M (+13.4%)
and EPS $0.28 against −$0.11 last year. The dispersion matters more than the average: median target
**$55**, the four newest notes at **$55–$65**, and a stale **$36** dragging the consensus to $53.33.
**Stop $55.27 / $55.12, 1.35×ATR, and it cannot be lowered** — no raise qualifies either, since the
2.0×ATR candidate sits below it. **What would change the view:** a guidance cut, an AtriClip
competitive read-out, or the re-underwrite — which needs the rules question above settled first,
because a literal reading of "would it be bought today" fails a +70% winner on entry gates written
for new capital.

**CON (Concentra) — HOLD | Conviction 3/5**
3 shares at $35.01, −0.8% from a $35.31 entry, five sessions. **Driver:** U.S. hiring, supported
this week by a flash composite PMI of 58.4, the strongest since 2021. Stop $33.19 / $33.04 at
**1.94×ATR** — the best-placed stop in the book after last week's correction, sitting below the
10-session low and the 50-day SMA. Strong Buy, $41.00 target; earnings 2026-11-05.
**What would change the view:** two consecutive weak payroll prints, or visit volumes lagging
headline hiring.

**HOPE (Hope Bancorp) — HOLD | Conviction 3/5 (driver under watch)**
15 shares at $13.83, −0.9% from entry, five sessions. **Driver has turned against the thesis:** the
10-year spiked above 5.12% on 9/23, the highest since 2007, with the market now pricing more hikes
— and the buy case was funding costs continuing to fall as CDs reprice. One week of repricing is not
the three-week reversal that invalidates a driver, so the position stands. Stop $13.48 / $13.38 at
1.38×ATR, below the 10-session low, restoration unused. The MANUBANK Commercial Banking Unit close
is still due early Q4. **What would change the view:** a hawkish October FOMC, or deposit costs
turning back up at the 10/27 print.

**Portfolio.** Three positions, 32.7% cash, and **no buy for the first time in this phase** — the
right outcome rather than a disappointing one: one name in a hundred cleared the RISK-OFF defensive
profile and it was the name already rejected on its merits. The gap is −2.27% and moves almost
entirely with ATRC, which is both the book's best asset and its concentration risk. **Before the
next report:** settle the re-underwrite reading before 2026-10-05; watch whether ATRC's
post-inclusion giveback continues toward a stop that cannot be lowered; and watch the October FOMC
against HOPE's driver. The next screen runs 2026-10-03, and the mid-October earnings squeeze the
funnel rules warned about begins the week after — expect thinner lists, not richer ones.
Re-entry bans: PAR expired ~9/29, VTS ~9/30.

## Sources

- stockanalysis.com — https://stockanalysis.com/stocks/ATRC/ — ATRC $58.38 at Friday's close, day range $57.89–$61.32, 52-week high $61.70, volume 1,591,611, PE 277 / forward 245, consensus Buy at $53.33 — accessed 2026-09-28
- stockanalysis.com — https://stockanalysis.com/stocks/ATRC/forecast/ — the dispersion: 10 analysts, 5 Strong Buy / 3 Buy / 2 Hold, median $55, low $36, high $65; Stifel $65 (2026-09-17), Needham $64 (09-02), Piper $60 (08-27), BTIG $55 (08-24), Canaccord $55 (08-12); FY26 revenue $606.23M and EPS $0.28 — accessed 2026-09-28
- StockTitan SEC filings — https://www.stocktitan.net/sec-filings/ATRC/ — Form 4: CTO Salvatore Privitera sold 4,519 shares at $60.00 on 2026-09-23 under a 10b5-1 plan adopted 2026-02-20, 142,745 held after; Form 144 notes a third September sale — accessed 2026-09-28
- StockTitan news feeds — ATRC, CON and HOPE (https://www.stocktitan.net/news/TICKER/) — the daily news checks: nothing since 2026-09-04, 09-14 and 09-02 respectively — accessed 2026-09-28
- stockanalysis.com quote pages for all ten stage-1 names — VFF, SIBN, LTC, MQ, CYRX, CNK, ADMA, HNI, GCT, EGY (https://stockanalysis.com/stocks/TICKER/) — the revenue, EPS, rating, target, 52-week position and next-earnings figures behind every kill above — accessed 2026-09-28
- CNBC — https://www.cnbc.com/2026/09/23/treasury-yields-oil-inflation-fed.html — the 10-year above 5.12%, highest since 2007, on flash PMI 58.4 and hawkish Fed commentary — accessed 2026-09-23
- `Start Your Own/watchlist.csv` and `watchlist_extended.csv`, generated 2026-09-28 — the screen: 23 of the top 50 above their 50-day SMA, and 1 of 100 names clearing the RISK-OFF defensive profile

---

*Week 55 Research Report. Generated 2026-09-28 by Claude Code.*
