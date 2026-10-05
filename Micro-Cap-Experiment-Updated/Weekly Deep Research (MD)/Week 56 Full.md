# Week 56 — Research Report (2026-10-05)

Run Monday morning rather than over the weekend, after a disk-full failure stopped Saturday's
attempt. **No buy, for the second week running** — and this time the report carries the evidence
for *why* the screen cannot produce one, because that question now matters more than the week's
candidate list.

## 1. Scoreboard

| Metric | Value | Note |
|---|---|---|
| Gap vs S&P-equivalent, since re-base (2026-09-11) | **−3.73%** (−$27.61: $712.20 vs $739.81) | the scoreboard |
| Gap vs S&P-equivalent, since inception (2025-09-19) | −8.12% (−$62.92: $712.20 vs $775.12) | context only; TWR alpha −3.24% |
| Current drawdown from peak | −3.33% | clear — DE-RISK at −20%, CASH at −30% |
| Regime | **RISK-OFF**, IWM −3.78% vs its 50-day ($281.52 vs $292.58) | 24 sessions, since 2026-08-31 |
| Equity · cash · deployable above the 15% floor | $712.20 · $605.25 (85.0%) · **$498.42 (70.0%)** | floor $106.83 |
| Buys sought · stage-1 checks | 3 · 15 | from `<research_trigger>` |

All figures are Friday 2026-10-02 closes. The run happened just after Monday's open, so the
`<price_volume>` block holds partial intraday prints; they are not used anywhere in this report.

The gap widened 1.46pp in a week, and the arithmetic is simple: **ATRC's exit banked +$62.82 but
removed the only asset that could move the book.** What remains is one 15% position and 85% cash,
so the scoreboard now tracks the S&P almost mechanically — down when it rises, flat when it falls.

## 2. Deployment — the research funnel

Capacity under RISK-OFF: up to 3 catalyst plays at standard sizing, plus screener-sourced entries
at **half the risk budget (1%)** and only on the defensive profile. Buys sought: 3. **Blackout:
ATRC (9 sessions left) and HOPE (7) are banned from re-entry and were excluded from the draw
rather than spending quick checks** — the first week the computed `<blackout>` did that work.

### The constraint, measured

The FOCUS this week was to stop asserting that the RISK-OFF gate is tight and measure it. Across
all 100 screened names:

| Threshold | Names clearing it |
|---|---|
| `rank_low_vol` ≥ 0.90 | 54 / 100 |
| `vol_5_50` > 1.0 | 73 / 100 |
| **`near_high` ≥ −5%** | **16 / 100** |
| **All three — the defensive profile** | **2 / 100** |
| All three **and** above the 50-day SMA (the hard entry rule) | **1 / 100** |

Pairwise, the binding constraint is unambiguous: `low_vol` ∧ `vol_5_50` leaves **37** names, while
`low_vol` ∧ `near_high` leaves **6** and `near_high` ∧ `vol_5_50` leaves **8**. The scarce
condition is proximity to the 60-day high, by a factor of three to five over the other two.

**And that is structural, not a bad week.** `near_high` requires a name within 5% of its 60-day
high; RISK-OFF is *defined* as the index trading more than 1% below its 50-day average. The gate
demands stocks near their highs in precisely the regime defined by the market not being near its
own. **The allowance is anti-correlated with the condition that invokes it.** Week 55 produced one
qualifier, Week 56 produced one, and both times it was **LTC** — the same name, passed on merit
each time at under 5% consensus upside against a Hold.

This is an observation for the rules process, not a change made here. The RISK-OFF screener
allowance already carries a sunset at Phase 4; this is evidence to bring to that review, and the
pre-registered test should probably ask whether `near_high` is the right third condition rather
than whether the allowance survives at all.

### Stage 1 — quick checks

| # | Ticker | Source | Sector | Mkt cap | Rev growth | vs 50d | Next earnings | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | ELME | screener #1 | Real Estate | $156M | +3.4% | +6.04% | 2026-10-22 | PASS · negative-earnings |
| 2 | LTC | screener #3 | Real Estate | $2.32B | +59.4% | +4.18% | 2026-11-03 | PASS · valuation |
| 3 | AMN | screener #5 | Health Care | $1.41B | +24.0% | +7.02% | 2026-11-05 | PASS · valuation |
| 4 | PLX | screener #9 | Health Care | $218M | +30.1% | +6.73% | ~2026-11 | PASS · profile, no coverage |
| 5 | CCO | screener #11 | Communication Services | $1.22B | +9.1% | +0.34% | 2026-11-05 | PASS · negative-earnings |
| 6 | DRH | screener #13 | Real Estate | $2.56B | +1.0% | +0.41% | 2026-11-03 | PASS · profile, no catalyst |
| 7 | LPG | screener #14 | Energy | $2.47B | +80.5% | +13.02% | 2026-11-05 | PASS · valuation |
| 8 | WK | screener #24 | Technology | $3.94B | +19.7% | +1.90% | 2026-11-04 | PASS · profile (`near_high`) |
| 9 | AVPT | screener #29 | Technology | $3.00B | +25.0% | +5.77% | 2026-11-05 | PASS · profile (`low_vol`) |
| 10 | CHEF | screener #31 | Consumer Defensive | $4.79B | +11.2% | +5.76% | 2026-10-28 | PASS · valuation |
| 11 | GO | screener #33 | Consumer Defensive | $1.14B | +5.1% | +5.76% | 2026-11-03 | PASS · negative-earnings |
| 12 | SBH | screener #34 | Consumer Cyclical | $1.49B | +1.1% | +0.02% | 2026-11-12 | PASS · valuation |
| 13 | NTCT | screener #40 | Technology | $2.96B | +5.8% | +5.00% | 2026-11-05 | PASS · valuation |
| 14 | LILAK | screener #42 | Communication Services | $1.66B | +1.3% | +0.87% | 2026-11-04 | PASS · negative-earnings |
| 15 | TH | screener #49 | Industrials | $1.88B | +11.9% | +4.27% | 2026-11-05 | PASS · negative-earnings |

**Spread:** ranks 1–15 → 7 names; ranks 16+ → 8; GICS sectors → **8**; below $2Bn → **8**. All four
quotas met. No name was killed by the earnings guard — the whole list reports between 2026-10-22
and 2026-11-12, 13 sessions out or more. **Prohibited-business check: 14 of 15 clear.** The
exception is flagged below.

**The kills, in one line each:**
- **ELME** — TTM net income **−$185.31M**, EPS −$2.10, no analyst coverage, and a 52-week range of
  $1.26–$17.68: a collapse, not a base.
- **LTC** — the one name clearing the profile above its 50-day, for the third week, and still a
  **Hold with +4.65%** of upside. 4.65% does not pay for 2% of risk.
- **AMN** — the target sits **below the price** (−0.74%), Hold, and **forward PE 43.2 against
  trailing 13.6** says the market expects earnings to fall by two-thirds.
- **PLX** — revenue +30.1% and EPS +205% at PE 11.5, but **no analyst coverage at all** on a $218M
  cap, and `rank_low_vol` 0.64 fails the profile. **Also flagged for the prohibited-business
  check** — see below.
- **CCO** — net income **−$106.01M**, beta 2.02, +1.25% to target.
- **DRH** — revenue +1.0%, `near_high` −8.85 fails the profile, no dated catalyst.
- **LPG** — revenue +80.5% at PE 7.6, but the target is **5.54% below the price**, and a
  freight-rate driver would need freshness verification before any entry.
- **WK** — **the best name the gate removed.** Revenue +19.7%, Strong Buy, **+24.1%** to target,
  beta 0.49, forward PE 19.3. Killed solely by `near_high` −11.58 against the −5% threshold.
- **AVPT** — revenue +25.0%, +18.3% to target; `rank_low_vol` 0.578 fails the profile.
- **CHEF** — +3.96% to target at the 52-week high, PE 55.9, beta 1.44.
- **GO** — net income **−$381.25M**, EPS −$3.88, and a target 6.61% below the price.
- **SBH** — revenue +1.1% and EPS +3.8% for 6.58% of upside; `near_high` −6.07 also fails.
- **NTCT** — +6.43% to target, `near_high` −9.54 fails the profile.
- **LILAK** — net income −$100M, EPS −$0.50, and a target **15.19% below the price**.
- **TH** — net income −$37.68M, EPS −$0.38, beta 1.51. A Strong Buy with +33.8% upside does not
  survive losses at that size.

**⚠️ PLX — an exclusion question I could not settle, and did not decide.** Protalix BioTherapeutics
builds recombinant proteins on its ProCellEx plant-cell platform. The quote page lists **Country:
United States**, headquartered in Hackensack, New Jersey — so the page does **not** support an
Israeli-affiliation exclusion. The company's manufacturing and research base is widely understood
to be in Israel, which the rules' exclusion covers under *affiliation* rather than domicile. I
killed PLX on grounds I could verify — the profile failure and the absence of analyst coverage —
and I am **not** recording a prohibited-business determination I have not established.
**This needs a decision outside the report:** if Protalix counts, it belongs on `screener.py`'s
ticker blocklist so it never reaches a funnel again; if it does not, that should be written down
too, because it will keep reappearing on the screen.

### Stage 2 — full research

**None.** No stage-1 name survived. The **No Candidates Rule** applies: hold cash and explain why;
do not force trades. The explanation this week is the measurement above, not a judgement about the
individual names.

### Research log

All 15 stage-1 names logged (batch, 2026-10-05, week 56): **15 PASS, 0 BUY** — negative-earnings 5
(ELME, CCO, GO, LILAK, TH), valuation 6 (LTC, AMN, LPG, CHEF, SBH, NTCT), weak-catalyst 4 (PLX,
DRH, WK, AVPT). `research_log.csv` now holds **40 rows across weeks 54–56: 2 buys, 38 passes**,
each carrying its `pct_vs_sma50` margin.

## 3. Exact orders

**No buys — nothing cleared the RISK-OFF rules.** One order stands over from Friday:

- **Action:** UPDATE STOP
- **Ticker:** CON
- **Shares:** N/A (3 held)
- **Order type:** N/A — GTC stop-limit replacement
- **Limit price:** N/A
- **Time in force:** GTC
- **Intended execution:** 2026-10-05, before the close
- **Stop loss / stop limit:** **$33.69 / $33.54** — the 2.0×ATR raise target, below Friday's low
  of $34.90 and below the 10-session low
- **Sizing:** risk falls from $6.36 (0.89% of equity) to **$4.86 (0.68%)**; no share change
- **Rationale:** the first qualifying stop raise this book has produced — raise size **0.5083×ATR**
  at $33.69 against the 0.50 minimum, with 1.99×ATR of room left

**The rounding matters and is why the level is $33.69, not $33.68.** The unrounded 2.0×ATR
candidate is $33.682573, which passes the raise-size test by 0.0007. Rounded *down* to $33.68 the
test reads 0.4981 — a fail. Rounded *up* to $33.69 it reads 0.5083 — a pass. Placing $33.68 would
put the stop at a level the rule does not permit.

**Cash after:** unchanged at **$605.25 (85.0% of equity)**, $498.42 deployable.

## 4. Holdings — by exception

| Ticker | Shares | Price | P&L | Stop (room in ATR) | Sessions held | Primary driver | Status |
|---|---|---|---|---|---|---|---|
| CON | 3 | $35.65 | **+1.0%** (+$1.02) | $33.19 / $33.04 (2.50×ATR) → **$33.69 / $33.54 pending** | 10 | U.S. hiring → occupational-health visit volumes | Stop raise qualifies; otherwise no change |

**CON — the only holding, and behaving.** It closed Friday at **$35.65, +1.94%**, on 1.4× volume
and **+1.96pp against a flat XLV**, having risen in each of the last two sessions while healthcare
fell or stalled. Nothing on the wire since 2026-09-14; **Strong Buy, $41.00 target (+15.0%)**;
earnings 2026-11-05, 24 sessions out. The hiring driver last printed +162,000 in August with
upward revisions, and the next payrolls report is the thing to watch.

**One thing to keep in view:** CON's once-per-entry restoration was spent on 2026-09-22 correcting
my original placement error. Once this raise is placed the stop can never move down again. At
1.99×ATR of room that is an acceptable trade, but it is one-way, and it is the reason the raise is
worth taking now rather than waiting for a larger one.

## 5. Risk checks after proposed trades

| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | **PASS** — CON 15.0% |
| Risk per trade | ≤2% of equity at the stop | **PASS** — $4.86 = 0.68% after the raise |
| Cash floor | ≥15% of equity | **PASS** — $605.25 = 85.0% against a $106.83 floor |
| Driver cap | ≤2 positions per primary driver | **PASS** — one holding, one driver |
| Sector cap | ≤3 positions per GICS sector | **PASS** — Health Care 1 |
| Position count | ceiling 5–6 | **PASS** — 1 position |
| Slippage | ≤10% of average daily dollar volume | **N/A** — no buys |
| Regime capacity | RISK-OFF: ≤3 catalyst; screener at half risk, defensive profile only | **PASS** — no new entries |
| Circuit breaker | current drawdown vs −20% / −30% | **PASS** — −3.33% |
| Trend-gate margin | each buy above its 50-day; name any pass under +1.0% | **N/A** — no buys |
| Exclusions | no prohibited business, binary thesis, earnings inside 10 sessions, or a ticker inside its `<blackout>` ban | **PASS** — ATRC and HOPE excluded by the blackout; PLX flagged, not determined |

Every applicable row reads PASS.

## 6. Thesis summary

**CON (Concentra) — HOLD | Conviction 3/5**
3 shares at $35.65, **+1.0%** from a $35.31 entry, 10 sessions held, 15.0% of equity. **Driver:**
U.S. hiring, which sets occupational-health visit volumes; August payrolls +162,000 with upward
revisions, next print early October. Strong Buy, $41.00 target, earnings 2026-11-05. **A stop raise
qualifies for the first time: $33.69 / $33.54**, the 2.0×ATR target, cutting risk from 0.89% to
**0.68%** of equity. Place it before the close. Note the rounding — $33.68 fails the raise-size
test, $33.69 passes — and note that CON's restoration is already spent, so this stop will never
move down again. **What would change the view:** two weak payroll prints, or visit volumes lagging
headline hiring.

**Portfolio.** One position, **85% cash**, and a **second consecutive no-buy week** — but the
reason is now measured rather than asserted. **Of 100 screened names, 16 clear `near_high` ≥ −5%,
2 clear the full defensive profile, and 1 clears it while also above its 50-day SMA** — LTC, for
the third week, at under 5% upside on a Hold. The binding condition is proximity to the 60-day
high, and it is scarce *because* the regime that invokes the allowance is defined by the market
being below its own average. The gate is anti-correlated with its own precondition. **Before the
next report:** place CON's stop raise; decide the PLX exclusion question so it stops recurring;
and take the `near_high` measurement to the rules process rather than to another weekend report —
this is the third week it has produced the same answer, and the Phase 4 sunset review is where it
belongs. Re-entry bans: ATRC ~2026-10-15, HOPE ~2026-10-13.

## Sources

- stockanalysis.com quote pages for all 15 stage-1 names — ELME, LTC, AMN, PLX, CCO, DRH, LPG, WK, AVPT, CHEF, GO, SBH, NTCT, LILAK, TH (https://stockanalysis.com/stocks/TICKER/) — revenue, EPS, rating, target, 52-week position and next-earnings behind every kill above — accessed 2026-10-05
- stockanalysis.com — https://stockanalysis.com/stocks/PLX/company/ — Protalix listed as Country: United States, headquartered in Hackensack, New Jersey; the page does not establish or exclude Israeli affiliation — accessed 2026-10-05
- stockanalysis.com — https://stockanalysis.com/stocks/CON/ — CON $35.65, Strong Buy, $41.00 target, earnings 2026-11-05 — accessed 2026-10-05
- StockTitan — https://www.stocktitan.net/news/CON/ — news check: nothing since 2026-09-14 — accessed 2026-10-05
- `Start Your Own/watchlist.csv` and `watchlist_extended.csv`, generated 2026-10-05 — the threshold counts: 54 / 73 / 16 of 100 on the three profile conditions, 2 on all three, 1 also above its 50-day
- `trading_script.py` `<research_trigger>` and `<market_regime>` — the trigger, the blackout and the regime; IWM $281.52 vs a $292.58 50-day

---

*Week 56 Research Report. Generated 2026-10-05 by Claude Code.*
