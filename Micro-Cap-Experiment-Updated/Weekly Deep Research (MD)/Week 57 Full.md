# Week 57 — Research Report (2026-10-12)

The first report under the 2026-10-11 rule set, and the first in which the **screener lane is
closed**. The catalyst lane was the only route to a position. All four qualifying categories were
searched and **none produced a candidate** — that is this week's finding, and the categories and
dates are below so it can be checked rather than taken on trust.

## 1. Scoreboard

| Metric | Value | Note |
|---|---|---|
| Gap vs S&P-equivalent, since re-base (2026-09-11) | **−4.21%** (−$31.50: $716.82 vs $748.32) | the scoreboard |
| Gap vs S&P-equivalent, since inception (2025-09-19) | −8.57% (−$67.22: $716.82 vs $784.04) | context only; TWR alpha −3.84% |
| Current drawdown from peak | −2.71% | clear — DE-RISK at −20%, CASH at −30% |
| Regime | **RISK-OFF**, IWM −4.27% vs its 50-day ($278.94 vs $291.38) | **29 sessions**, since 2026-08-31 |
| Equity · cash · deployable above the 5% floor | $716.82 · $605.25 (84.4%) · **$569.41 (79.4%)** | floor $35.84 — cut from 15% on 10/11 |
| Buys sought · stage-1 checks | 3 · 15 | from `<research_trigger>` |
| Lanes | **catalyst 0/3 · screener 1/0 — CLOSED** | CON grandfathered |

The gap is −4.21% and 84% of the book is cash, so the scoreboard now moves almost inversely to the
index: it widens on up weeks and narrows on down ones, independent of anything the holding does.
That is the cost of an empty book, and it is the reason the catalyst lane matters more than any
single candidate would.

## 2. Deployment — the research funnel

**Capacity: the catalyst lane only.** `SCREENER_CAP["RISK-OFF"] = 0` since 2026-10-11, so the
screener lane is shut and **a screener-sourced buy would be a rule violation**. Three catalyst
slots are open at the full **2% risk budget ($14.34)**, with no defensive-profile test — that
profile no longer exists. CON is grandfathered: capacity gates govern new capital, not holdings.

**Blackout:** ATRC (4 sessions left) and HOPE (2) were excluded from the draw rather than spending
quick checks on them.

### Stage 0 — catalyst sourcing

Not a quota this week but the whole report. Every qualifying category in `portfolio_rules.md` was
searched. A qualifying catalyst is **a resolved fact with a dated earnings impact**, as distinct
from the prohibited binary case — an unresolved event with a dated announcement.

| Category | Searched | Result |
|---|---|---|
| S&P DJI / FTSE Russell index events | S&P DJI press archive, read directly | **Empty.** Newest release is 2026-10-05 (Freshworks). All three October adds are past their effective dates |
| Raised guide with the next print dated | Guidance-raise announcements, October 2026 | 2 found, both killed — see below |
| Dated contract or backlog conversion | Contract awards, October 2026 | **Empty**, and the names that surface are prohibited |
| Spin-off or refinancing with an announced date | Spin-off record and distribution dates, Q4 2026 | **Empty.** The one live timeline completed 2026-10-08 |

### Stage 1 — the five names examined

| # | Ticker | Source | Catalyst | Result |
|---|---|---|---|---|
| 1 | ZIM | catalyst:guidance-raise | FY26 adj. EBITDA guide raised to $2.7–3.0Bn, 2026-10-06 | **PASS · prohibited** |
| 2 | NNBR | catalyst:guidance-raise | FY26 sales and EBITDA raised 2026-09-22, next print 10/28 | **PASS · negative-earnings** |
| 3 | WK | catalyst:index | Joined S&P SmallCap 600, effective 2026-10-06 | **PASS · effective date passed** |
| 4 | FRSH | catalyst:index | Joined S&P SmallCap 600, effective 2026-10-08 | **PASS · effective date passed** |
| 5 | TORO | catalyst:spin-off | AI OKTO LPG-carrier spin-off, distributed ~2026-10-08 | **PASS · completed** |

**Why each failed:**

- **ZIM** — raised FY26 guidance on 10/06, a textbook resolved-fact catalyst. **ZIM Integrated
  Shipping is Israeli-affiliated, so it is killed at the prohibited-business check before any other
  gate is applied.** This is the second week running that the exclusion has removed an otherwise
  qualifying name — PLX was the first, and the same lesson applies: neither was catchable from an
  industry label.
- **NNBR** — the only name that cleared the hard gates. Guidance raised 2026-09-22, next print
  **2026-10-28, which is 12 sessions out and so clears the 10-session guard**; revenue +5.1%;
  market cap $370M. It fails on quality: **TTM net income −$47.92M, EPS −$0.96, beta 2.66**. A
  lossmaking micro-cap at nearly three times market beta is not what the catalyst lane is for,
  and a Strong Buy with a +68.9% target does not change that.
- **WK and FRSH** — both genuinely joined the S&P SmallCap 600, on 10/06 and 10/08. **An index add
  counts only before its effective date.** That rule exists because ATRC ran +12.8% into its
  2026-09-21 inclusion and then gave back 11.8% in the nine sessions after it. Buying either now
  would be buying the giveback.
- **TORO** — the AI OKTO spin-off had a live timeline a fortnight ago: record date 10/01,
  distribution on or about **10/08**. It has completed, so there is no dated forward catalyst left.

**Research log:** 5 names logged (2026-10-12, week 57), all `--source "catalyst:<category>"`:
prohibited 1, negative-earnings 1, weak-catalyst 3. **`research_log.csv` now holds 45 rows across
weeks 54–57 — and for the first time, 5 of them are catalyst-sourced rather than screener-sourced.**
No `lane-closed` rows were written, because no quote-page fetches were spent on screener names once
the lane was known to be shut.

### Stage 2 — full research

**None.** No catalyst candidate survived stage 1. The **No Candidates Rule** applies: hold cash and
explain why.

**The honest shape of this week's failure matters.** Two of the five were disqualified by *timing* —
index adds whose effective dates had passed — and one by the **prohibited-business** rule. Only one
name, NNBR, failed on its own merits. So the catalyst lane was not short of events; it was short of
events that were still ahead of us and attached to a company this book will own. That is a
different problem from "the screen produced nothing", and it suggests the lane will produce
candidates when the December index announcement lands, if not before.

## 3. Exact orders

**No orders.** No candidate cleared, and the only open lane is the catalyst lane.

**Explicitly withdrawn: the CON stop raise to $35.36 / $35.21** recommended on 2026-10-11. It was
justified solely by the 2.0×ATR tests passing, which the 10-11 rule set now names as insufficient.
**CON's stop stays at $34.49 / $34.34.** See section 4.

**Cash:** unchanged at **$605.25 (84.4% of equity)**, $569.41 deployable above the 5% floor.

## 4. Holdings — by exception

| Ticker | Shares | Price | P&L | Stop (room in ATR) | Sessions held | Primary driver | Status |
|---|---|---|---|---|---|---|---|
| CON | 3 | $37.19 | **+5.3%** (+$5.64) | $34.49 / $34.34 (**2.95×ATR**) | 15 | U.S. hiring → occupational-health visit volumes | FULL — ran up; stop held, no add |

**CON — flagged FULL because it has run, not because anything is wrong.** The script's wording
changed on 10/11 and now reads *"ran up — stop 2.9×ATR behind; raise ELIGIBLE to ~$35.36 but
default is HOLD (needs a written thesis change)."* That is the right flag to act on: check the
thesis and the add case, not the raise.

**Thesis: intact and better than at entry.** Revenue +14.0% TTM, EPS $1.53 (+29.4%), net income
+33.1%, forward PE 21.6, beta 0.63. Strong Buy with a $41.00 target, now +10.25% above the price
after a 5.3% run. The company raised FY26 guidance at its Q2 print (revenue $606.0M, +10.0%; net
income $67.3M, +45.7%). **Driver:** U.S. hiring, which sets pre-placement and workers-comp visit
volumes. Earnings **2026-11-05, 16 sessions out**.

**News check:** one item since the last report — a 2026-10-08 investor-conference notice. Immaterial.

**No stop raise.** The 2.0×ATR candidate is $35.36 and the tests pass, and that is not sufficient.
Nothing has changed about the thesis that would justify the exception: the company has not
re-guided, no analyst has moved, and a 5.3% advance on 1.0× volume is the position working as
underwritten. The measured cost of raising whenever eligible is **10.3pp of test return**, a 98%
stop-out rate and an average hold of 14 sessions against 57. **Stop stays $34.49 / $34.34**, risk
$2.46 = **0.34% of equity**.

**No add, and the reason is the lane, not the name.** CON is screener-sourced, and that lane is
closed in RISK-OFF. Grandfathering covers *holding* the position, not putting new capital into it —
an add is new capital. Worth recording because the arithmetic now permits what the rule forbids:
with fractional shares and a 5% floor, the 2% risk budget against a $2.70 stop distance would
support roughly 5.3 more shares inside the 30% ceiling. The lane closure is what stops it.

## 5. Risk checks after proposed trades

| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | **PASS** — CON 15.6% against a $215.05 ceiling |
| Risk per trade | ≤2% of equity at the stop | **PASS** — $2.46 = 0.34% |
| Cash floor | ≥5% of equity | **PASS** — $605.25 = 84.4% against a $35.84 floor |
| Driver cap | ≤2 positions per primary driver | **PASS** — one holding |
| Sector cap | ≤3 per GICS sector — apply greedily | **PASS** — Health Care 1 |
| Position count | ceiling 10 | **PASS** — 1 |
| Slippage | ≤10% of average daily dollar volume | **N/A** — no orders |
| Regime capacity | RISK-OFF: catalyst 3 slots at 2%; **screener lane closed** | **PASS** — no new entries; no screener buy proposed |
| Stop policy | default is DO NOT RAISE; an exception must name a thesis change | **PASS** — raise eligible and declined, in writing |
| Circuit breaker | current drawdown vs −20% / −30% | **PASS** — −2.71% |
| Exclusions | prohibited business, binary thesis, earnings inside 10 sessions, `<blackout>` | **PASS** — ZIM killed as prohibited; ATRC and HOPE excluded by blackout |

Every applicable row reads PASS.

## 6. Thesis summary

**CON (Concentra) — HOLD | Conviction 3/5**
3 shares at $37.19, **+5.3%** from a $35.31 entry, 15 sessions, 15.6% of equity. **Driver:** U.S.
hiring → occupational-health visit volumes. Revenue +14.0%, EPS +29.4%, forward PE 21.6, beta 0.63;
Strong Buy at $41.00 (+10.3%); earnings 2026-11-05. **Stop stays $34.49 / $34.34** at 2.95×ATR,
risk 0.34% of equity. A raise to $35.36 is eligible and **declined**: the tests passing is not a
reason, and nothing about the thesis has changed. **No add** — CON is screener-sourced and that
lane is closed in RISK-OFF; grandfathering covers holding, not new capital. **What would change the
view:** two weak payroll prints, visit volumes lagging headline hiring, or the 11/05 print
re-guiding down.

**Portfolio.** One position, **84% cash, third consecutive no-buy week** — but for a different
reason than the previous two. Weeks 55 and 56 were blocked by a defensive profile that has since
been removed; this week the catalyst lane was genuinely open and **produced no candidate that was
both still ahead of us and ownable**. Three of the five names examined failed on timing or
exclusion rather than quality. **Before the next report:** the December S&P rebalance announcement
is the next scheduled source of index catalysts, roughly early December, and is worth watching for
from late November; blackouts expire this week (HOPE ~10/13, ATRC ~10/15), returning two researched
names to the eligible set; and the mid-October earnings squeeze now blocks most of the screen
anyway, so the catalyst lane stays the only realistic route until the regime turns.

## Sources

- S&P Dow Jones Indices press archive — https://press.spglobal.com/index.php?s=2429&l=100 — newest index release 2026-10-05 (Freshworks, effective 10/08); nothing announced since, so no pending add with a future effective date — accessed 2026-10-12
- S&P DJI — https://press.spglobal.com/2026-10-01-Vylor-Added-to-the-S-P-500-Twilio-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600 — the 10/06 SmallCap 600 changes, including Workiva — accessed 2026-10-12
- S&P DJI — https://press.spglobal.com/2026-10-05-Freshworks-Set-to-Join-S-P-SmallCap-600 — Freshworks effective 2026-10-08 — accessed 2026-10-12
- ZIM investor relations — https://investors.zim.com/news/news-details/2026/ZIM-Raises-Full-Year-2026-Guidance/default.aspx — FY26 guidance raise dated 2026-10-06 — accessed 2026-10-12
- GlobeNewswire — https://www.globenewswire.com/news-release/2026/09/22/3366935/0/en/nn-inc-raises-full-year-2026-guidance-ranges-for-net-sales-and-adjusted-ebitda.html — NN Inc raised FY26 guidance 2026-09-22, Q3 print 2026-10-28 — accessed 2026-10-12
- stockanalysis.com — https://stockanalysis.com/stocks/NNBR/ — NNBR $3.95, revenue +5.1%, TTM net income −$47.92M, EPS −$0.96, beta 2.66, earnings 10/28 — accessed 2026-10-12
- Nasdaq / GlobeNewswire — https://www.nasdaq.com/press-release/toro-corp-announces-record-date-proposed-spin-its-lpg-carrier-business-2026-09-21 — Toro AI OKTO spin-off, record date 10/01, distribution ~10/08 — accessed 2026-10-12
- stockanalysis.com — https://stockanalysis.com/stocks/CON/ — CON $37.19, revenue +14.0%, EPS $1.53, forward PE 21.6, beta 0.63, Strong Buy $41.00, earnings 2026-11-05 — accessed 2026-10-12
- StockTitan — https://www.stocktitan.net/news/CON/ — news check: one 2026-10-08 investor-conference notice, immaterial; Q2 print raised FY26 guidance — accessed 2026-10-12

---

*Week 57 Research Report. Generated 2026-10-12 by Claude Code.*
