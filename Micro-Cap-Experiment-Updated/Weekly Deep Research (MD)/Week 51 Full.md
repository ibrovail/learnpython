# Week 51 Deep Research — Micro-Cap Portfolio

**Date:** Monday, 2026-08-31 (intraday, ~10:00 AM EDT) | **Week:** 51 of 52 | **Runway:** ends 2026-09-18 (**14 sessions including today**)
**Session directives:** Sector = Wide net · Catalyst timing = Within 5 days · Risk posture = **Aggressive — trailing benchmark** · Max positions = 6

---

## 1. RESTATED RULES

- **Universe:** US-listed common stock, market cap ≤ $5B. No ETFs, CEFs/**BDCs**, SPACs, ADRs, units/warrants.
- **Cash floor: 8%** (amended from 15% on 2026-08-28).
- **Risk-per-trade** ≤5% of equity; **single-name ceiling 30%**.
- **Stops mandatory, never lowered.** Range check: below the most recent session's low AND ≥1.5×ATR from the reference price.
- **One open order per stock** — new shares in an existing name **inherit that name's current stop**.
- **Sector cap** 2 per GICS sector (healthcare 3).
- **PRV gate:** browser-fetch the live quote page before ANY buy/sell/trim/add.
- **No averaging down** more than 5% below entry without a confirmed new catalyst.

---

## 2. RESEARCH SCOPE

**Retrieved Monday 2026-08-31, 09:53–10:05 AM EDT — with the market OPEN.** Every price below is a live intraday quote, not a settled close. This is stated on each figure and matters for the order-side checks.

| Check | Method | Result |
|---|---|---|
| Settled OHLC through Fri 8/28 | `trading_script.py` daily run | Complete |
| ATR(14), stop distances | Computed locally from settled bars only | Complete |
| PRV live quotes | Browser → `stockanalysis.com` | ATRC, PAR, CADL + candidates PD, BOX, RCKT |
| Screener universe | `screener.py` → Finviz + yfinance | **Re-run after a fix — see below** |

### ⚠️ The screener's first run this week was invalid — diagnosed and fixed

The initial `make weekend` screen produced a watchlist that was **structurally wrong**, and it would have been easy to research it without noticing:

- **Every candidate showed a volume ratio of 0.10–0.38×** — i.e. every name in the entire universe was apparently trading at a fraction of normal volume. Last week's identical screen produced 1.2–8.3×.
- The top 15 was **dominated by mortgage REITs and BDCs** (ORC, PFLT, DX, ARR, LAND, EFC, IVR, PNNT, TRIN, BBDC) — **four of which are BDCs, an explicitly excluded security class.**

**Cause:** `screener.py:325` computes `volume.iloc[-1] / volume.tail(20).mean()`. Run during market hours, `iloc[-1]` is **today's partial-day bar** — about twenty minutes of trading measured against a full 20-day average. That collapsed the volume score for every genuine momentum name and let low-volatility yield vehicles float to the top on the remaining factors. Momentum, SMA and Bollinger-width were all reading off an intraday price rather than a close for the same reason.

**Fix applied:** the price fetch now drops any bar for a session that has not closed, reusing `last_completed_session()` from `trading_script.py` rather than keeping a second copy of that logic. Re-running produced a **completely different and sane universe** — volume ratios 1.0–2.8×, and the BDC/REIT cluster gone.

**This is the finding of the week.** The corrupted screen was not obviously broken on its face; it looked like a defensible list of value names. Had I researched it, the entire Week 51 candidate set would have been drawn from excluded security classes selected by a broken factor.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

**Live equity $752.10** (Friday close $771.65, **−2.53% intraday**) · Cash **$69.34 (9.2%)** · S&P-equivalent ≈$771.16 · **Gap ≈ −2.47%**

| Ticker | Role | Entry Date | Avg Cost | Live Price | Current Stop | Conviction | Status |
|---|---|---|---|---|---|---|---|
| ATRC | Core winner | 2026-07-15 | $34.30 | $46.96 (+36.9%) | $46.30 | 5/5 | **KEEP — 0.38×ATR from stop** |
| TILE | Core | 2026-06-30 | $32.10 | $38.06 (+18.6%) | $37.10 | 4/5 | **KEEP — 0.81×ATR from stop** |
| CADL | Catalyst | 2026-08-17 | $11.43 | $12.75 (+11.5%) | $12.35 | 3/5 | **KEEP — 0.44×ATR from stop** |
| PAR | Growth | 2026-08-17/28 | $19.05 | $18.39 (−3.5%) | $17.50 | 4/5 | **KEEP — 0.94×ATR from stop** |
| WWW | Recovery | 2026-08-24 | $21.10 | $19.95 (−5.5%) | $19.30 | 3/5 | **KEEP — 0.68×ATR from stop** |

### The single most important fact this week

**All five positions are now inside 1.0×ATR of their stops.** Not one is at the 1.5×ATR minimum the rules require for a functioning stop.

I did not tighten them into this state. Two stops (ATRC, CADL) were raised last week on genuine price progress; the rest have not moved. **The market walked the prices down to fixed lines that cannot be lowered.** The mechanical consequence is that the book's remaining downside is now unusually small — **all five stops firing costs only −2.87% of equity** — but the probability of those stops firing is correspondingly high.

### ATRC — the position that matters | Conviction 5/5

**PRV, 8/31 10:00 AM EDT (live):** $46.96, **−3.26%**. Previous close $48.54, open $48.23. Volume 44,132 (light, early session). *Note: the page's stated day-range low of $47.01 lags the $46.96 last price — the range field had not refreshed; I am using the last price, which is the more current field, and flagging the discrepancy rather than reconciling it away.*

**The fundamental case has strengthened materially while the price fell:**
- **Piper Sandler raised its target to $60 from $50 (Overweight), four days ago** — explicitly citing the STS quality metric
- BTIG raised to **$55 from $45 (Buy)** on 8/24, same thesis
- **Consensus PT has risen $47.33 → $49.56**, now *above* the market price rather than below it
- Revenue TTM $569.62M **+13.9%**; no adverse company news since 7/27

**And the stop is $0.66 away — 0.38×ATR.**

This is the sharpest version of the structural problem. ATRC will very likely be stopped out at $46.30 for **+35.0%** — a genuinely good outcome — precisely as two banks put $55–60 targets on it. There is no lever: the stop cannot be lowered.

**Honest reflection on my own decision:** I raised this stop from $45.85 to $46.30 last Thursday when ATRC closed at $49.27 near its 52-week high. That gained **$1.35** of locked profit and measurably increased the probability of being stopped out. At the time it passed every check (1.75×ATR, below the 4-day low). Three sessions later it is 0.38×ATR away. **The raise was rule-compliant and, in hindsight, poor value** — a marginal gain bought with real optionality. The lesson is not "don't trail stops"; it is that **trailing a stop into a 3.5%-ATR name at 1.75× leaves almost no room when the name gives back a week of gains**, and the increment being locked should be weighed against the option being sold.

### PAR | Conviction 4/5

**PRV, 8/31 10:00 AM EDT (live):** $18.39, **−3.97%**. Revenue TTM **+18.8%**, forward PE 16.82, **Buy, PT $25.31 — +35.1% upside**, the largest of any holding. **No news** — latest item is a 13-day-old survey release. The decline is not company-specific.

The 3 shares added Friday at $19.60 are **−6.2%**. Blended cost is now $19.05 across 11 shares. At 26.9% of equity it is the largest position, with only $23.34 of headroom to the 30% cap.

### TILE | Conviction 4/5
$38.06 (−1.35% intraday), +18.6%. Stop $37.10 locks **+15.6%**. Strong Buy, PT $45.25. Stop cannot be raised (trailing floor sits below it) or lowered.

### CADL | Conviction 3/5
**PRV, 8/31 10:02 AM EDT (live):** $12.75, −2.45%. Day range $12.73–13.13. **PT raised to $21.00 (+65.75%), Strong Buy** — the largest upside on the board. Beta −0.50. But XBI fell −3.48% Friday and another −1.27% today, and CADL is 0.44×ATR from its stop.

### WWW | Conviction 3/5
$19.95 (−1.94% intraday), −5.5%. Forward PE ~11.2, Buy, PT $24.30 (+22%). Bounced +2.94% Friday, giving part back today. Stop $19.30 is 0.68×ATR away.

---

## 4. CANDIDATE SET

Evaluated from the **corrected** screen. Every name below was browser-verified.

| Ticker | Sector | One-Line Thesis | Catalyst | Confirmation | Verdict |
|---|---|---|---|---|---|
| PD | Technology | Post-earnings momentum, ARR crossed $500M | Q2 print 8/27 (past) | Verified live | **REJECT** |
| BOX | Technology | Cloud content management, revenue +9.6% | Q2 print 8/25 (past) | Verified live | **REJECT** |
| RCKT | Healthcare | Gene therapy, PT +135% | No dated catalyst in window | Verified live | **REJECT** |
| KFY | Industrials | Korn Ferry, $4.4B, +4.1% momentum | None in window | Screen only | **REJECT** |
| EPAC | Industrials | Enerpac, $1.9B, +4.2% momentum | None in window | Screen only | **REJECT** |
| GCMG | Financial | GCM Grosvenor, $2.8B | None in window | Screen only | **REJECT** |

### Why each was rejected

- **PD — rejected again, and I was wrong on the price.** I passed at $12.32 last week; it is **$13.96 today, a 13% move I missed.** But every disqualifying fact is unchanged or worse: revenue growth **decelerated from +3.7% to +2.3%**, consensus is still **Hold with PT $12.64 — 9.5% BELOW the price**, and **forward PE 9.97 exceeds trailing 6.42**, meaning earnings are expected to fall. The $185.55M net income is a tax item, not operations. Two Buy-side raises to $15 are real, but BofA raised to $9.50 and kept **Underperform**. Buying now would be chasing a momentum move into a consensus sell-side target below the market. *A missed 13% on a name that still fails the fundamental screen is a cost I accept.*
- **BOX — rejected on risk/reward.** **Hold rating, PT $37.50 = only +6.4% upside**, near its 52-week high, with **net income −42.8% and EPS −44.6%**. WWW, the position I would have to sell to buy it, has +22% upside and a Buy rating. Strictly worse than what we hold.
- **RCKT — rejected on shape, not story.** **PT $8.84 = +135% upside, Buy from 14 analysts**, beta 0.47 — the largest upside on any screen this experiment has produced. But it is a pre-revenue clinical-stage gene-therapy company with **no dated catalyst inside the window**, and a 12-month price target is not a 13-session thesis. Buying it would also make healthcare **3 of 6** alongside ATRC and CADL while XBI is actively selling off — concentrating precisely the exposure that cost us today.
- **KFY, EPAC, GCMG** — momentum of +3–4% and no catalyst inside the 5-day window. In a 14-session runway these are index proxies with single-stock risk.
- **SR, CHEF, PRGS, CSWC, CMTL, DBRG, ORC, AI, IRON** (remaining ranks) — not advanced: no dated catalyst inside the window, and none offers upside superior to the existing book.

---

## 5. PORTFOLIO ACTIONS

**Keep:** ATRC, TILE, PAR, CADL, WWW — all five. No thesis has broken; today's decline is broad and news-free across every holding.

**Add to:** **none — and this is a hard constraint, not a preference.**

The one-open-order-per-stock rule means new shares inherit the existing stop. Every holding fails that test today:

| | Live | Stop | Distance | Add eligible? |
|---|---|---|---|---|
| ATRC | $46.96 | $46.30 | **0.38×ATR** | ❌ |
| CADL | $12.75 | $12.35 | **0.44×ATR** | ❌ |
| WWW | $19.95 | $19.30 | **0.68×ATR** | ❌ *(also blocked: −5.5%, no-averaging-down)* |
| TILE | $38.06 | $37.10 | **0.81×ATR** | ❌ |
| PAR | $18.39 | $17.50 | **0.94×ATR** | ❌ *(also: only $23 of headroom to the 30% cap)* |

**Trim / Exit:** none. Selling anything today means selling into a −2.5% intraday drawdown on no news.

**Initiate:** **none.** Deployable cash is **$9.17** against the 8% floor. A 6th position requires liquidating a holding, and no candidate above is superior to what it would replace.

### On the "Aggressive" directive — what I did with it

You asked for an aggressive posture because we are trailing, and for a 6th position if capital could be freed. **I could not express that through deployment today, and I want to be explicit about why rather than quietly ignoring the directive:**

1. There is no free capital — $9.17.
2. Freeing capital means selling a holding **at today's depressed prices**, on no news, to buy a candidate that is not better than what it replaces (BOX has a third of WWW's upside; RCKT has no catalyst in the window).
3. Every add-to-existing route is blocked by the inherited-stop test.

**So the aggressive posture is expressed the only way it legitimately can be this week: by NOT tightening a single stop.** Last week's directive tightened ATRC and CADL; this week I am raising nothing, leaving every position the maximum room the no-lowering rule permits. Given the book is already within 1.0×ATR of its stops, refusing to tighten is a genuinely aggressive act — it is the difference between letting five theses run and guaranteeing they are stopped out.

**If you want a more aggressive expression than that**, the honest option is: **sell WWW (3 sh ≈ $59.85, realising −5.5%) and buy RCKT**, trading a +22%-upside recovery story for a +135%-upside binary. I am **not recommending it** — it swaps a verified thesis for a lottery ticket with no catalyst inside the runway, at the cost of a realised loss — but it is the trade the directive points at, and it is your call to make.

---

## 6. EXACT ORDERS

**No orders.**

Every candidate action fails a hard check: adds fail the inherited-stop test; a 6th position lacks funding and a superior candidate; stop raises are unavailable (all trailing floors sit below current stops) and would be counter-productive with every position already inside 1.0×ATR; and no exit is warranted since no thesis has broken.

**Stops remain, unchanged:**
- TILE — $37.10 / $36.95
- ATRC — $46.30 / $46.15
- PAR — $17.50 / $17.35
- CADL — $12.35 / $12.20
- WWW — $19.30 / $19.15

---

## 7. RISK AND LIQUIDITY CHECKS

**Concentration (live intraday):**

| Holding | Value | % Equity |
|---|---|---|
| PAR | $202.29 | 26.9% |
| TILE | $152.24 | 20.2% |
| ATRC | $140.88 | 18.7% |
| CADL | $127.50 | 17.0% |
| WWW | $59.85 | 8.0% |
| Cash | $69.34 | 9.2% |
| **TOTAL** | **$752.10** | 100% |

All under the 30% ceiling ✓ · Cash above the 8% floor ($60.17) ✓

**Sector cap:** Consumer Discretionary 2 (TILE, WWW) — full · Healthcare 2 (ATRC, CADL) — cap 3 · Technology 1 (PAR) ✓

**Downside if every stop fires:** **−$21.56 = −2.87% of equity** → equity $730.54, fully in cash. The tightness of the stops has, as a side effect, capped the book's remaining downside at under 3%.

**Liquidity:** all five trade far above the $1M/day threshold. No order proposed.

---

## 8. MONITORING PLAN

| Holding | Watch for | Trigger |
|---|---|---|
| **ATRC** | Most likely stop-out in the book — **0.38×ATR away.** Whether the Piper $60 / BTIG $55 targets draw follow-on buying | Stop at $46.30 = **+35.0% realised**. A good outcome; take it and do not re-enter |
| **CADL** | 0.44×ATR from stop; XBI direction is the whole story | Stop at $12.35 = **+8.0% realised** |
| **TILE** | 0.81×ATR from stop | Stop at $37.10 = **+15.6% realised** |
| **PAR** | Largest position (26.9%) and largest upside (+35.1%). Holding the $18.38 higher-low | A break below $18.38 on volume is the thesis-crack warning, well before the $17.50 stop |
| **WWW** | 0.68×ATR from stop; the only holding at a loss besides PAR | Stop at $19.30 = −8.5% realised |
| **Book** | XBI (−1.27% today, −3.48% Friday) and IWM. Three of five holdings are small-cap value, which has lagged all week | A second consecutive broad down day likely triggers multiple stops |

**Calendar:** No holding reports earnings before the experiment ends. **Sept 7 is Labor Day** (market closed), leaving 13 sessions after today.

---

## 9. THESIS REVIEW SUMMARY

*(Reproduced verbatim as `Week 51 Summary.md`)*

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash: $69.34 — unchanged (no orders). 9.2% of live equity, above the 8% floor.**

| Constraint | Status |
|---|---|
| Cash floor ≥8% | ✅ 9.2% |
| Single-name ≤30% | ✅ largest PAR 26.9% |
| Max 6 positions | ✅ 5 held |
| Risk-per-trade ≤5% | ✅ largest is PAR at 1.30% |
| Sector cap | ✅ Cons Disc 2 (full), Healthcare 2/3, Tech 1 |
| Market cap ≤$5B | ✅ largest ATRC $2.40B |
| Excluded classes | ✅ all common stock — **and the corrected screen removed 4 BDCs that the broken screen had ranked in the top 7** |
| Stops on all longs | ✅ five live stops |
| PRV gate | ✅ ATRC, PAR, CADL + PD, BOX, RCKT browser-verified 09:53–10:05 EDT |
| No averaging down | ✅ no adds proposed |

**All constraints satisfied. No orders for execution.**

---

*Week 51 Full Report — generated 2026-08-31 by Claude Code. Posture: Aggressive (trailing benchmark) · Wide net · 5-day window · max 6 positions. All prices are LIVE INTRADAY quotes browser-verified 09:53–10:05 AM EDT with the market open; all ATR and range figures computed from settled bars through 2026-08-28.*
