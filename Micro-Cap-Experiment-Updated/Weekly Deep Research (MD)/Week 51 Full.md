# Week 51 Deep Research — Micro-Cap Portfolio

**Date:** Monday, 2026-08-31 (intraday, ~10:00 AM EDT) | **Week:** 51 of 52 | **Runway:** ends 2026-09-18 (**14 sessions including today**)
**Session directives:** Sector = Wide net · Catalyst timing = Within 5 days · Risk posture = **Aggressive — trailing benchmark** · Max positions = 6

---

## 1. RESTATED RULES

- **Universe:** US-listed common stock, market cap ≤ $5B. No ETFs, CEFs/**BDCs**, SPACs, ADRs, units/warrants.
- **Cash floor: 8%** (amended from 15% on 2026-08-28).
- **Risk-per-trade** ≤5% of equity; **single-name ceiling 30%**.
- **Stops mandatory.** Range check: below the most recent session's low AND ≥1.5×ATR from the reference price.
- **⚠️ RULE RELAXED THIS SESSION — "never lower a stop" suspended for one documented adjustment.** Authorized 2026-08-31 as a **single restoration** of ATRC, PAR and TILE to the `max(1.5×ATR, …)` band this rule set already mandates, which all five stops had drifted *below* through price movement alone. Scope, reasoning and limits are recorded in `portfolio_rules.md`. **The rule resumes in full immediately afterwards.**
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

**All five positions are inside 1.0×ATR of their stops.** Not one is at the 1.5×ATR minimum the rules require for a functioning stop — meaning every stop in the book had drifted *out of compliance with the rules' own ATR band*, which is what justified the one-time restoration in §6.

I did not tighten them into this state. Two stops (ATRC, CADL) were raised last week on genuine price progress; the rest have not moved. **The market walked the prices down to fixed lines that cannot be lowered.** The mechanical consequence is that the book's remaining downside is now unusually small — **all five stops firing costs only −2.87% of equity** — but the probability of those stops firing is correspondingly high.

### ATRC — the position that matters | Conviction 5/5

**PRV, 8/31 10:00 AM EDT (live):** $46.96, **−3.26%**. Previous close $48.54, open $48.23. Volume 44,132 (light, early session). *Note: the page's stated day-range low of $47.01 lags the $46.96 last price — the range field had not refreshed; I am using the last price, which is the more current field, and flagging the discrepancy rather than reconciling it away.*

**The fundamental case has strengthened materially while the price fell:**
- **Piper Sandler raised its target to $60 from $50 (Overweight), four days ago** — explicitly citing the STS quality metric
- BTIG raised to **$55 from $45 (Buy)** on 8/24, same thesis
- **Consensus PT has risen $47.33 → $49.56**, now *above* the market price rather than below it
- Revenue TTM $569.62M **+13.9%**; no adverse company news since 7/27

**And the stop is $0.66 away — 0.38×ATR.**

This was the sharpest version of the structural problem, and it is what forced the rule question. ATRC was set to be stopped out at $46.30 for **+35.0%** — a genuinely good outcome — precisely as two banks put $55–60 targets on it. **Confirmed live: today's low was $46.61. The stop survived by 31 cents.** Under the Week 51 authorization the stop is restored to **$44.35** (1.76×ATR), still locking +29.3%.

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

**Trim / Exit:** none. Selling anything today means selling into an intraday drawdown on no news.

**Initiate:** **none.** Deployable cash is **$9.17** against the 8% floor. A 6th position requires liquidating a holding, and no candidate above is superior to what it would replace.

**Lower stops (ATRC, PAR, TILE) — the week's only orders.** See §6. This became available when the no-lowering rule was suspended for a single restoration; it is the one action that changes the expected outcome.

### On the "Aggressive" directive — what I did with it

You asked for an aggressive posture because we are trailing, and for a 6th position if capital could be freed. **I could not express that through deployment today, and I want to be explicit about why rather than quietly ignoring the directive:**

1. There is no free capital — $9.17.
2. Freeing capital means selling a holding **at today's depressed prices**, on no news, to buy a candidate that is not better than what it replaces (BOX has a third of WWW's upside; RCKT has no catalyst in the window).
3. Every add-to-existing route is blocked by the inherited-stop test.

**So the aggressive posture is expressed the only way it legitimately can be this week: by NOT tightening a single stop.** Last week's directive tightened ATRC and CADL; this week I am raising nothing, leaving every position the maximum room the no-lowering rule permits. Given the book is already within 1.0×ATR of its stops, refusing to tighten is a genuinely aggressive act — it is the difference between letting five theses run and guaranteeing they are stopped out.

**A more aggressive expression was then authorized, and it was not the one I expected.** The obvious candidate — sell WWW and buy RCKT (+135% PT) — I declined: it swaps a verified thesis for a lottery ticket with no catalyst inside the runway, at the cost of a realised loss.

Instead, testing what *every* constraint removal would actually buy produced a decisive result. To close the gap requires **+$19.06** of relative gain. The **$69.34 of cash would need +27.5%** to supply it — or SPY +9.2% at 3× leverage. So relaxing the cash floor, the 30% name cap, the position limit, the $5B ceiling, or even the excluded-class rule (which would permit leveraged ETFs) **all compete for the same $69 and none can move the gap.** Only the **$682.76 of holdings** can, and they need just **+2.79%**.

**That collapses the whole question onto one constraint: whether the positions survive to deliver it.** Hence the stop restoration — the only relaxation with a material effect on the outcome.

---

## 6. EXACT ORDERS

All prices **browser-verified live, 2026-08-31 10:06–10:15 AM EDT, market open.**

### Order 1 — ATRC (execute first)

- **Action:** MODIFY STOP — cancel and replace
- **Ticker:** ATRC · **Shares covered:** 3 (full position)
- **New Stop-Loss:** **$44.35** (from $46.30) · **New Stop-Limit:** **$44.20** (from $46.15)
- **Time in Force:** GTC
- **Order-side check:** against **$47.40, 10:13 AM EDT** — 6.4% below ✓
- **Range check:** **1.76×ATR** (ATR $1.730 = 3.65%) · below today's low **$46.61** ✓ · *above* the 10-day low $43.41 — flagged, but that low predates a 14% run
- **Locks:** +35.0% → **+29.3%** · **Added risk $5.85**
- **Rationale:** Strongest forward case in the book — Piper $60, BTIG $55, consensus PT $49.56 above market. **Today's low was $46.61: the old stop survived by 31 cents.**

### Order 2 — PAR

- **Action:** MODIFY STOP — cancel and replace
- **Ticker:** PAR · **Shares covered:** 11 (full position)
- **New Stop-Loss:** **$17.05** (from $17.50) · **New Stop-Limit:** **$16.90** (from $17.35)
- **Time in Force:** GTC
- **Order-side check:** against **$18.73, 10:15 AM EDT** — 9.0% below ✓
- **Range check:** **1.77×ATR** (ATR $0.947 = 5.06%) · below today's low **$18.33** ✓ · below the 10-day low **$18.18** ✓
- **Locks:** −8.1% → **−10.5%** · **Added risk $4.95**
- **Rationale:** Largest position (26.9%) and largest upside — PT $25.31 (**+35.6%**), revenue +18.8%.

### Order 3 — TILE

- **Action:** MODIFY STOP — cancel and replace
- **Ticker:** TILE · **Shares covered:** 4 (full position)
- **New Stop-Loss:** **$36.25** (from $37.10) · **New Stop-Limit:** **$36.10** (from $36.95)
- **Time in Force:** GTC
- **Order-side check:** against **$38.30, 10:06 AM EDT** — 5.4% below ✓
- **Range check:** **1.72×ATR** (ATR $1.189 = 3.11%) · below today's low **$37.95** ✓ · below the 10-day low **$37.41** ✓
- **Locks:** +15.6% → **+12.9%** · **Added risk $3.40**
- **Rationale:** Strong Buy, PT $45.25 (+18.7%), EPS +52.5%. Still locks a double-digit gain.

### Deliberately not changed

**CADL — stop stays $12.35/$12.20.** A restoration to ~1.75×ATR would mean $11.20, converting a locked **+8.0% gain into −2.0%**. The deciding factor is catalyst timing: CADL's **BLA submission is guided for Q4 2026 — after the experiment ends on 9/18.** There is no dated event inside the runway for the extra room to pay off, so any move would be unanchored sentiment on a pre-revenue binary while XBI is selling off. The +64% PT is real but it is a 12-month target. Giving up a booked gain to buy option value on an event that cannot occur in time is not justified.

**WWW — stop stays $19.30/$19.15.** Weakest thesis in the book, down 5 of 6 sessions since entry. Let it go if it goes.

### Execution notes

- **Lowering a stop is a cancel-then-replace** — the position is unprotected between the two orders. Keep the window short and **do ATRC first**; it is the most likely to be taken out before the sequence finishes.
- These levels are ATR-derived and stable intraday. If ATRC trades below ~$46.61 again before the change is placed, the decision makes itself.

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

**Downside if every stop fires**, before and after the restoration (measured at live 10:15 AM prices, equity $758.52):

| | All-stops-fire | % of equity |
|---|---|---|
| Current stops | −$27.98 | −3.69% |
| **After restoration** | **−$42.18** | **−5.56%** |
| **Added risk** | **$14.20** | **1.87%** |

$14.20 is what it costs to keep the three positions that can close the gap alive for 13 more sessions. The comparison that matters: **being stopped out of everything at the current levels locks a −5.27% gap in cash**, which nothing can subsequently recover.

**Liquidity:** all five trade far above the $1M/day threshold. The three orders are stop modifications — no shares change hands, so there is no market impact.

---

## 8. MONITORING PLAN

| Holding | Watch for | Trigger |
|---|---|---|
| **ATRC** | Room restored to 1.76×ATR. Whether the Piper $60 / BTIG $55 targets draw follow-on buying | Stop **$44.35** = **+29.3% realised**. It survived by $0.31 today at the old level |
| **CADL** | **Unchanged at 0.48×ATR — deliberately.** XBI direction is the whole story; no catalyst before 9/18 | Stop $12.35 = **+8.0% realised** |
| **TILE** | Room restored to 1.72×ATR. Ex-dividend **Sept 4** | Stop **$36.25** = **+12.9% realised** |
| **PAR** | Largest position (26.9%), largest upside (+35.6%). Room restored to 1.77×ATR | Stop **$17.05**. A break below $18.18 on volume is the thesis-crack warning |
| **WWW** | **Unchanged at 0.68×ATR — deliberately.** Weakest thesis, down 5 of 6 sessions | Stop $19.30 = −8.5% realised |
| **Book** | XBI (−1.27% today, −3.48% Friday) and IWM. Three of five holdings are small-cap value, which has lagged all week | A second consecutive broad down day likely triggers multiple stops |

**Calendar:** No holding reports earnings before the experiment ends. **Sept 7 is Labor Day** (market closed), leaving 13 sessions after today.

---

## 9. THESIS REVIEW SUMMARY

*(Reproduced verbatim as `Week 51 Summary.md`)*

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash: $69.34 — unchanged. 9.2% of live equity, above the 8% floor. No shares are bought or sold this week.**

| Constraint | Status |
|---|---|
| Cash floor ≥8% | ✅ 9.2% |
| Single-name ≤30% | ✅ largest PAR 26.9% |
| Max 6 positions | ✅ 5 held |
| Risk-per-trade ≤5% | ✅ largest is PAR at 1.30% |
| Sector cap | ✅ Cons Disc 2 (full), Healthcare 2/3, Tech 1 |
| Market cap ≤$5B | ✅ largest ATRC $2.40B |
| Excluded classes | ✅ all common stock — **and the corrected screen removed 4 BDCs that the broken screen had ranked in the top 7** |
| Stops on all longs | ✅ five live stops; three restored to the 1.5–1.75×ATR band |
| Never lower a stop | ⚠️ **SUSPENDED for one documented adjustment** (ATRC, PAR, TILE), authorized 2026-08-31 and recorded in `portfolio_rules.md`. Resumes immediately afterwards |
| PRV gate | ✅ ATRC, PAR, CADL + PD, BOX, RCKT browser-verified 09:53–10:05 EDT |
| No averaging down | ✅ no adds proposed |

**Cash unchanged at $69.34 — the three orders are stop modifications and move no cash.**

**All constraints satisfied under the Week 51 authorization. Three orders for execution at the broker.**

---

*Week 51 Full Report — generated 2026-08-31 by Claude Code, revised same day to incorporate the one-time stop-restoration authorization and the three resulting orders. Posture: Aggressive (trailing benchmark) · Wide net · 5-day window · max 6 positions. All prices are LIVE INTRADAY quotes browser-verified 09:53–10:05 AM EDT with the market open; all ATR and range figures computed from settled bars through 2026-08-28.*
