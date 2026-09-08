# Week 52 Deep Research — Micro-Cap Portfolio (FINAL)

**Date:** Monday, 2026-09-07 (Labor Day, markets closed) | **Week:** 52 of 52 | **Runway:** ends 2026-09-18 — **9 sessions remain**
**Session directives:** Sector = Wide net · Catalyst timing = Within 5 days · Risk posture = **Aggressive, deploy the cash** · Max positions = 5 · **Endgame = hold through the close**

---

## 1. RESTATED RULES

- US-listed common stock, market cap ≤ $5B. No ETFs, CEFs/BDCs, SPACs, ADRs, units/warrants.
- **Cash floor 8%** · **risk-per-trade ≤5%** · **single-name ceiling 30%** · **sector cap 2 (healthcare 3)**
- **Binary event play: max 1 position, ≤15% of equity.**
- Stops mandatory. **Anti-ratchet:** no raise under 0.5×ATR, and it must leave ≥1.5×ATR of room.
- **Mechanical restoration:** a stop drifting below 1.5×ATR on price alone may be reset once per position. **ATRC, PAR and TILE have used theirs; CADL has not.**
- **PRV gate:** browser-fetch the live quote page before any buy/sell/trim/add.
- **Thesis-input freshness:** any commodity/rate/FX-dependent thesis requires the driver's current level *and direction*.

---

## 2. RESEARCH SCOPE

Retrieved **2026-09-07, 21:25–21:40 EDT**, with markets closed for Labor Day. All prices are Friday 9/04 settled closes plus verified after-hours prints — no intraday ambiguity this week.

| Check | Method | Result |
|---|---|---|
| Data currency | `--check-data-current` | Passes — last completed session 9/04 |
| Holdings + candidates | Browser → `stockanalysis.com` | ATRC, PAR, CADL + TYRA, VTS, DAN, NAVI, GNW, ITGR |
| Oil driver (VTS) | yfinance CL=F / BZ=F / XLE | WTI **$92.45, +7.8% 5d, +12.6% 20d** |
| Screener | `screener.py` | 15 candidates |

**Screen quality note:** this week's watchlist is the weakest of the final stretch — 20-day momentum spans just **+1.1% to +13.7%**, against +11% to +34% on the Week 50 screen, and the list is dominated by low-beta financials and insurers. That is a stalled tape, and it materially constrained what was available to buy.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

**Equity $772.00 · Cash $272.17 (35.3%) · S&P-equivalent $774.71 · Gap −0.35% · TWR alpha +5.71%**

| Ticker | Role | Entry | Avg Cost | Price (9/04) | Stop | Conviction | Status |
|---|---|---|---|---|---|---|---|
| ATRC | Core winner | 2026-07-15 | $34.30 | $51.52 (+50.2%) | $48.50 | 5/5 | KEEP |
| PAR | Growth | 2026-08-17/28 | $19.05 | $19.77 (+3.8%) | $17.05 | 4/5 | KEEP |
| CADL | Catalyst | 2026-08-17 | $11.43 | $12.78 (+11.8%) | $12.35 | 3/5 | KEEP |

**ATRC** — +50.2%, the position that carried the book. Three target raises in twelve days (BTIG $55, Piper $60, **Needham $64**), all Buy/Overweight, on the STS quality metric and the BoxX-NoAF trial. **The +60% partial trigger is $54.88, 6.5% away.** Stop $48.50 locks **+41.4%**. Worth restating: Needham attributed its raise to *peer multiple expansion*, and BoxX-NoAF data lands **H1 2027** — outside this experiment. This is a re-rated multiple, not a delivered fundamental, which is why the stop matters more than the target.

**PAR** — the best fundamental profile in the book: revenue **+18.8%**, forward PE ~16.8, Buy with **PT $25.31 (+28%)**. Rose **+4.60%** on Friday. At 28.2% of equity it is **$14 from the 30% cap** — no adds possible.

**CADL** — +11.8%, Strong Buy, **PT $21.00 (+64%)**, beta −0.50. Stop $12.35 sits at ~0.5×ATR on a 6.7%-ATR name, so it remains the likeliest exit. **CADL still holds its unused restoration allowance**, deliberately preserved: the CAN-2409 BLA is guided for Q4 2026, outside the runway, so buying room would purchase option value on an event that cannot occur in time.

**No stop changes are available.** Every trailing floor either sits below its stop or fails the anti-ratchet minimum — the same result as 9/03, and the rule working as designed.

---

## 4. CANDIDATE SET

| Ticker | Sector | Thesis | Catalyst | Verdict |
|---|---|---|---|---|
| **VTS** | Energy | Non-operated Williston/Rockies oil, revenue +7.9%, 9.8% yield | Oil at $92.45, +12.6% in 20 days | **✅ BUY** |
| TYRA | Healthcare | FGFR3 precision oncology, Strong Buy PT +77% | **SURF302 Phase 2 data, Sept 9** | ⚠️ **Declined — see §5** |
| DAN | Cons. Disc. | Auto parts, PT +16.8% | None in window | Reject |
| NAVI | Financials | Student lending | None | Reject |
| GNW | Financials | Mortgage/LTC insurance | None | Reject |
| ITGR | Healthcare | Medical device CDMO | Earnings Oct 22 (outside) | Reject |

**Why each was rejected**

- **NAVI — disqualified outright.** Revenue TTM **−42.8%**, net income −$49M, **Hold** rating with **PT $9.07, 5.8% BELOW the price.** Shrinking revenue is this book's single strongest disqualifier.
- **DAN — earnings quality fails.** Revenue +3.7%, and **forward PE 8.96 against trailing 3.09** — the $1.12B net income is a one-off rolling off, so the "PE 3" is an illusion. Same profile that disqualified PD twice. Beta 2.00.
- **GNW — unrateable.** **No analyst coverage** (rating n/a, PT n/a), so conviction cannot exceed 2/5 under the screener rule. Revenue +1.7%, trading 3% below its 52-week high.
- **ITGR — Hold with PT below market.** $126.50 against **PT $121.80 (−3.7%)**, sitting at its 52-week high, revenue +2.7%, earnings Oct 22 — after the finish.
- **KFY, EPC, ANAB, OGN, ATAI, BRBS, NMIH, CFFN, FBRT** — not advanced: 20-day momentum of +1% to +6% with no dated catalyst inside a 9-session runway. In this window those are index proxies carrying single-stock risk.

### Selected: VTS — Vitesse Energy | Conviction 4/5

**PRV, 9/04 close + AH 7:30 PM EDT:** $17.78 close, **after-hours $17.85 (+0.39%)**. Market cap $748.61M. Revenue TTM **$258.95M, +7.9%**. Net income −$11.25M (EPS −$0.28). **Buy, PT $21.00 (+18.1%).** Beta **0.63**. 52-week range $15.00–26.27 — **32% below its high.** ADV **$7.8M**.

**Thesis-input freshness — the driver verified, current and rising:**

| | Level (9/07) | 5-day | 20-day |
|---|---|---|---|
| WTI crude | **$92.45** | **+7.8%** | **+12.6%** |
| Brent | $97.07 | +7.3% | +10.7% |
| XLE | $64.06 | +2.2% | +11.4% |

Weekly WTI closes: 82.40 → 87.06 → 83.40 → 91.48 → **92.45**. The LXU rule voids a thesis whose driver has fallen ≥3 consecutive weeks; this driver is **rising into the entry**, which is the mirror image of that failure.

**Entry-discipline checks — all pass:** +5.0% over the 20-day SMA (limit ≤20%) · +10.0% over the 50-day (limit ≤40%) · not days 1–3 of a breakout · earnings Aug 3, long past · ADV $7.8M against a $107 order — immaterial.

**Why it fits this specific week:** ATR is just **2.33%** — the lowest-volatility name on the screen. With the gap at −0.35% and 9 sessions left, a low-beta (0.63) energy name with a rising driver adds exposure without adding the variance that could turn a near-tie into a clear loss. It also fills the **empty Energy sector**.

**Bear case, stated plainly:**
1. **The oil move is geopolitical** — US/Iran tensions and a Venezuela policy shift — not a demand-driven re-rating. A war premium can unwind in days, and VTS would follow.
2. **GAAP lossmaking** (−$0.28 EPS) despite the 9.8% distribution.
3. **⚠️ Ex-dividend Sept 15 — inside our window.** The quarterly payment (~$0.44/share) mechanically reduces the price on the ex-date, and **the ledger tracks `shares × price` with no dividend credit.** On 6 shares that is a **~$2.63 uncredited drag (0.34% of equity)** — a known, quantified cost of holding through the 15th, not a surprise.

---

## 5. TYRA — the binary I am declining, and why

TYRA is the **only candidate on the entire screen with a dated catalyst inside the window**, which is exactly what the 5-day timing directive asked for. It deserves a full account rather than a one-line rejection.

**The setup:** On Sept 4 the company announced a conference call for **initial results of oral dabogratinib from the SURF302 Phase 2 study in IR NMIBC, on September 9, 2026** — two sessions away. The stock closed **+15.25% at $28.65** on 1.86M shares, with after-hours $29.30. **Strong Buy from 15 analysts, PT $50.82 (+77%).** Beta 0.73, ADV $13M. It sits **−1.4% below its 50-day SMA** and 29.5% below its 52-week high, so it is a depressed name bouncing, not an extended one.

**Sizing, if taken:** 4 shares = $114.60 = 14.8% of equity, just inside the 15% binary cap.

**Why I am declining it:**

1. **A stop cannot protect a binary.** Phase 2 oncology readouts gap. If the data disappoints, TYRA opens 30–40% lower and the stop fills far below its level. The protection this book relies on simply does not function here — at 4 shares a −35% gap costs **−$40 (−5.2% of equity)**, turning a −0.35% gap into roughly −5.5% with 7 sessions left. Unrecoverable.
2. **I have no edge on the outcome.** The screener rule is explicit that a screen score is sourcing, not conviction, and conviction may rise only on independent evidence. I have evidence that analysts like the *stock* and that a date exists — none whatsoever about the *data*. Buying is a gamble, not an investment, and I should name it as such.
3. **The rumor is already bought.** A +15.25% move on the mere scheduling of a call means optimism is priced. We would be buying the top of the anticipation, before the resolution.
4. **It is day 1 of a >10% breakout**, which entry discipline tells us to avoid — the ARLO Week 34 rule.
5. **It resolves the experiment on a coin flip.** The gap is −0.35%: effectively a tie. A symmetric zero-edge bet converts "narrow finish" into "50/50 clear win or clear loss" — and this experiment exists to measure whether a disciplined process generates alpha. Settling that question on one clinical readout destroys the measurement regardless of which way it lands.

**This is your call, not mine, and you have overridden me correctly before.** If you want it, the order is: **BUY 4 TYRA, limit $29.30 (the last verified AH print), stop $24.10 — below the 10-day low of $24.17** — with the understanding that the stop is decoration against a gap. Under the pre-catalyst exit rule, also set a **+30% alert at $38.10** and, if it triggers, cancel the stop and sell ~1/3 into the spike.

---

## 6. PORTFOLIO ACTIONS

**Keep:** ATRC (5/5), PAR (4/5), CADL (3/5) — no thesis has broken.
**Add to:** none. PAR is $14 from the 30% cap; CADL fails the inherited-stop test at ~0.5×ATR; ATRC has room for exactly 1 share but sits at a 52-week high with consensus PT ($51.67) at the price — that is chasing.
**Trim / Exit:** none.
**Initiate:** **VTS, 6 shares.**

**On the unused fifth slot:** you authorized up to two new names and I am taking one. The screen did not produce a second that survives its own filters — NAVI's revenue is shrinking 42.8%, DAN's earnings base is evaporating, GNW has no coverage, ITGR's target is below its price. **Forcing a second position to satisfy a directive would be the exact failure this book has documented repeatedly.** Holding $165 rather than buying the sixth-best name on a weak screen is the disciplined answer, and `portfolio_rules.md` says so directly: *if no candidates pass all filters, hold cash and explain why.*

---

## 7. EXACT ORDERS

### Order 1 — BUY VTS

- **Action:** BUY
- **Ticker:** VTS — Vitesse Energy (NYSE)
- **Shares:** 6
- **Order Type:** LIMIT
- **Limit Price:** **$17.85**
- **Time in Force:** DAY
- **Intended Execution:** 2026-09-08 (Tuesday)
- **Stop-Loss:** **$16.65** — below the 10-day low of $16.69; **2.73×ATR** below entry
- **Stop-Limit:** **$16.50**
- **Special Instructions:** The limit equals the **last verified price — $17.85 after-hours, Sept 4, 7:30 PM EDT**; no Tuesday pre-market exists yet with markets closed today. **Do not chase above $18.30** — WTI rose ~1% while equities were shut, so VTS may gap; if it opens above that, stand down and I will reprice against a live pre-market quote in Tuesday's daily. **Skip condition: do not buy if VTS opens below $16.94** (its 20-day SMA), which would void the setup.
- **Rationale:** Only candidate passing every filter — revenue +7.9%, Buy with +18.1% upside, 32% below its 52-week high, filling the empty Energy sector, on a driver verified rising (+12.6% in 20 days), at the lowest ATR on the screen (2.33%).

**No other orders.** Stops unchanged: ATRC $48.50/$48.35 · PAR $17.05/$16.90 · CADL $12.35/$12.20.

---

## 8. RISK AND LIQUIDITY CHECKS

**Post-trade allocation** (VTS at $17.85 × 6 = $107.10):

| Holding | Value | % Equity |
|---|---|---|
| PAR | $217.47 | 28.2% |
| ATRC | $154.56 | 20.0% |
| CADL | $127.80 | 16.6% |
| **VTS (new)** | **$107.10** | **13.9%** |
| Cash | $165.07 | 21.4% |
| **TOTAL** | **$772.00** | 100% |

All under the 30% ceiling ✓ · Cash above the 8% floor ($61.76) ✓ · 4 of 5 positions ✓

**Sector:** Technology 1 (PAR) · Healthcare 2 of 3 (ATRC, CADL) · **Energy 1 (VTS, new)** · Consumer Discretionary 0 ✓

**Risk per position if stopped:**

| Holding | Shares | Price − Stop | Max loss | % Equity |
|---|---|---|---|---|
| ATRC | 3 | $51.52 − $48.50 | $9.06 | 1.17% ✓ |
| PAR | 11 | $19.77 − $17.05 | $29.92 | 3.88% ✓ |
| CADL | 10 | $12.78 − $12.35 | $4.30 | 0.56% ✓ |
| **VTS** | 6 | $17.85 − $16.65 | **$7.20** | **0.93%** ✓ |
| **Aggregate** | | | **$50.48** | **6.54%** |

Every position within the 5% risk-per-trade limit. **Liquidity:** a $107 order against $7.8M ADV is 0.0014% — immaterial, far inside the 10%-of-ADV slippage guard.

---

## 9. MONITORING PLAN

| Holding | Watch | Trigger |
|---|---|---|
| **ATRC** | **+60% partial at $54.88 — 6.5% away.** If reached, run all four deferral criteria fresh | Stop $48.50 = **+41.4% realised**. My lean is to **take** the partial rather than defer a third time: the BoxX-NoAF catalyst is H1 2027, so nothing can deliver inside the runway |
| **PAR** | Largest position (28.2%), best fundamentals, $14 from the 30% cap — the cap will block adds even on strength | Stop $17.05. A break below $18.50 on volume is the warning |
| **CADL** | Likeliest exit at ~0.5×ATR. **Restoration allowance still unused** if it drifts further on price alone | Stop $12.35 = **+8.0% realised** |
| **VTS** | **Oil is the whole thesis.** A US/Iran de-escalation headline unwinds the war premium fast | **Ex-dividend Sept 15** — expect a ~$0.44 price drop the ledger will not credit. Do not misread it as a thesis break |
| Book | IWM vs its 50-day; XBI for CADL | Regime was RISK-OFF at Friday's close |

**Calendar to the finish:** no holding reports earnings before 9/18. The only dated events are **VTS ex-dividend (Sept 15)** and, if taken, **TYRA's SURF302 readout (Sept 9)**.

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash: $272.17 → $165.07** after the VTS purchase — **21.4% of equity, above the 8% floor.**

| Constraint | Status |
|---|---|
| Cash floor ≥8% | ✅ 21.4% |
| Single-name ≤30% | ✅ largest PAR 28.2% |
| Max 5 positions | ✅ 4 held |
| Risk-per-trade ≤5% | ✅ largest PAR 3.88% |
| Sector cap | ✅ Healthcare 2/3, Tech 1, Energy 1 |
| Market cap ≤$5B | ✅ VTS $749M |
| Excluded classes | ✅ all common stock |
| Stops on all longs | ✅ four live stops |
| Full shares | ✅ |
| PRV gate | ✅ 3 holdings + 6 candidates browser-verified 21:25–21:40 EDT |
| Thesis-input freshness | ✅ oil verified current and rising |
| Binary event cap | ✅ none taken |

**All constraints satisfied. One order for execution on 2026-09-08.**

---

*Week 52 Full Report — the final scheduled research window. Generated 2026-09-07 by Claude Code. Posture: Aggressive · Wide net · 5-day catalyst window · max 5 positions · hold through the 9/18 close. All prices are 9/04 settled closes plus verified after-hours prints; ATR and range figures computed from settled bars through 2026-09-04.*
