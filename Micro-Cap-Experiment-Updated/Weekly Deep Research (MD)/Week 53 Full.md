# Week 53 Deep Research — Micro-Cap Portfolio (FINAL WEEK)

**Date:** Monday, 2026-09-14 (pre-market) | **Week:** 53 — final | **Runway:** ends 2026-09-18 — **5 sessions remain, including today**
**Session directives:** Sector = Wide net · Catalyst timing = Within 5 days (fixed by the calendar) · Risk posture = **Aggressive, deploy the cash** · Positions = **add 0–1, only a name that clears every filter** · ATRC partial = **take 1 share if it closes ≥ $54.88**

---

## 1. RESTATED RULES

- US-listed common stock, market cap ≤ $5B. No ETFs, CEFs/BDCs, SPACs, ADRs, units/warrants.
- **Cash floor 8%** · **risk-per-trade ≤5%** · **max loss per new position ≤2% of equity** (else reduce size) · **single-name ceiling 30%** · sector cap 2 (healthcare 3).
- **Market regime filter:** IWM below its 50-day SMA restricts new initiations to high-conviction, catalyst-driven plays.
- Stops mandatory. **Anti-ratchet:** no raise under 0.5×ATR, must leave ≥1.5×ATR. **Mechanical restoration:** once per position — ATRC, PAR used theirs; VTS and any new position have not.
- **PRV gate** before any buy/sell. **Thesis-input freshness:** time-varying drivers verified within 10 trading days, current level *and* direction. **Four-category** check for unexplained moves. **PT dispersion** pulled for any position of consequence.

---

## 2. RESEARCH SCOPE

Retrieved **2026-09-14, 07:03–07:55 AM EDT**, pre-market. Holdings priced at settled Friday 9/11 closes; pre-market prints at this hour carried no usable volume and were not relied upon.

| Check | Method | Result |
|---|---|---|
| Data currency | `--check-data-current` | Passes — last completed session 9/11 |
| Screener | `screener.py` | 15 candidates; weakest momentum of the final stretch |
| Candidate technicals + earnings dates | yfinance | No earnings inside the window for any candidate |
| PRV quote pages | Browser | CXW, BRBS, NMAX, LFST, BZH |
| CXW analyst dispersion | Browser → forecast page | All 5 analysts Buy; range $40–$45 |
| CXW / GEO news | Browser → StockTitan, quote pages | Latest releases Aug 5–10 |
| Detention-policy driver | WebSearch (discovery) → browser (dating) | Negative items dated Feb 2026 |
| Selloff dating | yfinance 12-month daily returns | Feb 12–20, 2026 |
| VTS dividend | yfinance + S&P Global table | Ex-date Sep 15 confirmed |

### Three corrections made during this session

1. **VTS ex-dividend date is Tuesday Sept 15.** yfinance's calendar field showed Sept 14 and I briefly reported that as a correction; the S&P Global dividend table (ex-date Sep 15, record date Sep 15, pay date **Sep 30**, $0.4375) is authoritative. The original date stands. The payment lands after the experiment ends, so the ledger will book the price drop and never the cash (~$2.63).
2. **CXW's forward PE of 2.11 is a data error.** The forecast table shows FY2026 net income of **$1.50B** against $116.5M the prior year — a misentry. On adjusted EPS the forward P/E is **~20.8× (FY26)** and **~14.7× (FY27)**; P/FFO 12.86.
3. **The search summary mis-dated the private-prison selloff as August 2026.** Price history places it in **February 2026** (GEO −14.9% Feb 12, −13.4% Feb 20; CXW −10.3% Feb 20). The underlying warehouse article is dated Feb 24, 2026.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

**Equity $733.51 · S&P-equivalent $768.52 · Gap −4.56% · Cash $259.33 (35.4%) · TWR alpha +0.55%**
**Regime: RISK-OFF** — IWM $288.89, below its 50-day SMA (−2.96% as of 9/10).

| Ticker | Role | Avg Cost | Close 9/11 | P&L | Stop | Conviction | Status |
|---|---|---|---|---|---|---|---|
| ATRC | Core winner | $34.30 | $54.54 | **+59.0%** | $50.30 | 5/5 | KEEP — partial pending |
| PAR | Growth | $19.05 | $18.00 | −5.5% | $17.05 | 3/5 | KEEP |
| VTS | Ballast | $17.85 | $18.76 | +5.1% | $17.50 | 4/5 | KEEP |

**ATRC** closed **$0.34 below its +60% partial trigger** ($54.88) after crossing it intraday at $55.16. All four deferral criteria technically hold. Per this session's directive, **1 share is sold if ATRC closes at or above $54.88** — consensus PT $51.67 sits below the price, the BoxX-NoAF catalyst is H1 2027, and five sessions favour banking a third. Stop $50.30 locks +46.6%; its next raise fails the anti-ratchet (0.29×ATR).

**PAR** recovered to $18.00 and 1.06×ATR from its stop after testing it all week. Its restoration is spent. Analyst targets range $16–$30 with UBS and RBC at $16 — the lowest conviction in the book, reduced to 3/5.

**VTS** is doing its job as ballast on a verified oil driver (WTI >$100). Stop $17.50 was sized to survive **tomorrow's ex-dividend** with ~1.7×ATR intact.

---

## 4. CANDIDATE SET

**Screen quality:** 20-day momentum mostly −2.5% to +10.8%; **six of fifteen closed below their 20-day SMA** and fail the relative-strength entry rule outright (DBRG, CBZ, WKC, VREX, RAMP, KRP). **No candidate reports earnings inside the window** — no binary events anywhere on the list.

| Ticker | Sector | Revenue | Rating / PT | Verdict |
|---|---|---|---|---|
| **CXW** | Industrials | **+24.3%** | **Strong Buy, $41.80 (+19.7%)** | ✅ **BUY** |
| BRBS | Financials | **−13.2%** | no coverage | ❌ shrinking revenue |
| NMAX | Comm. Services | +11.1% | Buy, $18.50 | ❌ lossmaking, fwd PE 111, ATR 4.97%, no beta history |
| LFST | Healthcare | +20.4% | Buy, **$14.20 (+9.5%)** | ❌ thin upside, at 52-wk high, $4.96B at the cap |
| BZH | Cons. Disc. | **−12.8%** | Hold, $27.50 (−17%) | ❌ **deal-pinned** |
| UTZ, DV | — | — | — | ❌ same deal-pin signature (ATR ~0.3%) |
| SPSC | Technology | — | — | ❌ 5-day momentum −1.8% |
| NATL | Technology | — | — | ❌ 20-day momentum −0.8% |

**On the deal-pin finding.** BZH is a beta-2.18 homebuilder that closed **0.00%** in an **8-cent daily range** ($33.25–$33.33) while trading 17% *above* its analyst target — the signature of a stock pinned to an acquisition price. UTZ and DV share its ~0.3% ATR with near-zero momentum. A pinned stock cannot generate return in five sessions regardless of the deal's merits.

### Selected: CXW — CoreCivic | Conviction 4/5

**PRV (9/11 close):** $34.93, market cap $3.45B, 52-week range $15.74–35.30 (new high **$35.05 on Sept 8**). Revenue TTM **$2.48B, +24.3%**; net income $127.9M; EPS $1.25 **+31.8%**. Beta **0.60**. ATR **3.37%**. ADV **$57.8M**. Next earnings **Nov 4** — outside the window.

**Analyst dispersion — the best case the rule can return.** Five analysts, **every one Buy or Strong Buy**, unchanged April through September. Targets **$40 low, $41.80 average, $42 median, $45 high** — **the lowest is +14.5% above the price.** Post-Q2 raises: JonesTrading $30→$42, Northland $32→$40, Benchmark $41→$45. **StoneX maintained $45 on Sept 11.**

**The catalyst that satisfies the RISK-OFF filter.** On Aug 10 CoreCivic entered a **$500M accelerated share repurchase** — about **14.5% of its market cap** — under an expanded $755.8M program, with ~12.4M shares delivered initially and final settlement before the end of Q2 2027. This is active, ongoing buying through our window, the same standard applied to VTS's verified oil driver when it was bought under RISK-OFF in Week 52. Q2 revenue rose **+27.3%** and guidance was revised with per-share metrics raised.

**Entry-discipline checks:** +3.9% over the 20-day SMA (≤20% ✓) · +8.4% over the 50-day (≤40% ✓) · 5-day move +3.8%, not days 1–3 of a >10% breakout ✓ · no earnings in window ✓.

**Thesis-input freshness — the driver verified, not reversing.** CXW's demand rests on federal immigration detention, a policy-driven input.

| Evidence | Date | Direction |
|---|---|---|
| ICE $45B multi-year detention appropriation | Jul 2025 | Supportive |
| GEO/CXW selloff on ICE insourcing report | **Feb 12–20, 2026** | Negative — **priced and recovered** |
| ICE warehouse purchases ([AIC](https://www.americanimmigrationcouncil.org/blog/ice-buys-warehouses-immigration-detention/)) | **Feb 24, 2026** | Negative — structural |
| FY26 appropriations: −5,500 beds, ICE funding maintained ([source](https://berardiimmigrationlaw.com/how-the-2026-federal-spending-bill-affects-immigration-enforcement-and-your-rights/)) | Early 2026 | Mildly negative |
| GEO: two new 5-year ICE contracts (~$165M/yr) + Rivers facility (~$80M/yr) | **Jul 29 – Aug 6** | Supportive |
| GEO and CXW both set **new 52-week highs** | **Aug 26 / Sep 8** | Supportive |
| ICE building its own 100 beds at Batavia ([source](https://nystateofpolitics.com/state-of-politics/new-york/capital-tonight/2026/09/02/ice-cites-new-york-law-in-planned-100-bed-expansion-at-batavia-detention-facility)) | **Sep 2** | Negative but immaterial in scale |
| StoneX maintains $45 on CXW | **Sep 11** | Supportive |

Weekly closes over five weeks: **CXW rising (0 down-weeks, +6.4%)**; **GEO down three consecutive weeks** (−1.3%, −1.6%, −0.9%, about −3.8%) from its Aug 26 high with no company news; **XLI down four weeks (−7.6%)**. GEO's three-week run touches the LXU threshold numerically, but it is a shallow pullback from a fresh high amid broad Industrials weakness under a RISK-OFF regime. LXU's disqualifier was the **driver itself** falling five straight weeks; here the newest dated driver evidence is new contracts and new highs.

**Bear case, stated plainly:**
1. **ICE insourcing is real and ongoing.** Its Detention Reengineering Initiative targets 92,600 beds via ICE-owned facilities, with a **Sept 30 deadline**. A fresh acquisition headline could gap CXW the way February's report did (−10.3% in one day). A stop does not protect a gap.
2. **Bought at a 52-week high**, in a sector down four weeks, under a RISK-OFF regime.
3. **Its role is modest.** Realistic five-session drift for a 0.60-beta name with buyback support is small; this position will not close a −4.56% gap. It improves the expected finish slightly and replaces idle cash.

---

## 5. PORTFOLIO ACTIONS

**Keep:** ATRC (5/5), VTS (4/5), PAR (3/5).
**Trim:** **ATRC — 1 share, conditional** on a close ≥ $54.88 (§6, Order 2).
**Initiate:** **CXW, 5 shares.**
**Add to / Exit:** none. No stop changes available: ATRC's next raise fails the anti-ratchet, PAR's floor sits below its stop, VTS was raised on 9/10.

---

## 6. EXACT ORDERS

### Order 1 — BUY CXW

- **Action:** BUY
- **Ticker:** CXW — CoreCivic (NYSE)
- **Shares:** 5
- **Order Type:** LIMIT
- **Limit Price:** **$34.93**
- **Time in Force:** DAY
- **Intended Execution:** 2026-09-14
- **Stop Loss:** **$32.85** — 1.77×ATR below entry; the wider of 1.75×ATR and the most recent swing low ($33.73); below the 5-day low of $33.15
- **Stop Limit:** **$32.70**
- **Special Instructions:** The limit equals the **last verified price — the $34.93 close, Sept 11, 4:00 PM EDT**. The 4:00 AM "pre-market $35.45" mirrored Friday's change with no volume and was not used. **Re-verify a live pre-market quote with volume around 9:25 AM.** **Do not chase above $35.50** (above the $35.30 52-week high). **Skip if it opens below $33.15** (5-day low), which would break the setup.
- **Rationale:** The only candidate clearing every filter — revenue +24.3%, all five analysts Buy with the lowest target +14.5% above market, an active $500M buyback worth 14.5% of market cap, beta 0.60, no earnings in the window, driver verified not reversing.

### Order 2 — CONDITIONAL: SELL 1 ATRC

- **Action:** SELL (partial profit, +60% tier)
- **Ticker:** ATRC · **Shares:** 1 of 3
- **Trigger:** **only if ATRC closes at or above $54.88**
- **Execution:** at the next session — cancel the existing stop, place a DAY limit sell for 1 share at or above the prior close, then **re-place the stop on the remaining 2 shares at $50.30 / $50.15**
- **Rationale:** consensus PT $51.67 is below the price and the BoxX-NoAF catalyst is H1 2027 — banking a third of the book's best position is worth more than the remaining upside over five sessions. Scale: +$4.24 if ATRC then stops out, −$5.46 if it runs to $60.

---

## 7. RISK AND LIQUIDITY CHECKS

**Post-trade allocation** (CXW 5 × $34.93 = $174.65):

| Holding | Value | % Equity |
|---|---|---|
| PAR | $198.00 | 27.0% |
| **CXW (new)** | **$174.65** | **23.8%** |
| ATRC | $163.62 | 22.3% |
| VTS | $112.56 | 15.3% |
| Cash | $84.68 | 11.5% |
| **TOTAL** | **$733.51** | 100% |

All under 30% ✓ · Cash above the 8% floor ($58.68) ✓ · **Sectors:** Technology 1, Healthcare 1, Energy 1, Industrials 1 ✓

**Risk if stopped:**

| Holding | Shares | Price − Stop | Max loss | % Equity |
|---|---|---|---|---|
| ATRC | 3 | $54.54 − $50.30 | $12.72 | 1.73% |
| PAR | 11 | $18.00 − $17.05 | $10.45 | 1.42% |
| VTS | 6 | $18.76 − $17.50 | $7.56 | 1.03% |
| **CXW** | 5 | $34.93 − $32.85 | **$10.40** | **1.42%** |
| **Aggregate** | | | **$41.13** | **5.61%** |

**CXW gap tail (not captured by the stop):** a repeat of February's −10.3% day costs **~$18.00 (2.45% of equity)**.
**Liquidity:** a $175 order against $57.8M ADV is immaterial.

---

## 8. MONITORING PLAN

| Holding | Watch | Trigger |
|---|---|---|
| **ATRC** | Close vs **$54.88** | Close ≥ $54.88 → sell 1 share next session; stop stays $50.30 on the rest |
| **CXW** | **ICE facility / insourcing headlines**; the DRI's Sept 30 deadline | A −5% session with an ICE headline is a thesis event — reassess same day, don't wait for the stop |
| **PAR** | 1.06×ATR from stop; restoration spent | Stop $17.05 |
| **VTS** | **Ex-dividend Tuesday Sept 15** (~$0.44 drop, no ledger credit); oil above $100 | Do not misread the ex-div drop; a sharp oil reversal is the real risk |
| Book | RISK-OFF regime; XLI's four-week slide | — |

**Calendar:** VTS ex-dividend Sep 15. ICE DRI deadline Sep 30 (after the finish). No earnings for any holding or candidate before Sept 18.

---

## 9. THESIS REVIEW SUMMARY

*(Reproduced as `Week 53 Summary.md`)*

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash: $259.33 → $84.68** after CXW — **11.5% of equity, above the 8% floor.**

| Constraint | Status |
|---|---|
| Cash floor ≥8% | ✅ 11.5% |
| Single-name ≤30% | ✅ largest PAR 27.0% |
| New-position max loss ≤2% | ✅ CXW 1.42% |
| Risk-per-trade ≤5% | ✅ |
| Sector cap | ✅ one name per sector |
| Market cap ≤$5B | ✅ CXW $3.45B |
| Regime filter (RISK-OFF) | ✅ catalyst-driven: active $500M ASR |
| Relative strength | ✅ CXW above its 20-day SMA |
| Distance from base | ✅ +3.9% / +8.4% |
| Earnings in window | ✅ none |
| PRV gate | ✅ |
| PT dispersion | ✅ all Buy, lowest target +14.5% |
| Thesis-input freshness | ✅ negative items dated Feb 2026 and priced; newest evidence supportive |

**All constraints satisfied. One order plus one conditional order.**

---

*Week 53 Full Report — final week. Generated 2026-09-14 by Claude Code. Posture: Aggressive · Wide net · 5-session window · add 0–1. Prices are 9/11 settled closes; ATR and range figures from settled bars through 9/11. Policy sources were found by WebSearch and dated by opening each in the browser.*
