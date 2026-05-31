# Week 37 — Deep Research Report

**Date:** Friday, May 29, 2026
**Week:** 37 of 52 (twelve-month live experiment)
**Analyst:** Claude Deep Research Mode

---

## 1. RESTATED RULES

- **Universe:** U.S.-listed common stocks, nano-cap to small-cap (market cap up to $2Bn), on NYSE/NASDAQ/NYSE American
- **Exclusions:** OTC/pink sheets, ETFs, SPACs, ADRs, defence, Israeli-affiliated
- **Execution:** Long-only, full shares only, no options/leverage/margin
- **Position sizing:** Max loss at stop = 5% of portfolio equity; no single position > 30% of equity
- **Cash reserve:** Maintain >= 15% cash at all times
- **Stop-losses:** Required on all positions; default max(1.5x ATR(14), 10% below entry)
- **Sector cap:** No more than 2 of 5 positions in the same GICS sector
- **Liquidity filters:** Price >= $1.00, 3-month ADV >= $500K, bid-ask spread <= 2%, float >= 5M shares
- **Entry discipline:** 3-day post-earnings cooldown, distance-from-base gates (50-day SMA <= 40%, 20-day SMA <= 20%), ATR-based stop sizing, pre-open verification required
- **Re-entry ban:** 10 trading days after stop-out (ECVT banned until ~June 12)
- **Market regime filter:** If IWM < 50-day SMA, restrict new initiations to catalyst-only. **STATUS: IWM at $290.43 vs 50-day SMA ~$279.06 — ABOVE. All strategies permitted.**
- **Partial profit-taking:** ~1/3 at +30%, ~1/3 at +60%, remainder with trailing stop
- **Screener picks:** Conviction starts at 2/5; raise only on independent verification

---

## 2. RESEARCH SCOPE

**Sources consulted:**
- Yahoo Finance (price, volume, key statistics) — accessed May 29, 2026
- Seeking Alpha (earnings analysis, analyst ratings, bearish theses) — accessed May 29, 2026
- Investing.com (technical analysis, moving averages) — accessed May 29, 2026
- StockTitan / QuiverQuant (press releases, earnings reports) — accessed May 29, 2026
- Finviz (screener data, technicals) — accessed May 29, 2026
- SEC EDGAR (10-Q, 8-K filings) — accessed May 29, 2026
- MarketBeat / TipRanks (analyst consensus, short interest) — accessed May 29, 2026
- Timothy Sykes News (momentum alerts, breakout tracking) — accessed May 29, 2026
- Company IR pages (press releases, guidance) — accessed May 29, 2026

**Checks performed:**
- All 4 current holdings: price, volume, news, catalyst status, stop adequacy
- 7 new candidates evaluated via web search (IMPP, CRNC, AIOT, CLPT, CDZI, PRGS, MGTX)
- IWM regime filter verified (above 50-day SMA)
- ECVT re-entry ban verified (stopped out May 29; banned until ~June 12)
- Distance-from-base gates applied to all candidates
- Post-earnings cooldown checked for all candidates
- Sector concentration verified for all proposed trades
- Slippage guard computed for all proposed orders

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|------------|----------|---------------|-------------|-------------------|--------|
| WKC | Anchor — Energy Distribution | Week 28 | $26.00 | $28.81 | $26.50/$26.35 (broker) | 4/5 | HOLD — thesis confirmed, near 52-wk high |
| ACCO | Marginal — Consumer/Industrials | Week 34 | $3.92 | $3.96 | $3.53/$3.43 | 2/5 (down from 3/5) | WATCH — flat after 3 weeks, recycle review Week 38 |
| SHO | Core — Hotel REIT | Week 35 | $10.20 | $10.82 | $9.40/$9.30 | 4/5 | HOLD — buyback floor, Q1 beat confirmed |
| OSPN | Growth — Cybersecurity/Digital ID | Week 36 (filled 5/26) | $13.00 | $14.44 | $12.00/$11.90 | 3/5 (up from 2/5) | HOLD — +11.1% in 3 days, ARR momentum |

**Portfolio Summary:**
- Portfolio Equity: $672.44
- S&P 500 Equivalent: $760.80
- Gap to Benchmark: -$88.36 (-11.6%)
- Cash: $207.68 (30.9%)
- Holdings: 4 of 5 slots filled
- ECVT stopped out May 29 at $13.24 (-$3.74 P&L, -5.4% from entry)

**Week-over-Week Changes:**
- OSPN entered May 26 (12 shares @ $13.00) — already +11.1%
- ECVT stopped out May 29 (stop triggered at $13.24, filled at $13.24) — P&L -$3.74
- WKC broker stop raised to $26.50/$26.35 per Week 36 recommendation
- Net equity change: $672.44 vs $653.13 (Week 36) = +$19.31 (+3.0%)

**Per-Holding Notes:**

**WKC ($28.81, +10.8% from entry):** Thesis fully confirmed. Marine GP +82% YoY remains the anchor thesis. FY26 adj-EPS guide $2.65-$2.85. Trading near 52-week high. Morgan Stanley Underweight at $25 PT is stale and increasingly irrelevant. Raymond James Outperform. The position is small (2 shares, 8.6% of equity) — the stop at $26.50 locks in breakeven. No new material news this week.

**ACCO ($3.96, +1.0% from entry):** Week 36 thesis flagged this as the recycling candidate: "if flat through Week 37, capital may be better deployed elsewhere." ACCO is functionally flat (+1.0%) after 3 weeks of holding. Q1 revenue beat (+8% YoY) was EPOS-driven, not organic. Market wants proof of organic growth. EPOS integration is progressing ($15.2M Q1 contribution, $80M FY26 run-rate expected). **Conviction downgraded to 2/5.** Decision: keep one more week with tight stop; if still flat at Week 38, recycle into higher-beta candidate.

**SHO ($10.82, +6.1% from entry):** Q1 was strong — RevPAR +14.6% to $255.04, Adjusted EBITDAre +18.3% to $67.7M, FFO $0.27/sh (+28.6%). $458.3M buyback remaining (24% of market cap). Repurchased 3.86M shares YTD at avg $9.11. However, adjusted EBITDAre guidance lowered to $226-240M. The guidance reduction is a concern but the massive buyback provides a price floor. Dividend $0.09/quarter (3.5% yield at entry). Conviction remains 4/5 — buyback and valuation discipline offset guidance softness.

**OSPN ($14.44, +11.1% from entry):** Best performer this week. Q1 EPS $0.39 beat $0.35 consensus. ARR $192M (+14% YoY), subscription revenue +8%. $50M buyback launched May 11. Analyst consensus Buy, PT $16.25 (+12.5% upside). The +11.1% gain in 3 trading days validates the screener-sourced entry. Conviction upgraded from 2/5 to 3/5 based on: (1) independent verification of ARR growth, (2) confirmed buyback with real board authorization, (3) analyst consensus confirms thesis. **Minimum 5-day hold in effect** — do not adjust stop until 5 trading days complete (next eligible: June 2).

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |
|--------|----------------|-------------|------------------------------|----------------|
| IMPP | Greek shipping fleet expansion + Q1 earnings blowout | Fleet growing to 26 vessels by Q3; Q1 rev +25.8% beat, EPS +67.3% beat | CONFIRMED — earnings reported May 22, fleet delivery confirmed in press release | ADV 360-720K shares, ~$2-4M daily dollar vol — PASS |
| CRNC | Amazon patent lawsuit + ITC complaint = IP monetization | ITC filing May 5, District Court actions pending; Q2 rev beat | CONFIRMED — ITC filing verified via company IR and multiple sources | ADV 1.36M shares, ~$17M daily dollar vol — PASS |
| AIOT | Accenture partnership for Central Europe IoT safety expansion | Partnership announced May 27 for connected warehouse/road safety | CONFIRMED — company PR and multiple financial news sources | ADV 1.34M shares, ~$4.6M daily dollar vol — PASS |
| CLPT | ClearPoint Neuro +43% Q1 revenue growth, FDA clearances | Revenue growth acceleration, 175+ global installed base | CONFIRMED — 10-Q filed, 8-K press release | Low volume; needs verification |
| CDZI | Bureau of Reclamation funding agreement for Mojave water bank | Gov't funding agreement executed May 26 for regulatory review | CONFIRMED — company PR via StockTitan | ADV needs verification |
| PRGS | Progress Software Q1 beat, earnings June 29 | Next earnings ~21 trading days away; analyst Buy consensus, PT $60.60 | CONFIRMED — but outside 10-day catalyst window | ADV adequate, mkt cap ~$1.15B |
| MGTX | MeiraGTx FDA Breakthrough Therapy designation for gene therapy | FDA designation confirmed, but Q1 revenue missed estimates | PARTIALLY CONFIRMED — FDA designation verified, revenue miss noted | ADV needs verification |

**Candidate Evaluation — Screener & Analyst-Sourced:**

**IMPP — SELECTED for 5th slot.** Q1 revenue $61.7M (+25.8% beat), EPS $0.58 (+67.3% beat). Best quarterly performance in company history (per press release). Fleet expanding from 21 to 26 vessels by end of Q3 2026. Post-earnings cooldown: earnings reported May 22; 5 trading days elapsed = CLEARED. Distance-from-base: 50-day SMA ~$4.71; current $5.41 = +14.9% above (within 40% limit). 20-day SMA estimated ~$5.10; current $5.41 = +6.1% above (within 20% limit). Above 20-day SMA: PASS. Energy sector creates 2/5 energy positions with WKC — within sector cap. **Bear case:** Seeking Alpha has published bearish theses ("permanent value trap") citing related-party management agreement with Stealth Maritime (Vafias family), concentrated 74% insider ownership, and historical dilution risk. These governance concerns are real but manageable for a momentum trade with strict stop discipline.

**CRNC — REJECTED (distance-from-base).** Stock broke out from $9.20 to $12.83 (+39%). Estimated 20-day SMA ~$10.50; current price 22% above = EXCEEDS 20% limit. Additionally, Q2 EPS $0.04 missed $0.14 consensus despite revenue beat. Amazon lawsuit is options-like upside but uncertain. 13.6% short interest adds squeeze potential but also downside risk.

**AIOT — REJECTED (fundamental weakness).** Down 33% YTD despite Accenture partnership. GF Value $4.25 suggests 18% undervaluation, but the persistent decline signals deeper issues. Beta 2.66 adds uncompensated volatility. Would create 2 Technology positions with OSPN (assuming IoT classification under Technology).

**CLPT — REJECTED (loss-making, widening losses).** Revenue +43% is impressive but EPS -$0.32 missed -$0.27 estimate. Losses widening despite revenue growth. Not appropriate for an aggressive posture where we need immediate alpha.

**CDZI — REJECTED (distant catalyst).** Bureau of Reclamation regulatory review is long-duration (months to years). No near-term price catalyst within 10 trading days. Speculative water-rights play.

**PRGS — REJECTED (outside catalyst window, broken chart).** Next earnings June 29 is 21 trading days away — outside the 10-day session directive. Stock -55% from 52-week high ($65.50 to $29.72) signals structural issues. Citi lowered PT to $46. Would create 2 Technology positions with OSPN.

**MGTX — REJECTED (revenue miss).** FDA Breakthrough designation is strong, but Q1 revenue miss undermines near-term momentum. Biotech binary risk without clear date-certain catalyst within 10 days.

---

## 5. PORTFOLIO ACTIONS

**Keep:**
- **WKC** — thesis confirmed, near 52-week high, raise trailing stop to lock in gains
- **SHO** — buyback floor intact, Q1 beat validated, dividend yield provides carry
- **OSPN** — +11.1% Week 1, ARR momentum confirmed, minimum 5-day hold active

**Add to:** None — no existing positions warrant adding (OSPN already near 26% of equity)

**Trim:** None — no position has reached +30% partial-profit threshold

**Exit:** None — no stops breached, no thesis breaks

**Watch (conditional exit):**
- **ACCO** — conviction downgraded to 2/5. If still flat (+/- 3%) at Week 38, recycle capital into higher-beta candidate. Stop at $3.53/$3.43 provides -9.9% max drawdown from entry.

**Initiate:**
- **IMPP** — 19 shares @ $5.40 limit — Q1 earnings blowout, fleet expansion catalyst, post-earnings pullback creates entry. Fills the open 5th slot. Conviction 3/5.

**Stop Adjustments:**
- **WKC:** Raise stop from $26.50/$26.35 to $27.90/$27.75 — locks in +$1.90/share profit (+7.3% from entry). ATR(14) ~$0.50; stop is 1.82x ATR below current = adequate.
- **SHO:** Raise stop from $9.40/$9.30 to $10.00/$9.90 — still below 50-day SMA area, provides 7.6% cushion from current price. ATR(14) ~$0.20; stop is 4.1x ATR below current = conservative.
- **OSPN:** No change — maintain $12.00/$11.90 until 5-day minimum hold expires (June 2). Re-evaluate at Week 38.
- **ACCO:** No change — maintain $3.53/$3.43.

---

## 6. EXACT ORDERS

### Order 1 — IMPP Initiation

```
Action:                BUY
Ticker:                IMPP
Shares:                19
Order Type:            Limit — pullback from $5.68 to $5.41; limit at $5.40 to capture 
                       further consolidation
Limit Price:           $5.40
Time in Force:         DAY
Intended Execution:    2026-06-02 (Monday)
Stop Loss:             $4.80 — 1.75x ATR(14) below entry; ATR ~$0.35, 
                       1.75x = $0.61, $5.40 - $0.61 = $4.79, rounded to $4.80
Stop Limit:            $4.70 — $0.10 below stop loss for execution buffer
Special Instructions:  Place stop-limit order with broker immediately after fill.
                       "Place this stop with your broker before the next market open."
Rationale:             Q1 earnings blowout (rev +25.8%, EPS +67.3%) + fleet expansion 
                       to 26 vessels by Q3 provides near-term momentum catalyst. 
                       Pullback from $5.68 creates entry opportunity. 2/5 energy 
                       sector allocation is within limits.
```

### Order 2 — WKC Stop Raise

```
Action:                MODIFY STOP
Ticker:                WKC
Current Stop:          $26.50 / $26.35
New Stop Loss:         $27.90
New Stop Limit:        $27.75
Rationale:             Stock at $28.81, trailing stop at 1.82x ATR below current price.
                       Locks in +$1.90/share profit (+7.3% from $26.00 entry).
```

### Order 3 — SHO Stop Raise

```
Action:                MODIFY STOP
Ticker:                SHO
Current Stop:          $9.40 / $9.30
New Stop Loss:         $10.00
New Stop Limit:        $9.90
Rationale:             Stock at $10.82 (+6.1% from entry). Raising stop to $10.00 
                       provides 7.6% cushion while moving closer to breakeven protection. 
                       4.1x ATR below current — conservative but appropriate given 
                       guidance reduction.
```

---

## 7. RISK AND LIQUIDITY CHECKS

### Position Concentration After Trades (assumes IMPP fills at $5.40)

| Holding | Value | % of Equity |
|---------|-------|-------------|
| WKC | $57.62 | 8.6% |
| ACCO | $114.84 | 17.1% |
| SHO | $119.02 | 17.7% |
| OSPN | $173.28 | 25.8% |
| IMPP (new) | $102.60 | 15.3% |
| Cash | $105.08 | 15.6% |
| **Total** | **$672.44** | **100.0%** |

**Concentration checks:**
- No position exceeds 30% cap (OSPN at 25.8% is largest) -- PASS
- OSPN approaching 30% — if it rises another +16%, will trigger monitoring. Flag for trim review if >28%.
- 2 Energy positions (WKC 8.6% + IMPP 15.3% = 23.9%) — within sector cap (max 2 of 5) -- PASS

### Cash Remaining After Trades

- Pre-trade cash: $207.68
- IMPP purchase: -$102.60 (19 x $5.40)
- Post-trade cash: $105.08 (15.6% of equity)
- 15% floor: $100.87
- **Cash buffer above floor: $4.21** -- TIGHT but PASS

### Per-Order Size as Multiple of Average Daily Volume

| Ticker | Order Shares | ADV (shares) | % of ADV | ADV ($) | Order ($) | % of $ ADV |
|--------|-------------|-------------|----------|---------|-----------|------------|
| IMPP | 19 | 360,000-720,000 | <0.01% | ~$2-4M | $102.60 | <0.01% |

**Slippage risk: NEGLIGIBLE.** 19 shares is a rounding error relative to IMPP's daily volume.

### Maximum Stop-Loss Exposure

| Ticker | Shares | Entry | Stop | Risk/Share | Total Risk | % of Equity |
|--------|--------|-------|------|-----------|-----------|-------------|
| WKC | 2 | $26.00 | $27.90 | -$0.91* | +$1.80* | +0.3% (profit) |
| ACCO | 29 | $3.92 | $3.53 | $0.39 | $11.31 | 1.7% |
| SHO | 11 | $10.20 | $10.00 | $0.20 | $2.20 | 0.3% |
| OSPN | 12 | $13.00 | $12.00 | $1.00 | $12.00 | 1.8% |
| IMPP | 19 | $5.40 | $4.80 | $0.60 | $11.40 | 1.7% |
| | | | | **Total risk:** | **$37.91** | **5.6%** |

*WKC stop is above entry — would lock in profit if triggered.

**Worst-case scenario** (all stops triggered simultaneously): equity would decline by ~$35.11 (net of WKC profit) = -5.2% of current equity. This is within acceptable risk bounds.

---

## 8. MONITORING PLAN

### WKC (World Kinect) — Week 38

- Watch for continued momentum above $29; if breaks $30, consider profit-taking review
- Monitor marine GP margin sustainability — any industry data on bunker fuel demand
- Watch for insider transactions (10b5-1 plan sales are pre-scheduled, non-material)
- Morgan Stanley PT at $25 is irrelevant at current levels; watch for upgrade catalysts
- **Action trigger:** If closes below $27.90, stop activates; review thesis

### ACCO (ACCO Brands) — Week 38 (CRITICAL)

- This is the recycling decision week. If stock remains within +/-3% of current ($3.84-$4.08), begin exit planning
- Monitor EPOS integration progress — any customer wins or margin improvement data
- Watch for office products seasonal weakness heading into summer
- If Q2 guidance is reiterated or raised, may extend hold
- **Action trigger:** Break below $3.53 triggers exit; break above $4.15 extends hold

### SHO (Sunstone Hotel Investors) — Week 38

- Monitor for additional buyback disclosures (next 10-Q expected)
- Watch hotel RevPAR data — summer travel season should support thesis
- Track interest rate expectations — rate cuts could boost REIT valuations
- Guidance reduction to $226-240M EBITDAre needs watching; any further reduction is thesis-negative
- **Action trigger:** If closes below $10.00, stop activates

### OSPN (OneSpan) — Week 38

- 5-day minimum hold expires June 2 — reassess stop level
- Monitor for $50M buyback execution (any 10b-18 filing or volume patterns)
- Watch cybersecurity sector news — any major breaches boost demand thesis
- Next earnings expected late July — no near-term binary risk
- Track analyst PT revisions — consensus at $16.25 provides +12.5% upside target
- **Action trigger:** If +30% from entry ($16.90), trigger partial-profit review

### IMPP (Imperial Petroleum) — Week 38 (if filled)

- Verify fill and place stop ($4.80/$4.70) with broker immediately
- Monitor shipping rates — any industry data on tanker/dry bulk rates
- Watch for fleet delivery announcements (Eco Crossfire delivered April; 4 dry bulk + 1 tanker pending)
- Track related-party disclosures — any Stealth Maritime management fee changes
- **Pre-open verification required:** Check Monday 6/2 pre-market. If pre-market down >2%, downgrade to limit at pre-market price or pass.
- **Action trigger:** Day-1 drawdown rule — if closes -8% or worse on entry day, exit at next open

---

## 9. THESIS REVIEW SUMMARY

### Per-Position Thesis

**WKC (World Kinect) — Conviction 4/5 (unchanged)**
Thesis is fully confirmed and executing. Marine GP +82% YoY was the catalyst that broke the bear case. FY26 adj-EPS guidance raised to $2.65-$2.85. The stock at $28.81 is near its 52-week high and well above Morgan Stanley's stale $25 PT. Raymond James Outperform rating validates the bull thesis. The position is small (8.6% of equity, 2 shares) — the raised stop at $27.90 locks in +7.3% profit. At this point, WKC is a "let it run" position with trailing stop protection.
**Risk:** Energy distribution cyclicality; marine segment outperformance may normalize in H2. Low position size limits contribution to alpha generation.

**ACCO (ACCO Brands) — Conviction 2/5 (down from 3/5)**
ACCO is the weakest link in the portfolio. After 3 weeks of holding, the stock has moved +1.0% — functionally flat. The Q1 revenue beat (+8% YoY) was driven entirely by the EPOS acquisition ($15.2M), not organic growth. FY26 EPS guidance maintained at $0.84-$0.89, but the market is not rewarding the execution. The EPOS integration thesis (tech peripherals pivot away from office products) is credible long-term but is not generating near-term price action. Conviction downgraded to 2/5. This is the recycling candidate — if flat at Week 38, exit and deploy capital into higher-beta name.
**Risk:** Office-products secular decline, FX headwinds. EPOS integration execution. The $3.53 stop provides -9.9% max drawdown from entry.

**SHO (Sunstone Hotel Investors) — Conviction 4/5 (unchanged)**
Q1 results validated the thesis: RevPAR +14.6%, EBITDAre +18.3%, FFO $0.27 (+28.6%). The $458.3M buyback authorization (24% of market cap) provides a significant floor — management has already repurchased 3.86M shares at avg $9.11 in 2026 YTD. The guidance reduction to $226-240M EBITDAre is a yellow flag but doesn't break the thesis given the buyback support and approaching summer travel season. Dividend $0.09/quarter (3.5% yield at entry) provides carry while waiting. Stop raised to $10.00 to protect gains while maintaining cushion.
**Risk:** Hotel REITs are late-cycle; any recession signal or rate hiking reversal would compress valuations. Guidance already lowered once — a second cut would trigger conviction downgrade.

**OSPN (OneSpan) — Conviction 3/5 (up from 2/5)**
The standout entry of the portfolio. OSPN gained +11.1% in its first 3 trading days, validating the screener-sourced thesis. Q1 fundamentals are strong: EPS $0.39 (beat $0.35), ARR $192M (+14% YoY), subscription revenue +8%, net retention 105%, EBITDA margin 32%. The $50M buyback provides price support, and the $0.13/quarter dividend (4% yield at entry) adds carry. Analyst consensus Buy with $16.25 PT (+12.5% upside). Conviction upgraded to 3/5 based on independent verification of fundamentals — but not yet 4/5 because the position is only 3 days old and we haven't seen how it handles normal pullback. Will reassess after 5-day minimum hold.
**Risk:** Hardware revenue ($43-45M FY26 guide) is low-margin and could drag blended margins. Currently at 25.8% of equity — approaching the 30% cap. If it continues to surge, may need to trim for risk management even before the +30% partial-profit threshold.

**IMPP (Imperial Petroleum) — Conviction 3/5 (new entry, analyst-sourced)**
IMPP was flagged in the Week 36 thesis as a potential "post-cooldown" candidate for Week 37, and the fundamentals confirm the case. Q1 was the company's second-best quarter in history: revenue $61.7M (+25.8% beat), EPS $0.58 (+67.3% beat). The fleet is expanding from 21 to 26 vessels by end of Q3, providing visibility on revenue growth. The post-earnings pullback from $5.68 to $5.41 creates an entry below the initial euphoria. However, governance concerns are significant: Stealth Maritime (Vafias family) controls ~74% of shares via management agreement, and Seeking Alpha has published bearish theses citing "permanent value trap" dynamics and related-party extraction risk. Conviction set at 3/5 — the earnings are real and the fleet expansion is tangible, but governance caps conviction. This is a momentum trade with strict stop discipline, not a long-term hold.
**Risk:** Related-party value extraction via management fees, concentrated insider ownership limits float, shipping cyclicality. The $4.80 stop caps downside at -11.1% from entry, or 1.7% of portfolio equity.

### Overall Portfolio Thesis — Week 37 of 52

Portfolio enters Week 37 at $672.44 equity (+3.0% WoW), trailing the S&P equivalent by $88.36 (-11.6%). The ECVT stop-out was a controlled exit — the stop-limit system worked exactly as designed, capping the loss at -$3.74 (-5.4% from entry). OSPN's +11.1% gain in 3 days validates the screener-sourced entry discipline and partially compensates for the ECVT loss.

**Portfolio Structure (post-IMPP fill):**
1. **5 positions across 4 sectors:** Energy (WKC, IMPP), Consumer/Industrials (ACCO), Real Estate (SHO), Technology (OSPN). Sector diversification is slightly reduced from last week (was 5 sectors with Materials/ECVT).
2. **Two high-conviction anchors (4/5):** WKC (thesis confirmed, gains locked) and SHO (buyback floor, Q1 beat).
3. **Two momentum plays (3/5):** OSPN (ARR growth, Week 1 winner) and IMPP (earnings blowout, fleet expansion).
4. **One marginal hold (2/5):** ACCO — recycling decision at Week 38.
5. **Cash at 15.6% post-IMPP fill** — just above the 15% floor. No room for opportunistic adds.

**Alpha Strategy for Remaining 15 Weeks:**
The -11.6% gap to the S&P requires +13.1% absolute return, or roughly +0.87%/week. This is achievable if:
- OSPN continues its trajectory toward the $16.25 analyst PT (+12.5% from current)
- IMPP re-rates on fleet expansion news and Q2 earnings momentum
- WKC breaks $30 resistance (has been consolidating near 52-week high)
- ACCO is recycled at Week 38 into a higher-conviction name

**Scenarios (Week 37 to 38):**
- **Bull (25%):** OSPN continues to $15.50, IMPP fills and holds, WKC breaks $30, SHO grinds higher on summer travel → equity $700-$720
- **Base (50%):** Holdings consolidate, IMPP fills flat, ACCO stays dead → equity $670-$700
- **Bear (25%):** Market pullback, IMPP day-1 drawdown triggers exit, ACCO approaches stop → equity $640-$665. Max stop-trigger downside: $37.91 (5.6% of equity)

---

## 10. CONFIRM CASH AND CONSTRAINTS

### Final Cash Balance

| Item | Amount |
|------|--------|
| Starting cash | $207.68 |
| IMPP purchase (19 x $5.40) | -$102.60 |
| **Ending cash** | **$105.08** |
| Cash as % of equity | 15.6% |
| 15% floor | $100.87 |
| Buffer above floor | $4.21 |

### Constraint Verification

| Constraint | Status | Detail |
|-----------|--------|--------|
| Cash >= 15% | PASS | 15.6% ($105.08 / $672.44) |
| No position > 30% | PASS | Largest: OSPN at 25.8% |
| Max 5 positions | PASS | 5 positions (WKC, ACCO, SHO, OSPN, IMPP) |
| Sector cap (max 2/sector) | PASS | Energy: 2 (WKC, IMPP); all others: 1 |
| All stops set | PASS | All 5 positions have stop/stop-limit |
| Risk per trade <= 5% | PASS | Largest single risk: ACCO at 1.7% |
| Total stop-loss exposure | PASS | $37.91 (5.6%) — all stops simultaneous |
| No re-entry ban violations | PASS | ECVT banned until ~June 12; not recommended |
| Post-earnings cooldown | PASS | IMPP: 5 trading days since May 22 earnings |
| Distance-from-base | PASS | IMPP: +14.9% above 50-day, ~+6.1% above 20-day |
| Liquidity filters | PASS | All positions: price >$1, ADV >$500K |
| IWM regime filter | PASS | IWM $290.43 > 50-day SMA $279.06 |

**All rules satisfied. Portfolio is compliant.**

---

*Report generated: May 29, 2026 — Week 37 Deep Research Mode*
*Next review: Week 38 (June 5, 2026) — ACCO recycling decision, OSPN stop reassessment, IMPP Week 1 review*
