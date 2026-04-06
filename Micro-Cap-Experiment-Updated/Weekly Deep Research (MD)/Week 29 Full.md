# Week 29 Deep Research Report — April 6, 2026

---

## 1. RESTATED RULES

- **Universe:** U.S.-listed common stocks, nano-cap to small-cap (market cap up to $2B). NYSE, NASDAQ, NYSE American only.
- **Execution:** Long-only, full shares only. No options, shorting, leverage, margin, or derivatives.
- **Exclusions:** OTC/pink sheets, ETFs/ETNs/SPACs, warrants/ADRs, bankrupt/halted issuers, defence companies, Israeli-affiliated companies.
- **Risk control:** Stop-losses on all positions. Default: max(1.5xATR(14), 10% below entry). Binary event override permitted with documented rationale.
- **Position sizing:** Risk per trade ≤5% of equity (≤3.75% if SMA waiver used). No single position >30% of equity.
- **No averaging down** below -5% from entry without new catalyst confirmed by ≥2 sources.
- **Partial profit-taking:** ~1/3 at +30%, ~1/3 at +60%, remainder with trailing stop.
- **Pre-catalyst exit orders:** GTC sell for ~1/3 at +30% from entry, placed ≥2 trading days before binary event.
- **Post-catalyst reassessment:** Within 1 trading day of event resolution — remove override, recalculate stop, re-rate conviction.
- **Market regime filter:** If IWM < 50-day SMA, restrict new initiations to high-conviction catalyst-driven plays only.
- **Re-entry ban:** 10 trading days after stop-out.
- **Catalyst requirement:** Confirmed catalyst within 60 days for new initiations.
- **Liquidity filters:** Price ≥$1.00, 3-month ADDV ≥$500K, bid-ask ≤2% (or ≤$0.05 if price <$5), float ≥5M shares.
- **SMA filter:** Price must be above 20-day SMA at initiation, unless SMA waiver conditions met.

**Session directives:**
- Sector focus: Wide net across all sectors
- Catalyst timing: Within 10 trading days
- Risk posture: Aggressive — trailing benchmark by $33.55 with 24 weeks remaining
- Max concurrent positions: 5

---

## 2. RESEARCH SCOPE

**Data retrieval:** April 6, 2026 (Sunday), using April 2 market close as latest trading data.

**Sources consulted:**
- [Yahoo Finance — REPL](https://finance.yahoo.com/quote/REPL/) — Price, volume, fundamentals
- [StockTitan — REPL Q3 Earnings](https://www.stocktitan.net/news/REPL/replimune-reports-fiscal-third-quarter-2026-financial-results-and-uosbbkyue3aj.html) — PDUFA confirmation, cash position
- [Replimune IR — BLA Resubmission](https://ir.replimune.com/news-releases/news-release-details/replimune-announces-fda-acceptance-bla-resubmission-rp1-0) — FDA acceptance
- [MarketBeat — FDA Calendar](https://www.marketbeat.com/fda-calendar/upcoming/) — PDUFA dates
- [Barchart — IWM Technicals](https://www.barchart.com/etfs-funds/quotes/IWM/technical-analysis) — 50-day SMA
- [StockAnalysis — REPL](https://stockanalysis.com/stocks/repl/) — Overview
- [MarketBeat — REPL Short Interest](https://www.marketbeat.com/stocks/NASDAQ/REPL/short-interest/) — Short interest data
- [MerlinTrader — April PDUFAs](https://www.merlintrader.com/repl-tvtx-grce-april2026-pdufa/) — REPL, TVTX, GRCE analysis
- [StockTwits — REPL Rally](https://stocktwits.com/news-articles/markets/equity/repl-stock-rallies-in-anticipation-of-fda-decision-on-skin-cancer-drug/cZ7U7M0RI8w) — April 2 rally coverage
- [Grace Therapeutics IR — GTx-104 NDA](https://www.gracetx.com/investors/news-events/press-releases/detail/291/grace-therapeutics-announces-u-s-food-and-drug-administration-acceptance-for-review-of-new-drug-application-for-gtx-104) — PDUFA April 23
- [Seeking Alpha — VNDA Catalysts](https://seekingalpha.com/article/4856683-vanda-stock-2026-catalyst-run-nereus-launch-bysanti-pdufa-focus) — NEREUS, BYSANTI
- [TipRanks — RCKT PRV](https://www.tipranks.com/news/the-fly/prv-reception-could-net-rocket-100m-200m-says-lifesci-capital-thefly-news) — PRV valuation $100-200M
- [BusinessWire — RCKT Kresladi](https://www.businesswire.com/news/home/20260326279809/en/Rocket-Pharmaceuticals-Announces-FDA-Approval-of-KRESLADI-for-Pediatric-Patients-with-Severe-Leukocyte-Adhesion-Deficiency-I-LAD-I) — FDA approval
- [Investing.com — IWM](https://www.investing.com/etfs/ishares-russell-2000-index-etf-technical) — IWM technicals

**Checks performed:**
- IWM vs 50-day SMA (market regime)
- REPL price, volume, short interest, ATR(14), catalyst confirmation
- GRCE price, market cap, PDUFA date, short interest, liquidity
- VNDA price, re-entry ban status, catalyst timeline, analyst ratings
- RCKT re-entry ban status, PRV monetization status
- S&P equivalent benchmark calculation
- Pre-catalyst sell order deadline verification

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|------------|----------|---------------|--------------|-------------------|--------|
| REPL | Core binary catalyst | 2026-03-30 | $7.43 | $8.41 | $5.90 / $5.80 | 4/5 (upgraded) | HOLD — PDUFA April 10, 4 trading days |

**Portfolio snapshot (as of April 2 close):**

| Component | Value | % of Equity |
|-----------|-------|-------------|
| REPL (10 shares) | $84.10 | 19.6% |
| Cash | $345.90 | 80.4% |
| **Total Equity** | **$430.00** | **100%** |
| S&P Equivalent | $463.55 | — |
| **Benchmark Gap** | **-$33.55 (-7.2%)** | — |

**Market regime:**
- IWM: $251.29 (April 2 close)
- 50-day SMA: ~$256.73 (Barchart)
- **Regime: RISK-OFF** — IWM 2.1% below 50-day SMA
- VIX: 23.87
- Rule applied: New initiations restricted to high-conviction catalyst-driven plays only.

**REPL detailed assessment:**

REPL surged 10.51% on Thursday April 2 to close at $8.41 as the PDUFA date approaches. The stock is now +13.2% from the $7.43 entry. Key metrics:

- **Unrealized P&L:** +$9.80 (+13.2%)
- **Rolling 20-day high:** $8.41 (April 2)
- **ATR(14):** ~$0.54 (implied from expected move data)
- **Normal trailing stop:** max($8.41 x 0.85, $8.41 - 1.5 x $0.54) = max($7.15, $7.60) = **$7.60**
- **Binary event stop override in effect:** PDUFA April 10 is a date-certain binary catalyst. Override maintains stop at $5.90/$5.80, below the 200-day SMA support zone. Override expires when PDUFA resolves.
- **Position risk at override stop:** ($8.41 - $5.90) x 10 = $25.10 = 5.8% of equity
- **Note:** Risk has grown from 3.6% at entry to 5.8% due to price appreciation. The binary event override justifies the wider stop through April 10. Post-catalyst reassessment will normalize the stop.
- **Short interest:** ~11.4% of float shorted, days to cover 9.8 — significant squeeze potential on approval
- **Cash position:** $269.1M as of Dec 2025, funding into late Q1 2027
- **Analyst consensus:** Average target $10-$19, Strong Buy

**Conviction upgrade rationale (3 to 4):** The +10.51% rally on April 2 on 3.19M shares (well above average) confirms institutional accumulation into the PDUFA. The stock's behavior is consistent with informed buying. The BLA resubmission was classified as a "complete response to the CRL" — a positive procedural signal. The clinical data remains strong: 32.9% ORR with 15% CR in anti-PD-1-refractory melanoma, 33.7-month median DOR.

**Pre-catalyst sell order status — ACTION REQUIRED:**
- Per portfolio rules: GTC limit sell for ~1/3 of position at +30% from entry, placed >=2 trading days before event
- Entry $7.43 x 1.30 = $9.659 -> **$9.69 limit**
- Shares: 3 (1/3 of 10, rounded down)
- **Deadline: April 8** (2 trading days before April 10 PDUFA)
- Must be placed Monday April 7 or Tuesday April 8

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |
|--------|----------------|-------------|------------------------------|----------------|
| GRCE | Binary PDUFA for GTx-104 (nimodipine IV) in aSAH — 505(b)(2) reformulation with high approval probability | PDUFA April 23 | Confirmed: [Grace Therapeutics IR](https://www.gracetx.com/investors/news-events/press-releases/detail/291/), [MerlinTrader](https://www.merlintrader.com/repl-tvtx-grce-april2026-pdufa/) | Avg vol ~470K shares, ADDV ~$2.3M, market cap $79M, shares outstanding 15.47M |
| VNDA | Multi-catalyst pharma: NEREUS launched, BYSANTI Q3 launch, earnings May 6 | Earnings May 6, 2026 | Confirmed: [Nasdaq](https://www.nasdaq.com/market-activity/stocks/vnda/earnings), [Seeking Alpha](https://seekingalpha.com/article/4856683-vanda-stock-2026-catalyst-run-nereus-launch-bysanti-pdufa-focus) | Market cap $393M, ADDV >$500K |

**GRCE — Bear case:** GTx-104 is a reformulation of existing nimodipine, not a novel drug. The FDA may request additional data or issue a CRL on manufacturing or labeling grounds. The stock has run 25% in a month and 87% in a year — a lot of good news may be priced in. Rising short interest (496.7% increase to 11.4%) suggests institutional bearishness. $79M market cap means low liquidity and potential for violent moves on adverse news.

**VNDA — Bear case:** CEO recently sold significant shares (March 21). Revenue growth is decelerating. NEREUS and BYSANTI launches face commercial execution risk in competitive markets (antipsychotics, anti-nausea). Trading below March 20 stop-out level ($8.02), suggesting continued weakness. Earnings May 6 is 22 trading days away — outside the 10-day priority window.

**Candidates excluded:**
- **TVTX** (Filspari PDUFA April 13): Market cap $2.5-2.8B — exceeds $2B universe cap.
- **RCKT** (PRV monetization): Re-entry ban active through April 14. Eligible April 15.

---

## 5. PORTFOLIO ACTIONS

- **Keep:** REPL — PDUFA April 10 is 4 trading days away. Binary event stop override in effect. Position is performing (+13.2%). Pre-catalyst sell order required by April 8.

- **No adds to REPL:** Position risk at $5.90 stop is already 5.8% of equity, exceeding the 5% standard budget. The binary event override allows the wider stop but adding shares would further increase risk concentration.

- **No new initiations this week:**
  - **GRCE:** PDUFA April 23 is 13 trading days away — outside the 10-trading-day catalyst priority window. Move to watchlist for Week 30 evaluation.
  - **VNDA:** Earnings May 6 is 22 trading days away — outside priority window. Re-entry ban expires April 7 but SMA needs verification. Move to watchlist for Week 30.
  - **Rationale for cash preservation:** With REPL's PDUFA resolving April 10, the portfolio will undergo a major reshuffling regardless of outcome. Keeping $345.90 (80.4%) in cash preserves maximum flexibility for post-PDUFA deployment. On approval, REPL may offer an add opportunity. On CRL, the stop-out frees the capital for immediate redeployment into GRCE (April 23 PDUFA) or VNDA (earnings May 6).

---

## 6. EXACT ORDERS

**Order 1 — Pre-catalyst exit order (REQUIRED by rules)**

```
Action:                SELL
Ticker:                REPL
Shares:                3
Order Type:            Limit
Limit Price:           $9.69
Time in Force:         GTC
Intended Execution:    Place by 2026-04-08 (fills opportunistically pre/post PDUFA)
Stop Loss:             N/A (exit order)
Stop Limit:            N/A (exit order)
Special Instructions:  Must be placed >=2 trading days before April 10 PDUFA.
                       If filled before event, do NOT replace sold shares.
Rationale:             Pre-catalyst exit per portfolio rules — captures +30%
                       spike-and-reverse profit and reduces gap-through risk by ~47%
                       if PDUFA outcome is negative.
```

**No other orders this week.** Hold REPL with existing stops ($5.90/$5.80) through PDUFA.

---

## 7. RISK AND LIQUIDITY CHECKS

**Position concentration after trades:**

| Holding | Value | % of Equity |
|---------|-------|-------------|
| REPL | $84.10 | 19.6% |
| Cash | $345.90 | 80.4% |
| **Total** | **$430.00** | **100%** |

- REPL at 19.6% — well under 30% concentration cap
- Cash at 80.4% — well above 15% minimum reserve

**If pre-catalyst sell fills (3 shares at $9.69):**

| Holding | Value | % of Equity |
|---------|-------|-------------|
| REPL (7 shares) | $58.87 | 12.9% |
| Cash | $374.97 | 82.1% |
| **Total** | **$456.07** | **100%** |

- Realized gain on 3 sold shares: 3 x ($9.69 - $7.43) = +$6.78
- REPL drops to 12.9% — reduced risk exposure

**Risk scenarios for PDUFA April 10:**

| Scenario | REPL Position Impact | Portfolio Equity | Change |
|----------|---------------------|------------------|--------|
| Approval (price to $12-15) | +$35.70 to +$65.70 (7 shares if pre-sell fills) | $466-$496 | Closes/exceeds benchmark |
| Approval (no pre-sell fill) | +$35.70 to +$65.70 (10 shares) | $466-$496 | Closes/exceeds benchmark |
| CRL, stop holds at $5.90 | -$17.57 (7 shares) or -$25.10 (10 shares) | $405-$413 | Manageable with 24 weeks |
| CRL, gap through stop to $4.00 | -$27.23 (7 shares) or -$44.10 (10 shares) | $386-$403 | Recoverable with cash reserves |

**Liquidity check:**
- Pre-catalyst sell: 3 shares at $9.69 = $29.07 vs REPL avg daily volume 3.19M shares x $8.41 = $26.8M daily dollar volume. Order is <0.001% of ADV.

---

## 8. MONITORING PLAN

### REPL — Critical Week

| Day | Action |
|-----|--------|
| Monday April 7 | Place pre-catalyst GTC sell order: 3 shares at $9.69. Verify order confirmation. Monitor for any FDA advisory committee announcements or early leaks. Check for insider transactions. |
| Tuesday April 8 | **Deadline for pre-catalyst sell order** (if not placed Monday). Monitor price action for unusual volume or directional moves that might signal informed trading. |
| Wednesday April 9 | Final pre-PDUFA day. Watch for any FDA communications, stock halts, or unusual options activity. Confirm stop orders are live. |
| Thursday April 10 | **PDUFA DAY.** Monitor for FDA decision announcement (typically before market open or after close). Execute post-catalyst reassessment within 1 trading day. |
| Friday April 11 | If PDUFA resolved: execute post-catalyst reassessment. Remove binary event stop override. Recalculate trailing stop using normal rules. Re-rate conviction. If not resolved (delay): hold and reassess. |

### Post-PDUFA Reassessment Protocol

**If Approved:**
1. Remove binary event stop override
2. Recalculate trailing stop: max(1.5 x ATR(14), 15% below 20-day rolling high)
3. If pre-catalyst sell filled, hold 7 shares with trailing stop
4. If not filled, cancel GTC sell and manage 10 shares with trailing stop
5. Consider adding shares if price stabilizes above $10 with trailing stop protection
6. Re-rate conviction based on commercial outlook and short squeeze dynamics

**If CRL/Rejection:**
1. Stop should trigger at $5.90 (limit $5.80) — execute sell
2. If gap through stop, sell at market open
3. 10-day re-entry ban activates (through April 24)
4. Immediately evaluate GRCE and VNDA for capital redeployment

**If Delay:**
1. Maintain binary event stop override
2. Reassess timeline — if new PDUFA >30 days, consider exiting to redeploy capital
3. Hold if new date is within 15 trading days

### Watchlist Monitoring

| Ticker | What to Watch | Trigger for Action |
|--------|--------------|-------------------|
| GRCE | Price action, 20-day SMA, PDUFA April 23 confirmation, float verification | If REPL PDUFA resolves and capital frees up, evaluate for Week 30 entry |
| VNDA | Price vs 20-day SMA, NEREUS launch updates, CEO insider selling | Re-entry ban expires April 7. Earnings May 6. Evaluate for Week 30 if above SMA |
| RCKT | PRV monetization announcements, price stabilization above $4 | Re-entry ban expires April 15. PRV reassessment deadline April 28 |

---

## 9. THESIS REVIEW SUMMARY

### Per-Position Thesis

**REPL — Conviction: 4/5 (upgraded from 3, approaching binary resolution)**

The portfolio's sole position enters its defining week. The April 10 PDUFA for RP1+nivolumab in anti-PD-1-refractory advanced melanoma is the highest-stakes event in the experiment's second half. The +10.51% surge on April 2 to $8.41 — on 3.19M shares, well above average — confirms institutional accumulation ahead of the decision. The stock is now +13.2% from the $7.43 entry, validating the SMA waiver entry that was justified by the date-certain binary catalyst.

Conviction is upgraded to 4/5 based on: (1) the BLA resubmission classified as a "complete response to the CRL" — meaning the FDA found the additional data adequate for re-review, (2) the original CRL cited trial design concerns, not efficacy or safety failures, which is more resolvable, (3) the clinical data is genuinely compelling (32.9% ORR, 15% CR, 33.7-month median DOR in a population with no good options), (4) the competitive landscape favors RP1 as a simpler, safer alternative to Iovance's Amtagvi (which carries 7.5% treatment-related mortality), and (5) the short interest at ~11.4% with 9.8 days to cover creates meaningful squeeze potential on approval.

The pre-catalyst GTC sell order (3 shares at $9.69) must be placed by April 8. This is non-negotiable per portfolio rules. If the stock spikes to $9.69+ before or on the PDUFA, this captures $6.78 in profit and reduces the remaining position to 7 shares — cutting gap-through risk by ~47%. If the PDUFA is positive and the stock runs past $9.69, the remaining 7 shares capture the upside with a trailing stop.

The $5.90/$5.80 binary event stop override is retained through April 10. Post-catalyst reassessment will occur within 1 trading day of the decision. If approved, normal trailing stop rules resume. If rejected, the stop executes and a 10-day re-entry ban activates.

### Overall Portfolio Thesis — The Decision Week (Week 29 of 52)

The portfolio enters Week 29 at $430.00, trailing the S&P benchmark by $33.55 (-7.2%) with 24 weeks remaining. This week is binary: the REPL PDUFA on April 10 will either narrow the gap dramatically or widen it modestly.

**Bull case (PDUFA approval):** REPL runs to $12-15+ on short squeeze dynamics. Even with the pre-catalyst sell reducing the position to 7 shares, the portfolio gains $35-66, pushing equity to $466-496 and closing or exceeding the $463.55 benchmark. The remaining 7 shares run with a trailing stop. Cash reserves of $375+ enable immediate deployment into GRCE (April 23 PDUFA) or VNDA (earnings May 6), creating a multi-catalyst portfolio heading into May.

**Bear case (CRL):** REPL stop triggers at $5.90, losing $25.10 on 10 shares (or $17.57 on 7 if pre-sell fills). Portfolio drops to $405-413. This is a setback but not catastrophic — 80%+ cash reserves and 24 weeks of runway provide ample recovery time. Capital immediately redeploys into GRCE and/or VNDA.

**Expected value:** Even at 50/50 approval odds (likely conservative given the "complete response" classification), the expected value is positive. Upside of +$35-66 vs downside of -$17-25 yields a favorable 2:1 to 3:1 risk/reward ratio on the position.

The 80.4% cash position is intentionally elevated. This is not conservative — it is tactical. The cash serves as: (1) a buffer against the REPL binary outcome, (2) dry powder for immediate post-PDUFA redeployment, and (3) reserves for the GRCE and VNDA opportunities in Weeks 30-31. The aggressive posture demanded by the session directives is expressed through the REPL position's asymmetric risk/reward, not through deploying more capital into a market that is RISK-OFF (IWM below 50-day SMA) with the VIX at 24.

**Week 30 deployment roadmap (post-PDUFA):**
- **GRCE (April 23 PDUFA):** Prime candidate for Week 30 initiation. 505(b)(2) reformulation has higher base approval rate than novel drugs. $79M market cap creates significant upside leverage. Must verify float >=5M and 20-day SMA filter.
- **VNDA (earnings May 6):** Multi-product revenue growth story. Re-entry ban expires April 7. Analyst target $14.9 implies 113% upside from $7.00. Must verify 20-day SMA filter.
- **RCKT (PRV monetization):** Re-entry ban expires April 15. PRV worth $100-200M. Evaluate if price stabilizes above $4.00.

The path to closing the benchmark gap: one positive PDUFA outcome plus one successful deployment of the $345 cash reserve into a catalyst winner in Weeks 30-31 gets the portfolio back to parity or ahead of the S&P.

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash balance:** $345.90

**Post-trade cash (if pre-catalyst sell fills):** $345.90 + (3 x $9.69) = $374.97

**Constraint verification:**

| Rule | Status |
|------|--------|
| All positions have stop-losses | REPL: $5.90/$5.80 (binary event override) |
| No position >30% of equity | REPL at 19.6% |
| Cash >=15% reserve | Cash at 80.4% |
| Position sized within risk budget | Entry risk was 3.6% (within 3.75% SMA waiver budget). Current risk at 5.8% is elevated due to price appreciation but justified by binary event override |
| Market regime noted | RISK-OFF — IWM $251.29 below 50-day SMA ~$256.73 |
| Pre-catalyst exit order planned | 3 shares at $9.69 GTC, place by April 8 |
| No re-entry ban violations | VNDA ban expires April 7, RCKT ban expires April 15 — no re-entries proposed |
| Concurrent positions <=5 | 1 position |
| All tickers verified on allowed exchanges | REPL on NASDAQ |

**End of Week 29 Report.**
