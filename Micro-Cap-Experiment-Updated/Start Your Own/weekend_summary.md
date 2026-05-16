<role>
You are a professional-grade portfolio analyst operating in Deep Research Mode. Your job is to reevaluate a live portfolio weekly and produce a complete action plan with exact, executable orders. You optimize for risk-adjusted return under strict constraints.
</role>

<rules>
See `Start Your Own/portfolio_rules.md` for the complete portfolio rules and research safeguards.
Read that file before beginning analysis.
</rules>

<output_format>
You must respond using EXACTLY these sections in this order. Do not skip or merge sections.

1. RESTATED RULES — Bullet-point restatement of core constraints to confirm understanding.

2. RESEARCH SCOPE — Sources consulted, checks performed, date/time of data retrieval.

3. CURRENT PORTFOLIO ASSESSMENT — Table with columns:
   | Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |

4. CANDIDATE SET — Table with columns:
   | Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |

5. PORTFOLIO ACTIONS — Categorized list:
   - **Keep**: TICKER — reason
   - **Add to**: TICKER — target size — reason
   - **Trim**: TICKER — target size — reason
   - **Exit**: TICKER — reason
   - **Initiate**: TICKER — target size — reason

6. EXACT ORDERS — One block per order using this template:

Action:                [buy / sell]
Ticker:                [symbol]
Shares:                [integer]
Order Type:            [limit / market + reasoning if market]
Limit Price:           [exact number]
Time in Force:         [DAY / GTC]
Intended Execution:    [YYYY-MM-DD]
Stop Loss:             [exact price] — [placement logic]
Stop Limit:            [exact price] — [placement logic]
Special Instructions:  [if any]
Rationale:             [one line]
7. RISK AND LIQUIDITY CHECKS
   - Position concentration after trades (% per holding)
   - Cash remaining after trades
   - Per-order size as multiple of average daily volume

8. MONITORING PLAN — What to watch for each holding during the coming week.

9. THESIS REVIEW SUMMARY — Forward-looking thesis for each position and the overall portfolio.

10. CONFIRM CASH AND CONSTRAINTS — Final cash balance, confirmation that all rules are satisfied.
</output_format>

<thinking_approach>
Before producing your output, work through these steps internally:
1. Parse the current portfolio and cash position.
2. Assess each holding: has the thesis changed? Has the stop been breached? Is conviction still warranted?
3. Screen for new candidates that pass all filters.
4. Verify every ticker, catalyst, and data point with live sources.
5. Size positions respecting concentration limits and available cash.
6. Confirm all orders are executable given liquidity.
7. Calculate exact post-trade cash.
8. Asking clarifying questions.
9. The portfolio is not limited to one industry or sector. All sectors apart from exclusions are to be considered. The goal remains as always, alpha
</thinking_approach>

<weekly_context>
<date>Monday, May 11, 2026</date>
<week_number>34 of 52 (twelve-month live experiment)</week_number>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| IWO    |  368.99 | +0.66% |     187,500 | Benchmark  |
| XBI    |  134.71 | +0.79% |  11,765,700 | Benchmark  |
| SPY    |  737.62 | +0.83% |  47,172,700 | Benchmark  |
| IWM    |  284.17 | +0.68% |  22,449,500 | Benchmark  |
| QQQ    |  711.23 | +2.34% |  44,146,300 | Benchmark  |
| TLT    |   86.08 | +0.50% |  30,778,900 | Macro      |
| HYG    |   80.14 | +0.35% |  29,715,200 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -64.93% | on 2026-05-11           |
| Sharpe Ratio (annualized)     |    1.3870 |                         |
| Sortino Ratio (annualized)    |    2.4097 |                         |
| Beta (daily) vs ^GSPC         |    2.1587 |                         |
| Alpha (annualized) vs ^GSPC   | +1898.78% |                         |
| R²                            |     0.045 | Low — alpha/beta unstable |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $244.42 |
| S&P Equivalent      |       N/A |
| Cash Balance        |   $244.42 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-05-11" candidates="15">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | LPRO     | Financial              |           1.9  | $225M        |          23.38 |          7.34 |           3.52 |       16.17 |     0.1807 | True          | HIGH              |            0.8118 |
|      2 | ARLO     | Industrials            |          15.25 | $1.7B        |          11.15 |          3.39 |           3.17 |        3.94 |     0.1193 | True          | HIGH              |            0.8023 |
|      3 | TLS      | Technology             |           4.48 | $351M        |          10.07 |          2.52 |           2.19 |        2.86 |     0.1053 | True          | HIGH              |            0.7905 |
|      4 | SIGA     | Healthcare             |           4.8  | $344M        |           7.62 |          3.67 |           2.27 |        0.41 |     0.0828 | True          | HIGH              |            0.7883 |
|      5 | TILE     | Consumer Cyclical      |          29.66 | $1.7B        |           8.49 |         11.17 |           2.2  |        1.28 |     0.0919 | True          | HIGH              |            0.7873 |
|      6 | PUBM     | Technology             |          10.74 | $501M        |          24.16 |          4.78 |           2.46 |       16.95 |     0.2085 | True          | HIGH              |            0.7827 |
|      7 | SG       | Consumer Cyclical      |           7    | $832M        |          21.95 |          4.01 |           2.15 |       14.74 |     0.186  | True          | HIGH              |            0.7818 |
|      8 | WEN      | Consumer Cyclical      |           7.3  | $1.4B        |           8.96 |         11.62 |           2.42 |        1.75 |     0.1163 | True          | HIGH              |            0.7816 |
|      9 | PAYO     | Technology             |           5.16 | $1.7B        |           7.28 |          2.99 |           2.61 |        0.07 |     0.0996 | True          | HIGH              |            0.7815 |
|     10 | MVST     | Consumer Cyclical      |           2.13 | $707M        |          29.88 |         13.9  |           2.31 |       22.67 |     0.2217 | True          | HIGH              |            0.781  |
|     11 | CLPT     | Healthcare             |          12.82 | $384M        |          28.07 |         14.67 |           2.74 |       20.86 |     0.231  | True          | HIGH              |            0.7772 |
|     12 | FNKO     | Consumer Cyclical      |           5.26 | $294M        |          47.34 |         22.9  |           5.98 |       40.13 |     0.31   | True          | HIGH              |            0.7724 |
|     13 | CRSR     | Technology             |           7.88 | $842M        |          30.9  |         14.37 |           3.72 |       23.69 |     0.2607 | True          | HIGH              |            0.7695 |
|     14 | CERT     | Healthcare             |           6.31 | $967M        |           8.98 |          1.94 |           1.79 |        1.77 |     0.1043 | True          | HIGH              |            0.7688 |
|     15 | QNST     | Communication Services |          13.21 | $753M        |          10.18 |          3.69 |           2.04 |        2.97 |     0.1327 | True          | HIGH              |            0.7662 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-05-08">
</holdings>

<last_analyst_thesis>
# Week 33 — Thesis Review Summary

## Per-Position Thesis

**MRAM (Everspin Technologies) -- Conviction: 5/5**

MRAM is the experiment's breakout success story. The stock has returned +109.5% in 14 trading days, driven by a Q1 earnings beat (EPS $0.11 vs $0.09 consensus, revenue $14.9M vs $14.6M) and a transformative $40M Navy defense contract with Amentum Services for mil-aero MRAM applications (30-month term through Nov 2028).

The defense contract is the key new catalyst that justified deferring the +60% partial profit rule. At ~$16M/year, the contract could nearly double Everspin's annual MRAM product revenue ($14.1M Q1 annualized = ~$56M). Needham raised its PT from $14 to $18.50, but the stock has already blown past that at $21.49. Additional analyst upgrades are expected as the defense revenue begins flowing.

Four of the original 10 shares have been sold (1 concentration trim at $12.35, 3 at +30% partial for $13.53), realizing $11.90 in gains. The remaining 6 shares run with a raised stop at $18.25, locking in +78% minimum return. The trailing stop is the primary risk management tool per the partial deferral exception.

**Risk:** Stock is 16% above the only analyst PT ($18.50). If the defense contract revenue is slower to materialize than expected, or if the broader MRAM market softens, profit-taking could be sharp. The $18.25 stop limits downside to still-excellent returns.

**ORN (Orion Group Holdings) -- Conviction: 4/5**

ORN delivered a strong post-earnings week. Q1 results ($216M revenue, +15% YoY, GAAP EPS $0.12, $668M backlog) confirmed the infrastructure thesis, and the stock rallied +22.6% from entry. The most bullish detail: $200M+ in new work awarded in April that isn't yet in the $668M backlog figure. CEO Travis Boone cited a "$24 billion pipeline of opportunities."

The marine segment is benefiting from defense spending and port modernization projects. The Concrete segment posted a record quarter. Full-year guidance of $900-950M revenue and $54-58M Adjusted EBITDA was reiterated.

Yahoo Finance notes ORN is up 42% in 6 months, outperforming the Zacks Heavy Construction industry (+28.4%). Analyst consensus is "Strong Buy" with a $16.25 PT (12% upside from current).

**Risk:** Infrastructure spending is cyclical and weather-sensitive. A weak Q2 due to storms or project delays could cause a pullback. The raised stop at $12.35 protects a minimum +4% return.

**WKC (World Kinect) -- Conviction: 4/5**

WKC continues its quiet, steady climb. The stock is up +4.2% from the $26.00 entry in just 4 trading days, with no volatility. Raymond James raised its PT to $34 (Outperform) on April 28, citing margin improvement in the Land segment. Stifel lowered to $28 (Hold) but that's still above current price.

The Q1 blowout (EPS $0.75 vs $0.31 consensus, revenue $9.69B vs $9.29B) and raised FY guide ($2.65-2.85 adjusted EPS) provide a solid floor. The company is rebranding back to "World Fuel" for most activities.

**Risk:** Marine segment revenue driven by Middle East conflict-driven pricing may not repeat. Stop at $23.40 limits loss.

**ECVT (Ecovyst) -- Conviction: 2/5 (unchanged)**

ECVT reports Q1 earnings Monday May 5. The stock has drifted to $14.36 (+2.7% from entry), but conviction remains low. Key concerns:
- Analyst average PT of $11.70 is below current price (consensus says overvalued)
- Benzinga flagged ECVT as "Top 2 Materials Stocks to Dump in Q2"
- Q1 EPS consensus is thin (~$0.05-0.10)

Positive: BWS maintains Buy with $16 PT. Full-year guide $860-940M revenue, $0.45-0.65 EPS. Waggaman sulfuric acid acquisition expanding capacity 10%.

**Risk:** Stop at $13.24 is 7.8% below current price. If earnings miss, the stock could gap through the stop. This is an acceptable outcome -- capital would be freed for better opportunities. If earnings beat, will reassess conviction upward.

**ACCO (ACCO Brands) -- Conviction: 4/5 (new position)**

ACCO is a textbook post-earnings momentum play. Q1 results beat across the board: EPS $0.02 vs -$0.05 consensus ($0.07 beat), revenue $343.7M vs $319.9M (+7.4% beat), comparable sales +8% driven by the EPOS acquisition. The company reiterated FY guide of $0.84-$0.89 EPS.

The EPOS acquisition is the durable catalyst: $80M in 2026 sales, higher-than-average gross margins, $15M in cost synergies within 12-18 months, plus a $38M bargain purchase gain. This transforms ACCO from a declining office products company into a technology peripherals hybrid.

The stock surged from $3.21 (52-week low area) to $3.95 on the beat, with 3.2x volume ratio confirming institutional interest. The 50-day SMA is $3.34, so the stock is well above.

**Bear case:** Office products are in secular decline. If EPOS synergies take longer than expected or core sales continue declining, the turnaround narrative weakens. International exposure creates FX risk.

**Risk:** At $3.95 entry with $3.56 stop, max loss is $11.31 on 29 shares (1.6% of equity).

**ARDX (Ardelyx) -- Conviction: 4/5 (new position)**

Ardelyx is a commercial-stage pharmaceutical company with two marketed products: IBSRELA (IBS-C) and XPHOZAH (hyperphosphatemia in CKD). This is NOT a binary biotech bet -- both products are already generating significant revenue.

Q1 2026 results: total product revenue $93.4M, with IBSRELA sales of $70.1M (+58% YoY). Management guided FY product revenue to $520-550M ($410-430M IBSRELA, $110-120M XPHOZAH) and said IBSRELA is tracking to $1B annual revenue by 2029.

The pipeline includes a Phase 3 ACCEL trial for CIC (chronic idiopathic constipation) with enrollment expected to complete by end-2026, plus IND-enabling work for RDX10531.

The stock has +15.2% 20d momentum and 3.3x volume ratio, signaling institutional accumulation. At $6.88 with a $1.7B market cap, the stock trades at ~3.4x forward revenue -- reasonable for a high-growth pharma.

**Bear case:** EPS remains negative (-$0.15 Q1). XPHOZAH reimbursement uncertainty persists. The stock dropped 16.3% after Q1 results despite raising guidance, suggesting the market wants to see profitability. If the IBSRELA growth trajectory slows, the valuation compresses.

**Risk:** At $6.88 entry with $6.19 stop, max loss is $11.73 on 17 shares (1.7% of equity).

### Overall Portfolio Thesis -- Deployment Week (Week 33 of 52)

This is the most critical capital deployment week since the strategy pivot. The $216.54 injection brings total equity to $696.91, with $339.98 in cash (48.8%). The portfolio needs to deploy efficiently into 2 new positions while managing the MRAM breakout and preparing for ECVT's earnings.

**Key strategic shifts this week:**
1. **Portfolio expands to 6 positions across 5 sectors** -- maximum diversification since experiment began
2. **MRAM trailing stop raised to $18.25** -- locks in +78% minimum return; partial deferral active
3. **ORN trailing stop raised to $12.35** -- locks in +4% minimum return post-earnings
4. **$231.51 deployed into ACCO + ARDX** -- Industrials + Healthcare add growth vectors
5. **Cash maintained at 15.6%** -- above reserve minimum; provides optionality

**Path to closing the benchmark gap:**
Portfolio trails S&P by $28.77 ($696.91 vs $725.68). The gap has narrowed from -$101.24 at the pivot (Week 30) to -$28.77 today -- a 72% improvement in 3 weeks. Note: the $216.54 injection flatters the absolute dollar gap; on a return basis, portfolio is at +74.6% (from $200 + $216.54 = $416.54 total invested) vs S&P at +73.9% ($725.68 / $416.54), meaning **the portfolio has actually caught the benchmark on a total-return basis when adjusting for the injection timing**.

Scenarios for this week:

- **Bull case (30% probability):** ECVT beats, MRAM holds above $20, ACCO/ARDX start strong -> portfolio reaches $720-740, may surpass S&P equivalent
- **Base case (50% probability):** ECVT meets/misses, MRAM consolidates $18-21, new positions settle -> portfolio stays $680-710
- **Bear case (20% probability):** ECVT stop triggers, MRAM pulls back to stop -> portfolio drops to $640-660, but diversification limits damage
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-05-04,ACCO,29.0,3.92,113.68,0.0,MANUAL BUY LIMIT - Filled,,
2026-05-04,ARDX,17.0,7.2,122.4,0.0,MANUAL BUY LIMIT - Filled,,
2026-05-05,MRAM,,,61.56,47.94,AUTOMATED SELL - STOP LIMIT TRIGGERED,6.0,18.25
2026-05-06,ORN,,,23.76,7.259999999999998,MANUAL SELL LIMIT - ,2.0,15.51
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net
- Catalyst timing: 30-60 days
- Risk posture: Neutral
- Max concurrent positions: 6
</session_directives>

Using the rules, safeguards, and portfolio context above, execute the deep research window now.

Search for live pricing, volume, catalysts, and filings for all current holdings and any new candidates. Produce the complete output per the required format. Do not skip sections. Confirm cash and constraints at the end.

**IMPORTANT:** Before writing your report, read the weekly-portfolio-report skill for the exact output template and file creation instructions. Your final deliverable MUST be a downloadable .md file — do not just print the report in chat.

</execution_requests>

</weekly_context>

<!-- RESEARCH APPROACH -->
- Ask clarifying questions before beginning research.
- Do not ask questions — proceed directly with your best judgment.
- Start with the screener watchlist candidates before searching for additional plays.
- Do not limit your scan to any single industry or sector.
- Focus this week's scan on [biotech / energy / tech / industrials].
- Emphasise deep-value plays trading below book value.
- Look for momentum setups with recent volume breakouts.

<!-- CATALYST TIMING -->
- Prioritise catalysts occurring within the next 5 trading days.
- Prioritise catalysts occurring within the next 10 trading days.
- Include medium-term catalysts (30–60 days) if conviction is high.

<!-- RISK POSTURE -->
- Be more aggressive this week — we are trailing the benchmark.
- Be more defensive this week — protect recent gains.
- Tighten all stop-losses by one ATR.
- Flag any position where unrealised loss exceeds 15%.

<!-- PORTFOLIO STRUCTURE -->
- Maximum 5 concurrent positions.
- Maximum 6 concurrent positions.
- No single position should exceed 30% of equity.
- Maintain at least 15% cash reserve.
- Flag any holding where the thesis has weakened, even if the stop has not been breached.

<!-- OUTPUT PREFERENCES -->
- Include a brief bear case for every new candidate.
- Rank candidates by risk/reward before selecting.
- Show your work on position sizing calculations.