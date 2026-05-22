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
<date>Monday, May 18, 2026</date>
<week_number>35 of 52 (twelve-month live experiment)</week_number>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| ECVT   |   14.56 | -2.74% |   1,000,500 | Holding    |
| ORN    |   15.04 | -2.84% |     373,200 | Holding    |
| WKC    |   28.45 | +1.46% |     741,400 | Holding    |
| ACCO   |    3.76 | -4.08% |   1,078,600 | Holding    |
| IWO    |  362.25 | -2.68% |     253,400 | Benchmark  |
| XBI    |  130.69 | -3.08% |   9,436,100 | Benchmark  |
| SPY    |  739.17 | -1.20% |  60,290,000 | Benchmark  |
| IWM    |  277.60 | -2.41% |  35,384,700 | Benchmark  |
| QQQ    |  708.93 | -1.51% |  51,687,900 | Benchmark  |
| TLT    |   83.66 | -1.48% |  50,726,700 | Macro      |
| HYG    |   79.46 | -0.49% |  54,377,000 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    2.3375 |                         |
| Sortino Ratio (annualized)    |    8.0139 |                         |
| Beta (daily) vs ^GSPC         |    2.1053 |                         |
| Alpha (annualized) vs ^GSPC   | +1555.43% |                         |
| R²                            |     0.044 | Low — alpha/beta unstable |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $656.12 |
| S&P Equivalent      |   $743.58 |
| Cash Balance        |   $342.18 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-05-18" candidates="15">
|   rank | ticker   | sector             |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:-------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | SG       | Consumer Cyclical  |           8.09 | $961M        |          16.07 |         20.03 |           2.24 |       15.98 |     0.1896 | True          | HIGH              |            0.8353 |
|      2 | SHO      | Real Estate        |          10.15 | $1.9B        |           4.32 |         -1.17 |           1.88 |        4.23 |     0.1244 | True          | HIGH              |            0.8334 |
|      3 | NAC      | Financial          |          11.82 | $1.9B        |           0.6  |         -1.5  |           1.6  |        0.51 |     0.0268 | False         | HIGH              |            0.8315 |
|      4 | PRA      | Financial          |          24.51 | $1.3B        |          -0.81 |         -0.53 |           1.88 |       -0.9  |     0.0106 | False         | HIGH              |            0.8312 |
|      5 | KW       | Real Estate        |          11.03 | $1.5B        |           1.1  |          0    |           1.49 |        1.01 |     0.0202 | True          | HIGH              |            0.8307 |
|      6 | BCAT     | Financial          |          15.4  | $1.6B        |           4.05 |         -1.22 |           1.48 |        3.96 |     0.0839 | True          | HIGH              |            0.8301 |
|      7 | FBIO     | Healthcare         |           2.41 | $78M         |           3.88 |          7.59 |           2.63 |        3.79 |     0.1556 | True          | HIGH              |            0.8222 |
|      8 | AVAH     | Healthcare         |           7.72 | $1.7B        |          12.37 |         17.15 |           2.27 |       12.28 |     0.1919 | True          | HIGH              |            0.8221 |
|      9 | FBRT     | Real Estate        |           9.04 | $696M        |          -0.99 |          2.84 |           1.7  |       -1.08 |     0.0515 | False         | HIGH              |            0.8157 |
|     10 | WLTH     | Technology         |          11.28 | $1.7B        |           2.73 |         -2.08 |           1.78 |        2.64 |     0.1271 | True          | HIGH              |            0.8148 |
|     11 | VTS      | Energy             |          18.5  | $772M        |           3.82 |          0.43 |           1.3  |        3.73 |     0.0772 | True          | HIGH              |            0.8134 |
|     12 | EHAB     | Healthcare         |          13.8  | $707M        |           0.58 |          0.07 |           1.32 |        0.49 |     0.0074 | True          | MEDIUM            |            0.8125 |
|     13 | CRMD     | Healthcare         |           7.55 | $592M        |          -0.4  |         -1.44 |           2.28 |       -0.49 |     0.1106 | False         | HIGH              |            0.8109 |
|     14 | LPG      | Energy             |          40.87 | $1.8B        |          10.82 |          1.14 |           1.39 |       10.73 |     0.1672 | True          | HIGH              |            0.8041 |
|     15 | RLX      | Consumer Defensive |           2.16 | $2.0B        |           0.93 |          0.47 |           1.35 |        0.84 |     0.073  | False         | HIGH              |            0.8035 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-05-15">
<holding ticker="ECVT" shares="5" avg_cost="13.99" current_price="14.56" stop_loss="13.24" stop_limit="13.14" />
<holding ticker="ORN" shares="5" avg_cost="11.88" current_price="15.04" stop_loss="13.50" stop_limit="13.40" />
<holding ticker="WKC" shares="2" avg_cost="26.00" current_price="28.45" stop_loss="23.40" stop_limit="23.30" />
<holding ticker="ACCO" shares="29" avg_cost="3.92" current_price="3.76" stop_loss="3.53" stop_limit="3.43" />
</holdings>

<last_analyst_thesis>
# Week 34 — Thesis Review Summary

## Per-Position Thesis

**ECVT (Ecovyst) — Conviction 4/5 ↑ from 2/5**
Q1 2026 was a clean beat (EPS $0.11 vs $0.06; revenue $215M vs $194.95M; sales +50% YoY; adj. EBITDA +87%). FY26 guide raised to $890–970M revenue / $180–195M adj. EBITDA. Calabrian acquisition ($190M) on track for Q2 close at ~2x pro forma leverage. Stock $14.14 with $13.24 stop — adequate cushion. Thesis pivots from speculative to confirmed; this is now a growth-by-acquisition compounder, not a deep-value bet.
**Risk:** Sulfur cost pass-through dynamics could compress mix optics in 2H. Stop locks +6% min if triggered.

**ORN (Orion Group) — Conviction 4/5**
JPMorgan Overweight initiation drove +15.9%; Roth MKM Buy reaffirmed; J.E. McAmis acquisition adds Pacific marine capability. Federal infrastructure + data-center build + shipbuilding tailwinds intact. INDOPACOM/PDI MACC large-task-order timing slipped ~1 year — biggest medium-term risk to the thesis. Domestic backlog and concrete segment cushion the slip.
**Risk:** Project timing is lumpy; weather and federal CR risk. Stop $12.35 locks +25% if triggered.

**WKC (World Kinect) — Conviction 3/5 ↓ from 4/5**
Marine segment Q4 gross profit -22% YoY signals the post-conflict bunker-pricing normalization is biting. Morgan Stanley cut PT to $25; analyst stance Hold-biased. Stock at $27.07 trades above mean target. No clean 30–60 day catalyst.
**Risk:** Earnings re-rating could roll over. Flagged for review; will exit on close below $25 or stop trigger at $23.40.

**ACCO (ACCO Brands) — Conviction 4/5**
Q1 confirmed the EPOS-led transformation: revenue $343.7M vs $320.2M, EPS $0.02 vs -$0.05, EPOS contributed $15.2M. FY26 guide reiterated ($0.84–0.89 EPS); peripherals targeting 25% of revenue by year-end. Cost-synergy update next quarter is the next durable catalyst.
**Risk:** Office-products secular decline; FX exposure. Stop $3.53 locks small loss.

**ARDX (Ardelyx) — Conviction 4/5**
Q1 product revenue $93.4M (+38% YoY) with IBSRELA at $70.1M (+58%). FY26 guidance reaffirmed; ACCEL Phase 3 CIC enrollment EOY 2026, topline 2H27. Commercial-stage with two revenue products, not a binary biotech. Stop $6.48 is tight at -4.4%; will widen on confirmed strength.
**Risk:** XPHOZAH reimbursement; market wants profitability. Stop is the primary risk control.

**ARLO (Arlo Technologies) — Conviction 4/5 (new position)**
Q1 2026 inflection: revenue $150M (+26%), non-GAAP EPS $0.28 (+86%), paid-account adds 318k vs 190–230k target. Services now 60% of revenue at 85.4% gross margin growing 31%. $50M buyback authorized. FY26 guide $550–580M. Subscription mix re-rating is the 30–60 day catalyst.
**Risk:** Hardware competition from Ring/Wyze; FY guide implies 2H deceleration; ad-spend cycle. Stop $13.85 caps loss at 1.8% of equity.

## Overall Portfolio Thesis — Week 34 of 52

Portfolio enters Week 34 at ~$676 equity (mark-to-market) with five confirmed-thesis holdings and one fresh add (ARLO). After Week 33's deployment of $216.54 + MRAM stop-out, the structural picture is:

1. **Six positions across six sectors** — maximum diversification of the experiment.
2. **All current holdings post-Q1** — every name has now reported and survived its earnings window, which sharply reduces single-print drawdown risk for the next 4–6 weeks.
3. **30–60 day catalyst posture** — ECVT Calabrian close (Q2), ACCO peripherals mix milestone, ARDX IBSRELA tracking to FY guide, ARLO post-print re-rating. ORN AGM 5/19 and TLS earnings 5/11 are the near-term watch-items but not portfolio-moving.
4. **Cash 18% post-trade** — above 15% reserve, room for one defensive trim if WKC thesis decays further.

**Scenarios (Week 34→35):**
- **Bull (30%):** ARLO re-rates to $17–18, ECVT pre-Calabrian-close pop, ARDX recovers entry → equity $710–730.
- **Base (50%):** Holdings consolidate, ARLO drifts $15–16 → equity $670–690.
- **Bear (20%):** WKC rolls under $25 and we exit at $24, ARDX stops out at $6.48, broader small-cap risk-off → equity $620–640. Maximum aggregate stop-trigger downside is ~$48 (7.1%).

**Path vs. benchmark:** MRAM was the alpha engine. Without it, the new portfolio must rely on breadth rather than a single explosive winner. ARLO + ECVT inflection narratives are the two best candidates to fill that role over Weeks 34–36.
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-05-11,ARLO,8.0,15.25,122.0,0.0,MANUAL BUY LIMIT - Filled,,
2026-05-12,ARLO,,,122.0,-12.400000000000006,MANUAL SELL LIMIT - ,8.0,13.7
2026-05-12,ARDX,,,122.4,-12.239999999999997,MANUAL SELL LIMIT - ,17.0,6.48
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net
- Catalyst timing: Within 10 days
- Risk posture: Neutral
- Max concurrent positions: 5
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