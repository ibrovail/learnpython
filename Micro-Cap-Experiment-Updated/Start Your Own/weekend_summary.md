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
<date>Sunday, June 14, 2026</date>
<week_number>39 of 52 (twelve-month live experiment)</week_number>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| WKC    |   31.72 | +0.89% |     551,400 | Holding    |
| SHO    |   11.72 | +0.17% |   1,370,600 | Holding    |
| IMPP   |    5.25 | -0.57% |     352,400 | Holding    |
| TDAY   |    7.74 | +0.91% |   1,145,400 | Holding    |
| INN    |    6.20 | -1.74% |   1,830,900 | Holding    |
| IWO    |  380.39 | +0.66% |     274,900 | Benchmark  |
| XBI    |  133.79 | +0.79% |   7,822,900 | Benchmark  |
| SPY    |  741.75 | +0.54% |  56,939,800 | Benchmark  |
| IWM    |  292.95 | +0.87% |  34,388,500 | Benchmark  |
| QQQ    |  721.34 | +0.59% |  51,168,400 | Benchmark  |
| TLT    |   85.77 | -0.24% |  23,092,600 | Macro      |
| HYG    |   79.94 | +0.00% |  29,868,800 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    2.2643 |                         |
| Sortino Ratio (annualized)    |    7.7530 |                         |
| Beta (daily) vs ^GSPC         |    1.8942 |                         |
| Alpha (annualized) vs ^GSPC   | +1247.16% |                         |
| R²                            |     0.041 | Low — alpha/beta unstable |
| Time-Weighted Return (cum)    |    +9.37% | injection-neutral       |
| S&P 500 Return (cum)          |   +12.06% | same window             |
| TWR Alpha (cum)               |    -2.68% | TWR minus S&P           |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $691.54 |
| S&P Equivalent      |   $745.89 |
| Cash Balance        |   $139.19 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-06-14" candidates="15">
|   rank | ticker   | sector            |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | GT       | Consumer Cyclical |           6.4  | $1.8B        |          13.48 |         10.34 |           1.61 |        7.95 |     0.1478 | True          | HIGH              |            0.8371 |
|      2 | TILE     | Consumer Cyclical |          31.87 | $1.9B        |          13.62 |          4.49 |           1.65 |        8.09 |     0.1528 | True          | HIGH              |            0.835  |
|      3 | ADAM     | Real Estate       |           9.31 | $837M        |           6.77 |          4.49 |           1.69 |        1.24 |     0.0746 | True          | HIGH              |            0.8292 |
|      4 | JBI      | Industrials       |           5.35 | $730M        |          11.23 |          8.3  |           1.23 |        5.7  |     0.1445 | True          | HIGH              |            0.8099 |
|      5 | BCAT     | Financial         |          16.05 | $1.7B        |           4.22 |          1.78 |           1.52 |       -1.31 |     0.0613 | True          | HIGH              |            0.8087 |
|      6 | CBZ      | Industrials       |          35.26 | $1.9B        |          21.92 |          5    |           1.09 |       16.39 |     0.1875 | True          | HIGH              |            0.8076 |
|      7 | AMLX     | Healthcare        |          14.59 | $1.6B        |           9.82 |         12.14 |           1.38 |        4.29 |     0.1521 | True          | HIGH              |            0.8069 |
|      8 | VYX      | Technology        |           7.85 | $1.1B        |          17.87 |         13.28 |           1.73 |       12.34 |     0.2229 | True          | HIGH              |            0.8051 |
|      9 | PDM      | Real Estate       |           9.1  | $1.1B        |          16.67 |          2.02 |           1.53 |       11.14 |     0.2121 | True          | HIGH              |            0.8038 |
|     10 | AIP      | Technology        |          41    | $1.9B        |          21.63 |         16.28 |           1.22 |       16.1  |     0.214  | True          | HIGH              |            0.8015 |
|     11 | ACCO     | Industrials       |           4.04 | $373M        |           7.45 |          3.86 |           1.15 |        1.92 |     0.1011 | True          | HIGH              |            0.7997 |
|     12 | EVTC     | Technology        |          26.28 | $1.6B        |          11.21 |         16.28 |           1.33 |        5.68 |     0.1677 | True          | HIGH              |            0.7991 |
|     13 | ARHS     | Consumer Cyclical |           7.17 | $1.0B        |          24.7  |         15.27 |           1.31 |       19.17 |     0.2447 | True          | HIGH              |            0.7965 |
|     14 | MLKN     | Consumer Cyclical |          16.24 | $1.1B        |          12.31 |         11.85 |           1.24 |        6.78 |     0.1776 | True          | HIGH              |            0.7934 |
|     15 | INN      | Real Estate       |           6.2  | $752M        |          18.32 |          3.68 |           1.06 |       12.79 |     0.1935 | True          | HIGH              |            0.7887 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-06-12">
<holding ticker="WKC" shares="2" avg_cost="26.00" current_price="31.72" stop_loss="28.80" stop_limit="28.65" />
<holding ticker="SHO" shares="11" avg_cost="10.20" current_price="11.72" stop_loss="10.60" stop_limit="10.50" />
<holding ticker="IMPP" shares="19" avg_cost="5.20" current_price="5.25" stop_loss="4.80" stop_limit="4.70" />
<holding ticker="TDAY" shares="16" avg_cost="8.15" current_price="7.74" stop_loss="7.20" stop_limit="7.10" />
<holding ticker="INN" shares="22" avg_cost="5.75" current_price="6.20" stop_loss="5.15" stop_limit="5.05" />
</holdings>

<last_analyst_thesis>
# Week 38 — Thesis Review Summary

**Date:** 2026-06-07 | **Week:** 38 of 52

---

## Per-Position Thesis

**WKC (World Kinect) — HOLD | Conviction 4/5**
Energy distribution and services company riding strong fuel demand and operational efficiency. Q1 EPS beat by 142% ($0.75 vs $0.31 est). FY26 guidance $2.65–$2.85 implies P/E of ~10.5-11.3× at current price — still reasonable for the growth trajectory. At 52-week high territory. Key risk: analyst consensus diverging (Hold/$28.75 PT) suggests limited upside from here. Raising stop to $28.80 locks in +10.8% gain while allowing room for continued appreciation. **Thesis intact but approaching valuation ceiling.**

**SHO (Sunstone Hotel Investors) — HOLD | Conviction 4/5**
Premium upscale hotel REIT benefiting from RevPAR recovery and strategic portfolio repositioning. Q1 exceeded expectations by 200bp. Raised FY26 outlook. Andaz Miami Beach renovation delivering results. Active share repurchase program. FIFA World Cup 2026 provides potential tailwind for hotel REITs broadly. Trading above analyst consensus PT ($10), suggesting either re-rating or that analysts need to raise targets. Raising stop to $10.60 to lock in profit after strong run. **Thesis strengthening — operating fundamentals improving faster than expected.**

**IMPP (Imperial Petroleum) — HOLD WITH TIGHT STOP | Conviction 2/5 ↓**
Entered as a tanker rate catalyst play following Strait of Hormuz closure. Q1 delivered spectacularly (revenue $61.7M, net income $28M, second-best quarter ever). However, the stock has declined from $5.58 (May 27) to $4.97 despite the beat, suggesting the market views the rate environment as peaking or transitory. The thesis was rate-driven upside — if the market is pricing in rate normalization, the thesis is weakening. Only 3.4% from stop ($4.80). Let the stop manage risk — max loss $3.23 (0.5% of equity). **Thesis weakening — post-earnings price action signals market skepticism.**

**TDAY (USA TODAY Co.) — INITIATING | Conviction 3/5**
Digital transformation play entering Communication Services as a new sector. The AI licensing revenue surge (+125.6% to $18.8M from Meta/Microsoft deals) represents a new, high-margin revenue stream that could fundamentally change the company's trajectory. Digital revenue at an all-time high 47.8% of total, digital subscriptions growing 6.2% with record ARPU. While total revenue still declining (-4% YoY), the rate of decline is improving and management says they're "nearing" the inflection point. If AI licensing revenue continues growing and digital crosses 50% of revenue, the stock could re-rate from a legacy media company to a digital/AI content platform. **Thesis: AI monetization creates new growth narrative that the market hasn't fully priced.**

**INN (Summit Hotel Properties) — INITIATING | Conviction 3/5**
Hotel REIT positioned for FIFA World Cup 2026 tourism catalyst (June 11 start). Q1 RevPAR beat expectations by 200bp with March RevPAR accelerating to +4%. FY26 AFFO guide 75–85c represents 7-10% yield at current price. Conservative balance sheet (no debt maturities until 2028, $34.8M cash), active buyback ($6M), and strong dividend (7.46%). Volume ratio at 5.13× suggests institutional accumulation ahead of World Cup catalyst. **Thesis: FIFA World Cup creates a 6-week demand surge that lifts RevPAR and sentiment for hotel REITs.**

---

## Overall Portfolio Thesis

The portfolio enters Week 38 with a clear strategic imperative: **close the 8.8% gap to the S&P 500 benchmark over the remaining 14 weeks.** The Week 38 restructuring deploys excess cash (reducing from 42% to 21%) into two diversified plays — TDAY (AI/digital transformation) and INN (FIFA World Cup tourism) — while recycling the underperforming ACCO position. The portfolio now spans three sectors (Energy, Real Estate, Communication Services) with five positions, maximum diversification within constraints.

Key portfolio characteristics post-restructuring:
- **Fully deployed:** 5 of 5 slots filled, 79.5% invested vs 57.9% previously
- **Diversified:** 3 GICS sectors, no sector exceeding 2 positions
- **Risk-managed:** 6.4% aggregate stop-risk, each position ≤2.3% individual risk
- **Catalyst-loaded:** INN (FIFA World Cup June 11), TDAY (AI monetization ongoing), WKC/SHO (trailing momentum)
- **Weak links identified:** IMPP (2/5 conviction, 3.4% from stop) — will likely self-resolve via stop within 1-2 weeks

---

*Week 38 Summary generated 2026-06-07 by Claude Code*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-06-08,ACCO,,,113.68,-2.029999999999987,MANUAL SELL LIMIT - ,29.0,3.85
2026-06-08,TDAY,16.0,8.15,130.4,0.0,MANUAL BUY LIMIT - Filled,,
2026-06-08,INN,22.0,5.75,126.5,0.0,MANUAL BUY LIMIT - Filled (pre-market),,
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net across all sectors
- Catalyst timing: Within 10 trading days
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