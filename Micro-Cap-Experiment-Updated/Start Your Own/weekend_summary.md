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
<date>Sunday, April 12, 2026</date>
<week_number>30 of 52 (twelve-month live experiment)</week_number>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| IWO    |  331.87 | -0.26% |     248,300 | Benchmark  |
| XBI    |  129.44 | -1.81% |  10,033,200 | Benchmark  |
| SPY    |  679.46 | -0.07% |  41,916,700 | Benchmark  |
| IWM    |  261.30 | -0.25% |  22,329,700 | Benchmark  |
| QQQ    |  611.07 | +0.14% |  33,831,300 | Benchmark  |
| TLT    |   86.49 | -0.24% |  13,237,300 | Macro      |
| HYG    |   79.96 | -0.40% |  33,005,600 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    1.9667 |                         |
| Sortino Ratio (annualized)    |    5.8291 |                         |
| Beta (daily) vs ^GSPC         |    2.1379 |                         |
| Alpha (annualized) vs ^GSPC   | +1016.19% |                         |
| R²                            |     0.054 | Low — alpha/beta unstable |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $378.80 |
| S&P Equivalent      |   $480.04 |
| Cash Balance        |   $378.80 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-04-12" candidates="15">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | RBBN     | Technology             |           2.47 | $434M        |          17.62 |         10.27 |           4.07 |       11.65 |     0.1519 | True          | HIGH              |            0.8663 |
|      2 | SHO      | Real Estate            |           9.55 | $1.8B        |           6.94 |          5.76 |           1.47 |        0.97 |     0.0749 | True          | HIGH              |            0.8497 |
|      3 | UMH      | Real Estate            |          15.65 | $1.3B        |           6.83 |          6.9  |           2.11 |        0.86 |     0.1021 | True          | HIGH              |            0.8438 |
|      4 | PDM      | Real Estate            |           7.12 | $890M        |           7.39 |          7.72 |           1.24 |        1.42 |     0.0996 | True          | HIGH              |            0.8239 |
|      5 | XHR      | Real Estate            |          15.86 | $1.5B        |          12.01 |          7.74 |           0.96 |        6.04 |     0.1097 | True          | HIGH              |            0.8162 |
|      6 | TV       | Communication Services |           2.98 | $1.3B        |           4.56 |          2.05 |           1.16 |       -1.41 |     0.0567 | True          | HIGH              |            0.8145 |
|      7 | DHY      | Financial              |           1.88 | $195M        |           1.08 |         -0.53 |           3.75 |       -4.89 |     0.044  | True          | HIGH              |            0.813  |
|      8 | HTBK     | Financial              |          13.37 | $824M        |          10.22 |          4.45 |           1.21 |        4.25 |     0.136  | True          | HIGH              |            0.8065 |
|      9 | ECVT     | Basic Materials        |          13.93 | $1.5B        |          21.34 |          5.37 |           1.57 |       15.37 |     0.2147 | True          | HIGH              |            0.806  |
|     10 | SPWH     | Consumer Cyclical      |           1.48 | $57M         |          12.12 |         13.85 |           1.5  |        6.15 |     0.1638 | True          | HIGH              |            0.8059 |
|     11 | OCSL     | Financial              |          11.97 | $1.1B        |           7.84 |          3.01 |           1.08 |        1.87 |     0.1142 | True          | HIGH              |            0.8038 |
|     12 | SEMR     | Technology             |          11.94 | $1.8B        |           0.17 |         -0.08 |           1.81 |       -5.8  |     0.0042 | True          | HIGH              |            0.795  |
|     13 | PEB      | Real Estate            |          13.62 | $1.5B        |          17.72 |          7.16 |           1.01 |       11.75 |     0.1678 | True          | HIGH              |            0.7937 |
|     14 | MRAM     | Technology             |          10.26 | $237M        |          16.33 |          8    |           1.57 |       10.36 |     0.2119 | True          | HIGH              |            0.7907 |
|     15 | SWBI     | Industrials            |          14.34 | $638M        |           3.91 |         -2.85 |           1.26 |       -2.06 |     0.1026 | False         | HIGH              |            0.7903 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-04-10">
</holdings>

<last_analyst_thesis>
# Week 29 — Thesis Review Summary

## Per-Position Thesis

**REPL — Conviction: 4/5 (upgraded from 3, approaching binary resolution)**

The portfolio's lead position enters its defining week. The April 10 PDUFA for RP1+nivolumab in anti-PD-1-refractory advanced melanoma is the highest-stakes event in the experiment's second half. REPL closed at $8.54 on April 6, continuing the rally (+1.55%) after the +10.51% surge on April 2. The stock is now +14.9% from the $7.43 entry, validating the SMA waiver entry that was justified by the date-certain binary catalyst.

Conviction is upgraded to 4/5 based on: (1) the BLA resubmission classified as a "complete response to the CRL" — meaning the FDA found the additional data adequate for re-review, (2) the original CRL cited trial design concerns, not efficacy or safety failures, which is more resolvable, (3) the clinical data is genuinely compelling (32.9% ORR, 15% CR, 33.7-month median DOR in a population with no good options), (4) the competitive landscape favors RP1 as a simpler, safer alternative to Iovance's Amtagvi (which carries 7.5% treatment-related mortality), and (5) the short interest at ~11.4% with 9.8 days to cover creates meaningful squeeze potential on approval.

The pre-catalyst GTC sell order (3 shares at $9.69) must be placed by April 8. If the stock spikes to $9.69+ before or on the PDUFA, this captures $6.78 in profit and reduces the remaining position to 7 shares — cutting gap-through risk by ~47%.

**GRCE — Conviction: 3/5 (new initiation, binary PDUFA play)**

GRCE represents the strongest available binary catalyst outside of REPL. The April 23 PDUFA for GTx-104 (IV nimodipine for aSAH) is a well-constructed regulatory setup: the NDA is supported by positive Phase 3 STRIVE-ON data showing 19% reduction in hypotension, 29% more patients with favorable functional outcomes, and dramatically better dose compliance (54% vs 8% at >=95% RDI) compared to oral nimodipine. There is no competing IV nimodipine product, and oral nimodipine — the only current treatment — has been unchanged for decades.

The risk/reward is highly asymmetric at the ~$63M market cap. On approval, the analyst target of $12 implies 186%+ upside from the $4.20 entry — significantly better than the originally planned $5.15 entry. The 11.4% short interest (surging 496.7% recently) with 3.3 days to cover creates significant squeeze potential. On CRL, the stop at $3.63/$3.53 limits the loss to $13.68 (3.2% of equity).

Conviction is capped at 3/5 because: (1) cash is thin at $18.7M — a CRL would create financing pressure, (2) the stock has already run significantly from its $1.75 52-week low, pricing in some approval probability, (3) rising short interest suggests some institutional skepticism, and (4) the company has no other revenue-generating assets — single-product binary risk. The April 4 selloff (-20.66%) appears to be profit-taking/liquidity-driven rather than thesis-breaking, which provided the favorable $4.20 entry.

The pre-catalyst GTC sell order (8 shares at $5.46, +30% from $4.20 entry) must be placed by April 21, at least 2 trading days before the April 23 PDUFA.

## Overall Portfolio Thesis — The Dual-Catalyst Deployment (Week 29 of 52)

The portfolio enters Week 29 at $428.18, trailing the S&P benchmark by ~$37 (-8%) with 24 weeks remaining. The user directive to bypass the RISK-OFF regime and push harder for alpha is the correct call at this juncture — 80% cash with a benchmark deficit and 24 weeks of runway was too passive for the experiment's objectives.

The updated strategy deploys into two independent binary PDUFA events in April:

**REPL (April 10):** RP1 melanoma BLA resubmission. 3 trading days to decision. Conviction 4/5. Position: 10 shares ($85.40, 19.9% of equity).

**GRCE (April 23):** GTx-104 IV nimodipine NDA. 12 trading days to decision. Conviction 3/5. Position: 24 shares at $4.20 ($100.80, 23.5% of equity). Filled in 24-hour market on April 6 — lower entry than planned ($5.15) improves risk/reward materially.

**Why two concurrent PDUFAs:** The events are independent — REPL's RP1 (oncolytic immunotherapy for melanoma) and GRCE's GTx-104 (IV nimodipine for brain hemorrhage) have zero regulatory overlap. This means the portfolio's probability of at least one positive outcome is substantially higher than either alone. At conservative 50/50 odds per PDUFA, the probability of at least one approval is 75%.

**Combined scenarios:**
- Both approve: portfolio surges to $602-681, **crushing the benchmark by $137-216**
- One approves, one CRLs: portfolio reaches $449-590, **closing or exceeding the benchmark**
- Both CRL (25% probability): portfolio drops to ~$389, a setback but survivable with $245+ cash and 24 weeks

The 57.2% post-trade cash position maintains the buffer needed for: (1) absorbing potential stop-outs on either position, (2) deploying into VNDA (earnings May 6) or RCKT (PRV, ban expires April 15) after the April PDUFAs resolve, and (3) maintaining the 15% minimum cash reserve even in worst-case scenarios.

The path to closing the benchmark gap no longer depends on a single coin flip. With two independent catalysts, the expected value of the portfolio has improved materially. The lower GRCE fill price ($4.20 vs $5.15) provides better risk/reward — more upside per share, less capital at risk, and higher cash reserve. This is the aggressive, calculated deployment the experiment requires at this stage.
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-04-07,GRCE,24.0,4.2,100.8,0.0,MANUAL BUY LIMIT - Filled,,
2026-04-07,GRCE,,,4.2,-13.92,AUTOMATED SELL - STOP LIMIT TRIGGERED,24.0,3.62
2026-04-08,REPL,,,52.01,-10.71,AUTOMATED SELL - STOP LIMIT TRIGGERED,7.0,5.9
2026-04-10,REPL,,,22.29,-16.77,MANUAL SELL LIMIT - ,3.0,1.84
</recent_trades>

<execution_requests>
<session_directives>
<!-- UPDATE: Paste the directives for the weekly execution requests -->
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