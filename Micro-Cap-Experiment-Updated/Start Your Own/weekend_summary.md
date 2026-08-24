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
<date>Monday, August 24, 2026</date>
<week_number>50 of 52 (twelve-month live experiment)</week_number>
<experiment_runway>ends 2026-09-18 (4 calendar weeks remaining)</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| TILE   |   38.94 | -0.36% |     314,900 | Holding    |
| ATRC   |   48.98 | +3.14% |     421,100 | Holding    |
| PAR    |   19.53 | +3.06% |     962,800 | Holding    |
| CADL   |   13.13 | +6.32% |   1,336,400 | Holding    |
| IWO    |  387.37 | +1.24% |     570,400 | Benchmark  |
| XBI    |  165.73 | +1.44% |  10,010,500 | Benchmark  |
| SPY    |  765.72 | +0.41% |  39,030,400 | Benchmark  |
| IWM    |  299.96 | +0.77% |  22,933,800 | Benchmark  |
| QQQ    |  713.44 | +0.35% |  33,297,600 | Benchmark  |
| TLT    |   82.05 | -0.35% |  24,221,600 | Macro      |
| HYG    |   79.61 | +0.06% |  37,523,200 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    2.1213 |                         |
| Sortino Ratio (annualized)    |    7.2033 |                         |
| Beta (daily) vs ^GSPC         |    1.5686 |                         |
| Alpha (annualized) vs ^GSPC   |  +771.58% |                         |
| R²                            |     0.035 | Low — alpha/beta unstable |
| Time-Weighted Return (cum)    |   +23.63% | injection-neutral       |
| S&P 500 Return (cum)          |   +15.72% | same window             |
| TWR Alpha (cum)               |    +7.91% | TWR minus S&P           |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $781.68 |
| S&P Equivalent      |   $770.27 |
| Cash Balance        |   $191.44 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-08-24" candidates="15">
|   rank | ticker   | sector            |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | SIBN     | Healthcare        |          19.96 | $908M        |          17.07 |          6.8  |           2.36 |       14.66 |     0.153  | True          | HIGH              |            0.8267 |
|      2 | ABUS     | Healthcare        |           5.21 | $1.0B        |          18.14 |         10.62 |           8.28 |       15.73 |     0.1986 | True          | HIGH              |            0.8039 |
|      3 | BEAM     | Healthcare        |          29.76 | $3.0B        |          18.71 |          9.82 |           1.44 |       16.3  |     0.1713 | True          | HIGH              |            0.8006 |
|      4 | CYRX     | Industrials       |          17.13 | $866M        |          13.07 |         11.81 |           1.75 |       10.66 |     0.1652 | True          | HIGH              |            0.782  |
|      5 | PD       | Technology        |          12.32 | $950M        |          26.23 |          5.3  |           1.8  |       23.82 |     0.252  | True          | HIGH              |            0.7784 |
|      6 | ELME     | Real Estate       |           1.66 | $150M        |           2.47 |          3.11 |           1.88 |        0.06 |     0.0319 | True          | HIGH              |            0.7758 |
|      7 | FBRT     | Real Estate       |           8.39 | $691M        |           7.15 |          4.48 |           1.81 |        4.74 |     0.1157 | True          | HIGH              |            0.775  |
|      8 | BKKT     | Technology        |           8.59 | $379M        |          13.47 |         20.65 |           2.51 |       11.06 |     0.193  | True          | HIGH              |            0.7748 |
|      9 | NVAX     | Healthcare        |           8.94 | $1.5B        |          19.36 |         10.78 |           2.58 |       16.95 |     0.2308 | True          | HIGH              |            0.7741 |
|     10 | HNI      | Consumer Cyclical |          49.23 | $3.5B        |          14.62 |          1.88 |           1.76 |       12.21 |     0.1974 | True          | HIGH              |            0.771  |
|     11 | GEO      | Industrials       |          32.71 | $4.3B        |           5.75 |          5.01 |           1.44 |        3.34 |     0.0804 | True          | HIGH              |            0.7708 |
|     12 | ANY      | Financial         |           2.61 | $24M         |          33.85 |         24.29 |           5.65 |       31.44 |     0.3249 | True          | HIGH              |            0.7707 |
|     13 | RXRX     | Healthcare        |           3.5  | $1.9B        |          16.28 |         12.54 |           1.4  |       13.87 |     0.195  | True          | HIGH              |            0.767  |
|     14 | IRD      | Healthcare        |           4.05 | $339M        |          20.9  |         10.96 |           1.29 |       18.49 |     0.2152 | True          | HIGH              |            0.7629 |
|     15 | VIR      | Healthcare        |           9.78 | $1.7B        |          11.14 |          4.15 |           1.2  |        8.73 |     0.1463 | True          | HIGH              |            0.761  |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-08-21">
<holding ticker="TILE" shares="4" avg_cost="32.10" current_price="38.94" stop_loss="37.10" stop_limit="36.95" />
<holding ticker="ATRC" shares="3" avg_cost="34.30" current_price="48.98" stop_loss="44.20" stop_limit="44.05" />
<holding ticker="PAR" shares="8" avg_cost="18.85" current_price="19.53" stop_loss="16.50" stop_limit="16.35" />
<holding ticker="CADL" shares="10" avg_cost="11.43" current_price="13.13" stop_loss="12.00" stop_limit="11.85" />
</holdings>

<last_analyst_thesis>
# Week 49 — Thesis Review Summary

**Date:** 2026-08-16 | **Week:** 49 of 52 | **Posture:** Aggressive (amended constraints)

---

## Per-Position Thesis

**ATRC (AtriCure) — KEEP | Conviction 4/5**
**+29.2%** and knocking on the $44.59 partial trigger, which stays **deferred** — all four criteria hold: the FY EPS guidance doubling ($0.09–0.15 → $0.24–0.32) is a genuine new catalyst, the $39.50 stop locks **+15.2%**, the position is 17.9% of equity, conviction 4/5. With 54% cash going in, converting the best performer into more idle cash was the wrong direction.

**TILE (Interface) — KEEP | Conviction 4/5**
+18.3%, holding its post-earnings gap above the $36–37 consensus PT. Stop $35.75 locks +11.4% (deliberately at 1.27×ATR, tighter than guideline).

**ARDT (Ardent Health) — KEEP | Conviction 3/5**
+1.0% on the 5-share residual after the trim. The mixed Q2 (revenue beat, EPS miss, **adj. EBITDA −32% YoY**) keeps conviction capped. Stop $10.35.

**PAR (PAR Technology) — INITIATE | Conviction 4/5** *(order revised 8/17)*
The cleanest setup in weeks: **revenue +18.7% to $133.4M, ARR $338M, adjusted EBITDA $14.3M (from $5.5M), FY guidance raised to $516–523M** — real growth, not a cost-cut story on a shrinking base (the trap that cost us on TDAY and nearly on FOXF). Technically it **ground** higher rather than gapping — three consecutive higher lows (16.82 → 17.65 → 18.38) — and it is 8 sessions past its print. Fills the **empty Technology sector**.
**Revised order: 8 shares, limit $18.55, stop $16.50/$16.35.** PAR fell to **$18.46 (−3.25%)** on 8/17 with no news, which *improved* the entry to +7.6% over the 20-day SMA — but forced the stop wider, since the original $17.05 computed to only 1.24×ATR at the lower price. **Skip condition: do not buy if PAR closes below $18.38** (the last higher low).
Bear case: GAAP still lossmaking (−$1.77 TTM EPS) and the stock sits **~65% below its 52-week high** ($11.59–$54.62) — a beaten-down grower, not a leader. Offsetting: analyst PT **$25.31 (+37%)**, forward P/E 17.4.

**CADL (Candel Therapeutics) — INITIATE | Conviction 3/5** *(order revised 8/17)*
The Week 45 queue name, finally investable under the amended healthcare cap. CAN-2409 Phase 3 in localized prostate cancer showed a **39% improvement in disease-free survival** (58-month median follow-up), holds RMAT designation, and the **BLA submission is guided for Q4 2026** with cash into 2028. **Strong Buy, PT $20.88 (+78%)** — the largest upside on the board.
**Revised order: 10 shares, limit $11.50 (from $11.85), stop $10.70/$10.55 unchanged** — repriced to the verified pre-market level after a −2.05% print with no adverse news.
Bear case: clinical-stage binary — a BLA delay or FDA question resets the thesis, and biotech financings dilute. It also trades near its 52-week high ($4.35–$11.95), the mirror image of PAR.

---

## Overall Portfolio Thesis

**Gap −4.8% · TWR alpha +0.28% · cash 54.4% — and that cash was the problem.** Two sessions this week made it explicit: the book gained nothing while the S&P-equivalent rose, because half the portfolio earned zero. With ~4.5 weeks left, holding cash was a guaranteed way to finish behind.

So the decisive act this week wasn't a trade — it was **amending the constraints that were blocking deployment**. Raising the **healthcare cap 2 → 3** and the **universe ceiling $2B → $5B** nearly doubled the screenable universe (856 → **1,593** names) and directly unlocked one of the two initiations.

One useful finding: the ceiling change did **not** produce a flood of quality — the newly eligible $2–5B entrants (SFNC $3.5B, RNST $4.0B) screened poorly with *negative* relative strength. The micro-cap tilt wasn't the binding constraint; **the sector cap was.**

Deployment: **PAR** (real growth, empty sector, non-gapped entry) and **CADL** (dated BLA catalyst) — taking the book to **5 positions and ~19.0% cash** at the revised limits.

**Pre-open verification changed both orders (8/17).** PAR's first pre-market scan read **+1.10% at $19.26** — but a re-scan returned the same 5:26 AM timestamp from two sources, and MarketWatch showed why: **before-hours volume of 332 shares.** The print was one tiny trade carrying no information; the real session went the other way to **$18.46 (−3.25%)**. Both limits were repriced to verified live prices, and PAR's stop was widened from $17.05 to $16.50 because the drop pushed the original level to 1.24×ATR — inside the noise band. **Lesson banked: read extended-hours volume alongside the timestamp — a 332-share print looks like a price but isn't one.**

**WWW (Wolverine World Wide) is deliberately left on the table** despite having the strongest fundamentals of the three — Q2 revenue $506M (+7%), adj. EPS +14%, operating margin +80bps, and guidance raised on revenue, EPS, margin *and* free cash flow. It closed $18.06 → $19.88 → $21.03: **day 2 of a +16.4% move.** Chasing that is exactly what cost us on PHAT and LXU and nearly on FOXF. **Queued #1 for Week 50** on a higher-low base or a pullback toward the ~$19.35 20-day SMA.

Three weeks to run: the cash is finally working, the incumbents are protected at +11% to +15% locked, and the remaining edge has to come from PAR and CADL delivering what their fundamentals say they should.

**Next:** execute the revised orders (PAR subject to its $18.38 skip condition); WWW consolidation watch for Week 50; ATRC partial deferral under review.

---

*Week 49 Summary generated 2026-08-16 by Claude Code (Aggressive posture, amended constraints). Orders revised 2026-08-17 after live pre-open verification.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-08-17,PAR,8.0,18.8499,150.7992,0.0,MANUAL BUY LIMIT - Filled,,
2026-08-17,CADL,10.0,11.43,114.3,0.0,MANUAL BUY LIMIT - Filled,,
2026-08-21,ARDT,,,10.8,-2.25,AUTOMATED SELL - STOP LIMIT TRIGGERED,5.0,10.35
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net
- Catalyst timing: Within 5 days
- Risk posture: Tighten stops
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