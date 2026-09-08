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
<date>Monday, September 07, 2026</date>
<week_number>52 of 52 (twelve-month live experiment)</week_number>
<experiment_runway>ends 2026-09-18 (2 calendar weeks remaining)</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| ATRC   |   51.52 | +0.00% |     714,100 | Holding    |
| PAR    |   19.77 | +0.00% |     815,800 | Holding    |
| CADL   |   12.78 | -0.00% |     535,300 | Holding    |
| IWO    |  377.54 | +0.00% |     302,700 | Benchmark  |
| XBI    |  163.81 | +0.00% |   3,804,400 | Benchmark  |
| SPY    |  770.19 | +0.00% |  34,015,600 | Benchmark  |
| IWM    |  296.01 | +0.00% |  13,951,100 | Benchmark  |
| QQQ    |  718.96 | +0.00% |  32,784,600 | Benchmark  |
| TLT    |   82.21 | +0.00% |  16,575,000 | Macro      |
| HYG    |   79.16 | +0.00% |  25,350,900 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    2.0608 |                         |
| Sortino Ratio (annualized)    |    6.9880 |                         |
| Beta (daily) vs ^GSPC         |    1.5544 |                         |
| Alpha (annualized) vs ^GSPC   |  +679.89% |                         |
| R²                            |     0.035 | Low — alpha/beta unstable |
| Time-Weighted Return (cum)    |   +22.10% | injection-neutral       |
| S&P 500 Return (cum)          |   +16.38% | same window             |
| TWR Alpha (cum)               |    +5.71% | TWR minus S&P           |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $772.00 |
| S&P Equivalent      |   $774.71 |
| Cash Balance        |   $272.17 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-09-07" candidates="15">
|   rank | ticker   | sector             |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:-------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | GNW      | Financial          |          10.35 | $3.9B        |           5.61 |          4.12 |           1.35 |        6.93 |     0.0647 | True          | HIGH              |            0.8486 |
|      2 | KFY      | Industrials        |          85.35 | $4.4B        |           2.04 |         -0.23 |           1.82 |        3.36 |     0.0428 | True          | HIGH              |            0.8431 |
|      3 | ITGR     | Healthcare         |         126.5  | $4.3B        |           1.14 |          1    |           1.93 |        2.46 |     0.0101 | True          | HIGH              |            0.8346 |
|      4 | ATAI     | Healthcare         |           7.39 | $2.7B        |           1.93 |          0.68 |           1.45 |        3.25 |     0.037  | True          | HIGH              |            0.8335 |
|      5 | OGN      | Healthcare         |          13.78 | $3.6B        |           1.17 |          0.15 |           1.48 |        2.49 |     0.0132 | True          | HIGH              |            0.8239 |
|      6 | NMIH     | Financial          |          44.83 | $3.4B        |           1.43 |          1.08 |           1.47 |        2.75 |     0.0442 | False         | HIGH              |            0.8237 |
|      7 | ANAB     | Healthcare         |          56.8  | $1.7B        |           2.4  |         -0.4  |           2.56 |        3.72 |     0.0877 | False         | HIGH              |            0.8134 |
|      8 | BRBS     | Financial          |           3.96 | $350M        |           6.45 |          1.54 |           1.22 |        7.77 |     0.0958 | True          | HIGH              |            0.8133 |
|      9 | TYRA     | Healthcare         |          28.65 | $1.7B        |           9.14 |         14.46 |           3.75 |       10.46 |     0.1384 | True          | HIGH              |            0.8128 |
|     10 | EPC      | Consumer Defensive |          28.87 | $1.3B        |           3.14 |          0.52 |           1.04 |        4.46 |     0.0512 | True          | HIGH              |            0.8061 |
|     11 | VTS      | Energy             |          17.78 | $749M        |          10.09 |          1.66 |           1.27 |       11.41 |     0.1234 | True          | HIGH              |            0.8055 |
|     12 | CFFN     | Financial          |           8.87 | $1.1B        |           1.14 |          3.02 |           1.42 |        2.46 |     0.0586 | True          | HIGH              |            0.8035 |
|     13 | FBRT     | Real Estate        |           8.59 | $713M        |           9.99 |          2.63 |           1.11 |       11.31 |     0.1121 | True          | HIGH              |            0.8031 |
|     14 | DAN      | Consumer Cyclical  |          32.04 | $3.5B        |           7.37 |          6.87 |           1.01 |        8.69 |     0.0912 | True          | HIGH              |            0.803  |
|     15 | NAVI     | Financial          |           9.63 | $903M        |          13.7  |          1.26 |           1.23 |       15.02 |     0.1386 | True          | HIGH              |            0.8016 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-09-04">
<holding ticker="ATRC" shares="3" avg_cost="34.30" current_price="51.52" stop_loss="48.50" stop_limit="48.35" />
<holding ticker="PAR" shares="11" avg_cost="19.05" current_price="19.77" stop_loss="17.05" stop_limit="16.90" />
<holding ticker="CADL" shares="10" avg_cost="11.43" current_price="12.78" stop_loss="12.35" stop_limit="12.20" />
</holdings>

<last_analyst_thesis>
# Week 51 — Thesis Review Summary

**Date:** 2026-08-31 | **Week:** 51 of 52 | **Posture:** Aggressive — trailing benchmark
**14 sessions remain (Sept 7 is Labor Day).** All prices live intraday, ~10:00 AM EDT.

---

## Per-Position Thesis

**ATRC (AtriCure) — KEEP | Conviction 5/5**
**+36.9%**, and the most consequential position on the board. The fundamental case has **strengthened sharply while the price fell**: **Piper Sandler raised its target to $60 from $50 (Overweight)** four days ago, following BTIG's **$55 from $45 (Buy)** on 8/24 — both citing the new STS quality metric driving adoption of concomitant ablation. **Consensus PT has risen $47.33 → $49.56 and now sits above the market.** Revenue TTM +13.9%. No adverse news since 7/27.
**And the stop was $0.66 away — 0.38×ATR. Confirmed live: today's low printed $46.61, clearing it by 31 cents.** ATRC was set to be stopped out at $46.30 for **+35.0%** exactly as two banks put $55–60 on it. That near-miss is what forced the rule question, and under the Week 51 authorization the stop is **restored to $44.35 (1.76×ATR), still locking +29.3%**.
*On my own decision:* I raised this stop $45.85 → $46.30 last Thursday at $49.27, near the 52-week high. It gained **$1.35** of locked profit and materially raised the stop-out probability. It passed every check at the time. **In hindsight it was poor value** — a marginal gain bought with real optionality. Trailing at 1.75×ATR on a 3.5%-ATR name leaves nothing when the name gives back one week.

**PAR (PAR Technology) — KEEP | Conviction 4/5**
−3.5% at $18.39 after a −3.97% session, on **no news** (latest item is 13 days old). Revenue **+18.8%**, forward PE 16.82, **Buy with PT $25.31 — +35.1%, the largest upside of any holding.** Now the biggest position at 26.9% of equity, with $23 of headroom to the 30% cap.
The 3 shares added Friday at $19.60 are **−4.4%** — a poor fill in hindsight, since the $19.60 limit caught near the top of that day's $19.15–19.92 range. Stop **restored $17.50 → $17.05** (1.77×ATR), below both today's low and the 10-day low.

**CADL (Candel Therapeutics) — KEEP, stop deliberately NOT restored | Conviction 3/5**
+11.9%. **PT raised to $21.00 (+64%), Strong Buy** — the largest upside on the board — and beta −0.50. But XBI fell −3.48% Friday and again today, and CADL sits **0.48×ATR from its $12.35 stop**.
It was the one name where I reversed my own initial view. Restoring it to ~1.75×ATR means $11.20, converting a locked **+8.0% gain into −2.0%**. The deciding factor is catalyst timing: **the BLA submission is guided for Q4 2026, after the experiment ends on 9/18.** No dated event exists inside the runway for that extra room to pay off, so the move would be unanchored sentiment on a pre-revenue binary. Surrendering a booked gain to buy option value on an event that cannot occur in time is not a trade worth making — the +64% PT is a 12-month target.

**TILE (Interface) — KEEP | Conviction 4/5**
+19.3%. Strong Buy, PT $45.25 (+18.7%), EPS +52.5%. Stop **restored $37.10 → $36.25** (1.72×ATR), still locking **+12.9%**. Ex-dividend Sept 4.

**WWW (Wolverine World Wide) — KEEP, stop deliberately NOT restored | Conviction 3/5**
−5.5%. Forward PE ~11.2, Buy, PT $24.30 (+22%). Bounced +2.94% Friday and gave part back today. Left at 0.68×ATR: the weakest thesis in the book, down 5 of 6 sessions since entry. If it goes, let it go.

---

## Overall Portfolio Thesis

**Gap: +1.52% (Aug 25) → −0.31% (Aug 28) → −2.47% (live).** Five sessions turned the widest lead of the experiment into the largest deficit since July. Today alone the book is **−2.53% against SPY's −0.37%** — and not one of the five declines is company-specific. Every holding was browser-checked; there is no news, no downgrade, no guidance change anywhere in the book.

**The defining fact of Week 51: all five positions are now inside 1.0×ATR of their stops** — ATRC 0.38×, CADL 0.44×, WWW 0.68×, TILE 0.81×, PAR 0.94×. None is at the 1.5× minimum the rules require for a stop to function as anything other than a coin flip. I did not tighten them into this state; the market walked the prices down to fixed lines that cannot be lowered.

This has one benign consequence and one severe one. Benign: **every stop firing costs only −2.87% of equity** — the book's remaining downside is capped under 3%. Severe: **no position can absorb new capital**, because the one-open-order-per-stock rule means new shares inherit a stop inside a single day's noise. That single test blocked every add this week.

**On the Aggressive directive.** It could not be expressed through deployment, and testing why produced the week's decisive result. To close the gap needs **+$19.06**. The **$69.34 of cash would require +27.5%** to supply it — or SPY +9.2% at 3× leverage. So relaxing the cash floor, the 30% name cap, the position limit, the $5B ceiling, or even the excluded-class rule (which would permit leveraged ETFs) **all compete for the same $69, and none of them can move the gap.** Only the **$682.76 of holdings** can, and they need just **+2.79%**.

That collapses everything onto one question — do the positions survive to deliver it — and onto the one constraint that governs it. **Being stopped into cash locks the deficit permanently**, because cash has no mechanism to recover a gap: liquidation at the old levels would have banked a **−5.27% gap** with 13 sessions and no way back.

So the authorized relaxation was **not** the obvious aggressive trade (sell WWW, buy RCKT's +135% PT — declined: a lottery ticket with no catalyst inside the runway, bought with a realised loss). It was a **one-time restoration of ATRC, PAR and TILE to the 1.5–1.75×ATR band these rules already mandate** and which every stop had drifted below through price movement alone. Cost: **$14.20, or 1.87% of equity**, taking all-stops-fire from −3.69% to −5.56%. CADL and WWW were left alone.

**The limit matters as much as the permission.** "Restore the band" is infinitely reusable as prices fall, and that ratchet — not the $14 — is why the no-lowering rule exists. This is bounded to a single documented adjustment, recorded in `portfolio_rules.md`; the rule resumes immediately afterwards.

**Next:** 13 sessions after today. No holding reports earnings before the end, no catalyst is dated inside the window, and $9 is deployable. Three positions now have room to survive a normal week; two do not, by choice. The result rests on whether ATRC, PAR and TILE convert that room into the +2.79% the book needs.

---

*Week 51 Summary generated 2026-08-31 by Claude Code, revised same day for the one-time stop-restoration authorization. All prices are live intraday quotes browser-verified 09:53–10:05 AM EDT; ATR and range figures computed from settled bars through 2026-08-28.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-09-01,WWW,,,21.1,-5.43,AUTOMATED SELL - STOP LIMIT TRIGGERED,3.0,19.29
2026-09-03,TILE,,,32.1,16.56,AUTOMATED SELL - STOP LIMIT TRIGGERED,4.0,36.24
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net
- Catalyst timing: Within 5 days
- Risk posture: Aggressive — deploy the cash
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