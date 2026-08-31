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
<date>Monday, August 31, 2026</date>
<week_number>51 of 52 (twelve-month live experiment)</week_number>
<experiment_runway>ends 2026-09-18 (3 calendar weeks remaining)</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| TILE   |   38.06 | -1.35% |      33,433 | Holding    |
| ATRC   |   46.92 | -3.35% |      43,325 | Holding    |
| PAR    |   18.40 | -3.92% |      75,957 | Holding    |
| CADL   |   12.71 | -2.75% |      86,075 | Holding    |
| WWW    |   19.95 | -1.94% |      28,462 | Holding    |
| IWO    |  376.10 | -0.74% |      30,157 | Benchmark  |
| XBI    |  160.31 | -1.27% |   1,531,139 | Benchmark  |
| SPY    |  766.54 | -0.37% |   4,048,377 | Benchmark  |
| IWM    |  293.73 | -0.68% |   3,099,470 | Benchmark  |
| QQQ    |  715.74 | -0.10% |   3,811,223 | Benchmark  |
| TLT    |   82.28 | -0.72% |   4,072,382 | Macro      |
| HYG    |   79.68 | -0.07% |   1,575,802 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Max Drawdown                  |   -24.99% | on 2026-03-20           |
| Sharpe Ratio (annualized)     |    2.0839 |                         |
| Sortino Ratio (annualized)    |    7.0730 |                         |
| Beta (daily) vs ^GSPC         |    1.5653 |                         |
| Alpha (annualized) vs ^GSPC   |  +714.36% |                         |
| R²                            |     0.035 | Low — alpha/beta unstable |
| Time-Weighted Return (cum)    |   +22.04% | injection-neutral       |
| S&P 500 Return (cum)          |   +16.28% | same window             |
| TWR Alpha (cum)               |    +5.76% | TWR minus S&P           |
</risk_metrics>
</market_data>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $771.65 |
| S&P Equivalent      |   $774.02 |
| Cash Balance        |    $69.34 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-08-31" candidates="15">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   momentum_5d |   volume_ratio |   rs_vs_iwm |   bb_width | above_sma20   | data_confidence   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|--------------:|---------------:|------------:|-----------:|:--------------|:------------------|------------------:|
|      1 | ORC      | Real Estate            |           6.58 | $1.3B        |           3.13 |         -2.08 |           0.33 |        5.8  |     0.0699 | False         | HIGH              |            0.8351 |
|      2 | PFLT     | Financial              |           7.45 | $739M        |           3.4  |          0.47 |           0.17 |        6.07 |     0.0599 | True          | HIGH              |            0.8333 |
|      3 | DX       | Real Estate            |          12.91 | $3.2B        |           1.21 |         -0.73 |           0.18 |        3.88 |     0.046  | False         | HIGH              |            0.8239 |
|      4 | ATAI     | Healthcare             |           7.34 | $2.7B        |           2.23 |         -1.34 |           0.12 |        4.9  |     0.0442 | True          | HIGH              |            0.81   |
|      5 | BBDC     | Financial              |           9.48 | $990M        |          10.56 |          0.26 |           0.14 |       13.23 |     0.1197 | True          | HIGH              |            0.8093 |
|      6 | PRGS     | Technology             |          44.17 | $1.8B        |           5.54 |          2.06 |           0.12 |        8.21 |     0.0838 | True          | HIGH              |            0.8069 |
|      7 | TRIN     | Financial              |          18.72 | $1.8B        |           3.06 |          0.08 |           0.14 |        5.73 |     0.0839 | True          | HIGH              |            0.7978 |
|      8 | ARR      | Real Estate            |          16.29 | $2.3B        |          -0.86 |         -0.68 |           0.18 |        1.81 |     0.0381 | False         | HIGH              |            0.7972 |
|      9 | FIRY     | Communication Services |          10.1  | $168M        |           8.02 |          7.68 |           0.23 |       10.69 |     0.1409 | True          | HIGH              |            0.7965 |
|     10 | JRSH     | Consumer Cyclical      |           5.68 | $72M         |          16.96 |          3.35 |           0.38 |       19.63 |     0.1936 | True          | HIGH              |            0.7933 |
|     11 | LAND     | Real Estate            |           9.01 | $389M        |          10.42 |          2.62 |           0.15 |       13.09 |     0.1452 | True          | HIGH              |            0.7895 |
|     12 | PNNT     | Financial              |           3.78 | $246M        |           7.39 |         -0.79 |           0.1  |       10.06 |     0.0979 | True          | HIGH              |            0.785  |
|     13 | EFC      | Real Estate            |          13.43 | $1.7B        |           0.94 |         -1.14 |           0.11 |        3.61 |     0.0451 | False         | HIGH              |            0.7848 |
|     14 | IVR      | Real Estate            |           7.37 | $793M        |          -0.42 |         -0.95 |           0.13 |        2.25 |     0.033  | False         | HIGH              |            0.7846 |
|     15 | AI       | Technology             |          10.64 | $1.6B        |           5.87 |          8.68 |           0.14 |        8.54 |     0.1157 | True          | HIGH              |            0.7842 |
</screener_watchlist>

**Screener Integration:**
- Evaluate AT LEAST the top 5 screener candidates via WebSearch before selecting.
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-08-31">
<holding ticker="TILE" shares="4" avg_cost="32.10" current_price="38.06" stop_loss="37.10" stop_limit="36.95" />
<holding ticker="ATRC" shares="3" avg_cost="34.30" current_price="46.92" stop_loss="46.30" stop_limit="46.15" />
<holding ticker="PAR" shares="11" avg_cost="19.05" current_price="18.40" stop_loss="17.50" stop_limit="17.35" />
<holding ticker="CADL" shares="10" avg_cost="11.43" current_price="12.71" stop_loss="12.35" stop_limit="12.20" />
<holding ticker="WWW" shares="3" avg_cost="21.10" current_price="19.95" stop_loss="19.30" stop_limit="19.15" />
</holdings>

<last_analyst_thesis>
# Week 50 — Thesis Review Summary

**Date:** 2026-08-24 | **Week:** 50 of 52 | **Posture:** Tighten stops · Wide net · 5-day catalyst window

---

## Per-Position Thesis

**ATRC (AtriCure) — KEEP | Conviction 5/5 (raised from 4/5)**
**+42.8%**, the book's engine. The PRV gate paid for itself this morning: **BTIG raised its target to $55 from $45 (Buy) eighteen minutes before my fetch**, on a **new STS Quality Metric that could expand adoption** of concomitant ablation — a Monday pre-market catalyst invisible in Friday's settled data. Seven consecutive up sessions with stair-stepped rising lows (41.65 → 46.99); revenue TTM +13.9%; news feed clean since 7/27.
**The +30% partial stays deferred** on all four criteria: new catalyst (FY EPS guidance doubling, now reinforced by the STS metric), stop locks **+33.7%** (requirement +15%), position 18.8% of equity (cap 30%), conviction 5/5. Per the rule, the trailing stop *replaces* the partial sell as the risk control.
**Stop raised $44.20 → $45.85** (1.86×ATR below price, below Friday's $46.99 low).
Bear case, stated honestly: **PE 232, forward PE 199, and a consensus PT of $47.33 that sits BELOW the market.** The stock has outrun what most analysts think it is worth. The stop, not the story, protects this gain.

**PAR (PAR Technology) — KEEP | Conviction 4/5**
+3.6% in four sessions. **Revenue TTM +18.8%** — the strongest top line in the book — and **PT $25.31 (+29.6%)**, the largest verified upside among holdings. Forward PE 17.82 on a still-GAAP-lossmaking business (EPS −$1.77), trading **64% below its 52-week high**: a repair story with real growth underneath, which is exactly the profile that worked here and the opposite of the shrinking-revenue trap that cost us TDAY.
**Stop raised $16.50 → $17.50**, cutting the worst case from **−12.5% to −7.2%** while keeping two full ATRs of room.

**TILE (Interface) — KEEP | Conviction 4/5**
+21.3%. Net income **+52.0%** and EPS **+52.5%** on revenue +6.5% — margin expansion, not financial engineering. Strong Buy, PT $45.25 (+16.2%).
**Stop held at $37.10 — tightening deliberately refused.** A one-ATR tighten would sit at $38.58, *above* Friday's $38.55 low, and $37.10 is already 1.24×ATR, tighter than the 1.5× minimum. Soft spot: forward PE (16.20) now exceeds trailing (15.77), meaning the market expects earnings growth to slow.

**CADL (Candel Therapeutics) — KEEP | Conviction 3/5**
+14.9%, and Friday's +6.32% is already giving back — **pre-market $12.85 (−2.17%)**. Strong Buy with a **PT $20.88 (+59%)**, the largest upside on the board, but the headline hides something: **the most recent analyst action is BofA raising its target to $12 while keeping Neutral — and $12 is BELOW the market.** The marginal analyst is not a buyer here.
**Stop held at $12.00 — already exactly at the 1.5×ATR floor**, and a one-ATR tighten ($12.75) would sit above Friday's $12.38 low. **This is the most likely stop-out in the book**, sitting just 1.13×ATR under the pre-market print. If it fires, that is +5.0% realized: accept it and do not re-enter.

**WWW (Wolverine World Wide) — INITIATE | Conviction 4/5**
**3 shares, limit $21.10, stop $19.30/$19.15.**
The Week 49 queue delivered. WWW was deliberately passed over last week at day 2 of a +16.4% move, with an explicit gate: re-consider "on a higher-low base or a pullback toward the ~$19.35 20-day SMA." Five sessions later the lows stepped **19.38 → 19.88 → 20.08 → 19.85 → 20.32** inside a 19.38–21.08 range, **volume declined right through the base (1.30M → 0.66M)**, and Friday's close reclaimed the range top while staying under the 8/14 high. Constructive consolidation, not distribution — and the stock is **0.4% below where I declined to chase it**, now with a base underneath.
It is the only candidate profitable and growing on every line: **revenue +7.2%, net income +28.3%, EPS $1.28 +25.3%, and forward PE 11.91 against trailing 16.40** — forward *below* trailing, the signature of expected earnings growth. Q2 raised FY guidance on revenue, EPS, margin and free cash flow, with Merrell +11.1% and Saucony +9.9%, inventory −17%, net debt −22%. At **36% below its 52-week high of $32.80**, this is a recovery, not an extension.
Bear case: **beta 1.74** — it falls harder than the book in a drawdown, which matters with three weeks left. Tariffs remain a live 70bp margin headwind. Sweaty Betty (−2.4%) and Work Group (−1.6%) are shrinking; two brands carry the story. And there is **no dated catalyst inside the 5-day window** — this is post-earnings drift on a raised guide, a weaker reason to act than a scheduled event.

---

## Overall Portfolio Thesis

**Gap +1.48% · TWR alpha +7.91% · equity $781.68 vs S&P-equivalent $770.27.** The book has crossed in front of the benchmark for the first time in weeks — from **−4.8% on 8/14 to +1.5% today**. The scoreboard now reads as a lead, and the job for the final three weeks changes accordingly: **protect it without going passive.**

**Two decisions define this report, and both are refusals.**

The first is what the "tighten stops" directive actually permits. Applied mechanically, it says raise every stop by one ATR. Applied honestly, **it is only available on two of four holdings.** TILE ($37.10 already at 1.24×ATR) and CADL ($12.00 exactly at the 1.5×ATR floor) cannot be tightened without placing the stop *above Friday's low* — inside the noise band, where a stop stops being protection and becomes a coin flip. That is not a theoretical concern: **ARDT was lost on 8/21 at $10.35, the exact low of the day, and then closed at $10.99** — a 6.2% recovery we forfeited because the stop sat 0.28×ATR from the price. The directive is followed where the range check allows it (ATRC +$1.65, PAR +$1.00) and refused where it does not. Following an instruction into a known failure mode is not discipline.

The second is the ATRC partial. The stock is at a 52-week high on 232× earnings with a consensus PT *below* the market — a genuine case for taking the +30% trim. But all four deferral criteria hold, a **$55 target landed pre-market this morning on a structural adoption driver**, and the raised stop now locks **+33.7%** — capturing most of a trim's protective value without selling a third of a three-share position into an upgrade.

**The real constraint this week was neither market nor conviction — it was the 15% cash floor.** Of $191.44 in cash, only **$74.19** is deployable, which funds three WWW shares where risk-per-trade would have permitted twenty-one. The position is one-seventh the size the risk budget allows. I am not reinterpreting a floor that was explicitly reaffirmed as *unchanged* in the Week 49 amendments — but the constraint is now the binding limit on final-stretch alpha, and amending it is a decision available to the user, as the healthcare cap and market-cap ceiling were. **I am not recommending it:** at Week 50 with a lead in hand, cash is a legitimate defensive asset.

One structural note: **no holding reports earnings before the experiment ends.** All four printed in August. The last three weeks are drift-and-stop management, not event trading — which makes the stop levels the most consequential decisions left on the board.

**Next:** execute the three orders; watch CADL as the likeliest stop-out; re-run the ATRC deferral criteria fresh if $54.88 (+60%) is reached; note that the Consumer Discretionary cap is now full, so any Week 51 candidate must come from another sector.

---

*Week 50 Summary generated 2026-08-24 by Claude Code. All prices browser-verified pre-market 2026-08-24 07:00–07:25 EDT; all ATR and range figures computed from price history.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
<!-- No trades this week -->
</recent_trades>

<execution_requests>
<session_directives>
- Sector focus: Wide net
- Catalyst timing: Within 5 days
- Risk posture: Aggressive — trailing benchmark
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