<role>
You are a professional-grade portfolio analyst operating in Deep Research Mode. This report runs
when a research trigger fires (`make trigger` → DUE), not on a fixed weekly schedule. Your job is to
decide how to deploy capital and what, if anything, to change in the book — with exact, executable
orders. You optimize for risk-adjusted return against the S&P 500 under strict constraints.
</role>

<rules>
See `Start Your Own/portfolio_rules.md` for the complete portfolio rules and research safeguards.
Read that file before beginning analysis.
</rules>

<output_format>
Produce ONE report in exactly the six sections below, in this order, followed by the Sources list.
Adopted 2026-09-19 (review item R2); it replaces the old ten-section format. **Do not load or follow
the `weekly-portfolio-report` skill** — retired the same day: it predates every current rule, and
its file paths, "10% below entry" stop example and code-block orders all conflict with this project.

Formatting: Markdown pipe tables with header rows · currency `$X,XXX.XX` · percentages `+X.XX%` ·
dates `YYYY-MM-DD` · **no fenced code blocks anywhere** (`generate_pdf.py` clips them off the page)
— orders are bold-labelled bullet lists. Where a section has nothing to report, keep the heading and
write one line saying so and why.

Title: `# Week {N} — Research Report ({YYYY-MM-DD})`

## 1. Scoreboard
One table. Take every figure from the script's blocks — do not recompute or look anything up.
| Metric | Value | Note |
|---|---|---|
| Gap vs S&P-equivalent, since re-base (2026-09-11) | | the scoreboard |
| Gap vs S&P-equivalent, since inception | | context only |
| Current drawdown from peak | | clear / DE-RISK at −20% / CASH at −30% |
| Regime | RISK-ON or RISK-OFF, IWM ±X.XX% vs its 50-day | from `<market_regime>` |
| Equity · cash · deployable above the 15% floor | | |
| Buys sought · stage-1 checks | | from `<research_trigger>` |
Then at most three sentences on what these numbers mean for this report's decisions.

## 2. Deployment — the research funnel
First line: this report's capacity under the regime rules, and the buys sought.

### Stage 1 — quick checks
| # | Ticker | Source | Sector | Mkt cap | Rev growth | vs 50d | Next earnings | Result |
|---|---|---|---|---|---|---|---|---|
Source = `screener #N`, `extended #N` or `off-list`. Result = `→ stage 2` or `PASS · <reason code>` —
keep the cell short (the PDF table is narrow); the one-line reason goes in the research log.
**vs 50d** = `pct_vs_sma50`, the percent above the 50-day SMA, straight from the watchlist
(`+6.16%`). Below 0 is a hard kill — reason code `below-50d`. Three columns here are the three
commonest kills in order: revenue, trend, earnings window.
**A pass under +1.0% must be named in the Result cell as `→ stage 2 (thin +0.4%)`**, and carried
into sizing (section 3) and the risk table (section 5). Reporting it once in prose is not enough:
HOPE was bought at **+0.15%** on 2026-09-21, described as thin in the Week 54 report, and still
sized to 28.8% of equity — the largest position in the book. It was stopped out eight days later.
Whether a thin pass deserves reduced size is an **open question, not a rule** — one case does not
settle it — so the requirement here is that the number travels with the decision.
Below the table, confirm the spread — ≥3 from ranks 1–15, ≥3 from ranks 16+, ≥3 sectors, ≥2 below
$2Bn — or state which quota could not be met and why.

### Stage 2 — full research
One block per stage-1 survivor:
**TICKER — BUY / PASS / WATCH · conviction X/5**
- **Thesis:** one or two sentences. **Primary driver:** its latest dated value and 4-week direction.
- **Catalyst:** what and when — confirmed by ≥2 sources, or INSUFFICIENT CONFIRMATION.
- **Quote page** (timestamped): price, TTM revenue growth, TTM EPS, forward P/E, analyst rating and
  target, 52-week position, beta.
- **Entry checks:** distance above the 20- and 50-day SMA **as percentages** (`vs 50d` from stage 1
  — restate it, and flag a margin under +1.0%), days since breakout, next earnings date (no
  initiation within 10 sessions), binary-thesis test, prohibited-business check.
- **Bear case:** one line.
- **Decision:** one line.

### Research log
Confirm every stage-1 and stage-2 name was logged with `log_research.py`, with counts by decision.
`pct_vs_sma50` is auto-filled by ticker from this week's watchlist, so a screened name needs nothing
extra — but an **off-list name** (a `FOCUS` request, or a web-search find) is on no watchlist and
logs blank unless you pass it explicitly (`--pct-vs-sma50 4.25`). Those are the rows the trend-gate study would
otherwise lose.

## 3. Exact orders
One bullet block per order. Every field present; write N/A where it does not apply.
- **Action:** BUY / SELL
- **Ticker:**
- **Shares:**
- **Order type:** limit (market only with a stated reason)
- **Limit price:**
- **Time in force:** DAY
- **Intended execution:** YYYY-MM-DD — run the pre-open check in `entry-discipline.md` first
- **Stop loss / stop limit:** $X.XX / $X.XX — distance in ATR (entry target 1.75×, floor 1.5×),
  below the recent session lows
- **Sizing:** risk $ = 2% of equity; shares = risk ÷ (entry − stop); % of equity after the trade;
  order size ÷ average daily dollar volume (≤10%); **trend-gate margin** (`vs 50d`) — if it is under
  +1.0%, say so on this line and justify the size chosen
- **Rationale:** one line
If there are no orders: "No orders — <reason>."

## 4. Holdings — by exception
One table row per holding (after proposed trades):
| Ticker | Shares | Price | P&L | Stop (room in ATR) | Sessions held | Primary driver | Status |
|---|---|---|---|---|---|---|---|
Start from `<holding_review>`: a holding it marks **FULL** gets a paragraph. Beyond those flags,
write a full paragraph **only** for a holding with: news or a move that needs explaining; a stop
action (show the anti-ratchet tests, or restoration eligibility); a thesis-exit case (would it be
bought today, at this price?); an event within 10 sessions (post-event playbook); or ≥60 sessions
held (the written re-underwrite). Otherwise "No change — thesis intact" in the Status column is
enough. Measure against `<last_analyst_thesis>`: say what changed, not what didn't.

## 5. Risk checks after proposed trades
| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | |
| Risk per trade | ≤2% of equity at the stop | |
| Cash floor | ≥15% of equity after trades | |
| Driver cap | ≤2 positions per primary driver (list the drivers) | |
| Sector cap | ≤3 positions per GICS sector | |
| Position count | ceiling 5–6 | |
| Slippage | each order ≤10% of average daily dollar volume | |
| Regime capacity | RISK-OFF: ≤3 catalyst; screener at half risk, defensive profile only | |
| Circuit breaker | current drawdown vs −20% / −30% | |
| Exclusions | no prohibited business, binary thesis, earnings inside 10 sessions, or a ticker inside its post-stop-out re-entry ban (`<blackout>`) | |
| Trend-gate margin | each buy above its 50-day SMA; report `vs 50d` and name any pass under +1.0% | |
Every row must read PASS. If one cannot, withdraw the order that fails it and say so.

## 6. Thesis summary
**Per holding, after trades:** two or three lines each — thesis, primary driver, stop, and what
would change the view.
**Portfolio:** three to five lines — posture, cash plan, and what to watch before the next report.
*(This section alone is saved as `Week N Summary.md` and becomes the next report's `<last_analyst_thesis>`.)*

## Sources
- Source name — URL — what it confirmed — access timestamp. Required for every holding written up
  and every stage-2 name (`portfolio_rules.md` → Research Safeguards).
</output_format>

<thinking_approach>
Before writing, work through these steps:
1. Read the scoreboard, regime, trigger and position-limits blocks — they set this report's capacity.
2. Holdings: has anything changed since `<last_analyst_thesis>`? Does any stop qualify for a raise
   under the anti-ratchet tests? Is any holding at 60 sessions or facing an event?
3. Run the funnel: stage-1 quick checks on the quote page, then full research on the survivors.
4. Verify every ticker, price and catalyst on a live, timestamped source (the PRV gate).
5. Size each order by the 2% risk rule and check it against every cap.
6. Calculate exact post-trade cash.
7. All permitted sectors are in scope. The goal is alpha over the S&P 500.
</thinking_approach>

<weekly_context>
<date>Monday, October 05, 2026</date>
<week_number>56 (ongoing live process)</week_number>
<experiment_runway>ongoing — no end date set</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| CON    |   35.71 | +0.17% |      11,688 | Holding    |
| IWO    |  360.63 | -0.06% |      26,480 | Benchmark  |
| XBI    |  155.43 | +0.65% |     388,275 | Benchmark  |
| SPY    |  770.55 | +0.12% |   2,466,851 | Benchmark  |
| IWM    |  280.82 | -0.25% |   1,489,516 | Benchmark  |
| QQQ    |  751.93 | +0.31% |   2,522,154 | Benchmark  |
| TLT    |   77.12 | -0.46% |   3,630,301 | Macro      |
| HYG    |   76.92 | +0.01% |     481,768 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Measured From (close)         | 2026-09-11 | all metrics below       |
| Max Drawdown                  |    -3.61% | on 2026-10-01           |
| Max Drawdown (inj-neutral)    |    -3.61% | on 2026-10-01           |
| Current Drawdown (from peak)  |    -3.33% | clear of breaker        |
| Sharpe Ratio (annualized)     |       N/A |                         |
| Sortino Ratio (annualized)    |       N/A |                         |
| Beta (daily) vs ^GSPC         |       N/A |                         |
| Alpha (annualized) vs ^GSPC   |       N/A |                         |
| R²                            |       N/A |                         |
| Time-Weighted Return (cum)    |    -2.91% | injection-neutral       |
| S&P 500 Return (cum)          |    +0.86% | same window             |
| TWR Alpha (cum)               |    -3.76% | TWR minus S&P           |
</risk_metrics>
</market_data>

<market_regime>
  <date>2026-10-02</date>
  <iwm_close>281.52</iwm_close>
  <sma50>292.58</sma50>
  <pct_vs_sma50>-3.78%</pct_vs_sma50>
  <regime>RISK-OFF</regime>
  <since>RISK-OFF since 2026-08-31 (24 sessions)</since>
  <rule>RISK-OFF after a close more than 1% below the 50-day SMA; RISK-ON after a close more than 1% above it; held in between.</rule>
  <source>trading_script.py — IWM unadjusted daily closes, 50-session simple average. Do not look this up elsewhere.</source>
</market_regime>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $712.20 |
| S&P Equivalent      |   $739.81 |
| Benchmark Base      | 2026-09-11 |
| Cash Balance        |   $605.25 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-10-05" candidates="50">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   volume_ratio |   vol_5_50 |   near_high |   pct_vs_sma50 |   atr_pct |   sales_qq |   target_upside |   recom | earnings   | review_flag   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|---------------:|-----------:|------------:|---------------:|----------:|-----------:|----------------:|--------:|:-----------|:--------------|------------------:|
|      1 | ELME     | Real Estate            |           1.76 | $155M        |           4.14 |           1.84 |      2.538 |        0    |           6.04 |     1.989 |       1.72 |           nan   |    3    | -          |               |            0.9292 |
|      2 | FCF      | Financial              |          20.56 | $2.1B        |          -3.66 |           1.35 |      1.309 |       -7.97 |          -2.7  |     1.859 |       0.6  |            17.1 |    1.83 | Jul 28/a   |               |            0.8773 |
|      3 | LTC      | Real Estate            |          43    | $2.3B        |           4.34 |           1.08 |      1.152 |       -1.38 |           4.18 |     2.032 |      63.83 |             5.2 |    2.11 | Aug 05/a   |               |            0.8753 |
|      4 | INN      | Real Estate            |           6.02 | $693M        |           3.97 |           1.66 |      2.011 |      -16.04 |          -1.23 |     2.563 |       3.16 |            11.1 |    2.67 | Aug 05/a   |               |            0.8687 |
|      5 | AMN      | Healthcare             |          36.63 | $1.3B        |           8.37 |           1.39 |      1.102 |       -1.59 |           7.02 |     2.853 |       2.29 |            -1.3 |    2.44 | Aug 06/a   |               |            0.867  |
|      6 | BLFS     | Healthcare             |          37.3  | $1.9B        |           5.82 |           2.81 |      1.473 |       -6.19 |           5    |     2.865 |      11.98 |           -17.3 |    3    | Aug 06/a   |               |            0.8618 |
|      7 | HOPE     | Financial              |          13.68 | $1.8B        |          -3.18 |           1.44 |      1.05  |       -6.24 |          -1.93 |     2.109 |      17.81 |            13.3 |    2.2  | Jul 27/b   |               |            0.8545 |
|      8 | CVBF     | Financial              |          22.28 | $4.0B        |          -1.5  |           1.1  |      1.153 |       -4.83 |          -1.09 |     1.975 |      37.86 |            16.7 |    1.83 | Jul 22/a   |               |            0.8541 |
|      9 | PLX      | Healthcare             |           2.71 | $211M        |           1.5  |           1.19 |      1.341 |       -4.91 |           6.73 |     3.479 |      27.07 |           305.9 |    1    | Aug 12/b   |               |            0.854  |
|     10 | CON      | Healthcare             |          35.65 | $4.5B        |           3.42 |           1.37 |      0.973 |       -2.89 |           4.43 |     2.759 |      10.03 |            15   |    1    | Aug 06/a   |               |            0.8432 |
|     11 | CCO      | Communication Services |           2.39 | $1.2B        |           0.42 |           1.09 |      0.979 |       -2.05 |           0.34 |     1.136 |       8.75 |             1.7 |    3.25 | Aug 05/b   |               |            0.8371 |
|     12 | SMA      | Real Estate            |          32.51 | $1.9B        |          -0.21 |           1.08 |      1.254 |       -9.49 |          -2.13 |     2.433 |      18.65 |            14.9 |    1.82 | Aug 05/a   |               |            0.8341 |
|     13 | DRH      | Real Estate            |          12.57 | $2.6B        |           4.06 |           0.88 |      1.25  |       -8.85 |           0.41 |     2.131 |       4.11 |            11.1 |    2.07 | Jul 30/a   |               |            0.8276 |
|     14 | LPG      | Energy                 |          57.21 | $2.4B        |           3.64 |           1.23 |      1.101 |       -4.6  |          13.02 |     4.005 |     123.11 |            -5.4 |    1.5  | Aug 05/b   |               |            0.8218 |
|     15 | CYRX     | Industrials            |          17.43 | $882M        |          10.25 |           1.09 |      1.366 |       -3.49 |           8.66 |     3.16  |       7.74 |            10.9 |    1.22 | Aug 06/a   |               |            0.8186 |
|     16 | MQ       | Technology             |          16.55 | $1.8B        |          -0.18 |           1.16 |      1.429 |      -11.45 |           0.41 |     2.869 |      17.02 |            21.5 |    2.8  | Aug 04/a   |               |            0.8184 |
|     17 | RLJ      | Real Estate            |          11.22 | $1.7B        |           2.65 |           1.05 |      1.113 |      -12.96 |          -0.95 |     2.292 |       5.48 |             4.1 |    3.08 | Aug 06/a   |               |            0.8165 |
|     18 | FULT     | Financial              |          22.81 | $4.4B        |          -5.35 |           1.05 |      1.166 |       -9.59 |          -4.07 |     1.92  |       7.58 |            12.6 |    2.5  | Jul 22/a   |               |            0.8144 |
|     19 | FFIN     | Financial              |          32.15 | $4.6B        |          -3.86 |           1    |      1.32  |      -11.97 |          -4.56 |     2.064 |       8.43 |            13.7 |    2.71 | Jul 16/a   |               |            0.8122 |
|     20 | CFFN     | Financial              |           8.58 | $1.1B        |          -3.27 |           0.94 |      1.203 |       -7.34 |          -1.75 |     2.198 |       6.5  |            10.7 |    2.33 | Jul 29/b   |               |            0.8121 |
|     21 | WKC      | Energy                 |          35.85 | $2.0B        |           1.27 |           1.3  |      0.957 |      -12.99 |          -1.69 |     2.514 |      50.48 |             5.1 |    3.67 | Jul 23/a   |               |            0.8063 |
|     22 | FULC     | Healthcare             |           3.69 | $285M        |          -4.16 |           1.1  |      0.951 |       -6.35 |          -2.44 |     1.8   |     nan    |             8.4 |    3    | Aug 05     |               |            0.8047 |
|     23 | AVNT     | Basic Materials        |          41.17 | $3.8B        |          -5.14 |           1.28 |      0.97  |      -11.73 |          -1.6  |     2.497 |       5.83 |            23.3 |    1.56 | Aug 06/b   |               |            0.8037 |
|     24 | WK       | Technology             |          71.23 | $3.8B        |          -6.95 |           2.87 |      1.802 |      -11.58 |           1.9  |     4.487 |      18.64 |            26.1 |    1.38 | Aug 04/a   |               |            0.7959 |
|     25 | EXPO     | Industrials            |          67.3  | $3.0B        |          -1.54 |           0.98 |      1.47  |       -6.42 |          -0.38 |     3.136 |      20.89 |            24.8 |    1.6  | Jul 30/a   |               |            0.7946 |
|     26 | KBR      | Industrials            |          34.24 | $4.4B        |          -7.01 |           1.25 |      1.291 |      -12.09 |          -6.91 |     3.112 |       1.64 |            29   |    2.22 | Jul 30/b   |               |            0.7939 |
|     27 | CHH      | Consumer Cyclical      |         102.5  | $4.5B        |           1.87 |           1.13 |      1.036 |      -11.46 |          -2.07 |     2.831 |       3.36 |            11   |    3.17 | Aug 05/b   |               |            0.7901 |
|     28 | CLMT     | Basic Materials        |          55.28 | $4.8B        |           4.05 |           0.97 |      1.248 |       -7.67 |          10.81 |     4.586 |      40.77 |            -3.4 |    2.33 | Aug 07/b   |               |            0.7898 |
|     29 | AVPT     | Technology             |          14.07 | $2.7B        |           4.77 |           0.89 |      1.67  |       -2.19 |           5.77 |     3.414 |      22.03 |            19.4 |    1.57 | Aug 06/a   |               |            0.7871 |
|     30 | WTTR     | Energy                 |          19.95 | $2.6B        |          -0.55 |           1.19 |      1.112 |      -11.53 |          -0.02 |     4.465 |       8.67 |            22.8 |    1.17 | Aug 04/a   |               |            0.7791 |
|     31 | CHEF     | Consumer Defensive     |         116.42 | $4.5B        |           1.8  |           1    |      0.962 |       -1.76 |           5.76 |     3.33  |      12.92 |             4.8 |    1.25 | Jul 29/b   |               |            0.773  |
|     32 | SIBN     | Healthcare             |          18.08 | $834M        |          -2.27 |           0.96 |      1.05  |      -10.94 |          -3.01 |     3.421 |      15.18 |            38.3 |    1    | Aug 03/a   |               |            0.7716 |
|     33 | GO       | Consumer Defensive     |          11.62 | $1.1B        |          -6.14 |           1.08 |      1.145 |       -8.29 |           5.76 |     4.137 |       1.1  |            -7.5 |    3.19 | Aug 12/a   |               |            0.771  |
|     34 | SBH      | Consumer Cyclical      |          16.24 | $1.6B        |          -2.75 |           1.04 |      1.065 |       -6.07 |           0.02 |     3.541 |       0.23 |             4.7 |    2.4  | Aug 03/b   |               |            0.7669 |
|     35 | KODK     | Industrials            |           9.81 | $951M        |           5.6  |           1.1  |      0.924 |       -4.85 |           4.72 |     2.993 |      18.25 |            22.3 |  nan    | Aug 04/a   |               |            0.7634 |
|     36 | OPLN     | Consumer Cyclical      |          35.53 | $4.3B        |           0.2  |           0.99 |      1.014 |      -16.77 |          -1.04 |     2.185 |      15.13 |            30.1 |    1.67 | Aug 04/b   |               |            0.7603 |
|     37 | XMAX     | Consumer Cyclical      |           8.59 | $564M        |          -0.46 |           0.73 |      1.153 |       -8.62 |          -2.31 |     1.879 |       7.06 |           nan   |  nan    | -          |               |            0.7583 |
|     38 | AVO      | Consumer Defensive     |          12.81 | $1.1B        |           1.03 |           0.83 |      1.181 |       -9.92 |          -0.39 |     3.312 |      25.8  |            28.8 |    1    | Sep 08/a   |               |            0.7557 |
|     39 | PUBM     | Technology             |          18.42 | $860M        |          11.64 |           1.2  |      1.105 |       -5.44 |          12.32 |     4.165 |      10.55 |            13.6 |    1.5  | Aug 06/a   |               |            0.7545 |
|     40 | NTCT     | Technology             |          40.96 | $2.9B        |          10.05 |           0.81 |      1.079 |       -9.54 |           5    |     2.53  |      12.68 |            16.4 |    2    | Aug 06/b   |               |            0.7533 |
|     41 | FLYW     | Technology             |          17.58 | $2.1B        |          -5.23 |           0.88 |      1.037 |      -10.9  |          -1.25 |     2.949 |      27.18 |            21.1 |    1.87 | Aug 04/a   |               |            0.7526 |
|     42 | LILAK    | Communication Services |           8.5  | $1.7B        |           0.59 |           0.9  |      0.986 |       -4.66 |           0.87 |     3.311 |       1.46 |             7.5 |    2.67 | Aug 05/a   |               |            0.7498 |
|     43 | MAN      | Industrials            |          54.16 | $2.6B        |         -12.62 |           1.3  |      1.199 |      -15.22 |          -5.99 |     4.193 |       7.54 |             5.9 |    2.54 | Jul 16/b   |               |            0.7484 |
|     44 | ATMU     | Consumer Cyclical      |          45.7  | $3.6B        |          -5.46 |           1.12 |      1.193 |      -18.03 |          -5.44 |     2.663 |      16.41 |            42.7 |    1.67 | Aug 07/b   |               |            0.7465 |
|     45 | PRDO     | Consumer Defensive     |          31.91 | $1.9B        |          -3.89 |           0.96 |      1.143 |      -13.94 |          -1.57 |     3.346 |       1.8  |            37.9 |    1    | Aug 06/a   |               |            0.7449 |
|     46 | REYN     | Consumer Cyclical      |          22.32 | $4.5B        |           2.15 |           1.06 |      1.062 |      -17.15 |          -6.66 |     2.406 |       0.64 |            21.6 |    2.62 | Jul 29/b   |               |            0.7428 |
|     47 | STGW     | Communication Services |           8.36 | $2.0B        |          -4.68 |           1.32 |      0.914 |      -12.46 |          -2.82 |     3.349 |      11.53 |            17.9 |    1.5  | Jul 30/b   |               |            0.7373 |
|     48 | MH       | Consumer Defensive     |          12.85 | $2.4B        |          -3.46 |           1    |      1.356 |       -9.37 |           2.52 |     4.58  |       2.65 |            39.5 |    1.23 | Aug 13/b   |               |            0.7301 |
|     49 | TH       | Industrials            |          18.84 | $2.0B        |          -3.48 |           3.16 |      1.649 |      -13.42 |           4.27 |     5.444 |      38.71 |            39.8 |    1    | Aug 10/b   |               |            0.7237 |
|     50 | IEP      | Energy                 |           6.67 | $5.0B        |          -2.2  |           1.07 |      0.949 |      -15.14 |          -6.19 |     2.078 |      35.3  |            79.9 |    1    | Aug 05/b   |               |            0.7234 |
</screener_watchlist>

**Screener Integration:**
- Every candidate has already passed the screener's hard gates: prohibited businesses, deal-pinned, >40% above the 50-day or >20% above the 20-day SMA, days 1-3 of a >10% breakout, post-earnings jump, shrinking revenue (Sales Q/Q < 0), liquidity. Gates run on Finviz-level data — the PRV gate (browser quote page) still applies to every name.
- `composite_score` (since 2026-09-15) = equal-weight ranks of low volatility, proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio and distance above the 50-day SMA — the six signals that passed the Phase 2 factor study. 20-day momentum is reported but no longer scored (no measurable ranking skill). The measured edge is small and mostly defensive, so rank is sourcing, never conviction.
- `review_flag` = industry that mixes prohibited and permitted businesses: read what the company does before any research.
- Evaluate AT LEAST the top 5 screener candidates before selecting (discover via WebSearch, verify on the browser quote page).
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-10-05">
<holding ticker="CON" shares="3" avg_cost="35.31" current_price="35.71" stop_loss="33.19" stop_limit="33.04" />
</holdings>

<position_limits>
| Ticker | Sector                 | Sessions Held | 60-Session Review |
|--------|------------------------|---------------|-------------------|
| CON    | Healthcare             |            10 | not yet           |
  <sector_counts>Healthcare: 1</sector_counts>
  <sector_cap_status>OK (cap 3 per sector)</sector_cap_status>
  <driver_cap>Max 2 positions may share a primary thesis driver. Name each holding's primary driver in this report — it cannot be derived from data and an unnamed driver is a rule violation (portfolio_rules.md).</driver_cap>
</position_limits>

<holding_review>
| Ticker | Close | Move (×ATR) | Sector | vs sector | Volume (×20d) | Stop room (×ATR) | Held | Est. next earnings | Review |
|--------|-------|-------------|--------|-----------|---------------|------------------|------|--------------------|--------|
| CON    | $35.65 | +0.68 | XLV -0.01% | +1.96pp | 1.4 | 2.50 | 10 | ~2026-11-05 (24s, est.) | **FULL** |
  <full_review ticker="CON">stop raise qualifies (to ~$33.68)</full_review>
  <rule>FULL review (portfolio_rules.md → Daily monitoring by exception) when a holding moved ≥1.5×ATR, traded ≥3× its average volume, has its stop within 1×ATR, has a qualifying stop raise, may report earnings within ~15 sessions, or was bought ≤3 sessions ago — or when the user asks ("full review TICKER"). Otherwise ONE LINE. Every holding still gets the live news-feed check; news the script cannot see upgrades a LINE to FULL. **Sector** is the holding's sector proxy ETF and its move for the same session; **vs sector** is the holding's move minus the proxy's, in percentage points — the first test of whether a move is idiosyncratic. Reported, not gated.</rule>
</holding_review>

<research_trigger>
  <cadence>trigger-based since 2026-09-17 (the weekly SCREEN is unchanged)</cadence>
  <funnel>buys sought: 3; stage-1 quick checks: 15 (extend to watchlist_extended.csv if the top 50 cannot supply them)</funnel>
  <blackout>re-entry banned (portfolio_rules.md: 10 sessions after a stop-out) -- ATRC, 9 sessions left; HOPE, 7 sessions left. Kill these at stage 1; do not spend a quick check on them.</blackout>
  <week_number>56</week_number>
  <last_report>2026-09-28 (4 sessions ago)</last_report>
  <status>DUE</status>
  <reason>free slot (1/5) with 70% deployable</reason>
  <reason>deployable cash 70% >= 25%</reason>
  <regime>RISK-OFF now; RISK-OFF at the last report (from regime_history.csv)</regime>
  <if_not_due>Produce a short monitoring note instead: stops, any position nearing 60 sessions, and the breaker line. Do not re-underwrite theses that nothing has changed for.</if_not_due>
</research_trigger>

<last_analyst_thesis>
# Week 55 — Thesis Summary (2026-09-28)

**Date:** 2026-09-28 | **Week 55, indefinite phase** | **Regime:** RISK-OFF (IWM −4.09% vs its 50-day, 19 sessions)
Equity $724.95 · Cash $237.33 (32.7%) · **Gap since re-base −2.27%** · Gap since inception −6.72% · Drawdown −1.60%
**No buy this week** — 1 of 100 screened names cleared the RISK-OFF defensive profile, and it was the name already passed on its merits.

---

**ATRC (AtriCure) — HOLD | Conviction 4/5**
+70.2% at $58.38, **54 sessions** held, re-underwrite due ~2026-10-05. Friday's −3.93% reversal from
a 52-week high survived all five unexplained-move checks: no wire news, no analyst action, a CTO
Form 4 that is 0.28% of the session's volume and cannot explain it, the index add complete on 9/21,
and a sector that rose. **Leading explanation: post-inclusion giveback**, which this book's own rule
predicts. **Driver:** U.S. Afib procedure volumes, unchanged; FY26 revenue consensus $606M (+13.4%)
and EPS $0.28 against −$0.11 last year. The dispersion matters more than the average: median target
**$55**, the four newest notes at **$55–$65**, and a stale **$36** dragging the consensus to $53.33.
**Stop $55.27 / $55.12, 1.35×ATR, and it cannot be lowered** — no raise qualifies either, since the
2.0×ATR candidate sits below it. **What would change the view:** a guidance cut, an AtriClip
competitive read-out, or the re-underwrite — which needs the rules question above settled first,
because a literal reading of "would it be bought today" fails a +70% winner on entry gates written
for new capital.

**CON (Concentra) — HOLD | Conviction 3/5**
3 shares at $35.01, −0.8% from a $35.31 entry, five sessions. **Driver:** U.S. hiring, supported
this week by a flash composite PMI of 58.4, the strongest since 2021. Stop $33.19 / $33.04 at
**1.94×ATR** — the best-placed stop in the book after last week's correction, sitting below the
10-session low and the 50-day SMA. Strong Buy, $41.00 target; earnings 2026-11-05.
**What would change the view:** two consecutive weak payroll prints, or visit volumes lagging
headline hiring.

**HOPE (Hope Bancorp) — HOLD | Conviction 3/5 (driver under watch)**
15 shares at $13.83, −0.9% from entry, five sessions. **Driver has turned against the thesis:** the
10-year spiked above 5.12% on 9/23, the highest since 2007, with the market now pricing more hikes
— and the buy case was funding costs continuing to fall as CDs reprice. One week of repricing is not
the three-week reversal that invalidates a driver, so the position stands. Stop $13.48 / $13.38 at
1.38×ATR, below the 10-session low, restoration unused. The MANUBANK Commercial Banking Unit close
is still due early Q4. **What would change the view:** a hawkish October FOMC, or deposit costs
turning back up at the 10/27 print.

**Portfolio.** Three positions, 32.7% cash, and **no buy for the first time in this phase** — the
right outcome rather than a disappointing one: one name in a hundred cleared the RISK-OFF defensive
profile and it was the name already rejected on its merits. The gap is −2.27% and moves almost
entirely with ATRC, which is both the book's best asset and its concentration risk. **Before the
next report:** settle the re-underwrite reading before 2026-10-05; watch whether ATRC's
post-inclusion giveback continues toward a stop that cannot be lowered; and watch the October FOMC
against HOPE's driver. The next screen runs 2026-10-03, and the mid-October earnings squeeze the
funnel rules warned about begins the week after — expect thinner lists, not richer ones.
Re-entry bans: PAR expired ~9/29, VTS ~9/30.

---

*Week 55 Summary. Generated 2026-09-28 by Claude Code.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
<!-- No trades this week -->
</recent_trades>

<execution_requests>
<session_directives>
- Research focus: Wide net across the screener for the 3 buys sought. Plus: if the RISK-OFF defensive profile again yields nothing, quantify the constraint - how many of the 100 screened names clear each of the three thresholds separately and together - so the question of whether the gate can ever be satisfied has evidence rather than one week's anecdote.
- Fixed by portfolio_rules.md, not chosen weekly: holding horizon 40–60 sessions; catalyst window 90 days, non-binary only; 2% risk per trade; 5–6 position ceiling (about 4 fit at current sizing); risk posture set by the regime filter and the drawdown circuit breaker.
- Research funnel (analysis-workflow.md Step 2): stage 1 = quick quote-page checks, ~5 per buy sought (the count is in <research_trigger>), spread across ranks, sectors and size, extending to watchlist_extended.csv (ranks 51–100) if the top 50 cannot supply them; stage 2 = full research on the survivors. Log every name at both stages with log_research.py.
</session_directives>

Using the rules, safeguards, and portfolio context above, execute the deep research window now.

Write the report in the six-section format in <output_format> above: scoreboard, deployment funnel, exact orders, holdings by exception, risk checks, thesis summary, then sources.

**IMPORTANT:** Do NOT load or follow the weekly-portfolio-report skill — retired 2026-09-19; <output_format> replaces it. Save the report files as set out in .claude/rules/analysis-workflow.md (Week N Full.md, Week N Summary.md = section 6 only, and the PDF) — not to /mnt/user-data/outputs.

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