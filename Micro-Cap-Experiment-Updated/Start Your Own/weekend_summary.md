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
<date>Monday, September 28, 2026</date>
<week_number>55 (ongoing live process)</week_number>
<experiment_runway>ongoing — no end date set</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| ATRC   |   58.38 | +0.00% |   1,591,200 | Holding    |
| CON    |   35.01 | -0.00% |     571,400 | Holding    |
| HOPE   |   13.83 | -0.00% |     858,700 | Holding    |
| IWO    |  360.15 | +0.00% |     578,700 | Benchmark  |
| XBI    |  155.03 | +0.00% |   8,234,900 | Benchmark  |
| SPY    |  771.35 | +0.00% |  36,629,600 | Benchmark  |
| IWM    |  281.97 | +0.00% |  22,604,100 | Benchmark  |
| QQQ    |  744.50 | +0.00% |  30,249,200 | Benchmark  |
| TLT    |   79.32 | +0.00% |  62,513,500 | Macro      |
| HYG    |   77.86 | +0.00% |  58,325,400 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Measured From (close)         | 2026-09-11 | all metrics below       |
| Max Drawdown                  |    -1.77% | on 2026-09-21           |
| Max Drawdown (inj-neutral)    |    -1.77% | on 2026-09-21           |
| Current Drawdown (from peak)  |    -1.60% | clear of breaker        |
| Sharpe Ratio (annualized)     |       N/A |                         |
| Sortino Ratio (annualized)    |       N/A |                         |
| Beta (daily) vs ^GSPC         |       N/A |                         |
| Alpha (annualized) vs ^GSPC   |       N/A |                         |
| R²                            |       N/A |                         |
| Time-Weighted Return (cum)    |    -1.17% | injection-neutral       |
| S&P 500 Return (cum)          |    +1.13% | same window             |
| TWR Alpha (cum)               |    -2.30% | TWR minus S&P           |
</risk_metrics>
</market_data>

<market_regime>
  <date>2026-09-25</date>
  <iwm_close>281.97</iwm_close>
  <sma50>294.00</sma50>
  <pct_vs_sma50>-4.09%</pct_vs_sma50>
  <regime>RISK-OFF</regime>
  <since>RISK-OFF since 2026-08-31 (19 sessions)</since>
  <rule>RISK-OFF after a close more than 1% below the 50-day SMA; RISK-ON after a close more than 1% above it; held in between.</rule>
  <source>trading_script.py — IWM unadjusted daily closes, 50-session simple average. Do not look this up elsewhere.</source>
</market_regime>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $724.95 |
| S&P Equivalent      |   $741.79 |
| Benchmark Base      | 2026-09-11 |
| Cash Balance        |   $237.33 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-09-28" candidates="50">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   volume_ratio |   vol_5_50 |   near_high |   pct_vs_sma50 |   atr_pct |   sales_qq |   target_upside |   recom | earnings   | review_flag   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|---------------:|-----------:|------------:|---------------:|----------:|-----------:|----------------:|--------:|:-----------|:--------------|------------------:|
|      1 | ELME     | Real Estate            |           1.73 | $155M        |           2.98 |           1.04 |      1.273 |       -1.7  |           5.75 |     2.023 |       1.72 |           nan   |    3    | -          |               |            0.8719 |
|      2 | CFFN     | Financial              |           8.68 | $1.1B        |           0.23 |           1.34 |      1.084 |       -6.26 |          -0.78 |     1.835 |       6.5  |             9.4 |    2.33 | Jul 29/b   |               |            0.8718 |
|      3 | CVBF     | Financial              |          22.53 | $4.0B        |           1.53 |           1.37 |      1.046 |       -3.76 |          -0.18 |     1.839 |      37.86 |            15.4 |    1.83 | Jul 22/a   |               |            0.8699 |
|      4 | FFBC     | Financial              |          32.4  | $3.4B        |          -1.28 |           1.91 |      1.242 |      -10.62 |          -2.53 |     1.865 |      12.98 |            16.9 |    2    | Jul 21/a   |               |            0.8661 |
|      5 | FCF      | Financial              |          20.91 | $2.1B        |           0.29 |           1.03 |      1.246 |       -6.4  |          -1.28 |     1.65  |       0.6  |            15.2 |    1.83 | Jul 28/a   |               |            0.8566 |
|      6 | LILAK    | Communication Services |           8.49 | $1.7B        |           0.12 |           1.7  |      1.089 |       -4.77 |           2.02 |     3.213 |       1.46 |             7.7 |    2.67 | Aug 05/a   |               |            0.8563 |
|      7 | FULC     | Healthcare             |           3.74 | $285M        |          -2.86 |           1.71 |      0.946 |       -5.08 |          -1.08 |     1.614 |     nan    |             7   |    3    | Aug 05     |               |            0.8563 |
|      8 | VFF      | Consumer Defensive     |           3.17 | $388M        |           8.93 |           1.45 |      1.039 |       -0.94 |          24.98 |     3.74  |       6.81 |            83.9 |    1.25 | Aug 10/b   |               |            0.8431 |
|      9 | SIBN     | Healthcare             |          18.59 | $834M        |          -3.23 |           1.12 |      1.112 |       -8.42 |           0.64 |     3.285 |      15.18 |            34.5 |    1    | Aug 03/a   |               |            0.8382 |
|     10 | PSEC     | Financial              |           2.19 | $1.1B        |          -2.67 |           1.26 |      1.073 |       -7.59 |          -1.62 |     2.087 |     208.67 |            -8.7 |    5    | Aug 20/a   |               |            0.8325 |
|     11 | AVO      | Consumer Defensive     |          13.14 | $1.1B        |           2.9  |           1.08 |      1.235 |       -7.59 |           2.3  |     4.067 |      25.8  |            25.6 |    1    | Sep 08/a   |               |            0.832  |
|     12 | FULT     | Financial              |          23.23 | $4.4B        |          -1.9  |           1.32 |      1.008 |       -7.93 |          -3.04 |     1.814 |       7.58 |            10.6 |    2.5  | Jul 22/a   |               |            0.8298 |
|     13 | WTTR     | Energy                 |          19.84 | $2.6B        |           2.9  |           2.72 |      1.364 |      -12.02 |          -0.97 |     4.397 |       8.67 |            23.5 |    1.17 | Aug 04/a   |               |            0.8274 |
|     14 | LTC      | Real Estate            |          42.94 | $2.3B        |           5.71 |           0.88 |      1.094 |       -1.51 |           4.25 |     2.266 |      63.83 |             5.4 |    2.11 | Aug 05/a   |               |            0.8239 |
|     15 | MQ       | Technology             |          16.96 | $1.8B        |           2.98 |           1.3  |      1.055 |       -9.26 |           2.66 |     2.771 |      17.02 |            18.6 |    2.8  | Aug 04/a   |               |            0.8088 |
|     16 | CYRX     | Industrials            |          17.41 | $882M        |           8.2  |           1.29 |      1.159 |       -3.17 |           9.35 |     3.401 |       7.74 |            11   |    1.22 | Aug 06/a   |               |            0.8063 |
|     17 | BXDC     | Real Estate            |          19.08 | $1.9B        |          -4.36 |           1.13 |      1.483 |      -14.25 |          -5.31 |     3.088 |     nan    |            23.6 |    2.17 | Aug 04/b   |               |            0.8013 |
|     18 | PUBM     | Technology             |          18.6  | $860M        |          10.58 |           1.17 |      1.247 |       -3.07 |          17.7  |     3.678 |      10.55 |            12.5 |    1.5  | Aug 06/a   |               |            0.7988 |
|     19 | CNK      | Communication Services |          37.77 | $4.4B        |           4.95 |           0.92 |      1.084 |       -3.1  |           5.72 |     3.018 |      15.51 |             6.5 |    1.77 | Jul 30/b   |               |            0.7975 |
|     20 | KRP      | Energy                 |          14.57 | $1.9B        |          -2.28 |           0.82 |      1.224 |       -6.91 |          -2.27 |     1.777 |      37.76 |            38.4 |    1.57 | Aug 07/b   |               |            0.797  |
|     21 | LFST     | Healthcare             |          11.84 | $4.5B        |          -3.82 |           1.32 |      1.169 |      -12.68 |          -1.08 |     4.063 |      26.08 |            19.9 |    1.45 | Aug 06/b   |               |            0.7886 |
|     22 | ADMA     | Healthcare             |           9.38 | $2.1B        |           0.43 |           1.26 |      1.116 |       -9.37 |           1.23 |     3.697 |       1.98 |            99   |    1.33 | Aug 05/a   |               |            0.7839 |
|     23 | ORIC     | Healthcare             |          13.16 | $1.4B        |          -0.6  |           1.75 |      1.105 |      -10.35 |           3.32 |     5.075 |     nan    |            60.2 |    1.13 | Aug 03/a   |               |            0.7836 |
|     24 | LADR     | Real Estate            |           9.08 | $1.2B        |          -7.54 |           1.2  |      1.226 |      -14.26 |          -6.55 |     1.88  |      15.51 |            32.2 |    1.29 | Jul 23/b   |               |            0.7807 |
|     25 | CUZ      | Real Estate            |          28.65 | $4.7B        |          -2.68 |           0.94 |      1.158 |      -13.05 |          -3.63 |     2.097 |      11.83 |            17.1 |    1.36 | Jul 30/a   |               |            0.7802 |
|     26 | EFC      | Real Estate            |          12.36 | $1.6B        |          -8.92 |           1.2  |      1.202 |      -10.69 |          -6.81 |     1.786 |      31.72 |            20.2 |    1.57 | Aug 06/a   |               |            0.7799 |
|     27 | INVA     | Healthcare             |          20.74 | $1.5B        |          -1.1  |           0.62 |      1.933 |       -8.75 |          -1.53 |     2.473 |      18.61 |            68.8 |    1.8  | Aug 05/a   |               |            0.7736 |
|     28 | PRDO     | Consumer Defensive     |          30.49 | $1.9B        |          -9.17 |           1.29 |      1.204 |      -17.77 |          -5.83 |     3.425 |       1.8  |            44.3 |    1    | Aug 06/a   |               |            0.7724 |
|     29 | HNI      | Consumer Cyclical      |          47.21 | $3.4B        |          -3.91 |           0.88 |      1     |       -6.35 |           0.82 |     2.726 |     120.72 |            47.7 |    1    | Jul 30/b   |               |            0.7641 |
|     30 | ACCO     | Industrials            |           4.31 | $406M        |           0.47 |           0.99 |      0.832 |       -5.48 |           1.36 |     2.867 |       5.14 |            85.6 |    1    | Jul 30/a   |               |            0.7592 |
|     31 | GCT      | Technology             |          53.98 | $1.9B        |           1.95 |           0.86 |      1.239 |       -4.07 |          11.04 |     4.773 |      27.6  |            21.3 |    1    | Aug 06/b   |               |            0.7562 |
|     32 | EXPO     | Industrials            |          64.18 | $3.0B        |         -10.28 |           1.16 |      0.957 |      -10.76 |          -4.31 |     2.672 |      20.89 |            30.9 |    1.6  | Jul 30/a   |               |            0.7507 |
|     33 | CLMT     | Basic Materials        |          53.65 | $4.8B        |          12.36 |           1.1  |      1.261 |      -10.39 |           9.73 |     4.579 |      40.77 |            -0.5 |    2.33 | Aug 07/b   |               |            0.7506 |
|     34 | AVPT     | Technology             |          13    | $2.7B        |          -7.14 |           1.67 |      0.88  |       -9.63 |          -1.35 |     3.371 |      22.03 |            29.2 |    1.57 | Aug 06/a   |               |            0.7482 |
|     35 | SBH      | Consumer Cyclical      |          17.18 | $1.6B        |           3.81 |           0.8  |      0.914 |       -0.64 |           6.91 |     3.235 |       0.23 |            -1   |    2.4  | Aug 03/b   |               |            0.7446 |
|     36 | KBR      | Industrials            |          34.54 | $4.4B        |          -9.18 |           1.08 |      1.019 |      -11.32 |          -6.18 |     3.054 |       1.64 |            27.9 |    2.22 | Jul 30/b   |               |            0.7427 |
|     37 | SPSC     | Technology             |          81.3  | $2.9B        |          -4.26 |           1.32 |      0.791 |       -9.1  |           6.27 |     4.217 |       5.56 |            -7.2 |    2.79 | Jul 30/a   |               |            0.7384 |
|     38 | MH       | Consumer Defensive     |          12.73 | $2.4B        |          -3.71 |           1.23 |      1.103 |      -10.22 |           4.36 |     4.625 |       2.65 |            40.8 |    1.23 | Aug 13/b   |               |            0.7378 |
|     39 | CRC      | Energy                 |          52.51 | $4.7B        |           0.54 |           1.01 |      1.019 |       -9.48 |          -1.4  |     3.452 |      33.01 |            44.2 |    1.38 | Aug 10/b   |               |            0.7343 |
|     40 | VIA      | Technology             |          29.41 | $2.4B        |           2.94 |           0.98 |      1.09  |       -4.33 |          19.05 |     4.819 |      26.67 |            19   |    1.1  | Aug 06/b   |               |            0.733  |
|     41 | AVA      | Utilities              |          35.11 | $2.9B        |          -6.77 |           1.18 |      1.039 |      -18.54 |          -8.71 |     1.646 |       0.49 |            14.6 |    3    | Aug 03/b   |               |            0.7321 |
|     42 | MDU      | Utilities              |          18.76 | $4.0B        |          -5.2  |           0.92 |      1.116 |      -13.19 |          -5.9  |     2.037 |       6.87 |            26.4 |    1.78 | Aug 06/b   |               |            0.7288 |
|     43 | FUL      | Basic Materials        |          50.06 | $2.7B        |         -11.16 |           1.4  |      1.872 |      -22.07 |         -10.22 |     3.694 |       5.17 |            39.3 |    1.67 | Sep 23/a   |               |            0.7263 |
|     44 | TALO     | Energy                 |          16.47 | $2.7B        |          -1.2  |           1.09 |      0.981 |      -12.04 |           2.24 |     4.189 |      56.53 |            22.5 |    1.69 | Aug 04/a   |               |            0.7239 |
|     45 | RES      | Energy                 |           5.84 | $1.3B        |          -8.75 |           1.15 |      1.479 |      -13.22 |          -3.78 |     3.755 |       9.52 |             9.6 |    3.5  | Jul 30/b   |               |            0.723  |
|     46 | SHEN     | Communication Services |          11.76 | $660M        |          -5.62 |           1.07 |      1.265 |      -20.65 |          -4.01 |     4.063 |       5.53 |           133.8 |    1.5  | Jul 29/b   |               |            0.7131 |
|     47 | TREX     | Industrials            |          44.79 | $4.5B        |          -2.14 |           1.28 |      0.835 |      -12.11 |          -1.44 |     3.028 |       7.79 |            22.1 |    2.23 | Aug 04/b   |               |            0.7075 |
|     48 | EGY      | Energy                 |           5.82 | $615M        |           0.87 |           0.91 |      0.986 |      -11.68 |           1.08 |     3.662 |      39.5  |            64.8 |    1    | Aug 06/a   |               |            0.7047 |
|     49 | ASH      | Basic Materials        |          69.06 | $3.2B        |          -7.14 |           0.81 |      0.983 |      -11.74 |          -3.59 |     2.63  |       7.34 |            16.1 |    1.64 | Jul 28/a   |               |            0.704  |
|     50 | LILA     | Communication Services |           8.48 | $1.7B        |          -0.93 |           0.83 |      0.763 |       -5.62 |           1.05 |     2.831 |       1.46 |             8.1 |    2.6  | Aug 05/a   |               |            0.7024 |
</screener_watchlist>

**Screener Integration:**
- Every candidate has already passed the screener's hard gates: prohibited businesses, deal-pinned, >40% above the 50-day or >20% above the 20-day SMA, days 1-3 of a >10% breakout, post-earnings jump, shrinking revenue (Sales Q/Q < 0), liquidity. Gates run on Finviz-level data — the PRV gate (browser quote page) still applies to every name.
- `composite_score` (since 2026-09-15) = equal-weight ranks of low volatility, proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio and distance above the 50-day SMA — the six signals that passed the Phase 2 factor study. 20-day momentum is reported but no longer scored (no measurable ranking skill). The measured edge is small and mostly defensive, so rank is sourcing, never conviction.
- `review_flag` = industry that mixes prohibited and permitted businesses: read what the company does before any research.
- Evaluate AT LEAST the top 5 screener candidates before selecting (discover via WebSearch, verify on the browser quote page).
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-09-25">
<holding ticker="ATRC" shares="3" avg_cost="34.30" current_price="58.38" stop_loss="55.27" stop_limit="55.12" />
<holding ticker="CON" shares="3" avg_cost="35.31" current_price="35.01" stop_loss="33.19" stop_limit="33.04" />
<holding ticker="HOPE" shares="15" avg_cost="13.96" current_price="13.83" stop_loss="13.48" stop_limit="13.38" />
</holdings>

<position_limits>
| Ticker | Sector                 | Sessions Held | 60-Session Review |
|--------|------------------------|---------------|-------------------|
| ATRC   | Healthcare             |            54 | in 6              |
| CON    | Healthcare             |             5 | not yet           |
| HOPE   | Financial              |             5 | not yet           |
  <sector_counts>Financial: 1, Healthcare: 2</sector_counts>
  <sector_cap_status>OK (cap 3 per sector)</sector_cap_status>
  <driver_cap>Max 2 positions may share a primary thesis driver. Name each holding's primary driver in this report — it cannot be derived from data and an unnamed driver is a rule violation (portfolio_rules.md).</driver_cap>
</position_limits>

<holding_review>
| Ticker | Close | Move (×ATR) | Volume (×20d) | Stop room (×ATR) | Held | Est. next earnings | Review |
|--------|-------|-------------|---------------|------------------|------|--------------------|--------|
| ATRC   | $58.38 | -1.11 | 1.1 | 1.35 | 54 | ~2026-10-22 (19s, est.) | **LINE** |
| CON    | $35.01 | +0.42 | 0.6 | 1.94 | 5 | ~2026-11-05 (29s, est.) | **LINE** |
| HOPE   | $13.83 | +0.43 | 1.0 | 1.38 | 5 | ~2026-10-26 (21s, est.) | **LINE** |
  <rule>FULL review (portfolio_rules.md → Daily monitoring by exception) when a holding moved ≥1.5×ATR, traded ≥3× its average volume, has its stop within 1×ATR, has a qualifying stop raise, may report earnings within ~15 sessions, or was bought ≤3 sessions ago — or when the user asks ("full review TICKER"). Otherwise ONE LINE. Every holding still gets the live news-feed check; news the script cannot see upgrades a LINE to FULL.</rule>
</holding_review>

<research_trigger>
  <cadence>trigger-based since 2026-09-17 (the weekly SCREEN is unchanged)</cadence>
  <funnel>buys sought: 1; stage-1 quick checks: 10 (extend to watchlist_extended.csv if the top 50 cannot supply them)</funnel>
  <week_number>55</week_number>
  <last_report>2026-09-20 (5 sessions ago)</last_report>
  <status>DUE</status>
  <reason>free slot (3/5) with 18% deployable</reason>
  <regime>RISK-OFF now; RISK-OFF at the last report (from regime_history.csv)</regime>
  <if_not_due>Produce a short monitoring note instead: stops, any position nearing 60 sessions, and the breaker line. Do not re-underwrite theses that nothing has changed for.</if_not_due>
</research_trigger>

<last_analyst_thesis>
# Week 54 — Thesis Summary (2026-09-20)

**Date:** 2026-09-20 | **Week 54, indefinite phase** | **Regime:** RISK-OFF (IWM −3.74% vs its 50-day)
Equity $727.32 · Cash $553.02 before trades, $201.74 after (27.7%) · **Gap since re-base −0.76%** · Gap since inception −5.28% · Current drawdown −1.28%

---

**ATRC (AtriCure) — HOLD | Conviction 4/5 (from 5/5)**
+69.4% at $58.10 and 49 sessions held. Joins the **S&P SmallCap 600 before Monday 2026-09-21's
open** as a straight addition — confirmed from the S&P DJI release of 2026-09-04 — which is what
Friday's 8.67M shares were; the structural bid ends there. **Driver:** U.S. Afib procedure volumes
and AtriClip/cryo adoption, supported by Q2 revenue +12.8% and raised FY26 guidance.
**Stop $55.27 / $55.12, 1.28×ATR of room, and it cannot be lowered** — restoration was spent on
8/31 and the 2.0×ATR raise target ($53.68) sits below it, so no raise qualifies either. No trim:
the case for one rests on price level and a completed flow event, not on new negative information,
and the rules reserve discretionary selling for a verified thesis break. **What would change the
view:** a guidance cut, a competitive read-out against AtriClip, or the 60-session re-underwrite
due around 2026-10-05 — whichever comes first.

**CON (Concentra Group Holdings) — INITIATE | Conviction 3/5**
4 shares at $35.47, stop $33.92 / $33.77 (1.75×ATR), $141.88 = 19.5% of equity, risk 0.85%.
Occupational health with revenue +14.0%, EPS +29.4%, 20.6× forward, beta 0.63 and a Strong Buy with
a $41.00 target. **Driver:** U.S. hiring — August payrolls +162,000 against +53,000 expected
(2026-09-04), with June and July revised up; improving over four weeks. **What would change the
view:** two consecutive weak payroll prints, or any sign that employer visit volumes are lagging
headline hiring.

**HOPE (Hope Bancorp) — INITIATE | Conviction 3/5**
15 shares at $13.96, stop $13.48 / $13.38 (1.75×ATR), $209.40 = 28.8% of equity, risk 0.99%.
Revenue +31.1%, EPS +173.8% off a depressed 2025 base, 10.1× forward, 4.0% yield, an $15.50 target.
**Driver:** the short-rate path through NIM — 2.96% in Q2, +6bp sequentially, deposit costs
−6bp — now partly offset by the FOMC's 25bp **hike** on 2026-09-16 to 3.75%–4.00%. **Catalyst:**
the SMBC MANUBANK Commercial Banking Unit acquisition, fully approved on 2026-09-02, closing early
Q4. **What would change the view:** a second hike with deposit costs turning back up, or the
MANUBANK close slipping out of Q4.

**Portfolio.** Three positions and 27.7% cash after the orders, which is the RISK-OFF rules working
rather than a market call: the half risk budget plus 1.75×ATR stops sizes positions at roughly
$200, and $443.92 of deployable cash bought two of the three buys the trigger asked for. Both
entries are deliberately low-beta (0.63 and 0.81) and both clear the defensive profile on all three
thresholds — the book is not chasing the −0.76% gap, and after nine sessions there is nothing to
chase. **Before the next report:** the pre-open check on both limits Monday morning; ATRC's first
week without an index bid, and its 60-session re-underwrite around 2026-10-05; HOPE's MANUBANK
close in early Q4; and the October FOMC, which is now a live risk to the second position's driver.
Re-entry bans: PAR until ~2026-09-29, VTS until ~2026-09-30.

---

*Week 54 Summary. Generated 2026-09-20 by Claude Code.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price
2026-09-21,CON,4.0,35.31,141.24,0.0,MANUAL BUY LIMIT - Filled,,
2026-09-21,HOPE,15.0,13.96,209.4,0.0,MANUAL BUY LIMIT - Filled,,
2026-09-22,CON,,,35.31,-0.3599999999999994,MANUAL SELL MARKET - Filled,1.0,34.95
</recent_trades>

<execution_requests>
<session_directives>
- Research focus: Wide net across the screener for the single buy sought. Plus ATRC: explain Friday 9/25's -3.93% reversal from a new high (no wire news, consensus PT unchanged, sector does not cover it) and prepare the 60-session re-underwrite, due in 6 sessions.
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