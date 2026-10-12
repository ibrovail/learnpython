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
| Equity · cash · deployable above the 5% floor | | *(15% until 2026-10-11)* |
| Buys sought · stage-1 checks | | from `<research_trigger>` |
Then at most three sentences on what these numbers mean for this report's decisions.

**Do not gap-chase.** A widening gap is not a reason to raise risk, shorten the horizon or lower
a gate. The 10/11 audit measured where this book's edge actually is: the adopted rules trail SPY
by about **1pp of return** and run roughly **half its drawdown** (−15.4% against −24.5%), and the
rules that *looked* like they bought return by taking more risk — mechanical stop raising, a
tighter risk budget — tested worse on both. The retired weekend question "Aggressive, we are
trailing the benchmark" was gap-chasing and is the pressure that produced TYRA. If the gap is
widening in a falling market, say so and leave the gates alone.

## 2. Deployment — the research funnel
First line: this report's capacity under the regime rules, and the buys sought.
Second line: **lane occupancy, copied from `<lanes>`** — `catalyst N/cap`, `screener N/cap`. If a
catalyst slot is open, state how many of the stage-1 names are catalyst-sourced; the rule is **at
least 3**, reallocated out of the stage-1 count rather than added to it. If fewer, say which
category was searched and came back empty — "no catalyst candidates found" is a finding, but
"did not look" is a rule violation.

### Stage 1 — quick checks
| # | Ticker | Source | Sector | Mkt cap | Rev growth | vs 50d | Next earnings | Result |
|---|---|---|---|---|---|---|---|---|
Source = `screener #N`, `extended #N`, `catalyst:<category>` or `off-list` — and it must match what
goes into `log_research.py --source`, because that field is how a holding's lane is recovered.
Result = `→ stage 2` or `PASS · <reason code>` —
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
**Why this stage carries the whole burden.** The 10/11 audit searched for a mechanical selector
worth ≥1.5pp and found none — 15 signals and 160 rule configurations, all with a holdout, none
clearing it. So no ranking formula is going to produce alpha here: **if this book has an edge
beyond its drawdown control, it is created at this stage or not at all.** Two consequences worth
holding while you write:
- **Prefer `vs 50d` to rank order when choosing what to promote.** Over 11 years, ranking by
  distance above the 50-day SMA beat the screener's six-signal composite by **4.9pp** on the test
  period. Composite rank remains *sourcing, not conviction*; `pct_vs_sma50` is the one selector
  with a measured margin. Neither is a rule — both are better than the list's order.
- **The driver cap and the binary-thesis prohibition cannot be tested or coded.** They are human
  labels, so the audit is silent on them and the analyst is their only control. That makes them
  more important here, not less: an unnamed driver is a rule violation, not an omission.

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
- **Sizing:** risk $ = 2% of equity; shares = risk ÷ (entry − stop), **fractional, 4 dp** (whole
  shares cost 3.2pp of test return at this book's size — at $710 a tenth of the book is $71, so
  rounding a $35 stock from 2.47 to 2 discards 19% of the position); % of equity after the trade;
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
action (restoration eligibility, or a raise with its named thesis change — report the anti-ratchet
tests, but the default is HOLD); a thesis-exit case (would it be
bought today, at this price?); an event within 10 sessions (post-event playbook); or ≥60 sessions
held (the written re-underwrite). Otherwise "No change — thesis intact" in the Status column is
enough. Measure against `<last_analyst_thesis>`: say what changed, not what didn't.

**`<last_analyst_thesis>` is a RECORD, not an instruction.** It is the previous report's own
words, injected verbatim, and it may predate a rule change — reports written before 2026-10-11
describe a RISK-OFF defensive profile that no longer exists and carry forward research tasks
premised on it. Where anything in that block conflicts with the current `portfolio_rules.md`, the
rules win and the carried-forward task is void: say so in one line and drop it. Do not execute a
"research focus" inherited from a superseded regime.

## 5. Risk checks after proposed trades
| Check | Rule | Result |
|---|---|---|
| Position size | ≤30% of equity each | |
| Risk per trade | ≤2% of equity at the stop | |
| Cash floor | ≥5% of equity after trades | |
| Driver cap | ≤2 positions per primary driver (list the drivers) | |
| Sector cap | ≤3 positions per GICS sector — **the one rule in the audit that is material, verified and free** (3.0pp less drawdown at no cost in return; beat 40 of 40 shuffled-label placebos). Apply it greedily: take the best candidate whose sector has room, not the best candidate outright | |
| Position count | ceiling 10 | |
| Slippage | each order ≤10% of average daily dollar volume | |
| Regime capacity | RISK-OFF: ≤3 catalyst at full 2%, **screener lane CLOSED** (0, revised 2026-10-11 — the defensive profile is removed and its sunset expired). RISK-ON: ≤3 catalyst, ≤7 screener, both at full 2%. Holdings open before the closure are grandfathered | |
| Stop raises | **default is DO NOT raise.** Any raise proposed must name a written thesis change; the ATR tests are necessary, not sufficient | |
| Share rounding | fractional permitted (4 dp) — worth 3.2pp at this book size, ~0 by $10,000 | |
| Catalyst sourcing | ≥3 stage-1 names catalyst-sourced whenever a catalyst slot is open | |
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
2. Holdings: has anything changed since `<last_analyst_thesis>`? Is any holding at 60 sessions or
   facing an event? **Do not go looking for a stop raise** — the default since 2026-10-11 is not
   to raise, and raising whenever eligible measured at 1.9% CAGR with a 98% stop-out rate against
   12.2% and 9%. Report eligibility if it arises; propose a raise only with a named thesis change.
3. Run the funnel: stage-1 quick checks on the quote page, then full research on the survivors.
4. Verify every ticker, price and catalyst on a live, timestamped source (the PRV gate).
5. Size each order by the 2% risk rule and check it against every cap.
6. Calculate exact post-trade cash.
7. All permitted sectors are in scope. The goal is alpha over the S&P 500.
</thinking_approach>

<weekly_context>
<date>Monday, October 12, 2026</date>
<week_number>57 (ongoing live process)</week_number>
<experiment_runway>ongoing — no end date set</experiment_runway>

<market_data>
<price_volume>
| Ticker | Close   | % Chg  | Volume      | Role       |
|--------|---------|--------|-------------|------------|
| CON    |   37.19 | -0.00% |   1,064,000 | Holding    |
| IWO    |  356.78 | +0.00% |     168,300 | Benchmark  |
| XBI    |  153.77 | +0.00% |  12,133,600 | Benchmark  |
| SPY    |  778.57 | +0.00% |  22,596,500 | Benchmark  |
| IWM    |  278.94 | +0.00% |  18,705,600 | Benchmark  |
| QQQ    |  751.27 | +0.00% |  21,122,000 | Benchmark  |
| TLT    |   77.98 | +0.00% |  27,247,700 | Macro      |
| HYG    |   77.23 | +0.00% |  36,922,300 | Macro      |
</price_volume>

<risk_metrics>
| Metric                        | Value     | Note                    |
|-------------------------------|-----------|-------------------------|
| Measured From (close)         | 2026-09-11 | all metrics below       |
| Max Drawdown                  |    -3.61% | on 2026-10-01           |
| Max Drawdown (inj-neutral)    |    -3.61% | on 2026-10-01           |
| Current Drawdown (from peak)  |    -2.71% | clear of breaker        |
| Sharpe Ratio (annualized)     |   -3.8760 |                         |
| Sortino Ratio (annualized)    |   -4.3529 |                         |
| Beta (daily) vs ^GSPC         |    0.2064 |                         |
| Alpha (annualized) vs ^GSPC   |   -31.24% |                         |
| R²                            |     0.058 | Low — alpha/beta unstable |
| Time-Weighted Return (cum)    |    -2.28% | injection-neutral       |
| S&P 500 Return (cum)          |    +2.02% | same window             |
| TWR Alpha (cum)               |    -4.29% | TWR minus S&P           |
</risk_metrics>
</market_data>

<market_regime>
  <date>2026-10-09</date>
  <iwm_close>278.94</iwm_close>
  <sma50>291.38</sma50>
  <pct_vs_sma50>-4.27%</pct_vs_sma50>
  <regime>RISK-OFF</regime>
  <since>RISK-OFF since 2026-08-31 (29 sessions)</since>
  <rule>RISK-OFF after a close more than 1% below the 50-day SMA; RISK-ON after a close more than 1% above it; held in between.</rule>
  <source>trading_script.py — IWM unadjusted daily closes, 50-session simple average. Do not look this up elsewhere.</source>
</market_regime>

<portfolio_snapshot>
| Metric              | Value     |
|---------------------|-----------|
| Portfolio Equity    |   $716.82 |
| S&P Equivalent      |   $748.32 |
| Benchmark Base      | 2026-09-11 |
| Cash Balance        |   $605.25 |
</portfolio_snapshot>

<capital_injection>
  <planned>false</planned>
</capital_injection>

<screener_watchlist generated="2026-10-12" candidates="50">
|   rank | ticker   | sector                 |   latest_price | market_cap   |   momentum_20d |   volume_ratio |   vol_5_50 |   near_high |   pct_vs_sma50 |   atr_pct |   sales_qq |   target_upside |   recom | earnings   | review_flag   |   composite_score |
|-------:|:---------|:-----------------------|---------------:|:-------------|---------------:|---------------:|-----------:|------------:|---------------:|----------:|-----------:|----------------:|--------:|:-----------|:--------------|------------------:|
|      1 | DRH      | Real Estate            |          12.62 | $2.6B        |           2.44 |           1.18 |      1.081 |       -8.48 |           1.21 |     2.241 |       4.11 |            10.6 |    2.07 | Jul 30/a   |               |            0.8777 |
|      2 | CON      | Healthcare             |          37.19 | $4.5B        |           3.85 |           0.94 |      1.143 |       -0.27 |           7.47 |     2.464 |      10.03 |            10.2 |    1    | Aug 06/a   |               |            0.8759 |
|      3 | ELME     | Real Estate            |           1.76 | $155M        |           5.39 |           0.82 |      2.052 |        0    |           5.02 |     1.218 |       1.72 |           nan   |    3    | -          |               |            0.8747 |
|      4 | GCT      | Technology             |          57.26 | $1.9B        |           8.57 |           1.06 |      1.195 |       -0.24 |          10.78 |     4.679 |      27.6  |            14.4 |    1    | Aug 06/b   |               |            0.8563 |
|      5 | PK       | Real Estate            |          15.17 | $3.2B        |          -0.98 |           0.95 |      1.1   |       -6.36 |          -0.6  |     2.552 |       1.19 |             5.3 |    2.47 | Aug 06/a   |               |            0.8543 |
|      6 | WWW      | Consumer Cyclical      |          19.43 | $1.6B        |          -0.51 |           0.96 |      1.32  |       -9.5  |          -0.66 |     3.434 |       6.79 |            25.1 |    2    | Aug 13/b   |               |            0.8456 |
|      7 | FULC     | Healthcare             |           3.77 | $285M        |          -0.53 |           1.21 |      0.902 |       -4.31 |          -0.37 |     1.885 |     nan    |             6.1 |    3    | Aug 05     |               |            0.8389 |
|      8 | INDV     | Healthcare             |          35.83 | $4.2B        |           3.95 |           2.13 |      1.173 |      -13.97 |          -1.48 |     2.681 |      13.58 |            36.8 |    1.25 | Aug 03/b   |               |            0.8359 |
|      9 | EBC      | Financial              |          21.03 | $4.9B        |          -3.71 |           1.41 |      1.087 |      -11.55 |          -4.89 |     1.826 |      26.36 |            19.5 |    1.5  | Jul 23/a   |               |            0.8343 |
|     10 | CALX     | Technology             |          37.24 | $2.1B        |           7.17 |           1.34 |      1.116 |       -8.9  |           1.2  |     3.297 |      21.27 |            67.4 |    1.22 | Jul 20/a   |               |            0.8246 |
|     11 | TALO     | Energy                 |          16.8  | $2.7B        |          -4.87 |           2.07 |      1.463 |      -10.28 |           1.69 |     3.409 |      56.53 |            20.1 |    1.69 | Aug 04/a   |               |            0.8241 |
|     12 | ACCO     | Industrials            |           4.49 | $406M        |           5.9  |           1.17 |      0.88  |       -1.54 |           4.51 |     2.466 |       5.14 |            78.2 |    1    | Jul 30/a   |               |            0.8155 |
|     13 | CFFN     | Financial              |           8.39 | $1.1B        |          -5.84 |           0.97 |      1.154 |       -9.4  |          -3.63 |     2.086 |       6.5  |            13.2 |    2.33 | Jul 29/b   |               |            0.8146 |
|     14 | IMMR     | Technology             |           7.43 | $242M        |           3.05 |           1.71 |      1.006 |       -7.7  |          -0.27 |     2.375 |       0.81 |            81.7 |    2    | Sep 14/b   |               |            0.8146 |
|     15 | FHI      | Financial              |          56.52 | $4.3B        |          -4.19 |           1.25 |      1.248 |      -15.89 |          -6.86 |     2.303 |      18.21 |             8.9 |    2.89 | Jul 30/a   |               |            0.8117 |
|     16 | SPSC     | Technology             |          84.59 | $2.9B        |           1.88 |           0.77 |      0.996 |       -5.42 |           5.61 |     3.182 |       5.56 |           -10.8 |    2.79 | Jul 30/a   |               |            0.8111 |
|     17 | PSNL     | Healthcare             |          16    | $1.8B        |          -1.78 |           0.85 |      1.012 |      -15.3  |           1.2  |     2.464 |      29.96 |             1.6 |    2.8  | Aug 04/a   |               |            0.8067 |
|     18 | XHR      | Real Estate            |          18.02 | $1.7B        |           2.1  |           0.88 |      1.167 |      -18.31 |          -3.2  |     2.359 |       2.75 |            17.2 |    2.17 | Jul 30/b   |               |            0.797  |
|     19 | XMAX     | Consumer Cyclical      |           8.74 | $564M        |          -1.02 |           1.2  |      0.814 |       -7.02 |          -0.92 |     2.117 |       7.06 |           nan   |  nan    | -          |               |            0.7964 |
|     20 | OPLN     | Consumer Cyclical      |          36.71 | $4.3B        |           3.47 |           1.05 |      0.924 |      -14.01 |           3.59 |     2.193 |      15.13 |            25.9 |    1.67 | Aug 04/b   |               |            0.7954 |
|     21 | FFIN     | Financial              |          30.77 | $4.6B        |          -7.49 |           1.09 |      1.204 |      -15.37 |          -7.73 |     2.087 |       8.43 |            18.8 |    2.71 | Jul 16/a   |               |            0.7952 |
|     22 | HLF      | Consumer Defensive     |          13.03 | $1.3B        |           8.13 |           1.1  |      0.841 |       -1.06 |           6.36 |     2.73  |       5.38 |            38.1 |    1.6  | Aug 05/a   |               |            0.791  |
|     23 | PFLT     | Financial              |           6.68 | $694M        |          -6.18 |           1.37 |      1.092 |      -12.57 |          -6.92 |     1.946 |     112.49 |            46   |    1.57 | Aug 10/a   |               |            0.7897 |
|     24 | CRNC     | Technology             |           8.43 | $360M        |           0.96 |           1.21 |      1.392 |      -14.42 |           1.3  |     3.68  |      11.81 |            27.5 |    2.6  | Aug 06/a   |               |            0.7875 |
|     25 | TDC      | Technology             |          30.59 | $2.6B        |           4.15 |           0.83 |      1.038 |      -12.45 |           6.6  |     2.769 |       0.49 |             9.5 |    2.73 | Aug 04/a   |               |            0.7873 |
|     26 | LILAK    | Communication Services |           8.12 | $1.7B        |          -7.73 |           1.68 |      1.279 |       -9.73 |          -4.46 |     3.651 |       1.46 |            12.6 |    2.67 | Aug 05/a   |               |            0.7863 |
|     27 | CNO      | Financial              |          52.99 | $5.0B        |          -4.57 |           0.77 |      0.995 |       -7.99 |          -2.92 |     2.046 |      11.28 |             9.1 |    2.71 | Jul 30/a   |               |            0.7855 |
|     28 | SBH      | Consumer Cyclical      |          15.9  | $1.6B        |          -0.69 |           0.99 |      1.123 |       -8.04 |          -2.66 |     3.459 |       0.23 |             6.9 |    2.4  | Aug 03/b   |               |            0.7828 |
|     29 | SHO      | Real Estate            |          11.31 | $2.1B        |           2.91 |           0.83 |      0.765 |       -5.36 |           1.37 |     1.743 |       6.67 |             5   |    2.5  | Aug 06/b   |               |            0.7804 |
|     30 | CYRX     | Industrials            |          17.35 | $882M        |           8.71 |           0.6  |      1.662 |       -5.5  |           6.62 |     3.602 |       7.74 |            11.4 |    1.22 | Aug 06/a   |               |            0.7785 |
|     31 | HIW      | Real Estate            |          29.04 | $3.3B        |          -5.28 |           1.22 |      1.207 |      -18.06 |          -5.61 |     2.605 |       7.93 |            11.3 |    2.8  | Jul 28/a   |               |            0.7781 |
|     32 | VSTM     | Healthcare             |           8.38 | $705M        |          11.58 |           1.22 |      1.302 |       -3.12 |          13.71 |     6.154 |    1775.43 |            75.1 |    1    | Aug 06/a   |               |            0.7776 |
|     33 | VCEL     | Healthcare             |          40.1  | $2.1B        |           1.26 |           0.88 |      1.27  |      -18.69 |          -3.49 |     3.276 |      22.48 |            51   |    1.25 | Jul 30/b   |               |            0.7765 |
|     34 | COLM     | Consumer Cyclical      |          57.04 | $2.9B        |          -1.09 |           0.83 |      0.962 |      -12.78 |          -0.93 |     2.684 |       1.5  |            22.4 |    2.55 | Jul 30/a   |               |            0.7741 |
|     35 | DNOW     | Industrials            |          15.57 | $2.8B        |          -0.45 |           0.99 |      0.92  |       -9.48 |          -1.14 |     2.827 |     108.12 |            20.7 |    1.4  | Aug 06/b   |               |            0.7703 |
|     36 | AVA      | Utilities              |          35.49 | $2.9B        |          -3.17 |           0.86 |      1.073 |      -17.66 |          -4.47 |     1.715 |       0.49 |            13.4 |    3    | Aug 03/b   |               |            0.7672 |
|     37 | XIFR     | Utilities              |          10.37 | $2.0B        |          -7.41 |           1.15 |      1.57  |      -19.67 |          -7.76 |     2.445 |       5.57 |            22.5 |    3    | Jul 28/b   |               |            0.7623 |
|     38 | AVO      | Consumer Defensive     |          11.73 | $1.1B        |          -8.39 |           2.16 |      1.703 |      -17.48 |          -8.45 |     3.956 |      25.8  |            40.7 |    1    | Sep 08/a   |               |            0.7617 |
|     39 | STGW     | Communication Services |           8.65 | $2.0B        |          -1.7  |           0.65 |      1.303 |       -9.42 |          -0.24 |     3.13  |      11.53 |            14   |    1.5  | Jul 30/b   |               |            0.7551 |
|     40 | KBR      | Industrials            |          35.34 | $4.4B        |          -3.97 |           0.73 |      1.146 |       -9.27 |          -3.5  |     3.258 |       1.64 |            25   |    2.22 | Jul 30/b   |               |            0.7532 |
|     41 | AVNT     | Basic Materials        |          37.79 | $3.8B        |          -8.19 |           3.31 |      1.257 |      -18.98 |         -10.49 |     3.295 |       5.83 |            34.3 |    1.56 | Aug 06/b   |               |            0.7507 |
|     42 | MIR      | Industrials            |          16.62 | $3.9B        |           8.06 |           1.64 |      1.464 |       -5.84 |           6.95 |     4.388 |      19.69 |            46   |    1.27 | Jul 28/a   |               |            0.7481 |
|     43 | ASH      | Basic Materials        |          70.57 | $3.2B        |           1.29 |           0.61 |      1.046 |       -9.81 |          -1.7  |     2.263 |       7.34 |            13.6 |    1.64 | Jul 28/a   |               |            0.7444 |
|     44 | TREX     | Industrials            |          44.78 | $4.5B        |           2.57 |           0.86 |      0.856 |      -12.13 |          -1.77 |     3.488 |       7.79 |            22.1 |    2.23 | Aug 04/b   |               |            0.7349 |
|     45 | HNI      | Consumer Cyclical      |          46.86 | $3.4B        |           0.34 |           0.75 |      0.829 |       -7.04 |          -1.85 |     3.23  |     120.72 |            48.8 |    1    | Jul 30/b   |               |            0.734  |
|     46 | RES      | Energy                 |           5.96 | $1.3B        |          -2.3  |           0.72 |      1.188 |      -11.44 |          -2.55 |     3.284 |       9.52 |             7.4 |    3.5  | Jul 30/b   |               |            0.7225 |
|     47 | EGY      | Energy                 |           6.01 | $615M        |          -4.15 |           0.87 |      0.969 |       -8.8  |           2.84 |     2.863 |      39.5  |            59.6 |    1    | Aug 06/a   |               |            0.7203 |
|     48 | SXC      | Basic Materials        |           9.94 | $810M        |           4.19 |           0.67 |      0.925 |       -6.93 |           2.67 |     3.55  |       9.49 |            -4.4 |    2    | Jul 30/b   |               |            0.7161 |
|     49 | LPG      | Energy                 |          57.64 | $2.4B        |           4.08 |           0.71 |      0.883 |       -3.89 |          11.21 |     3.986 |     123.11 |            -6.1 |    1.5  | Aug 05/b   |               |            0.7158 |
|     50 | WKC      | Energy                 |          36.41 | $2.0B        |           4    |           0.56 |      0.965 |      -11.63 |           0.65 |     2.482 |      50.48 |             3.5 |    3.67 | Jul 23/a   |               |            0.7148 |
</screener_watchlist>

**Screener Integration:**
- Every candidate has already passed the screener's hard gates: prohibited businesses, deal-pinned, >40% above the 50-day or >20% above the 20-day SMA, days 1-3 of a >10% breakout, post-earnings jump, shrinking revenue (Sales Q/Q < 0), liquidity. Gates run on Finviz-level data — the PRV gate (browser quote page) still applies to every name.
- `composite_score` (since 2026-09-15) = equal-weight ranks of low volatility, proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio and distance above the 50-day SMA — the six signals that passed the Phase 2 factor study. 20-day momentum is reported but no longer scored (no measurable ranking skill). The measured edge is small and mostly defensive, so rank is sourcing, never conviction.
- `review_flag` = industry that mixes prohibited and permitted businesses: read what the company does before any research.
- Evaluate AT LEAST the top 5 screener candidates before selecting (discover via WebSearch, verify on the browser quote page).
- For each screener candidate not selected, state why in one line.
- Respect the sector cap: max 2 positions in the same GICS sector.

<holdings date="2026-10-09">
<holding ticker="CON" shares="3" avg_cost="35.31" current_price="37.19" stop_loss="34.49" stop_limit="34.34" />
</holdings>

<position_limits>
| Ticker | Sector                 | Sessions Held | 60-Session Review |
|--------|------------------------|---------------|-------------------|
| CON    | Healthcare             |            15 | not yet           |
  <sector_counts>Healthcare: 1</sector_counts>
  <sector_cap_status>OK (cap 3 per sector)</sector_cap_status>
  <driver_cap>Max 2 positions may share a primary thesis driver. Name each holding's primary driver in this report — it cannot be derived from data and an unnamed driver is a rule violation (portfolio_rules.md).</driver_cap>
</position_limits>

<holding_review>
| Ticker | Close | Move (×ATR) | Sector | vs sector | Volume (×20d) | Stop room (×ATR) | Held | Est. next earnings | Review |
|--------|-------|-------------|--------|-----------|---------------|------------------|------|--------------------|--------|
| CON    | $37.19 | +0.64 | XLV +1.58% | +0.01pp | 1.0 | 2.95 | 15 | ~2026-11-05 (19s, est.) | **FULL** |
  <full_review ticker="CON">ran up -- stop 2.9xATR behind; raise ELIGIBLE to ~$35.36 but default is HOLD (needs a written thesis change)</full_review>
  <rule>FULL review (portfolio_rules.md → Daily monitoring by exception) when a holding moved ≥1.5×ATR, traded ≥3× its average volume, has its stop within 1×ATR, has a qualifying stop raise, may report earnings within ~15 sessions, or was bought ≤3 sessions ago — or when the user asks ("full review TICKER"). Otherwise ONE LINE. Every holding still gets the live news-feed check; news the script cannot see upgrades a LINE to FULL. **Sector** is the holding's sector proxy ETF and its move for the same session; **vs sector** is the holding's move minus the proxy's, in percentage points — the first test of whether a move is idiosyncratic. Reported, not gated.</rule>
</holding_review>

<research_trigger>
  <cadence>trigger-based since 2026-09-17 (the weekly SCREEN is unchanged)</cadence>
  <funnel>buys sought: 3; stage-1 quick checks: 15 (extend to watchlist_extended.csv if the top 50 cannot supply them)</funnel>
  <lanes>catalyst 0/3 (full 2% risk); screener 1/0 -- lane CLOSED in RISK-OFF since 2026-10-11; holdings already open are grandfathered (capacity gates govern new capital only) -- 3 catalyst slots open: source at least 3 dated non-binary catalyst candidates this report, the ONLY lane open in this regime</lanes>
  <blackout>re-entry banned (portfolio_rules.md: 10 sessions after a stop-out) -- ATRC, 4 sessions left; HOPE, 2 sessions left. Kill these at stage 1; do not spend a quick check on them.</blackout>
  <week_number>57</week_number>
  <last_report>2026-10-05 (4 sessions ago)</last_report>
  <status>DUE</status>
  <reason>free slot (1/10) with 79% deployable</reason>
  <reason>deployable cash 79% >= 25%</reason>
  <regime>RISK-OFF now; RISK-OFF at the last report (from regime_history.csv)</regime>
  <if_not_due>Produce a short monitoring note instead: stops, any position nearing 60 sessions, and the breaker line. Do not re-underwrite theses that nothing has changed for.</if_not_due>
</research_trigger>

<last_analyst_thesis>
# Week 56 — Thesis Summary (2026-10-05)

**Date:** 2026-10-05 | **Week 56, indefinite phase** | **Regime:** RISK-OFF (IWM −3.78% vs its 50-day, 24 sessions)
Equity $712.20 · Cash $605.25 (85.0%) · **Gap since re-base −3.73%** · Gap since inception −8.12% · Drawdown −3.33%
**Second consecutive no-buy week** — and the constraint is now measured: 16 of 100 screened names clear `near_high` ≥ −5%, 2 clear the full defensive profile, 1 also clears the 50-day rule.

---

**CON (Concentra) — HOLD | Conviction 3/5**
3 shares at $35.65, **+1.0%** from a $35.31 entry, 10 sessions held, 15.0% of equity. **Driver:**
U.S. hiring, which sets occupational-health visit volumes; August payrolls +162,000 with upward
revisions, next print early October. Strong Buy, $41.00 target, earnings 2026-11-05. **A stop raise
qualifies for the first time: $33.69 / $33.54**, the 2.0×ATR target, cutting risk from 0.89% to
**0.68%** of equity. Place it before the close. Note the rounding — $33.68 fails the raise-size
test, $33.69 passes — and note that CON's restoration is already spent, so this stop will never
move down again. **What would change the view:** two weak payroll prints, or visit volumes lagging
headline hiring.

**Portfolio.** One position, **85% cash**, and a **second consecutive no-buy week** — but the
reason is now measured rather than asserted. **Of 100 screened names, 16 clear `near_high` ≥ −5%,
2 clear the full defensive profile, and 1 clears it while also above its 50-day SMA** — LTC, for
the third week, at under 5% upside on a Hold. The binding condition is proximity to the 60-day
high, and it is scarce *because* the regime that invokes the allowance is defined by the market
being below its own average. The gate is anti-correlated with its own precondition. **Before the
next report:** place CON's stop raise; decide the PLX exclusion question so it stops recurring;
and take the `near_high` measurement to the rules process rather than to another weekend report —
this is the third week it has produced the same answer, and the Phase 4 sunset review is where it
belongs. Re-entry bans: ATRC ~2026-10-15, HOPE ~2026-10-13.

---

*Week 56 Summary. Generated 2026-10-05 by Claude Code.*
</last_analyst_thesis>

<recent_trades>
<!-- Trades from Monday through Friday of current week -->
<!-- No trades this week -->
</recent_trades>

<execution_requests>
<session_directives>
- Research focus: Wide catalyst net. The screener lane is CLOSED in RISK-OFF, so the catalyst lane is the only route: search all qualifying categories - index rebalance, dated contract or backlog conversion, a raised guide with the next print dated, spin-off or refinancing. If nothing qualifies, report that and name the categories searched.
- Fixed by portfolio_rules.md, not chosen weekly: holding horizon 40–60 sessions; catalyst window 90 days, non-binary only; 2% risk per trade; 10 position ceiling (about 4 fit at current sizing); risk posture set by the regime filter and the drawdown circuit breaker.
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
- Maintain at least 5% cash reserve.
- Flag any holding where the thesis has weakened, even if the stop has not been breached.

<!-- OUTPUT PREFERENCES -->
- Include a brief bear case for every new candidate.
- Rank candidates by risk/reward before selecting.
- Show your work on position sizing calculations.