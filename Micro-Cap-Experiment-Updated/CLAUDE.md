# Small-Cap Live Portfolio

An indefinite live process using Claude Code to manage a real-money **small-cap** stock portfolio
(market cap up to $5Bn), tracking alpha vs. the S&P 500. *Folder and file names still say
"Micro-Cap" / "chatgpt_" from the project's origins; they are kept because renaming breaks paths.* The fixed 52-week experiment closed at the
2026-09-11 close; there is no end date. Holding horizon is **40–60 trading sessions**.

## Tech Stack

| Tool | Version | Use |
|------|---------|-----|
| Python | 3.12 | Core |
| pandas | 2.2.2 | Data processing |
| numpy | 2.3.2 | Calculations |
| yfinance | 0.2.65 | Price data (Yahoo → Stooq fallback) |
| matplotlib | 3.8.4 | Performance charts |
| fpdf2 | 2.8.0 | Weekly report PDF generation |

## Project Structure

```
├── trading_script.py              # Core trading engine
├── inject_last_thesis.py          # Injects prior week's thesis into weekend_summary.md
├── generate_pdf.py                # Markdown → PDF converter
├── Makefile                       # Build + workflow shortcuts
├── requirements.txt               # Python dependencies
├── README_CLAUDE.md               # Claude Code workflow documentation
│
├── Start Your Own/                # Live experiment data + analysis prompts
├── Weekly Deep Research (MD)/     # Weekly analysis files (Full + Summary)
├── Weekly Deep Research (PDF)/    # Weekly PDF reports
├── Experiment Details/            # Methodology, prompts, disclaimer
└── Performance Results/           # Monthly performance charts
```

## Key Files

| File | Purpose |
|------|---------|
| `trading_script.py` | Trading engine: portfolio processing (482-741), daily analytics (987-1330), weekend summary (1368-1517) |
| `Start Your Own/portfolio_rules.md` | **Standing** portfolio rules — read before every analysis session |
| `Experiment Details/Rules Amendment History.md` | Why each rule exists; every dated amendment and its outcome |
| `Start Your Own/daily_analysis_prompt.md` | Daily 6-section format + weekend directive questions |
| `Start Your Own/weekend_summary.md` | Weekend deep research prompt (updated by `make weekend`) |
| `inject_last_thesis.py` | Injects Week N-1 Summary into `<last_analyst_thesis>` block |
| `generate_pdf.py` | Converts weekly MD report to PDF using fpdf2 |
| `Start Your Own/Generate_Graph.py` | Portfolio vs S&P 500 benchmark visualization |
| `perplexity_brief.py` | Peer/sector comparison computed locally + the Perplexity Finance URL for discovery |
| `screener.py` | Quantitative screener: Finviz universe → yfinance signals → ranked watchlist |

## Commands

```bash
make daily      # Run trading script after 4 PM (Claude auto-analyzes output)
make screen     # Run quantitative screener (outputs watchlist CSV)
make brief      # Discovery brief: verified peer/sector moves + the Perplexity URL (T=TICKER)
make trigger    # Is a full weekend report due? Ledger-only verdict, run before any question
make weekend    # Run screener + weekend analysis workflow (Claude auto-triggers deep research)
make setup      # Create venv + install deps
make graph      # Generate performance chart
make clean      # Remove venv
```

**Setup note**: This project is on iCloud-synced Desktop — use `make setup` to create the `.nosync` venv automatically. See `.claude/docs/setup-guide.md` for full details and troubleshooting.

## Environment Variables

- `ASOF_DATE=YYYY-MM-DD` — override today's date for backtesting (`trading_script.py:56-59`)

## Portfolio Rules

Complete rules (universe, execution limits, risk control, sizing, exclusions) are in `Start Your Own/portfolio_rules.md`. Read before every analysis session.

## Daily Workflow

1. Tell Claude: `Run daily: <inputs>` after 4 PM EST (Claude pipes inputs and runs the script)
   - No changes: `Run daily: no changes`
   - With trades: `Run daily: inject $143.08, buy 17 REPL limit $7.05 stop $5.90/$5.80`
   - Selling: `Run daily: sell 8 RCKT at $5.11`
   - Optional, any day: `Run daily: no changes, full review ATRC` — forces a full review of a holding
     the script marked LINE (holdings are reviewed by exception since 2026-09-19)
2. Claude auto-analyzes the XML output with live web search
3. Review recommendations; specify any trades in the next `Run daily:` command

**Note:** Always use `Run daily:` (not the `!` shell prefix on `make daily`) — Claude Code's `!` prefix does not support interactive stdin. Likewise, say `run weekend` (not `! make weekend`): Claude runs `make trigger` first; if a full report is due it asks one optional question (anything specific to research?) and runs `make weekend FOCUS="..."`; if not, it runs `make screen` and writes a short `Week N Monitor.md` note.

## Current State

- **Complete**: **Indefinite-system rules revision (9/17)** — horizon **40–60 sessions**; the
  trailing stop is the only **mechanical** exit (partials deleted); **binary-thesis entries
  prohibited**; risk-per-trade **2%**; **drawdown circuit breaker** (−20% de-risk / −30% cash, on
  the injection-neutral series from the re-based peak); driver cap 2 + sector cap 3; RISK-OFF
  capacity raised; weekend **report** is trigger-based while the **screen** stays weekly. Standing
  rules in `portfolio_rules.md`, reasons in `Experiment Details/Rules Amendment History.md`.
- **Complete**: **Phase 3.5** — on non-overlapping formation dates the composite clears t=2.0 at
  **no** horizon, and 20/40/60 sessions have too few independent observations for any verdict;
  effect sizes do rise with horizon. **Phase 4 (Dec) primary horizon = 20 sessions**; the
  40-session test waits for ~March 2027. Index changes are the fifth unexplained-move category.
- **In progress**: **Two positions after HOPE stopped out 9/29.** ATRC 3 sh **+69.2%** ($58.05),
  stop $55.27/$55.12 (**1.29×ATR**, cannot be lowered), **56 sessions — re-underwrite ~10/5**; the
  9/25 −3.93% reversal cleared all five unexplained-move checks and reads as **post-inclusion
  giveback**. **CON 3 @ $35.31** ($35.34, +0.1%), stop $33.19/$33.04 (2.37×ATR), 7 sessions — no
  raise qualifies (+0.37×ATR against the 0.50 minimum). Equity **$719.70**, cash **$439.53
  (61.1%)**, **$331.57 deployable (46.1%)**, **gap −2.06%** since re-base, drawdown −2.31%, regime
  RISK-OFF (21 sessions).
- **HOPE post-mortem (9/29)**: stopped out at $13.48 for **−$7.20 = 0.99% of equity — the budgeted
  risk to the cent**, 6 sessions held. **No rule failed; the entry did.** It was bought **+0.15%
  above its 50-day SMA** — the thinnest possible pass of the hard trend gate, flagged in the Week 54
  report as thin — and still sized to **28.8% of equity**, the largest position in the book. Its rate
  driver then inverted within three sessions (10-year highest since 2007 on 9/23). Two questions for
  a later review, **not rule changes on one case**: does a sub-1% trend-gate pass deserve reduced
  size or none, and should a rate-path thesis be entered days after an FOMC? *(`research_log.csv`
  now records `pct_vs_sma50`, auto-filled and backfilled, so the first is queryable — but 25 rows
  over two research dates cannot answer it yet: no date has both BUY and PASS names with forward
  data.)*
- **Complete (9/19)**: review items adopted — **one weekend question** (R1); **two-stage research funnel**
  sized to the buys sought, spread across ranks/sectors/size, extending to ranks 51–100 (R3); **research log** of buys *and* passes via `log_research.py`
  (R4); stop **raise target 2.0×ATR**, 1.5× floor applies at placement (R5); **regime computed**
  by the script → `<market_regime>`, `regime_history.csv`, flip trigger (R7); **small-cap** label +
  size control for Phase 4 (R8). **Six-section report** replaces the ten (R2);
  the `weekly-portfolio-report` skill is **retired** (predates every rule; conflicts with four).
  **Dailies by exception** — `<holding_review>` marks FULL/LINE (R6); **regime ±1% band** — 17 → 7
  regime changes a year, same RISK-OFF share (D3). Nothing left under discussion.
- **Complete (9/28)**: **the 60-session re-underwrite now says what it tests** — the **quality**
  tests (revenue, earnings path, driver freshness, non-binary, prohibited business, trend and
  distance-from-base, earnings guard, placeable stop inside the 30% ceiling), **not** the regime
  **capacity** gates (defensive profile, half risk budget, 90-day catalyst window, position
  ceiling), which govern new capital only. A failed quality test is an exit; a failed capacity gate
  is a hold that **must name the gate in writing**. Reasoning and the honest accounting are in
  `Rules Amendment History.md` (2026-09-28). Deliberately left open: the review cadence *after* a
  pass — decide it when ATRC actually passes.
- **Next**: **ATRC's 60-session review lands ~10/5** (56 sessions on 9/29) — plan is hold unless a
  quality test fails first; the prep is in the Week 55 report. Next screen **10/03** with two free
  slots and 46% deployable, so the trigger fires; but Week 55's funnel found nothing buyable in
  RISK-OFF and the **mid-October earnings squeeze** starts next week, so expect the same constraint
  with more cash behind it. **HOPE is re-entry-banned to ~10/13.**
- **Fixed (9/30 c)**: **session arithmetic now goes through the NYSE calendar**
  (`_sessions_between`, `_add_sessions`). Three hand-rolled conversions were holiday-blind while
  `last_completed_session()` had used `exchange_calendars` all along. The **30-session backstop**
  scaled days by 5/7 (off +2 over a quarter, drifting with the gap) and now counts closed sessions;
  the **earnings estimate** collapsed every past-due print to −1 via `pd.bdate_range`, which always
  passed the −3..15 test and so pinned a **permanent spurious FULL review** on any holding with a
  stale estimate.
- **Complete (9/30 b)**: the **re-underwrite trigger fires before the threshold**. The review is
  counted in sessions but only happens in a weekend report, so a holding crossing 60 mid-week was
  reviewed late by construction — ATRC would have been raised at **64 sessions**.
  `<research_trigger>` now emits a second reason within **5 sessions** of 60, naming the projected
  date. **The threshold is still 60**, not 55 — the trigger is early, the standard is not lowered.
- **Complete (9/30)**: the **trend-gate margin travels with the decision**. `research_log.csv`
  records `pct_vs_sma50` (auto-filled; Weeks 54–55 backfilled from `screener_history/`), and the
  six-section template now carries it in the **stage-1 table** (`vs 50d`), the stage-2 entry checks,
  the **Sizing** bullet and the risk table — **a pass under +1.0% must be named** and carried
  forward. Disclosure, not a size penalty: whether a thin pass deserves less size is still an open
  question. PDF width checked — the ninth column changes no wrapping.
- **Fixed (9/29)**: the **post-stop-out re-entry ban is now computed and printed** as
  `<blackout>` in `<research_trigger>`, with reason code `re-entry-ban` in `log_research.py` and the
  ban added to the stage-1 kill list. It had been prose in `portfolio_rules.md` that nothing
  surfaced — the first run found **VTS 1 session inside its own ban**, unnoticed.
- **Complete (9/29)**: **`research_log.csv` records `pct_vs_sma50`** — the candidate's margin above
  its 50-day SMA at decision time, auto-filled from that week's watchlist so it cannot be forgotten
  (`--pct-vs-sma50` overrides for off-list names). Weeks 54–55 backfilled from the committed
  `screener_history` snapshots, so the HOPE post-mortem's question — does a **thin** trend-gate pass
  predict a worse outcome? — is now queryable (HOPE +0.15% stopped out in 6 sessions; CON +6.16%
  held). `log_research.py` also refuses to append when the file's header no longer matches its
  columns, instead of writing silently misaligned rows.
- **Complete (9/30)**: **the daily names each holding's sector proxy.** `<holding_review>` gained
  **Sector** (the holding's sector-proxy ETF and its same-session move) and **vs sector** (the
  holding's move minus it, in pp). Origin: ATRC fell 3.19% on 9/30 and the review compared it with
  **XBI — a biotech ETF** carried in `DEFAULT_BENCHMARKS` from when this book held biotech — and
  concluded the sector did not cover the move. **XLV was −1.35%** and the device peers fell
  0.5–1.4%: about a third was sector. Reported, not gated. Plus **`perplexity_brief.py`**
  (`make brief T=ATRC`): peers picked from the cached universe by industry and log market cap,
  their moves and the sector proxy computed **locally from price history**, and the Perplexity
  Finance URL printed for candidate *explanations* with a dating checklist — discovery and
  evidence kept apart, as `price-data-integrity.md` requires.
- **Fixed (9/20–9/21)**: `inject_last_thesis.py` line-anchors its tags (it had deleted the weekend
  data block); `entry-discipline.md` now requires all three stop candidates — 1.75×ATR, the
  10-session low, the 50-day SMA — with the stop below the lowest, recomputed from the actual fill.

## Documentation

Reference docs (not auto-loaded — read on demand):

| File | Contents |
|------|----------|
| `.claude/docs/setup-guide.md` | Full iCloud venv setup, script arguments, troubleshooting |
| `.claude/docs/implementation-notes.md` | Key implementation patterns + full change history |
| `.claude/docs/architectural_patterns.md` | 10 core architectural patterns with code references |
| `.claude/rules/price-data-integrity.md` | Price sourcing: no WebSearch for live/AH prices, timestamp + order-side checks |
| `README_CLAUDE.md` | Claude Code daily/weekend workflow guide |
| `Experiment Details/Prompts.md` | Original ChatGPT prompts and integration notes |
