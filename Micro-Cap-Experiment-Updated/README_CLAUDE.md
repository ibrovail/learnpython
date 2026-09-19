# Claude Code Analysis Workflow

This document describes the Claude Code CLI workflow that replaces the manual copy-paste-to-ChatGPT approach for portfolio analysis.

---

## Overview

Instead of copying trading script output to ChatGPT, Claude Code acts as the analysis layer directly in the CLI. Claude pipes inputs to the trading script automatically; the XML output lands in the conversation and Claude auto-analyzes it with live web search.

**Key benefits:**
- No copy-paste between tools
- No separate terminal needed — everything runs inside Claude Code
- Persistent conversation context across the session
- Live web search for prices, catalysts, and ATR data
- Automated weekend report file creation (MD + PDF)
- Quantitative screener scans 1,000+ stocks across all sectors for sector-diverse candidate generation

---

## Daily Workflow

**After 4 PM EST on trading days:**

1. Tell Claude what to do using the `Run daily:` pattern:
   ```
   Run daily: no changes
   Run daily: inject $143.08, buy 17 REPL limit $7.05 stop $5.90/$5.80
   Run daily: sell 8 RCKT at $5.11
   ```
2. Claude constructs the input sequence and pipes it to the trading script automatically
3. Claude **automatically** detects the `<daily_summary>` XML output and runs the daily analysis — no additional prompt needed
4. Review Claude's recommendations and specify any trades in the next `Run daily:` command

**Why not run `make daily` via the `!` shell prefix?** Claude Code's `!` prefix does not support interactive stdin (`input()` calls fail with EOFError). The `Run daily:` pattern works around this by piping pre-constructed answers to the script via the Bash tool.

**Daily analysis output (6 sections):**
1. Market Regime Check — IWM vs its 50-day SMA, computed by the script (`<market_regime>`)
2. Holdings by exception — the script marks each holding FULL or LINE (`<holding_review>`); a full
   review (catalyst research, stop update, add-shares check) only when flagged or when you write
   "full review TICKER"; a live news check for every holding either way
3. New positions — screening or pass with rationale
4. Final decisions — exact action blocks (BUY / SELL / UPDATE STOP / HOLD)
5. Post-event playbook — only when a binary catalyst is ≤10 trading days away
6. Portfolio state — full snapshot with updated stops

---

## Weekend Workflow

**Cadence note (changed 2026-09-17):** the **screener runs every weekend**, but the full
research report is now **trigger-based** — it runs when there is something to decide (free slot
with capital, ≥25% deployable cash, a holding at 60 sessions, the circuit breaker armed, or 30
sessions since the last report). Saying `run weekend` now starts with `make trigger` (instant, ledger-only), which prints the
verdict **before** any directive question is asked. Due → the usual questions and full report.
Not due → `make screen` plus a short monitoring note saved as `Week N Monitor.md`. Thresholds and rationale are in
`portfolio_rules.md` → *Research cadence*.

**On Saturday or Sunday:**

1. Tell Claude to run the weekend analysis:
   ```
   run weekend
   ```
   **Do NOT invoke `make weekend` directly via the `!` shell prefix** — that bypasses Claude. Claude runs `make trigger` first, then (if a report is due) asks one question and runs `make weekend FOCUS="..."`.

2. If a report is due, Claude asks **one optional question**: anything specific to research (a
   ticker, sector or question)? Default is a wide net. The four questions used until 2026-09-19
   (sector, timing, risk posture, positions) are retired — the rules now fix all of those.

3. After you answer, Claude runs `make weekend` which:
   - Runs the quantitative screener (`screener.py`) to generate a sector-diverse watchlist
   - Checks if the portfolio CSV is current for the last trading day (exits with an error if not)
   - Runs `trading_script.py --weekend-summary` to update `weekend_summary.md` (injects screener watchlist as `<screener_watchlist>` XML block)
   - Injects the previous week's thesis summary into the `<last_analyst_thesis>` block
   - Prints the full `weekend_summary.md` content to the conversation

4. Claude **automatically** begins the deep research:
   - Evaluates at least the top 5 screener candidates via web search (with sector cap enforcement: max 2 positions per GICS sector)
   - Writes the report in the **six-section format** (since 2026-09-19): scoreboard, deployment
     funnel, exact orders, holdings by exception, risk checks, thesis summary — plus sources.
     The `weekly-portfolio-report` skill is retired and must not be used

5. After the report, Claude automatically saves three output files:
   - `Weekly Deep Research (MD)/Week X Full.md` — the full report
   - `Weekly Deep Research (MD)/Week X Summary.md` — Section 6 (Thesis summary) only (Section 9 before Week 54)
   - `Weekly Deep Research (PDF)/Week X.pdf` — PDF version of the full report

**If `make weekend` reports the portfolio is not current:**
```
⚠️  Portfolio data is not current for the last trading day. Ask Claude to 'run daily' first, then 'run weekend'.
```
Say `Run daily: no changes` (or with any needed trades), then say `run weekend` again. Claude will skip the daily analysis during this prerequisite run and wait for the weekend context.

**If you ran `trading_script.py` manually**, always pass `--data-dir "Start Your Own"` — otherwise the portfolio CSV is written to a different path and `make weekend` will report stale data even though the script ran successfully.

---

## Screener

The quantitative screener (`screener.py`) scans the full small-cap universe (up to $5Bn) to generate sector-diverse candidates. It runs automatically as part of `make weekend`, or standalone:

```
! make screen
```

**How it works:**
1. Pulls ~1,000 stocks from Finviz (market cap ≤$2B, price ≥$1, ADV ≥$500K)
2. Enriches with 30-day yfinance price/volume history
3. Calculates signals: 20-day and 5-day momentum, 1-day and 5/50-day volume ratios, relative strength vs IWM, Bollinger Band width, 20-day return volatility, distance from the 60-day high, distance above the 20- and 50-day SMAs, ATR%, breakout age, post-earnings reaction
4. Ranks by composite score (since 2026-09-15): the equal-weighted ranks of low volatility, proximity to the 60-day high, Bollinger squeeze, 5/50-day volume, 1-day volume ratio and distance above the 50-day SMA — the six signals that passed the Phase 2 factor study (`Experiment Details/Screener Factor Study — Phase 2.md`). 20-day momentum is reported but no longer scored: it showed no ranking skill. The legacy and deduplicated scores are recorded in `screener_history/` for an out-of-sample comparison
5. Applies the hard gates (prohibited businesses, deal-pinned, distance from 20/50-day SMA, fresh breakout, post-earnings jump, shrinking revenue, liquidity) **before** ranking, then outputs the top 50 survivors (max 6 per sector) to `Start Your Own/watchlist.csv` and the full gated universe to `Start Your Own/screener_history/screen_YYYY-MM-DD.csv`

**Allocation Framework** (see `portfolio_rules.md`):
- **Catalyst plays**: dated **non-binary** catalyst within 90 days. Sized by the standard 2% risk
  budget — no separate size cap. RISK-ON up to 2 positions, RISK-OFF up to 3.
  **Binary-thesis entries are prohibited** (a stop cannot bound an overnight gap).
- **Screener-sourced plays**: sourced from the screener watchlist, no catalyst required.
  RISK-ON up to 4 positions; RISK-OFF permitted at half the risk budget on a defensive profile
  (this allowance sunsets at Phase 4 unless the regime test confirms it).
- **Correlated-risk caps**: max **2 positions sharing a primary thesis driver**; max **3 positions
  in any one GICS sector**. Neither is gated in code — `trading_script.py` prints a
  `<position_limits>` block so both are visible at research time.
- **Exits**: the trailing stop is the only exit. No mechanical partials. Any holding at
  **60 sessions** must be re-underwritten in writing or exited.

---

## Key Files

| File | Purpose |
|------|---------|
| `Start Your Own/portfolio_rules.md` | Complete portfolio rules — read before every analysis |
| `Start Your Own/daily_analysis_prompt.md` | Daily 6-section output format + the one weekend question |
| `Start Your Own/weekend_summary.md` | Weekend deep research prompt — updated by `make weekend` |
| `screener.py` | Quantitative screener: Finviz universe → yfinance signals → ranked watchlist CSV |
| `Start Your Own/watchlist.csv` | Screener output — top 50 gate survivors ranked by composite score (max 6 per sector) |
| `Start Your Own/screener_history/` | Every screen's full universe with signals, fundamentals, gate results and scores — the dataset for factor research |
| `inject_last_thesis.py` | Injects previous week's thesis into `weekend_summary.md` |
| `generate_pdf.py` | Converts markdown reports to PDF (requires `fpdf2`) |
| `Weekly Deep Research (MD)/` | All weekly analysis files (Full + Summary) |
| `Weekly Deep Research (PDF)/` | PDF versions of weekly reports |

---

## Setup

Ensure `fpdf2` is installed for PDF generation:
```bash
source venv/bin/activate
pip install fpdf2
```

Or install all dependencies:
```bash
make install
```

---

## The Weekend Question

When a full report is due, Claude asks one optional question — **anything specific you want
researched?** — and passes the answer as `make weekend FOCUS="..."`. Leaving it empty means a
wide net. Retired 2026-09-19: sector focus, catalyst timing, risk posture and max positions. The
rules set all four; two of the old options (aggressive / tighten stops by one ATR) contradicted
them.
