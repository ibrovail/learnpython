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

**Why not `! make daily`?** Claude Code's `!` prefix does not support interactive stdin (`input()` calls fail with EOFError). The `Run daily:` pattern works around this by piping pre-constructed answers to the script via the Bash tool.

**Daily analysis output (6 sections):**
1. Market Regime Check — IWM vs 50-day SMA (live web search)
2. Per-holding review — price, P&L, catalyst research, trailing stop recalculation, add-shares check
3. New positions — screening or pass with rationale
4. Final decisions — exact action blocks (BUY / SELL / UPDATE STOP / HOLD)
5. Post-event playbook — only when a binary catalyst is ≤10 trading days away
6. Portfolio state — full snapshot with updated stops

---

## Weekend Workflow

**On Saturday or Sunday:**

1. Run the weekend command in the Claude Code CLI:
   ```
   ! make weekend
   ```
   This does three things automatically:
   - Checks if the portfolio CSV is current for the last trading day (exits with an error if not)
   - Runs `trading_script.py --weekend-summary` to update `weekend_summary.md`
   - Injects the previous week's thesis summary into the `<last_analyst_thesis>` block
   - Prints the full `weekend_summary.md` content to the conversation

2. Claude **automatically** begins the weekend workflow:
   - Asks 4 session directive questions (sector focus, catalyst timing, risk posture, max positions)
   - Updates `<session_directives>` in `weekend_summary.md` with your answers
   - Runs the full 10-section deep research report with extensive web search

3. After the report, Claude automatically saves three output files:
   - `Weekly Deep Research (MD)/Week X Full.md` — full 10-section report
   - `Weekly Deep Research (MD)/Week X Summary.md` — Section 9 (Thesis Review) only
   - `Weekly Deep Research (PDF)/Week X.pdf` — PDF version of the full report

**If `! make weekend` reports the portfolio is not current:**
```
⚠️  Portfolio data is not current. Run '! make daily' first, then re-run '! make weekend'.
```
Run `Run daily: no changes` (or with any needed trades), then re-run `! make weekend`. Claude will skip the daily analysis during this prerequisite run and wait for the weekend context.

**If you ran `trading_script.py` manually** (outside of `make daily`), always pass `--data-dir "Start Your Own"` — otherwise the portfolio CSV is written to a different path and `make weekend` will report stale data even though the script ran successfully.

---

## Key Files

| File | Purpose |
|------|---------|
| `Start Your Own/portfolio_rules.md` | Complete portfolio rules — read before every analysis |
| `Start Your Own/daily_analysis_prompt.md` | Daily 6-section output format + weekend session directive questions |
| `Start Your Own/weekend_summary.md` | Weekend deep research prompt — updated by `make weekend` |
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

## Session Directive Options

Each weekend, Claude asks these 4 questions before running the analysis:

| Question | Options |
|----------|---------|
| Sector focus | Wide net (default) / Biotech / Energy / Tech / Industrials |
| Catalyst timing | Within 5 days / Within 10 days / 30–60 days (medium-term, high conviction) |
| Risk posture | Neutral / Aggressive (trailing benchmark) / Defensive (protect gains) / Tighten stops 1 ATR |
| Max concurrent positions | 5 / 6 |
