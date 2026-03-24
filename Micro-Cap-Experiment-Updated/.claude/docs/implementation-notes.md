# Implementation Notes

Reference documentation for key implementation decisions and change history.

---

## Key Implementation Patterns

- **All orders are limit orders** — prevents slippage from lookahead bias (`trading_script.py:816-823`)
- **Weekends map to prior Friday** — `trading_script.py:175-196`
- **Dollar-weighted S&P 500 benchmark** — accounts for capital injections (`trading_script.py:1110-1159`)
- **Data fetching: Yahoo → Stooq fallback** — resilience against Yahoo outages (`trading_script.py:361-410`)
- **Generate_Graph.py date filtering** — capital injections filtered to portfolio date range (`Generate_Graph.py:350-358`)
- **Stop-limit logging** — buy flow collects both stop trigger price and stop-limit price; stop-limit defaults to stop price if omitted
- **Stop-loss trigger is interactive** — when a stop fires, script prints trigger/stop-limit/range and prompts for actual fill price, looping on invalid input (`trading_script.py:702-748`)
- **Backward-compatible CSV loading** — old CSVs without `Stop Limit` column load cleanly; missing values default to stop price (`trading_script.py:1761-1774`)

## Testing Approach

No formal unit tests. Validation is through:
- Manual review of daily results
- Weekly performance reports
- CSV audit trail

---

## Change History

### 2026-03-24 — Claude Code Analysis Layer

Replaced copy-paste-to-ChatGPT workflow with Claude Code as the analysis layer.

**Summary**: Claude now auto-triggers daily and weekend analysis from XML output produced by `make daily` / `make weekend`. Weekly research reports are automatically saved as MD and PDF files.

| File | Change |
|------|--------|
| `Makefile` | Added `daily` and `weekend` targets |
| `CLAUDE.md` | Added Claude Analysis Integration section; updated Daily Workflow |
| `.claude/rules/analysis-workflow.md` | Created — auto-trigger behavior rules |
| `Start Your Own/portfolio_rules.md` | Created — standalone portfolio rules |
| `Start Your Own/daily_analysis_prompt.md` | Created — 6-section daily format + weekend config questions |
| `inject_last_thesis.py` | Created — injects prior week's thesis into weekend_summary.md |
| `generate_pdf.py` | Created — markdown-to-PDF converter (fpdf2) |
| `requirements.txt` | Created — project dependencies |
| `README_CLAUDE.md` | Created — Claude Code workflow documentation |

---

### 2026-03-23 — Capital Injection Declaration in --weekend-summary

Added ability to declare a planned capital injection during the `--weekend-summary` run so it is factored into ChatGPT's position sizing for the coming week.

| File | Change |
|------|--------|
| `trading_script.py` | Added `planned_injection` param to `print_weekend_summary()`; interactive prompt in `main()` |
| `Start Your Own/weekend_summary.md` | Updated `<budget>` rule to reference `<capital_injection>` block |

---

### 2026-02-06 — Generate_Graph.py --end-date Fix

Fixed IndexError when `--end-date` is before the last capital injection date.

| File | Change |
|------|--------|
| `Start Your Own/Generate_Graph.py` | Filter capital injections to portfolio date range before `last_injection_date` calc (lines 350-358) |

---

### 2025-12 — Stop-Limit Order Support

Added stop-limit order support to buy flow and stop trigger handling.

| File | Change |
|------|--------|
| `trading_script.py` | Buy logging collects stop trigger + stop-limit; trigger flow prompts for fill price |

---

### 2025-10 — Extended Experiment + Phase 2 Rules

Extended experiment timeline to 12 months. Upgraded rules for Phase 2 alpha generation.

| File | Change |
|------|--------|
| `Start Your Own/weekend_summary.md` | Updated rules, universe, and output format |
| `trading_script.py` | Extended date range handling |
