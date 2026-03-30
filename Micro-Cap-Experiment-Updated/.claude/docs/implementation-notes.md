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

### 2026-03-29 — Portfolio Rules v2: Binary Event Framework

Upgraded `portfolio_rules.md` with 5 new rules derived from 28 weeks of live experiment data, particularly the RCKT PDUFA experience (approval spike missed, sell-the-news crash, stop proximity). All rules are sector-agnostic — designed for any date-certain binary catalyst (FDA, earnings, contract awards, permit rulings, drill results, patent rulings), not just biotech.

New rules: (1) 20-day SMA filter waiver for date-certain binary events within 15 trading days (with 25% position size reduction), (2) pre-catalyst GTC sell orders for 1/3 at +30% placed ≥2 days before event (also serves as gap risk mitigation), (3) binary event stop override at technical support (expires on resolution), (4) mandatory post-catalyst reassessment within 1 trading day (stop recalc, conviction re-rate, hold/trim/exit decision), (5) trailing stop precision — specified "20-day rolling high" lookback. Also added formal definition of "date-certain binary catalyst" to Entry Requirements.

| File | Change |
|------|--------|
| `Start Your Own/portfolio_rules.md` | Added 5 new rules to Risk Control + Liquidity Filters + Entry Requirements sections |
| `CLAUDE.md` | Updated Current State to reflect Week 28 status and rules v2 |

---

### 2026-03-29 — Makefile Weekend Fix + generate_pdf Unicode Sanitization

Fixed two bugs: (1) `make weekend` target had a Python `while` one-liner syntax error on Python 3.12; fixed with `exec()` wrapper. Also added `printf '\n'` pipe to handle the `--weekend-summary` capital injection prompt (same stdin issue as `make daily`). (2) `generate_pdf.py` failed on Unicode characters (em dash, bullet, arrows, checkmarks) not supported by Helvetica/latin-1; added `sanitize_latin1()` function that replaces all non-latin-1 characters before PDF rendering and added Unicode replacement map to `strip_inline()`.

| File | Change |
|------|--------|
| `Makefile` | Fixed `while` syntax error with `exec()` wrapper; added `printf '\n'` pipe for `--weekend-summary` stdin |
| `generate_pdf.py` | Added `sanitize_latin1()` function; replaced Unicode bullet chars in list rendering; added Unicode replacement map to `strip_inline()` |

---

### 2026-03-29 — NYSE-Aware make weekend Staleness Check + Manual Run Documentation

Fixed the `make weekend` portfolio staleness check to use `exchange_calendars` (XNYS) instead of the simple weekday roll-back (`while weekday >= 5: d -= 1 day`). The old logic always resolved to the most recent Monday–Friday date, which meant on holiday-shortened weeks (e.g., Good Friday) it expected Friday's data when the true last trading session was Thursday — causing a false "not current" failure. The new check finds the most recent NYSE session on or before today, matching the same holiday-aware logic added to `trading_script.py` in the prior session. Also added a note to `README_CLAUDE.md` documenting that manual runs of `trading_script.py` must include `--data-dir "Start Your Own"` or the CSV is written to the wrong path and `make weekend` will report stale data.

| File | Change |
|------|--------|
| `Makefile` | Replaced inline staleness check with NYSE-aware session lookup via `exchange_calendars` |
| `README_CLAUDE.md` | Added `--data-dir` requirement note for manual `trading_script.py` runs |
| `CLAUDE.md` | Updated Current State |

---

### 2026-03-27 — End-of-Week Auto-Skip for Daily Analysis

Added automatic detection of the last trading day of the week so that running `make daily` on a Friday (or holiday-shortened Thursday) skips the 6-section analysis and instead prompts the user to run `! make weekend`. The detection uses the `exchange-calendars` library (NYSE/XNYS calendar) to identify the final trading session of each calendar week — correctly handling NYSE holidays like Good Friday, where Thursday is the true last session. An `is_end_of_week` attribute is injected into the `<daily_summary>` XML tag (`"true"` or `"false"`), and a new Skip condition 2 in `analysis-workflow.md` triggers the skip when the attribute is `"true"`. A try/except fallback ensures the script degrades to Friday-only detection if the calendar library is unavailable.

| File | Change |
|------|--------|
| `requirements.txt` | Added `exchange-calendars>=4.5` dependency |
| `trading_script.py` | Added optional `exchange_calendars` import with `_HAS_XCALS` guard; added NYSE end-of-week logic emitting `is_end_of_week` attribute on `<daily_summary>` tag |
| `.claude/rules/analysis-workflow.md` | Restructured Daily Analysis section into two named skip conditions; added Skip condition 2 for end-of-week runs |
| `CLAUDE.md` | Updated Current State |

---

### 2026-03-25 — Telegram Daily Workflow (Designed, Not Implemented)

Designed a Telegram-driven daily workflow to replace the manual "Run daily:" trigger. At 5 PM Eastern (via launchd), a Telegram message is sent prompting the user for any trade changes. The user replies from their phone in the same structured format as "Run daily:". Claude Code (open in background) detects the reply and automatically runs the full piped-input daily workflow, followed by the 6-section analysis, and sends Section 4 (Final Decisions) back to Telegram.

**Status: Parked** — pending decision on automation level. Two paths documented:
- **Option A (Open laptop once)**: launchd sends prompt; user replies; Claude Code auto-detects reply on session start and runs everything. Requires Power Nap enabled.
- **Option B (Fully automated)**: Cloud agent or Claude API handles everything without opening the laptop. Requires Claude API key + remote agent infrastructure.

Full design in plan file: `~/.claude/plans/luminous-foraging-scroll.md`

| File | Planned Change |
|------|---------------|
| `telegram_bot.py` | CREATE — `--setup`, `--send-prompt`, `--send`, `--get-reply` modes |
| `Makefile` | ADD — `telegram-setup`, `telegram-prompt`, `telegram-send` targets |
| `~/Library/LaunchAgents/com.learnpython.microcap.telegram-prompt.plist` | CREATE — 5 PM weekday launchd schedule |
| `.claude/rules/analysis-workflow.md` | ADD — Telegram polling trigger section |
| `.gitignore` | ADD — `.telegram_config`, `.daily_state`, `.daily_reply_processed` |

---

### 2026-03-24 — Makefile Venv Fix and Piped-Input Daily Workflow

All Makefile targets (`daily`, `weekend`, `trade`, `graph`) now use `venv/bin/python` instead of bare `python`, which was resolving to system Python and causing the script to hang on import. Additionally, since Claude Code's `!` prefix does not support interactive stdin (`input()` calls fail with EOFError), the daily workflow now uses a piped-input pattern: the user tells Claude what inputs to provide (e.g., "Run daily: buy 17 REPL limit $7.05 stop $5.90/$5.80") and Claude constructs the answer sequence and pipes it to the script via the Bash tool. `! make weekend` is unaffected (no interactive prompts).

| File | Change |
|------|--------|
| `Makefile` | Changed all targets from `python`/`python3` to `venv/bin/python` |
| `CLAUDE.md` | Updated Daily Workflow section to document `Run daily:` pattern; updated Current State |

---

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
