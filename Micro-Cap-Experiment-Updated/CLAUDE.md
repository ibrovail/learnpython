# Micro-Cap Live Portfolio

An indefinite live process using Claude Code to manage a real-money small/micro-cap stock
portfolio, tracking alpha vs. the S&P 500. The fixed 52-week experiment closed at the
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
| `screener.py` | Quantitative screener: Finviz universe → yfinance signals → ranked watchlist |

## Commands

```bash
make daily      # Run trading script after 4 PM (Claude auto-analyzes output)
make screen     # Run quantitative screener (outputs watchlist CSV)
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
2. Claude auto-analyzes the XML output with live web search
3. Review recommendations; specify any trades in the next `Run daily:` command

**Note:** Always use `Run daily:` (not the `!` shell prefix on `make daily`) — Claude Code's `!` prefix does not support interactive stdin. Likewise, say `run weekend` (not `! make weekend`) so Claude asks the 4 session directive questions first, then runs `make weekend` with answers as CLI args (SECTOR, TIMING, RISK, POSITIONS).

## Current State

- **Complete**: **Indefinite-system rules revision (2026-09-17)** — 13 decisions, 6 rules deleted,
  3 new. Horizon **40–60 sessions**; trailing stop is the **only** exit (partials and the whole
  deferral apparatus deleted); **binary-thesis entries prohibited**; risk-per-trade **2%**;
  **drawdown circuit breaker** on the injection-neutral series (−20% de-risk / −30% cash); driver
  cap 2 + uniform sector cap 3; RISK-OFF capacity raised. `portfolio_rules.md` rewritten as a
  standing manual, history split to `Experiment Details/Rules Amendment History.md`. Found in
  passing: raw-equity max drawdown understated the true figure by **12.3 points** (−24.99% vs
  −37.26%).
- **Complete**: **Phase 3.5 reported** (`9f47b4a` pre-reg → `afbf106` raw → interpretation).
  **All six composite signals retain support at 40 sessions; the composite stands (rule 5).**
  Screener edge rises with horizon — top-50 excess **+0.78pp at 10 sessions → +5.57pp at 40 →
  +6.99pp at 60** — and Phase 2's quintile-spread weakness reverses. **But the study's own
  sample-size floor was mis-specified**: it counted raw dates, when overlapping weekly windows
  give an effective n of **1.75 at 40 sessions and 0.83 at 60**, so the t-statistics (up to 14.8)
  are not trustworthy. Verdict: *no evidence of breakdown at the adopted horizon*, not
  confirmation. `mom20` passes at 40s but stays out of the composite per pre-registered rule 8.
- **Position**: **ATRC 3 sh +66.6%** ($57.14), stop **$53.40 / $53.25**. Equity **$724.44**, cash
  **$553.02 (76.3%)**, regime RISK-OFF. **47 sessions held — 60-session re-underwrite due in ~13
  sessions.** Current drawdown from the re-based peak −1.67%, clear of the breaker.
- **Next**: 9/17 daily after the 4 PM close. Week 54 research can now deploy under RISK-OFF
  (up to 3 catalyst positions at standard sizing, plus half-size defensive screener entries) —
  the previous rules made high cash arithmetically unavoidable. Re-entry bans: PAR ~9/29,
  VTS ~9/30.

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
