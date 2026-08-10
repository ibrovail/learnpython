# Micro-Cap Experiment

A live 12-month trading experiment using Claude Code to manage a real-money micro-cap stock portfolio, tracking alpha generation vs. S&P 500 benchmark.

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
| `Start Your Own/portfolio_rules.md` | Portfolio rules — read before every analysis session |
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

- **Complete**: Week 48 deep research (8/9, Neutral) — 8/6 exits: WKC stop $37.66 (**+44.8%**, best trade of the experiment), TDAY stop $7.77 (−4.7%, Q2 revenue −8.3% YoY = broken thesis), ARDT trim (+2.8%); net realized +$7.08. TILE beat-and-raise 8/7 (+9.09%). Week 48 plan: EXIT SHO, raise TILE/ATRC/ARDT stops, **no initiation** (4 BDCs excluded, 4 healthcare sector-capped, 2 arbs, FOXF = day 2 of a +12% gap → queued #1). Script fix: daily save now preserves same-day SELL rows on re-run
- **In progress**: Week 48 of 52, **3 positions** (TILE/ATRC/ARDT); equity $734.97 (8/10), cash **$404.79 (55.1%)**. SHO exited 8/10 @ $11.07 (+8.5%) — beat sold into, and the exit beat the $10.90 stop that would have fired the same day. Stops: TILE $35.75 (+11.4% locked), ATRC $37.90 (+10.5%), ARDT $10.35. Aggregate stop-risk ≈ −$0.65 — the book is effectively all house money
- **Next**: **FOXF queued #1** — enter on consolidation (higher-low base >$20 or pullback toward the $18.70 20-day SMA), 1.75×ATR stop ≈ $19.37. **Strategic flag: 55% cash cannot close a −5.0% gap in 6 weeks** — if the Week 49 screen is again structurally blocked (4 BDCs excluded + 4 healthcare sector-capped this week), revisit the constraint set with the user

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
