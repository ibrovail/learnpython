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

- **Complete**: Week 51 research (8/31, Aggressive, run intraday). **Rule relaxed — "never lower a stop" suspended for ONE documented restoration** (recorded in `portfolio_rules.md`): all five stops had drifted *below* the `max(1.5×ATR,…)` band the rules themselves mandate. Orders: **ATRC $46.30→$44.35**, **PAR $17.50→$17.05**, **TILE $37.10→$36.25** (all ~1.75×ATR). CADL and WWW deliberately left tight. Also fixed a **screener partial-bar bug** — run during market hours, `volume.iloc[-1]` used today's incomplete bar, collapsing every volume ratio to 0.10–0.38× and floating **4 BDCs (excluded class) into the top 7**
- **In progress**: Week 51 of 52, **5 positions** (PAR/TILE/ATRC/CADL/WWW), live equity **$758.52**, cash $69.34 (9.2%, floor 8% since 8/28); **gap −1.64%**. The decisive finding: **$69 of cash cannot close a $19 gap** (needs +27.5%, or SPY +9.2% at 3× leverage) — only the **$683 of holdings** can, needing **+2.79%**. So every constraint except the stop rule is arithmetic noise. Restoration costs **$14.20 (1.87%)**, taking all-stops-fire −3.69% → −5.56%
- **Next**: 13 sessions (Sept 7 = Labor Day). **ATRC's old stop survived by 31 cents today** (low $46.61 vs $46.30) — that near-miss forced the rule question. No earnings, no catalysts. Being stopped into cash locks the deficit permanently; the whole outcome now rests on ATRC/PAR/TILE converting their restored room

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
