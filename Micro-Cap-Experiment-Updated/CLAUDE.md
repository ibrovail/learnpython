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

- **Complete**: Week 47 deep research (8/2, Aggressive) — initiated ARDT 10sh @ $10.80 into its 8/4 print, chosen over screener momentum (AMCX missed Q2, PHAT cut guide, DMC identity-mismatch). Also `Strategic Pivot — Week 46 Readout` (gap repaired −21.1%→−2.6% since the April pivot, but the lead round-tripped twice). Rules added 8/4: **earnings-night live check** (browser for BOTH the AH quote AND the actual release — never infer beat/miss from price; news recency now browser-verified) and **stop range check** (a stop above the day's low is inside the noise; compute ATR, target 1.5–1.75×; de-risk by size, not stop tightness)
- **In progress**: Week 47 of 52, 6 positions (WKC/SHO/TDAY/TILE/ATRC/ARDT); equity $739.77 (8/4), cash $65.54 (8.9% — user-authorized 15%-floor override for the ARDT buy); gap −4.7% (low-beta lag on a +1.8% SPY day), TWR alpha +0.34%. ARDT printed **mixed** 8/4: revenue beat $1.622B, but EPS $0.12 missed $0.17 and adj. EBITDA −32% YoY; +6.98% AH on +67% op cash flow + reaffirmed guidance
- **Next**: 8/5 verify ARDT regular-session price — if it holds $11.30+, consider selling 3–4 sh into strength (de-risk by size, keep the wide $9.55 stop, ~2.5×ATR); TDAY + SHO both report 8/6 (do the live quote AND live release check that same evening); WKC +60% partial alert $41.60 (stop $37.65 locks +44.8%); SHO stop-raise trigger $12

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
