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

- **ADOPTED (10/11) — the rule audit: eleven rules measured, four changed.** The first amendment
  produced by measurement rather than reasoning or a single-trade post-mortem. Built on a
  **survivorship-free** 11-year frame (499 tickers, 37.5% delisted), a **daily event-driven
  simulation** with real slippage, stops, sizing and share rounding, and a **train/test split at
  2021-06-30** — every figure from one clean ablation, so they are comparable.
  **Changed:** defensive profile **removed** with the screener lane **closing** in RISK-OFF instead
  (+2.8pp, its sunset expiring as written); **stops not raised by default** (+10.3pp, the largest
  effect); **fractional shares** (+3.2pp at $710, ~0 at $10,000 — a small-book rule); **ceiling
  5 → 10** (+3.9pp); **cash floor 15% → 5%**. **Kept with evidence:** regime filter (+3.8pp,
  **reversing Phase 3.8 R2**), **3-per-sector cap** (3.0pp less drawdown, free, beats **40 of 40**
  placebos), 2% risk, the $5Bn cap (**lifting it tested worse — recommendation withdrawn**), the
  Day-1 rule, and the **earnings guard — a removal I published then withdrew** after finding a
  session-index bug in my own test. Breaker fires in 2020, not in test; re-entry ban **inert**.
  **The goal was not met:** 12.2% test CAGR vs SPY 13.2%, at −15.4% drawdown vs −24.5%. The gap is
  **cap-weighting** — SPY beats equal-weighted S&P by 4.7pp — which is why mega-caps are a separate
  sleeve. (`Rules Amendment History.md`, 2026-10-11.)
- **In progress**: **One position, ~84% cash; no buy since 9/22.** CON 3 @ $35.31, last close **$37.19
  (+5.3%)**, stop **$34.49 / $34.34** as placed on 10/06. **Three raises were recommended in five
  sessions, two placed and irreversible, the third withdrawn on 10/11 under the new never-raise
  default.** The stop is safe — **2.95×ATR** of room (ATR(14) $0.916) against a 1.5×ATR floor, and
  max loss against entry is $2.46 = **0.34% of equity** — but note it now sits **above the
  10-session low ($34.05)**, so a −7.3% pullback takes it out. **No further raise without a
  written thesis change.** **Gap −3.73%** since re-base (−8.12% since inception), RISK-OFF; the
  5% floor lifted deployable to **79%**.
- **Decided — three names prohibited in eight days, and the detention exclusion now covers
  SUPPLIERS: PLX (10/05), AISP and FIRY (10/12).** The user's determinations, all blocklisted in
  `screener.py`. **All three were invisible to every coded check**, each failing a different one:
  PLX lists Country "United States", HQ Hackensack NJ, while its ProCellEx base is in Israel;
  **AISP** reads industry *Software — Infrastructure* and sells DHS video/data platforms, so it was
  not a detention *operator* under the old wording — now widened, a **vendor to a detention or
  immigration-enforcement agency is prohibited though it runs no facility**; **FIRY** (ex-Skillz) is
  NYSE/Las Vegas with Israel in its operating footprint. So `_PROHIBITED_TICKERS` is **a record of
  determinations, not a filter that generalises** — it catches a name the second time, never the
  first. Only the PRV prohibited-business check, run against what a company *does* rather than any
  field, catches a first encounter, which is why it is ordered first. Reasoning in
  `Rules Amendment History.md` (2026-10-12). *(The Week 56 PLX research-log row reads
  `weak-catalyst`: the determination post-dates it and the log is append-only.)*
- **Decided (10/12) — two TFSA sleeves, operating separately; no transfer path.** A second
  experiment account (mega-cap / no market-cap limit) is being initialised by the cloud session.
  **No capital moves between sleeves and no `--cash-flow` path is to be built** — reasoning in
  `Rules Amendment History.md` (2026-10-12 b), including that the gap arithmetic is already
  correct for withdrawals, and that the "lifting the $5Bn cap tested worse" result does **not**
  transfer to a sleeve with its own rules. **Idle cash here is a risk budget, not spare capital.**
- **Earlier state — moved out of this file 2026-10-12** to keep it inside the 200-line cap
  `documentation.md` sets: the 9/17 indefinite-system revision, Phase 3.5, the 9/19 R1–R8
  adoptions, the 9/28 re-underwrite clarification, the 9/20–21 fixes, the HOPE post-mortem,
  and the two 10/05 amendments. All of it is in **`.claude/docs/state-history.md`**, with the
  full reasoning in `Experiment Details/Rules Amendment History.md`.
- **Next**: **catalyst sourcing audit (10/12) — all 6 categories searched, 6 names, 0 survive**
  (`Weekly Deep Research (MD)/Week 58 Catalyst Sourcing Audit.md`). Week 57 searched only **3 of 6**
  and the missing three produced 4 of the names, so the lane was under-searched — but **the
  constraint is not sourcing**: 4 of 6 died on earnings quality or size, NNBR/BORR/FIRY on the same
  code. **Dated-catalyst small caps skew loss-making.** The **index check moves to the daily** (both
  October adds were intra-quarter M&A replacements with 3–5 days' notice; next release ~12/04), and
  **a demotion is not a promotion** (OLN left the MidCap 400 the same morning). **Correction:** the
  "≥6.67% minimum stop width" is `risk ÷ name_cap` = 2%/30% — **a ratio of two rule constants,
  independent of book size** — and it **excludes nothing** (size is the minimum of the three, so a
  tighter stop just deploys less risk). Tested: the ceiling binds **5%** of entries, so 2% risk is
  real, and no ceiling change is adopted. **The real exposure is under-deployment:** 86% of the
  100 watchlist names sit below the 3.81% ATR a 1.75×ATR stop needs, so the median candidate
  carries **1.49% risk — 74% of the budget**. It bites on **composite-ranked** names, not on
  `pct_vs_sma50` selections — a second argument for that selector. Blackouts
  expire this week (HOPE ~10/13, ATRC ~10/15). CON's $35.36 raise stays **declined in writing**.
- **Engineering, 9/29–10/11** (detail in `.claude/docs/implementation-notes.md`): session arithmetic
  routed through the **NYSE calendar** (the backstop had scaled days by 5/7, and `pd.bdate_range`
  collapsed past-due earnings estimates to −1, pinning a spurious FULL review); re-underwrite
  trigger fires within **5 sessions** of 60; **re-entry ban** prints as `<blackout>`;
  `research_log.csv` records **`pct_vs_sma50`**; `<holding_review>` names each sector proxy.
  **10/11:** `live_simulation.run()` takes earnings **dates** not pre-indexed sessions — the old
  signature silently shifted every index when a split ran separately, which produced the withdrawn
  earnings-guard verdict.
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
