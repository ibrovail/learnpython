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
  the injection-neutral series from the re-based peak); driver cap 2 + sector cap 3. *Several of
  these were measured for the first time on 10/11 — see the audit entry below.*
- **Complete**: **Phase 3.5** — on non-overlapping formation dates the composite clears t=2.0 at
  **no** horizon, and 20/40/60 sessions have too few independent observations for any verdict;
  effect sizes do rise with horizon. **Phase 4 (Dec) primary horizon = 20 sessions.** Index changes
  are the fifth unexplained-move category.
- **ADOPTED (10/11) — the rule audit: eleven rules measured, four changed.** The largest
  amendment the book has had and the first produced by measurement rather than reasoning or a
  single-trade post-mortem. Built on a **survivorship-free** 11-year frame (499 tickers, 37.5% of
  names delisted; `research/`), a **daily event-driven simulation** with real slippage, stops,
  sizing and share rounding, and a **train/test split at 2021-06-30** — every figure from one
  clean ablation (`research/rule_ablation.py`), so the numbers are comparable to each other.
  **Changed:** RISK-OFF **defensive profile removed**, the screener lane **closing** in RISK-OFF
  instead (+2.8pp, its sunset clause expiring as written); **stops not raised by default**
  (+10.3pp, the largest effect); **fractional shares permitted** (+3.2pp at $710, nothing at
  $10,000 — a small-book rule); **ceiling 5 → 10** (+3.9pp); **cash floor 15% → 5%** (+0.5pp). **Kept with evidence:** regime filter (+3.8pp, **reversing Phase 3.8's
  R2**), **3-per-sector cap** (3.0pp less drawdown at no cost, and it beats **40 of 40** shuffled-
  label placebos), 2% risk, the $5Bn cap (**lifting it tested worse — recommendation withdrawn**),
  the Day-1 rule, and the **10-session earnings guard — a removal I published and then withdrew**
  after finding a session-index bug in my own test. Breaker kept (fires in 2020, not in test); the
  re-entry ban is **inert**. **The goal was not met:** 12.2% test CAGR against SPY's 13.2%,
  at −15.4% drawdown against SPY's −24.5%. The gap is **cap-weighting**, not the rules — SPY beats
  equal-weighted S&P by 4.7pp on the same window — which is why mega-caps and leverage are a
  separate session. Reasoning in `Rules Amendment History.md` (2026-10-11).
- **In progress**: **One position, ~84% cash; Week 57 done — no buy.** CON 3 @ $35.31, last close **$37.19
  (+5.3%)**, stop **$34.49 / $34.34** as placed on 10/06. **Three raises were recommended in five
  sessions, two placed and irreversible, the third withdrawn on 10/11 under the new never-raise
  default.** The stop is safe — **2.95×ATR** of room (ATR(14) $0.916) against a 1.5×ATR floor, and
  max loss against entry is $2.46 = **0.34% of equity** — but note it now sits **above the
  10-session low ($34.05)**, so a −7.3% pullback takes it out. **No further raise without a
  written thesis change.** **Gap −3.73%** since re-base (−8.12% since inception), RISK-OFF; the
  5% floor lifted deployable to **79%**.
- **Decided (10/05) — PLX is prohibited (Israeli-affiliated) and is blocklisted.** The user's
  determination. It ranked **#9** on the 10/05 screen and **nothing in code or on the quote page
  would have caught it**: stockanalysis lists Country "United States", HQ Hackensack NJ, while its
  ProCellEx manufacturing and research base is in Israel. **The exclusion covers affiliation, not
  domicile** — which is why it cannot be screened by industry and must be checked per name. In
  `screener.py`'s `_PROHIBITED_TICKERS` since the **10/10** screen. *(The Week 56 research-log row
  reads `weak-catalyst`: the determination post-dates it and the log is append-only.)*
- **Adopted (10/05 b) — the catalyst lane gets a sourcing method, and since 10/11 it is the ONLY
  lane open in RISK-OFF.** The catalyst bucket had an eligibility test and no pipeline, so the
  funnel filled one bucket: **40 of 40 researched names came from the screener, 0 from catalyst
  discovery**, and the lane sat empty from 9/17 while the book held 85% cash. Fix: **Stage 0**
  reserves **≥3 stage-1 slots for catalyst names** whenever a slot is open (a *reallocation* — the
  screener lane returned 0 buys from 25 checks); `<research_trigger>` prints a **`<lanes>`** line;
  a qualifying catalyst is **a resolved fact with a dated earnings impact** (vs the prohibited
  binary case, an unresolved event with a dated announcement); **an index add counts only BEFORE
  its effective date** — ATRC ran +12.8% into 9/21 then gave back −11.82%. `source` in
  `log_research.py` is a controlled vocabulary (`catalyst:<category>`) because
  `trading_script.py` reads it to decide which cap a holding counts against. Reasoning in
  `Rules Amendment History.md` (2026-10-05 b).
- **Superseded (10/05) — `near_high` removed from the RISK-OFF defensive profile.** The whole
  profile went on 10/11, so its remaining legs and the deferred 0.90-threshold question are moot.
  What survives is the **open Phase 4 question of whether beta belongs in any future defensive
  test** — top-decile calm names had median beta **0.85 vs 0.84**, so `low_vol` never measured
  defensiveness. Full reasoning, including the corrected claim that beta had kept ATRC out of the
  profile (**beta was never a leg**), in `Rules Amendment History.md` (2026-10-05, 2026-10-11).
- **HOPE post-mortem (9/29)**: stopped out at $13.48 for **−$7.20 = 0.99% of equity, the budgeted
  risk to the cent**, 6 sessions held. **No rule failed; the entry did.** Bought **+0.15% above its
  50-day SMA** — the thinnest possible pass of the hard trend gate, flagged as thin in the Week 54
  report — and still sized to **28.8% of equity**. Its rate driver inverted within three sessions.
  **Still open after the 10/11 audit** (which tested rules, not entry quality): does a sub-1%
  trend-gate pass deserve reduced size or none? `pct_vs_sma50` is logged, so it is queryable once
  enough BUY rows mature.
- **Complete (9/19)**: review items adopted — one weekend question (R1); two-stage research funnel
  sized to the buys sought and extending to ranks 51–100 (R3); research log of buys *and* passes
  via `log_research.py` (R4); stop raise target 2.0×ATR (R5); regime computed by the script with a
  ±1% band → `<market_regime>`, `regime_history.csv` (R7, D3); small-cap label + size control
  (R8); six-section report (R2); dailies by exception via `<holding_review>` (R6). The
  `weekly-portfolio-report` skill is **retired** — it predates every current rule and conflicts
  with four. Detail in `Rules Amendment History.md` (2026-09-19 a–f).
- **Complete (9/28)**: **the 60-session re-underwrite now says what it tests** — the **quality**
  tests (revenue, earnings path, driver freshness, non-binary, prohibited business, trend and
  distance-from-base, earnings guard, placeable stop inside the 30% ceiling), **not** the regime
  **capacity** gates (defensive profile, half risk budget, 90-day catalyst window, position
  ceiling), which govern new capital only. A failed quality test is an exit; a failed capacity gate
  is a hold that **must name the gate in writing**. Reasoning and the honest accounting are in
  `Rules Amendment History.md` (2026-09-28). Deliberately left open: the review cadence *after* a
  pass — decide it when ATRC actually passes.
- **Next**: **Week 57 found no catalyst candidate** — four categories searched, 5 names logged (the
  first catalyst-sourced rows in 45). Three fell to **timing or exclusion**, not quality: ZIM
  (Israeli-affiliated — second week after PLX), WK and FRSH (SmallCap 600 adds past their effective
  dates), TORO (spin-off completed 10/08); only NNBR failed on merit (net income −$47.9M, beta
  2.66). The lane is short of events *still ahead of us* — **watch the December S&P rebalance
  announcement from late November.** Blackouts expire this week (HOPE ~10/13, ATRC ~10/15). CON: the
  $35.36 raise is eligible and **declined in writing**; no add into a closed lane.
- **Engineering, 9/29–10/11** (detail in `.claude/docs/implementation-notes.md`): session
  arithmetic routed through the **NYSE calendar** (the 30-session backstop had scaled days by 5/7,
  and `pd.bdate_range` collapsed every past-due earnings estimate to −1, pinning a permanent
  spurious FULL review); the re-underwrite trigger fires within **5 sessions** of 60; the
  post-stop-out **re-entry ban** prints as `<blackout>`; `research_log.csv` records
  **`pct_vs_sma50`**; `<holding_review>` names each holding's sector proxy. **10/11:**
  `live_simulation.run()` takes earnings **dates** rather than pre-indexed sessions — the old
  signature silently shifted every index when a split was run separately, which is what produced
  the withdrawn earnings-guard verdict.
- **Fixed (9/20–9/21)**: `inject_last_thesis.py` line-anchors its tags (it had deleted the weekend
  data block); entry stops need all three candidates — 1.75×ATR, 10-session low, 50-day SMA.
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
