# Strategic Pivot — Week 30 Readout

**Date:** April 12-13, 2026
**Author:** Claude Code (AI analyst) + portfolio manager
**Audience:** Anyone following the Micro-Cap Experiment

---

## Executive Summary

At Week 30 of a 52-week live-money micro-cap trading experiment, the portfolio hit an inflection point. Four consecutive binary bet failures wiped ~$97 from peak equity, the portfolio trailed its S&P 500 benchmark by $101, and a structural analysis revealed that the existing rules funneled every trade into biotech PDUFA plays — creating a single-sector, coin-flip dependency. Over a 24-hour working session, the strategy was diagnosed, redesigned, and rebuilt. A quantitative stock screener was created, portfolio rules were overhauled, and the first sector-diversified deployment plan was produced. This document captures the full arc: problem, options, decision, implementation, and next steps.

---

## 1. Problem Statement

### The Symptom

The portfolio entered Week 30 at $378.80 in 100% cash, trailing the S&P 500 equivalent of $480.04 by $101.24 (-21.1%). With 22 weeks remaining, the experiment needed to generate ~27% total return just to match the benchmark — let alone beat it.

### The Root Cause

A structural analysis of the portfolio rules revealed a funneling effect:

1. **Entry requirement:** "Catalyst within 60 days" — meant every new position needed a dated event
2. **Binary event framework:** "Pass/fail outcome expected to move the stock >=20%" — rewarded high-volatility plays
3. **Web-search-based candidate sourcing:** Claude's live web searches naturally surfaced headline-grabbing FDA decisions (PDUFAs) over quieter momentum setups in other sectors

These three forces combined to produce a portfolio that was effectively a series of biotech coin flips:

| Trade | Sector | Type | Outcome | P&L |
|-------|--------|------|---------|-----|
| RCKT | Biotech | PDUFA | Stopped out | -$25.89 |
| REPL (lot 1) | Biotech | PDUFA | Stopped out | -$10.71 |
| REPL (lot 2) | Biotech | PDUFA | CRL, manual exit | -$16.77 |
| GRCE | Biotech | PDUFA | Stopped out | -$13.92 |
| **Total** | | | **4/4 losses** | **-$67.29** |

The experiment had no sector diversification, no systematic candidate generation, and no mechanism to find alpha outside of binary healthcare events.

### Why This Matters

The experiment's stated goal is alpha generation vs. the S&P 500 over 12 months. A strategy that concentrates in one sector and depends on binary outcomes has:
- **High variance:** each trade is roughly 50/50, independent of skill
- **No edge accumulation:** wins and losses are event-driven, not thesis-driven
- **Survivorship risk:** a streak of losses (which happened) can put the benchmark gap out of reach

---

## 2. Options Considered

Three strategic paths were evaluated:

### Path A — Stay the Course (Refine Binary Bets)

Keep the existing rules but tighten risk management. Reduce binary bet sizing, add more pre-catalyst exit orders, and hope for better outcomes on the next set of PDUFAs.

**Pros:** No code changes, familiar workflow
**Cons:** Doesn't fix the structural bias. Same sector concentration. Same coin-flip dependency. The rules that produced 4 consecutive losses remain unchanged.

**Verdict:** Rejected. Repeating the same approach and expecting different results.

### Path B — Hybrid Quant Screener + Claude Analyst

Build a quantitative stock screener that systematically scans 1,000+ micro/small-cap stocks across ALL sectors, surface 15 ranked candidates per week, and feed them into the existing weekend analysis. Claude evaluates the screener's output and makes final picks — combining quantitative breadth with qualitative judgment.

**Pros:** Eliminates sector bias mechanically. Systematic candidate generation. Preserves the option for catalyst plays (capped). Builds on existing infrastructure.
**Cons:** Requires building new tooling. Screener signals (momentum, volume) may not capture fundamental value. Data quality risk from free sources.

**Verdict:** Selected. Best balance of impact, feasibility, and risk mitigation.

### Path C — Full ML Trading Model

Replace the analyst workflow entirely with a machine learning model (random forest, gradient boosting, or neural network) trained on historical micro-cap data to predict 5-20 day returns.

**Pros:** Removes human bias entirely. Can process more signals than a human analyst.
**Cons:** Requires labeled training data the experiment doesn't have. Overfitting risk is extreme on micro-caps. Black-box decisions are hard to audit. Would take weeks to build and validate — time the experiment doesn't have.

**Verdict:** Rejected for now. Possible future enhancement once the screener provides a data pipeline.

---

## 3. Path B — Design and Implementation

### Architecture

The hybrid approach has two layers:

1. **Quantitative layer** (`screener.py`): Scans the full micro/small-cap universe weekly, ranks by momentum + volume + volatility signals, outputs a sector-tagged watchlist
2. **Qualitative layer** (Claude Code): Evaluates screener candidates via live web research for catalysts, fundamentals, bear cases, and thesis quality. Makes final buy/sell decisions.

The screener surfaces candidates; Claude decides.

### What Was Built

#### screener.py (New File — ~250 lines)

A Python script that runs the full pipeline:

1. **Universe pull:** Connects to Finviz and pulls ~1,064 stocks matching: market cap <= $2B, price >= $1, average volume >= 500K shares. All sectors included — no filtering by industry.

2. **Signal enrichment:** Downloads 30 days of price/volume history from Yahoo Finance (with Stooq fallback) for each stock. Calculates:
   - 20-day momentum (price return)
   - 5-day momentum (short-term trend)
   - Volume ratio (current vs. 20-day average — detects accumulation)
   - Relative strength vs. IWM (outperformance vs. small-cap benchmark)
   - Bollinger Band width (volatility squeeze detection)
   - SMA position (above/below 20-day moving average)

3. **Composite scoring:** Ranks all stocks by a weighted composite:
   - 40% momentum (20-day return percentile rank)
   - 30% volume breakout (volume ratio percentile rank)
   - 30% volatility squeeze (inverse Bollinger Band width — tighter = higher rank)

4. **Output:** Top 15 candidates written to `Start Your Own/watchlist.csv` and printed as a formatted table. Each candidate includes sector tag, data confidence rating, and all signal values.

**Key design decisions:**
- **Rank, don't threshold:** Uses percentile ranks instead of hard cutoffs to avoid overfitting
- **Standard lookbacks only:** 5-day, 20-day, 50-day — no exotic parameters
- **Round weights:** 40/30/30 — no optimized coefficients
- **Data confidence flags:** Each stock rated HIGH/MEDIUM/LOW based on data completeness

#### Portfolio Rules Overhaul

Added four new sections to `portfolio_rules.md`:

**Allocation Framework:**
- **Catalyst plays** (existing binary bet rules): capped at 1 position, max 15% of equity. This prevents another REPL+GRCE scenario where two binary bets fail simultaneously.
- **Momentum/technical plays** (new): 3-4 positions sourced from screener. No catalyst date required — entry based on momentum, volume, and technical setup. 5-day minimum hold to prevent overtrading.

**Sector Diversification:**
- Max 2 of 5 positions in the same GICS sector. If 2 positions are in Healthcare, no new Healthcare candidates until one exits. Enforced mechanically via sector tags in the screener output.

**Slippage Guard:**
- Order size must not exceed 10% of average daily dollar volume. Prevents moving the market on thin micro-caps.

**Momentum Regime Freeze:**
- If IWM drops below its 50-day SMA mid-week, freeze all new momentum initiations. Prevents chasing momentum into a regime change.

#### Weekend Workflow Integration

- `make weekend` now auto-runs the screener before the staleness check
- The screener watchlist is injected as XML into the weekend analysis prompt
- Claude must evaluate at least the top 5 screener candidates before selecting positions
- For each rejected candidate, Claude states why in one line
- Sector cap is checked before finalizing any portfolio

#### Platform Constraint Documentation

During order placement, discovered that the trading platform does not support GTC limit sell orders — only GTC stop-limit sells. Updated portfolio rules to use price alerts + DAY limit sells as the workaround for pre-catalyst profit-taking targets.

### Weakness Mitigations

Every identified weakness has a specific countermeasure:

| Weakness | Mitigation | Implementation |
|----------|-----------|----------------|
| Biotech/sector bias | Max 2/5 positions per GICS sector + sector-tagged screener | `portfolio_rules.md` + `screener.py` |
| Binary bet variance | Catalyst plays capped at 1 position, 15% equity | `portfolio_rules.md` Allocation Framework |
| Data quality | Yahoo-to-Stooq fallback + confidence flags + Claude web verification | `screener.py` + `analysis-workflow.md` |
| Overfitting | Standard factors, round weights, rank not threshold | `screener.py` scoring logic |
| Trade costs / overtrading | 5-day minimum hold + slippage guard (<=10% ADV) | `portfolio_rules.md` |
| Momentum crash | IWM regime filter + composite score (not pure momentum) | `portfolio_rules.md` + `screener.py` |
| GTC limit sell unavailable | Price alerts + DAY limit sells | `portfolio_rules.md` Order Defaults |

---

## 4. First Deployment — Week 30 Report

The Week 30 deep research report was the first produced under the new system. Three parallel research agents evaluated 15 screener candidates plus additional non-screener finds across all sectors.

### Selected Positions

| Ticker | Sector | Type | Shares | Limit | Cost | % Equity | Key Thesis |
|--------|--------|------|--------|-------|------|----------|------------|
| MRAM | Technology | Momentum | 10 | $10.30 | $103.00 | 27.2% | MRAM tech leader, AEC-Q100 auto qualification May 2026, 238 design wins, earnings Apr 29 |
| ORN | Industrials | Momentum | 7 | $11.75* | $82.25 | 21.7% | Specialty marine/infrastructure construction, record backlog, earnings Apr 28, estimates up 44.7% |
| ECVT | Basic Materials | Momentum | 5 | $14.00* | $70.00 | 18.5% | Specialty catalyst company, $465M debt paydown, analyst upgrades, earnings late May |
| RBBN | Technology | Catalyst | 15 | $2.50 | $37.50 | 9.9% | Post-25% washout, Q1 earnings Apr 28 vs. lowered bar, Verizon expansion |

*Limit prices adjusted from original report based on pre-market and opening price action on April 14.

**Post-trade cash:** $86.05 (22.7%) — above 15% minimum

### What Changed vs. Previous Strategy

| Dimension | Before (Weeks 1-29) | After (Week 30+) |
|-----------|---------------------|-------------------|
| Candidate sourcing | Claude web search (ad hoc) | Quantitative screener (1,064 stocks) + Claude research |
| Sector exposure | 100% Healthcare/Biotech | Technology, Industrials, Basic Materials |
| Position count | 1-2 concentrated bets | 4 diversified positions |
| Thesis type | Binary (PDUFA pass/fail) | Momentum + fundamentals + catalyst |
| Max single stop-out | ~$25 (6.6% of equity) | ~$10.70 (2.8% of equity) |
| Max combined stop-out | ~$41 (9.7% of equity) | ~$31 (8.2% of equity) |
| Catalyst dependency | 100% of positions | 25% of positions (1 of 4) |

---

## 5. Next Steps

### Immediate (Week 30-31)

- **Monday April 14:** Execute 4 buy orders. Verify ORN price vs. 20-day SMA at open. Confirm all fills via `Run daily:` command.
- **By April 24:** Set price alert for RBBN at $3.25 (+30% from entry) for pre-catalyst profit-taking ahead of Apr 28 earnings.
- **April 28:** ORN and RBBN earnings. Mandatory post-catalyst reassessment within 1 trading day — recalculate stops, re-rate conviction, hold/trim/exit decision.
- **April 29:** MRAM earnings. Same post-catalyst protocol.

### Medium-Term (Weeks 31-40)

- **Weekly screener runs:** Each weekend, the screener regenerates the watchlist. As positions resolve (earnings pass, stops hit, profit targets reached), rotate into new screener candidates.
- **Performance tracking:** Compare the momentum/technical approach to the prior binary bet approach on a risk-adjusted basis. Track Sharpe, Sortino, and max drawdown under the new regime.
- **Screener refinement:** Monitor which signals (momentum, volume, volatility squeeze) are most predictive. Adjust composite weights if a clear pattern emerges after 8+ weeks of data — but avoid premature optimization.

### Long-Term (Weeks 40-52)

- **Benchmark gap assessment:** At Week 40, evaluate whether the diversified approach is closing the $101 gap. If on track, maintain course. If not, consider increasing position concentration (5 positions instead of 4) or extending catalyst play allocation.
- **ML exploration (Path C revisited):** By Week 40, the screener will have generated ~10 weeks of candidate data with known outcomes. This could serve as a labeled dataset for a simple ML model — random forest predicting 5-day returns from screener signals. Only pursue if the data volume is sufficient and the current approach is underperforming.
- **Experiment conclusion:** Week 52 final report comparing total return, risk metrics, and strategy evolution across the full 12 months.

---

## 6. Files Modified in This Session

| File | Change |
|------|--------|
| `screener.py` | **CREATED** — quantitative screener (Finviz universe, yfinance signals, composite ranking) |
| `Start Your Own/portfolio_rules.md` | Added Allocation Framework, Sector Diversification, slippage guard, momentum regime freeze, platform constraint |
| `.claude/rules/analysis-workflow.md` | Updated weekend flow with screener evaluation rules and sector cap checks |
| `Start Your Own/weekend_summary.md` | Added screener-first research directive |
| `trading_script.py` | Injected screener watchlist XML into weekend summary output |
| `Makefile` | Added `screen` target; `weekend` auto-runs screener |
| `requirements.txt` | Added `finvizfinance>=0.16`, `tabulate>=0.9` |
| `Weekly Deep Research (MD)/Week 30 Full.md` | **CREATED** — first 10-section report under new strategy |
| `Weekly Deep Research (MD)/Week 30 Summary.md` | **CREATED** — thesis review summary |
| `Weekly Deep Research (PDF)/Week 30.pdf` | **CREATED** — PDF version |
| `README_CLAUDE.md` | Corrected weekend workflow (say "run weekend" to Claude, not `! make weekend`) |
| `CLAUDE.md` | Updated Current State |
| `.claude/docs/implementation-notes.md` | Added session change log entries |

---

## 7. Key Metrics to Watch

| Metric | Current (Week 30) | Target (Week 52) |
|--------|-------------------|-------------------|
| Portfolio equity | $378.80 | > $480 (beat S&P equivalent) |
| Benchmark deficit | -$101.24 (-21.1%) | > $0 (positive alpha) |
| Sector concentration | 0 (100% cash) | <= 2 per GICS sector |
| Win rate (binary bets) | 0/4 (0%) | N/A (pivoted away) |
| Max drawdown | -24.99% | < -30% (survivable) |
| Sharpe ratio | 1.97 | > 1.5 (maintain) |
| Consecutive stop-outs | 4 | < 3 under new regime |

---

*This document captures the state of the experiment as of April 13, 2026. It will not be updated retroactively — future readouts will be separate documents.*
