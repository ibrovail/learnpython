# Implementation Notes

Reference documentation for key implementation decisions and change history.

---

### 2026-06-11 — Time-Weighted Return (TWR) Analytics

Added injection-neutral performance reporting to `_compute_portfolio_metrics`. Raw equity growth was misleading because $547.64 of total capital was injected across 5 tranches — equity could rise purely from contributions, not performance. TWR fixes this by chaining daily equity returns while removing the step-change on each injection day (start-of-day convention: `factor_t = Equity_t / (Equity_{t-1} + injection_on_t)`; each injection attributed to the first trading session on or after its date). Three new rows now print in both the daily and weekend `<risk_metrics>` tables: **Time-Weighted Return (cum)** (portfolio, injection-neutral), **S&P 500 Return (cum)** (^GSPC price return over the same window — injection-neutral by nature), and **TWR Alpha (cum)** (the difference). First live read: portfolio +9.54% vs S&P +11.49% since inception → −1.95% cumulative alpha, a far more trustworthy figure than the CAPM annualized alpha (+1287%, R²=0.04). `PortfolioMetrics` gained `twr`/`twr_spx`/`twr_alpha` fields (defaulted to NaN, so early-return paths are unaffected); `_print_risk_metrics` gained matching optional params.

| File | Change |
|------|--------|
| `trading_script.py` | Added TWR/S&P-cum/alpha computation in `_compute_portfolio_metrics`; 3 new fields on `PortfolioMetrics`; 3 new params + rows in `_print_risk_metrics`; updated both call sites (daily + weekend) |
| `CLAUDE.md` | Updated Current State |

### 2026-05-29 — Week 37 Deep Research

Weekend deep research for Week 37 of 52. Directives: Wide net / Within 10 days / Aggressive / Max 5 positions. Key events this week: OSPN entered 5/26 (12sh @ $13.00, +11.1% in 3 days), ECVT stopped out 5/29 @ $13.24 (-$3.74, -5.4% from entry). Portfolio at $672.44 equity (+3.0% WoW), trailing S&P by $88.36 (-11.6%). Evaluated 7 candidates (IMPP, CRNC, AIOT, CLPT, CDZI, PRGS, MGTX). CRNC rejected on distance-from-base (20-day SMA +22% exceeds 20% gate). IMPP selected for 5th slot: Q1 revenue +25.8% beat, EPS +67.3% beat, fleet expanding 21→26 vessels by Q3. Entry: 19sh @ $5.40 limit, stop $4.80/$4.70. ACCO downgraded 2/5 — recycling decision deferred to Week 38. WKC stop raised to $27.90/$27.75; SHO stop raised to $10.00/$9.90. Post-trade: 5 holdings across 4 sectors, cash 15.6%, worst-case aggregate stop-out -5.6% equity.

| File | Change |
|------|--------|
| `Weekly Deep Research (MD)/Week 37 Full.md` | **CREATED** — full 10-section deep research |
| `Weekly Deep Research (MD)/Week 37 Summary.md` | **CREATED** — Section 9 thesis review only |
| `Weekly Deep Research (PDF)/Week 37.pdf` | **CREATED** — PDF render of full report |
| `CLAUDE.md` | Current State refreshed for Week 37 |

---

### 2026-05-23 — Week 36 Deep Research

Weekend deep research for Week 36 of 52. Directives: Wide net / Within 10 days / Aggressive / Max 5 positions. ORN stopped out 5/21 (+13.6% gain on the lot). SHO upgraded 3/5→4/5 after hitting 52-week high ($10.66). OSPN selected from screener as new entry: 12sh @ $13.00 limit, stop $12.00/$11.90. WKC stop raised $25.00→$26.50. Post-trade: 5 holdings across 5 sectors, cash 21.7%.

| File | Change |
|------|--------|
| `Weekly Deep Research (MD)/Week 36 Full.md` | **CREATED** — full 10-section deep research |
| `Weekly Deep Research (MD)/Week 36 Summary.md` | **CREATED** — Section 9 thesis review only |
| `Weekly Deep Research (PDF)/Week 36.pdf` | **CREATED** — PDF render of full report |
| `CLAUDE.md` | Current State refreshed for Week 36 |

---

### 2026-05-18 — Week 35 Deep Research

Weekend deep research for Week 35 of 52. Directives: Wide net / Within 10 days / Neutral risk / Max 5 positions. Recommended Neutral (not Aggressive as user initially leaned) because two stop-outs the prior week argued against post-loss tilt. Key research findings: WKC thesis materially upgraded (Q1 print 4/23 blew out the marine segment that was the entire bear case — Marine GP +82% YoY, FY26 EPS guide raised to $2.65-2.85 from $2.20-2.40); ECVT $100M Term Loan B add-on May 14 funds Calabrian close (end of Q2); ORN AGM 5/19 administrative; ACCO Q1 beat with no price response. Evaluated top 5 screener picks (SG, SHO, NAC, PRA, KW) + 3 momentum names (AVAH, LPG, VTS). Skipped AVAH on post-earnings cooldown (5/14 print), LPG on stacked binary risk (ex-div 5/18 + earnings 5/20), KW on capped deal-arb. Top pick: **SHO** — Q1 RevPAR +14.6%, raised guide, $458M buyback remaining, post-print consolidation cleared. Order: 11 sh @ $10.20 limit DAY for Tuesday 5/19, stop $9.40/$9.30 (1.75× ATR). Post-trade: 5 holdings across 5 sectors, 35% cash, worst-case aggregate stop-out -2.0% equity.

| File | Change |
|------|--------|
| `Weekly Deep Research (MD)/Week 35 Full.md` | **CREATED** — full 10-section deep research |
| `Weekly Deep Research (MD)/Week 35 Summary.md` | **CREATED** — Section 9 thesis review only |
| `Weekly Deep Research (PDF)/Week 35.pdf` | **CREATED** — PDF render of full report |
| `CLAUDE.md` | Current State refreshed for Week 35 |

---

### 2026-05-12 — Double Exit: ARLO + ARDX

Two positions closed on 2026-05-12. (1) **ARLO** manual SELL LIMIT 8 @ $13.70 per `entry-discipline.md` day-1 drawdown rule, realized -$12.40 / -10.2%. (2) **ARDX** manual SELL LIMIT 17 @ $6.48 — stop level — that user flagged should have auto-executed. The script's auto-stop check did not trigger because the close ($6.61) sat above the $6.48 stop even though the intraday low touched it. Likely a close-vs-low logic gap in `trading_script.py:482-741` (process_portfolio) worth auditing before the next stop-eligible position. Final state: 4 holdings (ECVT, ORN, WKC, ACCO), cash $342.18, equity $657.64, 52% cash.

Process learning: the manual-sell flow asks "log another trade?" after each fill. Feeding the inputs as `"\ns\nTICKER\nSHARES\nPRICE\n\n\n"` (extra trailing newlines) makes the second prompt cleanly return Enter so the script proceeds to the analytics + CSV-save block without an EOF crash. Without the extra newline the portfolio CSV write is skipped and snapshot must be hand-patched.

| File | Change |
|------|--------|
| `Start Your Own/chatgpt_trade_log.csv` | 2026-05-12: appended ARLO sell (-$12.40) and ARDX sell (-$12.24) rows |
| `Start Your Own/chatgpt_portfolio_update.csv` | 2026-05-12 snapshot: 4 holdings, cash $342.18, equity $657.64 |
| `CLAUDE.md` | Current State: 4 holdings, both stops fired, auto-stop bug logged for next-step |

---

### 2026-05-11 — Entry Discipline Rules (Post-ARLO Failure)

ARLO was entered Monday 5/11 at $15.25 per the Week 34 deep research recommendation and closed -10.9% same-day (low $13.30, close $13.59) without negative news. Root causes: (1) bought 3 trading days after a +11.6% earnings pop at the post-print high; (2) entry ~60% above 50-day SMA, mean-reversion risk dominated; (3) stop $13.85 sat only ~1.5× ATR below entry — too tight for ARLO's normal daily range; (4) no pre-open verification; (5) screener score 0.802 HIGH was over-weighted as conviction; (6) broker stop not surfaced — recommendation didn't include explicit "place stop before open" reminder, so entry was naked. User instruction: exit at Tuesday open. New `entry-discipline.md` rules file codifies: post-earnings 3-day cooldown above +5%, ≤40% above 50-SMA limit, 1.75× ATR minimum stop sizing, pre-open verification, screener score = sourcing not conviction, day-1 drawdown auto-exit at -8%, broker-stop surfacing.

| File | Change |
|------|--------|
| `.claude/rules/entry-discipline.md` | **CREATED** — hard rules for new-position selection |
| `CLAUDE.md` | Current State updated to reflect ARLO failure and exit decision |
| `Start Your Own/chatgpt_portfolio_update.csv` | Patched: 2026-05-08 backfilled with closes; ARLO row added for 2026-05-11 (8sh @ $15.25, cur $13.59, PnL -$13.28); cash $122.42; total equity $659.40 |
| `Start Your Own/chatgpt_trade_log.csv` | ARLO BUY LIMIT 8 @ $15.25 logged 2026-05-11 |

---

### 2026-05-11 — Week 34 Deep Research + Stale `! make` References Purged

Ran Week 34 deep research: 5 holdings (ECVT/ORN/WKC/ACCO/ARDX) all post-Q1 earnings, ARLO selected as 6th position from the screener watchlist (Q1 +26% revenue, services 60% of mix at 85% GM, +318k paid accts vs 190–230k target; FY26 guide $550–580M). ECVT conviction upgraded 2/5 → 4/5 on a Q1 beat (+50% sales, +87% EBITDA) and raised FY guide; WKC downgraded 4/5 → 3/5 on marine softness and MS PT cut to $25. Recommended buy ARLO 8 sh @ $15.25 stop $13.85/$13.75. Also purged stale `! make daily` / `! make weekend` instructions from live docs — the correct invocations are `Run daily:` (Claude pipes inputs) and `run weekend` (Claude asks session directives, then runs `make weekend` with CLI args). The `!` shell prefix cannot drive interactive stdin or session directives.

| File | Change |
|------|--------|
| `Weekly Deep Research (MD)/Week 34 Full.md` | **CREATED** — 10-section report |
| `Weekly Deep Research (MD)/Week 34 Summary.md` | **CREATED** — thesis review |
| `Weekly Deep Research (PDF)/Week 34.pdf` | **CREATED** — PDF version |
| `Makefile` | Stale-data error message now points to `run daily` / `run weekend` |
| `.claude/rules/analysis-workflow.md` | Skip-condition wording updated to `run daily` / `run weekend` |
| `README_CLAUDE.md` | Replaced `! make daily` / `! make weekend` warnings with correct phrasing |
| `CLAUDE.md` | Workflow note clarified; Current State updated to Week 34 |

---

### 2026-04-13 — Week 30 Deep Research Report + README Weekend Flow Fix

Completed the first weekly deep research report using the hybrid quant screener (Path B). The screener surfaced 15 candidates across all sectors; 3 parallel research agents evaluated the top 10 screener picks plus additional non-screener candidates via web search. The report proposes 4 positions across 3 GICS sectors: MRAM and RBBN (Technology), ORN (Industrials), ECVT (Basic Materials) — compliant with the 2-per-sector cap. This marks the strategic pivot from PDUFA-dependent binary bets (4 consecutive stop-outs: RCKT, REPL x2, GRCE) to diversified momentum/technical plays. Also corrected `README_CLAUDE.md` weekend workflow: users must say "run weekend" to Claude (so it asks session directive questions first) instead of running `! make weekend` directly (which bypasses Claude and outputs raw markdown).

| File | Change |
|------|--------|
| `Weekly Deep Research (MD)/Week 30 Full.md` | **CREATED** — full 10-section report with 4 proposed positions |
| `Weekly Deep Research (MD)/Week 30 Summary.md` | **CREATED** — Section 9 thesis review summary |
| `Weekly Deep Research (PDF)/Week 30.pdf` | **CREATED** — PDF version of full report |
| `README_CLAUDE.md` | Rewrote Weekend Workflow section: "run weekend" flow with session directives before `make weekend` |
| `CLAUDE.md` | Updated Current State for Week 30 deployment plan |

---

### 2026-04-12 — Hybrid Quant Screener (Path B) and Allocation Framework

Built a quantitative screener (`screener.py`) that scans the full micro/small-cap universe across all sectors to eliminate the structural biotech bias caused by the catalyst-within-60-days rule funneling analysis toward PDUFA plays. The screener pulls ~1,000 stocks from Finviz (market cap ≤$2B, price ≥$1, ADV ≥$500K), enriches with yfinance price/volume history, and ranks by a composite score (40% momentum + 30% volume breakout + 30% volatility squeeze). Output is a 15-candidate CSV watchlist injected as a `<screener_watchlist>` XML block into the weekend summary. The `make weekend` target now auto-runs the screener first. Updated `portfolio_rules.md` with an Allocation Framework (catalyst plays capped at 1 position/15% equity for binary bets, momentum/technical plays allowed without catalyst date, 5-day minimum hold), Sector Diversification (max 2 of 5 positions per GICS sector), slippage guard (order ≤10% ADV), and momentum regime freeze (no new momentum initiations if IWM below 50-day SMA mid-week). Updated `analysis-workflow.md` to require evaluating at least the top 5 screener candidates before selecting, with sector cap enforcement.

| File | Change |
|------|--------|
| `screener.py` | **CREATED** — quantitative screener: Finviz universe → yfinance signals → composite ranking → watchlist CSV |
| `requirements.txt` | Added `finvizfinance>=0.16`, `tabulate>=0.9` |
| `trading_script.py` | `print_weekend_summary()` reads `watchlist.csv` and injects `<screener_watchlist>` XML block |
| `Start Your Own/portfolio_rules.md` | Added Allocation Framework, Sector Diversification, slippage guard, momentum regime freeze; relaxed catalyst requirement for momentum plays |
| `.claude/rules/analysis-workflow.md` | Weekend Step 2 now requires screener candidate evaluation and sector cap check |
| `Start Your Own/weekend_summary.md` | Added screener-first directive to research approach comments |
| `Makefile` | Added `screen` target; `weekend` target auto-runs screener before staleness check |
| `CLAUDE.md` | Updated Current State, Key Files, Commands |

---

### 2026-04-12 — Weekend Session Directives via CLI Args

Added CLI arguments (`--sector-focus`, `--catalyst-timing`, `--risk-posture`, `--max-positions`) to `trading_script.py --weekend-summary` so session directives are populated in the generated `<session_directives>` block instead of being left as a placeholder. The Makefile `weekend` target now accepts `SECTOR`, `TIMING`, `RISK`, `POSITIONS` variables and passes them through. The analysis-workflow rule was updated: Claude asks the 4 session directive questions in chat before running `make weekend`, then pipes answers as CLI args.

| File | Change |
|------|--------|
| `trading_script.py` | Added `session_directives` param to `print_weekend_summary()` and `main()`; added 4 CLI args |
| `Makefile` | Added SECTOR/TIMING/RISK/POSITIONS vars; builds `--*` flags dynamically |
| `.claude/rules/analysis-workflow.md` | Rewritten weekend flow: Step 1 ask questions, Step 2 run + analyze |

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
