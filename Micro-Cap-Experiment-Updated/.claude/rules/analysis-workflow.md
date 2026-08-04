# Analysis Workflow Rules

Read `Start Your Own/portfolio_rules.md` before any analysis session.

**Price sourcing:** all prices used in analysis or order recommendations must follow
`.claude/rules/price-data-integrity.md` — never source live/after-hours/pre-market
prices from WebSearch (use the browser tool), require a session label + timestamp,
and run the order-side check before naming any stop or limit level.

---

## Daily Analysis (Auto-trigger)

When `<daily_summary>` XML appears in the conversation, check for skip conditions before running analysis:

**Skip condition 1 — Stale-data prereq:** If you previously instructed the user to `run daily` as a stale-portfolio prereq step before `run weekend`, skip the 6-section analysis and say: "Daily data updated — please re-run `run weekend`."

**Skip condition 2 — End-of-week run:** If `<daily_summary>` contains `is_end_of_week="true"`, skip the 6-section analysis and say:
> End-of-week daily complete. Portfolio data is current as of [date]. Say `run weekend` to begin the deep research session.

If neither skip condition applies, **immediately run the daily portfolio analysis without waiting for a prompt.** Follow the 6-section format in `Start Your Own/daily_analysis_prompt.md`.

**Sourcing within the daily** (per `.claude/rules/price-data-integrity.md`):
- **IWM close** for the regime check comes from the `<daily_summary>` itself (the script prices IWM as a benchmark) — do NOT WebSearch a "live IWM price."
- **IWM 50-day SMA** is a slow-moving technical level; a dated technical-analysis page is acceptable, but note the date.
- **Any live/after-hours/pre-market price** (e.g. reacting to a post-close print) → **browser tool** with a timestamp, never WebSearch.
- **Catalyst dates, guidance, ATR, analyst PTs** → WebSearch is fine; date each claim and apply *Thesis-Input Freshness* (`.claude/rules/entry-discipline.md`) to any time-varying driver.

**Mandatory earnings-night live check (do NOT defer):** if any holding reports earnings **after the close on the day of the daily run** (or the previous evening, before a morning run), you MUST — *within that same daily analysis, before writing the Post-Event Playbook* — pull BOTH via the **browser tool**:
1. the **live after-hours quote** (session label + timestamp, reconcile against the close), AND
2. the **actual earnings release from a live, timestamped news source** (e.g. StockTitan/press-wire/IR) — the real revenue/EPS/EBITDA/guidance, not the pre-print consensus and not the AH price alone.

Do not write "the reaction is a tomorrow event" and defer it, and do not infer "beat/miss" from the price move alone. Origin: 2026-08-04 — ARDT reported after close and popped **+6.98% AH**, which read like a clean beat; the live release showed a **mixed print** (revenue beat $1.622B, but EPS $0.12 *missed* the $0.17 est and adj. EBITDA −32.3% YoY — the pop was on a +67% cash-flow jump + reaffirmed guidance). WebSearch had **none** of these numbers at that hour. Confirm the actual result, run the post-catalyst reassessment (re-rate conviction on the substance, plan the next-open stop change), and report it.

**News recency — the same discipline as prices, applied to news** (`.claude/rules/price-data-integrity.md`): WebSearch returns cached snippets and lags real time; it can surface an *older* article as the latest and miss a newer downgrade, guidance cut, or the print itself. So:
- **Time-sensitive / breaking / price-moving news and sentiment** — actual earnings results, same-day analyst rating/PT changes, M&A, halts, or the driver behind an **unexplained intraday move** — must be verified on a **live, timestamped source via the browser tool**, not on WebSearch alone.
- **Established, slower-moving facts** — a confirmed future earnings *date*, historical guidance, an analyst PT from several days ago — WebSearch is acceptable, but **date every claim** and re-verify live anything that could have changed in the last ~48h.
- If a holding is moving and you can't explain why, browser-check its live news feed before writing the review.

---

## Weekend Analysis (Two-step flow)

When the user asks to run the weekend analysis (e.g., "make weekend", "run weekend", "weekend summary"):

### Step 1 — Session Config (ask BEFORE running make weekend)

Ask the 4 session directive questions (defined in `Start Your Own/daily_analysis_prompt.md`):

**Q1 — Sector focus:** Wide net (default) | Biotech | Energy | Tech | Industrials
**Q2 — Catalyst timing:** Within 5 days | Within 10 days (default) | 30-60 days
**Q3 — Risk posture:** Neutral | Aggressive | Defensive | Tighten stops
**Q4 — Max concurrent positions:** 5 (default) | 6

Then run:
```bash
make weekend SECTOR="<answer>" TIMING="<answer>" RISK="<answer>" POSITIONS="<answer>"
```

The `make weekend` target automatically runs the screener first. If the screener fails (Finviz down, network issue), the weekend workflow continues — use WebSearch as a fallback for candidate sourcing.

### Step 2 — Analysis (auto-trigger when `<weekly_context>` XML appears)

When `<weekly_context>` XML appears in the conversation output, **immediately begin the deep research** — do NOT ask for further input:

1. **Screener candidate evaluation**: If a `<screener_watchlist>` block is present, evaluate AT LEAST the top 5 candidates via WebSearch for catalyst/fundamental info. For each screener candidate NOT selected, state why in one line. Include at least 2 candidates from different GICS sectors in the evaluation table. Screener candidates get priority over web-search-only finds.
2. **Run analysis**: produce the full 10-section deep research report (format defined in `Start Your Own/weekend_summary.md`) using WebSearch extensively for all holdings and new candidates
3. **Sector cap check**: Before finalizing positions, verify no more than 2 positions (of the up-to-6 book) are in the same GICS sector (per `portfolio_rules.md` Allocation Framework).
4. **Save outputs** immediately after the report completes:
   - Full report → `Weekly Deep Research (MD)/Week X Full.md`
   - Section 9 (Thesis Review Summary) only → `Weekly Deep Research (MD)/Week X Summary.md`
   - Convert full report to PDF → `Weekly Deep Research (PDF)/Week X.pdf`
     (run: `python generate_pdf.py "Weekly Deep Research (MD)/Week X Full.md" "Weekly Deep Research (PDF)/Week X.pdf"`)
   - Where X = the week number from `<week_number>` in the weekly context
