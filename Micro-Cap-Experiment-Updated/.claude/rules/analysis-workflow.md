# Analysis Workflow Rules

Read `Start Your Own/portfolio_rules.md` before any analysis session.

---

## Daily Analysis (Auto-trigger)

When `<daily_summary>` XML appears in the conversation, check for skip conditions before running analysis:

**Skip condition 1 — Stale-data prereq:** If you previously instructed the user to run `! make daily` as a stale-portfolio prereq step before `make weekend`, skip the 6-section analysis and say: "Daily data updated — please re-run `! make weekend`."

**Skip condition 2 — End-of-week run:** If `<daily_summary>` contains `is_end_of_week="true"`, skip the 6-section analysis and say:
> End-of-week daily complete. Portfolio data is current as of [date]. Run `! make weekend` to begin the deep research session.

If neither skip condition applies, **immediately run the daily portfolio analysis without waiting for a prompt.** Use WebSearch for live IWM data, catalyst updates, and ATR. Follow the 6-section format in `Start Your Own/daily_analysis_prompt.md`.

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
3. **Sector cap check**: Before finalizing positions, verify no more than 2 of 5 positions are in the same GICS sector (per `portfolio_rules.md` Allocation Framework).
4. **Save outputs** immediately after the report completes:
   - Full report → `Weekly Deep Research (MD)/Week X Full.md`
   - Section 9 (Thesis Review Summary) only → `Weekly Deep Research (MD)/Week X Summary.md`
   - Convert full report to PDF → `Weekly Deep Research (PDF)/Week X.pdf`
     (run: `python generate_pdf.py "Weekly Deep Research (MD)/Week X Full.md" "Weekly Deep Research (PDF)/Week X.pdf"`)
   - Where X = the week number from `<week_number>` in the weekly context
