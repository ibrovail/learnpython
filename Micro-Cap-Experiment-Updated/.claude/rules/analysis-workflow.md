# Analysis Workflow Rules

Read `Start Your Own/portfolio_rules.md` before any analysis session.

---

## Daily Analysis (Auto-trigger)

When `<daily_summary>` XML appears in the conversation, **immediately run the daily portfolio analysis without waiting for a prompt** — UNLESS you previously instructed the user to run `! make daily` as a weekend prep step (stale portfolio check). In that case, skip the 6-section analysis and say: "Daily data updated — please re-run `! make weekend`."

Use WebSearch for live IWM data, catalyst updates, and ATR. Follow the 6-section format in `Start Your Own/daily_analysis_prompt.md`.

---

## Weekend Analysis (Auto-trigger after `make weekend`)

When `<weekly_context>` XML appears in the conversation (output of `make weekend`), immediately begin the weekend workflow **without waiting for a prompt**:

1. **Session config**: ask the 4 session directive questions (defined in `Start Your Own/daily_analysis_prompt.md`)
2. **Update directives**: edit the `<session_directives>` block in `Start Your Own/weekend_summary.md` with the user's answers
3. **Run analysis**: produce the full 10-section deep research report (format defined in `Start Your Own/weekend_summary.md`) using WebSearch extensively for all holdings and new candidates
4. **Save outputs** immediately after the report completes:
   - Full report → `Weekly Deep Research (MD)/Week X Full.md`
   - Section 9 (Thesis Review Summary) only → `Weekly Deep Research (MD)/Week X Summary.md`
   - Convert full report to PDF → `Weekly Deep Research (PDF)/Week X.pdf`
     (run: `python generate_pdf.py "Weekly Deep Research (MD)/Week X Full.md" "Weekly Deep Research (PDF)/Week X.pdf"`)
   - Where X = the week number from `<week_number>` in the weekly context
