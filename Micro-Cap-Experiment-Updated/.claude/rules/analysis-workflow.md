# Analysis Workflow Rules

Read `Start Your Own/portfolio_rules.md` before any analysis session.

**Price sourcing:** all prices used in analysis or order recommendations must follow
`.claude/rules/price-data-integrity.md` — the **browser is the default source** for
live prices, valuation/analyst data, actual earnings results and news recency;
WebSearch is for discovery only. Require a session label + timestamp, and run the
order-side and range checks before naming any stop or limit level.

---

## ⛔ Pre-Recommendation Verification (PRV) — mandatory gate

**Before writing ANY buy, sell, trim, add, or exit recommendation for a ticker, fetch
that ticker's live quote page via the browser.** No exceptions, and not only when
something already looks wrong. This gate exists because every significant miss in this
experiment shared one shape: a recommendation written without the page that held the
deciding fact.

**Capture and state these fields** (one `stockanalysis.com/stocks/TICKER/` fetch returns
all of them):

| Field | Why it decides things |
|-------|----------------------|
| Live price + extended-hours + timestamp | The order-side check must run against a verified price |
| **Analyst rating + price target** | Upside/downside vs the recommendation; ARDT's +21% PT reversed an exit |
| **Forward P/E** (vs trailing) | Shows expected earnings direction; ARDT 9.51 fwd vs 18.98 trailing |
| **TTM revenue + growth %** | Growing vs shrinking is the single best filter this book has (TDAY −8.3% = exit; PAR +18.8% = buy) |
| **TTM EPS / net income** | FOXF's −$7.14 TTM EPS was invisible in the "beat and raise" headline |
| **52-week range** | Position within it: PAR was −65% from its high; CADL at its high |
| **Beta** | ARDT 0.70 = genuinely defensive, which changed the hold/exit call |
| Day range, volume, market cap | Range check inputs; volume validates whether a quote is meaningful |

**Also required, same gate:**
- **Primary thesis driver.** Name the time-varying input the thesis depends on (commodity/spot
  price, rate, FX cross, tariff/subsidy, named supply-demand condition). This feeds the **driver
  cap** — at most 2 positions may share one — which cannot be enforced in code. `trading_script.py`
  prints a `<position_limits>` block with live sector counts and a driver-cap reminder; the driver
  itself must be written by you. An unnamed driver is a rule violation, not an omission.
- **⛔ Prohibited-business check — first, before anything else.** Confirm the company is not a
  prison/detention operator, a weapons/defence/firearms business, a predatory lender, or
  Israeli-affiliated (`portfolio_rules.md` → *Exclusions*). The quote page's **Industry** field
  is the starting point; **Security & Protection Services** and **Credit Services** require
  reading what the company actually does. A name that fails is dropped without further
  research, however well it scores. Origin: 2026-09-14, CXW recommended despite being a
  private prison operator.
- **Extended-hours volume**, when quoting a pre/post-market price. A 332-share print is
  not a price (PAR, 2026-08-17). Report volume alongside the timestamp or don't use the quote.
- **Live news-feed check** (`stocktitan.net/news/TICKER/`) for any recommendation, to
  confirm nothing material landed since the last settled close.

**If the browser is unavailable or the page cannot be fetched:** say so explicitly and
either defer the recommendation or mark it **UNVERIFIED** — never present an
unverified recommendation as a normal one.

**Applies equally to exits.** There was previously no verification requirement on the
sell side at all, which is how the 2026-08-20 ARDT exit was recommended against a Buy
rating, +21% PT and 9.5× forward earnings. **A sell recommendation needs the same
evidence as a buy.**

---

## Daily Analysis (Auto-trigger)

When `<daily_summary>` XML appears in the conversation, check for skip conditions before running analysis:

**Skip condition 1 — Stale-data prereq:** If you previously instructed the user to `run daily` as a stale-portfolio prereq step before `run weekend`, skip the 6-section analysis and say: "Daily data updated — please re-run `run weekend`."

**Skip condition 2 — End-of-week run:** If `<daily_summary>` contains `is_end_of_week="true"`, skip the 6-section analysis and say:
> End-of-week daily complete. Portfolio data is current as of [date]. Say `run weekend` to begin the deep research session.

If neither skip condition applies, **immediately run the daily portfolio analysis without waiting for a prompt.** Follow the 6-section format in `Start Your Own/daily_analysis_prompt.md`.

**Sourcing within the daily** (per `.claude/rules/price-data-integrity.md`):
- **Regime (IWM close, 50-day SMA, RISK-ON/OFF)** comes from the script's `<market_regime>` block,
  computed from IWM daily closes and saved to `regime_history.csv` (since 2026-09-19). Do NOT look
  up the IWM price or its SMA on the web. If the block says UNAVAILABLE, say so and look it up.
- **Any live/after-hours/pre-market price** (e.g. reacting to a post-close print) → **browser tool** with a timestamp, never WebSearch.
- **Analyst PTs, ratings, forward P/E, TTM revenue/EPS, beta, 52-wk range** → **browser quote page**, not WebSearch (see the PRV gate above and the source hierarchy).
- **Catalyst dates + historical guidance** → WebSearch is acceptable for discovery; date each claim, browser-verify anything decision-relevant, and apply *Thesis-Input Freshness* (`.claude/rules/entry-discipline.md`) to any time-varying driver.
- **Any holding you are about to act on** (buy/sell/trim/add) → **PRV gate applies — fetch its quote page first.**

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

### Step 0 — Is a full report due? (run FIRST, before asking anything)

Cadence is **trigger-based** since 2026-09-17, not weekly. Run:

```bash
make trigger
```

It reads the ledger only (no downloads, ~3 seconds) and prints `<research_trigger>` with a
`<status>`, the reasons, and this weekend's `<week_number>`.

- **`DUE`** → Step 1 (one optional question) → `make weekend FOCUS="…"` → full report → save
  `Week N Full.md`, `Week N Summary.md` and the PDF, as below.
- **`NOT DUE`** → run **`make screen`** (the weekly screen always runs — Phase 4 needs every
  weekend's formation date), then write a **short monitoring note**: stops, any holding nearing its
  60-session re-underwrite, the circuit-breaker line, and anything that changed. Do **not**
  re-underwrite theses nothing has changed for. Save it as
  **`Weekly Deep Research (MD)/Week N Monitor.md`** — never as "Full": the 30-session backstop
  counts Full reports only, so a note saved as Full would silently reset it. No PDF, no Summary.
- A **regime flip** since the last Full report is now a computed trigger (from `regime_history.csv`).
- **Override** → run the full report anyway if the user asks.

If the portfolio is not current for the last session, the trigger still prints but `make weekend`
will stop: run the daily first.

### Step 1 — One optional question (only when Step 0 says DUE)

Ask **one** question (defined in `Start Your Own/daily_analysis_prompt.md`):

> **Anything specific you want researched this weekend** — a ticker, a sector, or a question?
> *(Default: no — wide net across all permitted sectors.)*

Then run:
```bash
make weekend FOCUS="<answer, or leave empty>"
```

**The four questions used until 2026-09-19 are retired.** Timing, risk posture and position count
are set by the rules, not chosen weekly: holding horizon 40–60 sessions, catalyst window 90 days
(non-binary only), 2% risk per trade, a 5–6 position ceiling (about 4 fit at current sizing), and
risk posture governed by the regime filter and the drawdown circuit breaker. Two of the retired
options worked against the rules — "Aggressive, we are trailing the benchmark" is gap-chasing (the
pressure behind TYRA), and "Tighten all stops by one ATR" contradicted the stop rules outright.

The `make weekend` target automatically runs the screener first. If the screener fails (Finviz down, network issue), the weekend workflow continues — use WebSearch as a fallback for candidate sourcing.

### Step 2 — Analysis (auto-trigger when `<weekly_context>` XML appears)

When `<weekly_context>` XML appears in the conversation output, **immediately begin the deep research** — do NOT ask for further input:

1. **Shortlist 8–10 candidates from the top 50 — a spread, not the top of the list.** The ranking
   order has no demonstrated skill on independent data (Phase 3.5 Part 3), and the top of the list
   tilts toward larger, calmer names (9/15: median $2.4Bn, 31 of 50 above $2Bn). So:
   - **Pre-filter from the watchlist columns first** — drop names with an `earnings` date inside
     the next 10 sessions (no-initiation guard), prohibited names, `REVIEW`-flagged names you
     cannot clear, and binary-thesis setups. Don't spend research on names that can't be bought.
   - **Then shortlist so that:** ≥3 come from ranks 1–15 **and** ≥3 from ranks 16–50; ≥3 different
     GICS sectors; **≥2 below $2Bn market cap.**
   - Use WebSearch to *discover* the story, then **browser-fetch the quote page of every
     shortlisted name** (PRV gate) — revenue growth, the book's best filter, lives there.
   - Off-list finds (web search, the user's `FOCUS`) are welcome **in addition** to the 8–10.
   - For every shortlisted name, bought or passed, state the decision and the reason in one line
     — and **log it** (item 5).
2. **Run analysis**: produce the full 10-section deep research report (format defined in `Start Your Own/weekend_summary.md`). Use WebSearch broadly for discovery, but **browser-verify every holding and every candidate you recommend acting on** (PRV gate).
3. **Correlated-risk check**: Before finalizing positions, verify both limits in
   `portfolio_rules.md` — at most **2 positions sharing a primary thesis driver** (named
   explicitly in the report) and at most **3 positions in any one GICS sector**. The
   `<position_limits>` block in the script output prints live sector counts and each holding's
   sessions held; the driver cap is yours to apply.
4. **60-session re-underwrite**: any holding whose `<position_limits>` row shows **60 or more
   sessions held** must be re-justified in writing this session — current thesis, current driver,
   current conviction — against the standard for a fresh buy, and exited if it fails.
5. **Log every shortlisted candidate** — bought, passed or put on watch — with
   `log_research.py`, one row each, **passes included**. Never edit `research_log.csv` by hand.
   ```bash
   venv/bin/python log_research.py --week 54 --ticker XYZ --source "screener #12" \
     --decision PASS --reason-code extended --reason "44% above the 50-day" --ref-price 12.34
   ```
   Buys are only half the evidence. Scoring the passes is the only way to learn whether research
   adds value beyond the screener list it chose from — the 9/19 trade review could not answer that,
   because passes were never recorded.
6. **Save outputs** immediately after the report completes:
   - Full report → `Weekly Deep Research (MD)/Week X Full.md`
   - Section 9 (Thesis Review Summary) only → `Weekly Deep Research (MD)/Week X Summary.md`
   - Convert full report to PDF → `Weekly Deep Research (PDF)/Week X.pdf`
     (run: `python generate_pdf.py "Weekly Deep Research (MD)/Week X Full.md" "Weekly Deep Research (PDF)/Week X.pdf"`)
   - Where X = the week number from `<week_number>` in the weekly context
