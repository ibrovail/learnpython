# Current-State history

Entries moved out of `CLAUDE.md` on 2026-10-12, when it passed the 200-line cap that
`.claude/rules/documentation.md` sets. Nothing here is current state — it is the trail of
how the rules got where they are, kept because the *reasoning* in these entries is often
what stops a settled question being reopened. The full account of each change is in
`Experiment Details/Rules Amendment History.md`; engineering detail is in
`.claude/docs/implementation-notes.md`.

---

- **Complete**: **Indefinite-system rules revision (9/17)** — horizon **40–60 sessions**; the
  trailing stop is the only **mechanical** exit (partials deleted); **binary-thesis entries
  prohibited**; risk-per-trade **2%**; **drawdown circuit breaker** (−20% de-risk / −30% cash, on
  the injection-neutral series from the re-based peak); driver cap 2 + sector cap 3. *Several of
  these were measured for the first time on 10/11 — see the audit entry below.*

- **Complete**: **Phase 3.5** — on non-overlapping formation dates the composite clears t=2.0 at
  **no** horizon, and 20/40/60 sessions have too few independent observations for any verdict;
  effect sizes do rise with horizon. **Phase 4 (Dec) primary horizon = 20 sessions.** Index changes
  are the fifth unexplained-move category.

- **Adopted (10/05 b) — the catalyst lane has a sourcing method, and since 10/11 it is the ONLY
  lane open in RISK-OFF.** It had an eligibility test and no pipeline, so **40 of 40 researched
  names came from the screener**. **Stage 0** reserves ≥3 stage-1 slots for catalyst names; a
  qualifying catalyst is **a resolved fact with a dated earnings impact** (the prohibited binary
  case is an unresolved event with a dated announcement); **an index add counts only BEFORE its
  effective date.** Full detail in `portfolio_rules.md` → *Catalyst plays* → *Sourcing*.

- **Superseded (10/05) — `near_high` removed from the RISK-OFF defensive profile**, and the whole
  profile went on 10/11, so its legs and the 0.90-threshold question are moot. What survives is the
  **open Phase 4 question of whether beta belongs in any future defensive test** — top-decile calm
  names had median beta **0.85 vs 0.84**, so `low_vol` never measured defensiveness (and **beta was
  never a leg**, correcting an earlier claim). `Rules Amendment History.md` (10-05, 10-11).

- **HOPE post-mortem (9/29)**: stopped out for **0.99% of equity — the budgeted risk to the cent**.
  **No rule failed; the entry did** — bought **+0.15% above its 50-day SMA**, the thinnest possible
  trend-gate pass, yet sized to **28.8% of equity**. **Still open** after the 10/11 audit (which
  tested rules, not entry quality): does a sub-1% pass deserve reduced size, or none?

- **Complete (9/19)**: R1–R8 adopted — one weekend question; two-stage funnel sized to the buys
  sought, extending to ranks 51–100; research log of buys *and* passes (`log_research.py`); regime
  computed by the script with a ±1% band → `<market_regime>`, `regime_history.csv`; six-section
  report; dailies by exception via `<holding_review>`. The `weekly-portfolio-report` skill is
  **retired** — it predates every current rule and conflicts with four. (`Rules Amendment
  History.md`, 2026-09-19 a–f.)

- **Complete (9/28)**: **the 60-session re-underwrite says what it tests** — the **quality** tests
  (revenue, earnings path, driver freshness, non-binary, prohibited business, trend and
  distance-from-base, earnings guard, placeable stop inside the 30% ceiling), **not** the regime
  **capacity** gates (half risk budget, 90-day window, position ceiling), which govern new capital
  only. A failed quality test is an exit; a failed capacity gate is a hold that **must name the gate
  in writing**. Left open: the review cadence *after* a pass. (`Rules Amendment History.md`, 9-28.)

- **Fixed (9/20–9/21)**: `inject_last_thesis.py` line-anchors its tags; entry stops need all three
  candidates — 1.75×ATR, 10-session low, 50-day SMA. *(Detail in implementation-notes.md.)*

