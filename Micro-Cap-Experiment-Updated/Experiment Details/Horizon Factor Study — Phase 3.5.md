# Horizon Factor Study — Phase 3.5

## Part 1 — Pre-registration

*Written and committed 2026-09-17, **before any factor result at 40 or 60 sessions was
computed.** The decision rules below are fixed; Part 2 applies them to whatever the data shows.
Discipline per `.claude/rules/research-methods.md`, established by Phase 2.*

### Question

**Do the six signals in the adopted composite retain ranking skill at the horizon the book
actually holds positions for?**

On 2026-09-17 the book adopted a **40–60 session holding horizon** for the indefinite phase.
Phase 2 measured 5, 10 and 20 sessions and set 10 as primary — chosen because 10 sessions "is
the default catalyst-timing directive, and it covers the 5-session minimum hold." Both of those
reasons were artifacts of the closed 52-week experiment. Neither survives it.

The adopted composite has therefore **never been measured at the horizon it is now used for.**

This is not a hypothetical concern. Phase 2 already demonstrated horizon-dependence in this
universe: 60-session momentum's top quintile **trailed its bottom quintile by 5.7pp** on mean
over 20 sessions — reversal, not continuation. A signal set selected at 10 sessions cannot be
assumed to hold at 40.

### Why this runs now rather than at Phase 4

Phase 4 was scheduled for ~December 2026, after about 12 more weekly screens. It cannot answer
this question: a 60-session forward measurement needs ~60 sessions of data **after** each
formation date, and screens only began 2026-09-14. December would yield 2–3 usable dates.

`research/factor_study.py` already rebuilds point-in-time universes from the committed
`universe_cache.csv` snapshots back to 2026-04-17. Formation dates from April to late June
**already carry 60+ sessions of forward data today.** The question is answerable now, on data
already in git.

**Phase 3.5 does not replace Phase 4.** It is a horizon-robustness check on the same period
Phase 2 used. Phase 4 remains the genuinely out-of-sample test.

### Signals tested

**The six adopted composite inputs** — `low_vol`, `near_high`, `squeeze`, `vol_5_50`,
`vol_ratio`, `vs_sma50` — computed exactly as `screener.py` does.

**The three composites:** `composite_score` (live), `composite_legacy` (the replaced 40/30/30),
`composite_dedup` (the four-signal version tracked for Phase 4).

**The dropped momentum signals** — `mom20`, `mom60_skip5`, `mom60_riskadj` — re-tested. They
were dropped on 10-session evidence. If momentum works at all in this universe, a longer
horizon is where it would appear, and the study must be willing to find that.

### Data

Identical to Phase 2 except for the forward horizons:

- **Universes:** committed `universe_cache.csv` snapshots from git history; each formation date
  uses the latest snapshot committed on or before it.
- **Formation dates:** every Friday from 2026-04-17 with full forward data at the horizon measured.
- **Prices:** yfinance daily bars, split- and dividend-adjusted.
- **Eligible at formation:** price ≥ $1; 20-session average dollar volume ≥ $500K; snapshot market
  cap ≤ $5B; security-type and prohibited-business exclusions.
- **Population ranked:** eligible stocks passing the price-computable Phase 1 gates.
- **New horizons: 40 and 60 sessions.** 5/10/20 re-reported for continuity with Phase 2.

**Expected coverage, estimated before running** (to be replaced by actuals in Part 2): roughly
**14 formation dates at 40 sessions** (2026-04-17 → ~2026-07-17) and **10 at 60 sessions**
(2026-04-17 → ~2026-06-19). Thin, and stated as such in advance.

### Measurement

Per signal and horizon: **mean rank IC**, **hit rate** (share of dates with IC > 0), and a
**Newey-West t-statistic**.

**Lag selection — fixed in advance.** Phase 2 used lags 0/1/3 for horizons of 5/10/20 sessions.
At 40 and 60 sessions with weekly formation dates the overlap is far more severe: consecutive
observations share 7 of 8, and 11 of 12, of their forward windows. Lags are therefore set to
**⌈horizon ÷ 5⌉** — 8 lags at 40 sessions, 12 at 60. A t-statistic computed with Phase 2's lag
structure at these horizons would be badly overstated, and is not reported.

**Also required** (per `research-methods.md`, all lessons from Phase 2's own errors):

- Quintile spread on **mean** returns beside every rank IC — never rank IC alone.
- IC against **winsorized** (1%/99%) returns.
- **Partial IC** controlling for `low_vol`.
- Pairwise mean cross-sectional **rank correlations** (redundancy).
- Results **split by the direction of the universe's median return**.
- **Group mean vs population mean, group median vs population median** — never mean vs median.

### Decision rules — fixed before results

1. **Primary horizon: 40 sessions** — the midpoint of the adopted 40–60 hold. 60 sessions
   secondary. 10 reported for continuity only, and cannot trigger any rule below.
2. **A signal RETAINS support** if, at 40 sessions: mean IC > 0, Newey-West t ≥ 2.0 at the
   prescribed lag, and IC > 0 on ≥ 60% of dates. All three.
3. **A signal LOSES support** if, at 40 sessions: mean IC ≤ 0, or t < 1.0.
4. **Unproven (1.0 ≤ t < 2.0):** reported, no action.
5. **If ≥ 4 of the 6 retain support:** the composite stands. Decision recorded; revisit at Phase 4.
6. **If 2–3 retain support:** the composite is declared **horizon-mismatched**. No immediate
   screener change — the universe is frozen until Phase 4 and this sample is thin — but Phase 4's
   scope expands to a full re-selection at 40 sessions, and the measured-edge note in
   `entry-discipline.md` is amended to state the horizon limitation explicitly.
7. **If ≤ 1 retains support:** the screener's ordering is not evidence-based at the book's
   horizon. Say so plainly in `entry-discipline.md`, downgrade the sourcing language to "the
   ordering is weak evidence at the adopted horizon," and make composite re-selection the
   primary task of Phase 4.
8. **Momentum re-test:** if `mom20`, `mom60_skip5` or `mom60_riskadj` passes rule 2 at 40 or 60
   sessions having failed at 10, it is **reported as a finding but does not re-enter the
   composite before Phase 4.** The universe is frozen, and one post-hoc horizon cannot overturn a
   pre-registered test on an overlapping sample.
9. **Gates are rules, not tuning parameters.** Report how each gated group performed at the new
   horizons. A gate changes only as a proposal to the user, never automatically.
10. **Sample-size floor.** Any horizon carrying fewer than **8 formation dates** with full forward
    data is reported as **descriptive only** and cannot trigger rules 5–7.

### Limitations — stated before results

- **Thin.** ~14 dates at 40 sessions, ~10 at 60, all inside one five-month market period.
- **Overlap is severe** at these horizons; the lag rule above mitigates but does not remove it.
- **Not independent evidence.** The six signals were selected on 10-session data from an
  overlapping sample. This is a horizon-robustness check on the same period, not out-of-sample
  validation. Phase 4 supplies that.
- **Universe definition changed twice** in the window: ~−25% around 2026-07-21, +55% on
  2026-08-16 when the ceiling rose to $5B.
- **Fundamentals untested** — no point-in-time history before 2026-09-14.
- **Regime:** almost all formation dates in this window are RISK-ON. This study cannot speak to
  the RISK-OFF question; that is pre-registered separately for Phase 4.
- Survivorship coverage reported per date.

---

## Part 2 — Results

*Not yet run. To be filled after the study executes, with raw output committed before
interpretation.*
