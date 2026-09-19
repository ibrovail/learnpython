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

*Run 2026-09-17. Pre-registration committed as `9f47b4a` before any 40/60-session result was
computed; the extended script and raw output committed as `afbf106` before this interpretation.
Everything under "Post-hoc" was written after seeing the results and is labelled as such.*

### Data as run

- **21 formation dates** (2026-04-17 → 2026-09-04) from **20 universe snapshots**.
- **2,370 of 2,421 tickers priced** (2025-11-03 → 2026-09-17).
- Dates with full forward data: 21 at 5 sessions, 20 at 10, 18 at 20, **14 at 40**, **10 at 60**.
- Forward-return coverage among survivors is 99.7–100% on every date that has data, so
  survivorship bias is negligible.
- **Regime: every one of the 14 dates at 40 sessions is RISK-ON.** As the pre-registration
  predicted, this study says nothing about the RISK-OFF question.

### The pre-registered verdict

**Rule 2 applied at 40 sessions — all six adopted signals retain support:**

| Signal | Mean IC | NW t | IC > 0 | Verdict |
|---|---|---|---|---|
| `low_vol` | 0.182 | 5.29 | 100% | **RETAINS** |
| `near_high` | 0.144 | 14.77 | 100% | **RETAINS** |
| `squeeze` | 0.130 | 3.88 | 93% | **RETAINS** |
| `vol_ratio` | 0.101 | 5.78 | 86% | **RETAINS** |
| `vol_5_50` | 0.094 | 6.62 | 93% | **RETAINS** |
| `vs_sma50` | 0.054 | 4.04 | 93% | **RETAINS** |
| `composite` (legacy 40/30/30) | 0.131 | 5.52 | 100% | retains |
| `vs_sma20` | 0.103 | 5.96 | 93% | retains (not in composite) |
| `mom20` | 0.068 | 6.41 | 93% | **retains — see rule 8** |
| `mom5` | 0.061 | 2.73 | 79% | retains (not in composite) |
| `mom60_skip5` | −0.020 | −0.74 | 29% | LOSES |
| `mom60_riskadj` | −0.027 | −0.95 | 21% | LOSES |
| `squeeze_own` | −0.039 | −2.81 | 29% | LOSES |

**6 of 6 retain support → rule 5 applies: the composite stands.** No horizon-mismatch is
declared. Rules 6 and 7 are not triggered.

**Rule 8 — momentum re-test.** `mom20` fails at 10 sessions (t 1.24) and passes at 40 (t 6.41).
Per the rule fixed in advance, this is **reported but does not re-enter the composite before
Phase 4.** Two things argue for that restraint independently: its IC against the *eligible*
universe is only 0.023 at 40 sessions, and at 60 sessions its eligible IC turns **negative
(−0.039)**. Both 60-session momentum signals remain negative at every horizon — the reversal
Phase 2 found is confirmed, not overturned.

**Rule 9 — gates unchanged.** Every gated group still underperformed survivors at 10 sessions:
>40% above the 50-day **−12.35pp** mean, >20% above the 20-day **−1.99pp**, fresh breakout
**−1.18pp**. Deal-pinned shows +0.69pp on **11 stock-dates** — too few to mean anything, and its
median is −0.64pp. No gate changes.

---

### ⚠️ Post-hoc — the pre-registration contained a flaw, and it is the decisive fact

**The t-statistics above are not trustworthy, and the sample-size floor I wrote to catch exactly
this did not catch it.**

Rule 10 set a floor of **8 formation dates**. At 40 sessions there are 14, and at 60 there are 10,
so both cleared it. But **the floor counted raw dates, when what matters is how many
*independent* observations those dates represent.** With weekly formation and an h-session
forward window, roughly h/5 consecutive dates share the same forward period:

| Horizon | Dates | Overlap | **Effective n** | NW lags | Lags as % of n |
|---|---|---|---|---|---|
| 5 | 21 | 1.0× | 21.0 | 0 | 0% |
| 10 | 20 | 2.0× | 10.0 | 1 | 5% |
| 20 | 18 | 4.0× | 4.5 | 3 | 17% |
| **40** | **14** | **8.0×** | **1.75** | 8 | **57%** |
| **60** | **10** | **12.0×** | **0.83** | 12 | **120%** |

Counted another way — how many genuinely non-overlapping windows fit in the span the formation
dates cover — there are **2.6 at 40 sessions and 1.7 at 60**. The IC series carries a mean lag-1
autocorrelation of **+0.42** at 40 sessions, confirming the dependence is real and not an
artifact of the arithmetic above.

**At 60 sessions the Newey-West lag length (12) exceeds the number of observations (10).** That is
not a conservative correction; it is an undefined one. The 60-session column should be read as a
single overlapping episode and nothing more.

This is why `near_high` prints **t = 14.77** on 14 dates and `low_vol` prints **t = 15.30** on 10.
Those numbers are counting near-duplicate observations as independent evidence. The underlying
IC series are genuinely tight and positive — `near_high` mean 0.144, sd 0.043, every date above
zero — but "14 consecutive weekly readings of substantially the same two-month period" is roughly
**two or three** independent looks, not fourteen.

**The honest verdict, therefore: there is no evidence that the six signals break down at the
adopted horizon. That is not the same as confirmation, and nothing here should be quoted as
strong support.** The pre-registered rules were applied mechanically as written, and their
outcome stands — but their evidential weight is far below what the t-statistics suggest.

*Lesson recorded in `.claude/rules/research-methods.md`: a sample-size floor must be set on
effective, overlap-adjusted observations, not raw formation dates. This is the same class of
error as Phase 2's mean-vs-median comparison — a pre-registered metric that looked rigorous and
silently flattered the result.*

### Post-hoc — findings that do NOT rest on a t-statistic

These are the results worth actually carrying forward. They still sit on overlapping data, but
they are effect sizes and monotone trends rather than significance claims.

**1. Phase 2's central weakness reverses at longer horizons.** Phase 2's most damaging diagnostic
was that the quintile spread on **mean** returns was near zero or negative — the signals avoided
volatile losers without finding winners. That flips with horizon:

| Q5 − Q1, mean returns | 10 sessions | 40 sessions | 60 sessions |
|---|---|---|---|
| `low_vol` | −0.5pp | **+5.0pp** | **+11.7pp** |
| `near_high` | +0.3pp | **+5.2pp** | **+8.3pp** |
| `squeeze` | −0.9pp | **+3.1pp** | **+10.7pp** |
| `vol_ratio` | −0.1pp | **+4.4pp** | **+9.1pp** |
| `composite` | −0.4pp | **+4.3pp** | **+7.8pp** |

**2. The screener's practical edge rises with horizon — by about half as much as first
reported.** Top 50 by composite versus the other gate survivors:

| Horizon | 5 | 10 | 20 | **40** | **60** |
|---|---|---|---|---|---|
| Mean vs mean | +0.09pp | −0.23pp | +0.76pp | **+2.46pp** | **+3.54pp** |
| Median vs median | +0.12pp | +0.36pp | +1.29pp | **+3.66pp** | **+4.80pp** |
| ~~As first printed (mean vs median)~~ | ~~+0.50~~ | ~~+0.78~~ | ~~+2.85~~ | ~~+5.57~~ | ~~+6.99~~ |

*Corrected 2026-09-19 — see Part 5.* The first version compared the top-50 **mean** with the
survivor **median**, exactly the error `research-methods.md` records from Phase 2. Survivor skew
(mean − median) grows from 0.41pp at 5 sessions to **3.45pp at 60**, so the bias grew with the
horizon and manufactured about half of the apparent rise. The claim that the edge at 40 sessions
is "roughly seven times" the 10-session edge is withdrawn. What survives is a smaller, still
monotone rise — on overlapping dates, so descriptive only.

**3. ~~The delivered-watchlist mystery widens.~~ The delivered-watchlist gap is mostly a few
lucky lists.** *Corrected 2026-09-19.* Two errors in the first version. It called these "the
watchlists actually researched" — they are not: they are the screener's **own committed output**
(`watchlist.csv` snapshots), the list research chose *from*, so they say nothing about the value
research adds. And the +2.87 / +9.19 / +11.37pp figures were mean vs median. Like-for-like:

| Delivered lists vs universe | 10s | 20s | 40s | 60s |
|---|---|---|---|---|
| Mean vs mean | +2.24pp | +4.08pp | +6.73pp | +8.51pp |
| **Share of lists that beat the universe** | 55% | 59% | **41%** | **27%** |
| Median vs median | +1.44pp | +3.02pp | +7.61pp | +7.37pp |

At 40 and 60 sessions **fewer than half the lists beat the universe**; a handful of early
(April–May) lists carry the average. There is little left to explain.

**4. `squeeze_own` gets worse the longer you hold it** (t −2.81 at 40, −3.34 at 60). Phase 2's
decision to drop it is reinforced.

**5. `vs_sma20` looks materially better at the adopted horizon** than at 10 sessions (IC 0.103 vs
0.042). This bears on a rule changed on 2026-09-17 — see below.

### Consequences

1. **Composite unchanged.** Rule 5. Revisit at Phase 4 with genuinely out-of-sample screens.
2. **`mom20` stays out of the composite.** Rule 8, reinforced by its negative eligible-universe IC
   at 60 sessions.
3. **Gates and the regime filter unchanged.** Rule 9; and this study has no RISK-OFF dates at all,
   so the RISK-OFF screener allowance keeps its Phase 4 sunset exactly as written.
4. **The horizon decision is supported, not undermined.** Findings 1 and 2 both point the same
   way: this signal set works better over 40–60 sessions than over 10.
5. **`entry-discipline.md`'s measured-edge language is updated** to state the edge by horizon
   rather than quoting the 10-session figure alone.
6. **The 20-day SMA downgrade keeps its conclusion but loses part of its stated reason.** On
   2026-09-17 the hard 20-day SMA entry gate was downgraded to a reported check requiring written
   justification, on the grounds that `vs_sma20` was rated UNPROVEN by Phase 2. At 40 sessions it
   tests well. The downgrade still stands — a candidate below its 20-day SMA is not blocked, only
   argued for — but the rationale is corrected in place rather than left misleading.
7. **Phase 4 must fix the sample-size rule.** Its floor is to be set on effective observations.
   On the current cadence, a credible 40-session test needs roughly **80+ weekly formation dates**
   — about 18 months of screens — or non-overlapping formation dates, which is the cheaper fix.

### Limits

- **Effective n of 1.75 at 40 sessions and 0.83 at 60.** This is the binding limitation and it
  dominates everything above.
- All 14 usable dates at 40 sessions are RISK-ON.
- The six signals were selected on 10-session data from an overlapping sample, so this is a
  horizon-robustness check on the same period, not out-of-sample validation.
- The universe definition changed twice mid-window (−25% in July, +55% in August).
- Fundamentals remain untested.

---

## Part 3 — Non-overlapping formation dates (adopted for Phase 4, 2026-09-17)

*Post-hoc relative to Phase 3.5's pre-registration: this basis was chosen after seeing that the
pre-registered one was unsound. It is **pre-registered for Phase 4** as of this date.*

### Method

A formation date at session index `i` owns the forward window `[i, i+h)`. Two dates are
independent only if their indices differ by at least `h`. `nonoverlapping_phases()` greedily
builds every maximal subset of dates satisfying that, starting from each of the first `h/5`
dates — so weekly dates yield several **phases**, each internally independent, which between them
use all the data.

Within a phase, **Newey-West is unnecessary and lags are 0**: there is no overlap left to correct
for. Phases are *not* independent of each other, so they are reported side by side and never
pooled. The spread across phases is the point — it shows how much the answer depends on which
slice was taken.

### Results — and they overturn Part 2's headline

| Horizon | Phases | Indep. obs per phase | `low_vol` mean IC | t range across phases | Verdict |
|---|---|---|---|---|---|
| 5 | 1 | **18** | 0.072 | 2.48 | testable |
| 10 | 2 | **9** | 0.092 | 1.64 – 2.26 | marginal |
| 20 | 4 | **4** | 0.123 | 1.36 – 2.56 | **descriptive only** |
| 40 | 5 | **2** | 0.172 | 1.39 – 31.35 | **descriptive only** |
| 60 | — | **<2** | — | — | **not computable** |

**The 40-session result in Part 2 does not survive.** With honest independence there are **two**
observations, not fourteen. The t-statistics on two points (up to 31) are meaningless for the
obvious reason. Per the ≥5-observation threshold now applied, 20, 40 and 60 sessions are all
**descriptive only — no verdict is available at the horizon the book actually holds.**

**What the pre-registered verdict is worth at horizons that ARE testable:**

| Signal | 5 sessions (18 obs) | 10 sessions (9 obs, 2 phases) |
|---|---|---|
| `low_vol` | t **2.48** | t 1.64 – 2.26 |
| `vol_ratio` | t **2.22** | t **2.05 – 2.51** |
| `near_high` | t **2.10** | t 1.16 – 1.93 |
| `squeeze` | t **2.02** | t 1.19 – 1.64 |
| `vs_sma50` | t 1.58 | t 0.79 – 1.57 |
| `vol_5_50` | t 1.56 | t **1.94 – 2.89** |
| **`composite`** | **t 1.29** | **t 0.52 – 1.60** |

**The composite does not clear t = 2.0 at any horizon on independent observations.** Individual
signals do at 5 sessions; `vol_ratio` is the only one that holds up across both phases at 10 —
consistent with Phase 2's finding that volume carries the one clearly independent piece of
information.

### What is NOT overturned

**Effect sizes rise monotonically with horizon on non-overlapping data too**, which is the result
the horizon decision rested on, and it does not depend on any t-statistic:

`low_vol` mean IC: **0.072 (5s) → 0.092 (10s) → 0.123 (20s) → 0.172 (40s)**.
`near_high`: 0.062 → 0.076 → 0.099 → 0.135. `composite`: 0.030 → 0.046 → 0.077 → 0.133.

And the point estimates agree closely with Part 2's pooled figures (`low_vol` 0.172 vs 0.182 at
40 sessions; `near_high` 0.135 vs 0.144). **The direction and magnitude are consistent; only the
confidence was fictional.** Several signals are positive on *both* 40-session observations.

### When each horizon becomes answerable

Independence needs `k·h` sessions of formation span plus `h` of forward data:

| Horizon | Indep. obs now | 5 obs by | 8 obs by | 20 obs by |
|---|---|---|---|---|
| 10 | 9 | *(have)* | *(have)* | 2027-02 |
| **20** | 4 | **2026-10** | **2026-12** | 2027-11 |
| 40 | 1–2 | 2027-03 | 2027-09 | 2029-07 |
| 60 | 0 | 2027-09 | 2028-05 | 2031-02 |

**Consequence for Phase 4 (December 2026): its primary horizon must be 20 sessions**, where it
will hold ~8 independent observations — enough for a real verdict. 40 sessions stays descriptive
until roughly **March 2027**, and 60 sessions is effectively out of reach on this dataset.

This does not undermine the 40–60 session holding decision. It means the *research* can only
validate at 20 sessions for now, and — since effect sizes grow monotonically with horizon —
20-session evidence is a conservative lower bound for a longer hold, not a contradiction of it.

### Cadence

**Weekly screens are kept.** Independence is set by the calendar span, not the sampling rate, so a
slower cadence would not add independent observations — it would only remove **phases** to
cross-check against:

| Horizon | Weekly screens | Monthly screens |
|---|---|---|
| 20 sessions | **4 phases** | 1 phase |
| 40 sessions | **8 phases** | 2 phases |

Weekly formation dates are what made this diagnostic possible at all. Keep them.

---

## Part 4 — Phase 4 scope, revised (decided 2026-09-17)

Phase 4 keeps its **December 2026** date, with its primary horizon changed.

- **Primary horizon: 20 sessions**, where Phase 4 will hold roughly **8 independent
  non-overlapping observations** — enough to deliver a real verdict under the pre-registered
  decision rules.
- **40 and 60 sessions: reported as descriptive only.** 40 sessions reaches 5 independent
  observations around **March 2027**; a follow-up re-runs the test then. 60 sessions reaches 5
  around September 2027 and 20 observations not until 2031 — it is effectively out of reach on
  this dataset and should not be treated as a pending answer.
- **Verdicts are computed on non-overlapping phases**, lags 0, reported side by side with the
  spread across phases. The pooled weekly estimate may be shown for continuity but **cannot
  trigger a decision rule.**
- **Sample-size floor is now on independent observations: ≥5**, replacing Phase 3.5's floor of 8
  raw formation dates.
- **Weekly screens continue.** They are what generate the phases: 4 at a 20-session horizon and 8
  at 40, against 1 and 2 for a monthly cadence.

**Why 20 sessions is an acceptable proxy for a 40–60 session book.** Effect sizes rise
monotonically with horizon on non-overlapping data (`low_vol` 0.072 → 0.092 → 0.123 → 0.172
across 5/10/20/40). A signal that works at 20 sessions is therefore a **conservative lower bound**
for a longer hold, not a contradiction of it. The gap is acknowledged rather than assumed away,
and March 2027 closes it.

Phase 4's other three questions are unchanged: the out-of-sample comparison of `composite_score`
vs `composite_legacy` vs `composite_dedup`, the first test of fundamentals, and the unexplained
delivered-watchlist gap (*corrected 2026-09-19:* mostly a few lucky early lists — at 40 sessions
fewer than half the lists beat the universe; see Part 5).
A **pre-registered regime test** is also required, since the RISK-OFF screener allowance sunsets
on its outcome and Phase 3.5 had no RISK-OFF dates at all.

---

## Part 5 — Correction (2026-09-19)

Two metrics in `factor_study.py` compared a group **mean** with a population **median**: the
top-50 spread and the delivered-watchlist excess. `research-methods.md` has forbidden that since
Phase 2, which caught the same flaw in its own pre-registered metrics and reported unbiased
versions in the diagnostics script. The primary script never received the fix, and Phase 3.5
quoted its biased columns as findings — including in the rules files — for two days.

The bias is not constant; it grows with horizon, because survivor skew (mean − median) does:

| Horizon | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| Survivor mean − median | 0.41pp | 1.02pp | 2.09pp | 3.11pp | 3.45pp |

So a mean-vs-median comparison doesn't merely inflate an edge; it inflates it **more the longer
the horizon**, which is precisely the shape of the result it produced ("the edge rises seven-fold
with holding period"). The like-for-like columns (`top50_mean_vs_mean`,
`top50_median_vs_median`, `mm{h}`, `dd{h}`) are now written beside the legacy ones.

**What changes:** the edge at 40 sessions is ~+2.5–3.7pp, not +5.6pp; the delivered-watchlist
"mystery" is withdrawn as a Phase 4 question; and the Part 2 description of those lists as
"researched" was wrong — they are screener output.

**What does not change:** every rank-IC result (rank-based, unaffected), the quintile spreads
(`quintile_spread()` is mean vs mean), the non-overlapping verdicts in Part 3, and the Phase 4
scope. The rise in effect size with horizon holds on both the fair metrics and the rank ICs.

**Lesson for the pipeline, not just the analysis:** a fix applied in a diagnostics script and not
back-ported to the primary script is not a fix. The primary script now carries the fair columns.
