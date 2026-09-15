# Screener Factor Study — Phase 2

## Part 1 — Pre-registration

*Written and committed 2026-09-14, **before any factor result was computed.** The decision
rules below are fixed; Part 2 applies them to whatever the data shows.*

### Question

Which signals should the screener's composite use, with what weights — and is option 3
(timing-based screener modes) justified by evidence? Phase 1 fixed the pipeline (hard gates
before ranking); the composite weights (40% momentum / 30% volume / 30% squeeze) were chosen by
judgment and have never been tested.

### Signals tested

**Current composite components** — computed exactly as `screener.py` does:

| Signal | Definition | Direction |
|---|---|---|
| `mom20` | close / close 19 sessions earlier − 1 | higher ranks better |
| `vol_ratio` | today's volume / 20-session average volume | higher |
| `squeeze` | Bollinger width = 4 × 20-session σ / 20-session SMA | **lower** ranks better |
| `composite` | 0.40 rank(mom20) + 0.30 rank(vol_ratio) + 0.30 rank(−width) | higher |

**Candidate signals:**

| Signal | Definition | Why test it |
|---|---|---|
| `mom5` | 5-session return | shorter horizon (option 3's short mode) |
| `mom60_skip5` | return from 60 to 5 sessions ago | intermediate momentum, skipping the reversal-prone latest week |
| `mom60_riskadj` | `mom60_skip5` / 60-session return σ | momentum per unit of risk |
| `vol_5_50` | 5-session / 50-session average volume | multi-day volume instead of one session |
| `squeeze_own` | Bollinger width / its own 60-session median | a true squeeze: tight *for that stock* |
| `low_vol` | 20-session return σ | **lower** ranks better — tests whether `squeeze` is just low volatility |
| `near_high` | close / 60-session high − 1 | proximity to highs |
| `vs_sma50`, `vs_sma20` | distance above the 50- / 20-session SMA | trend strength |

### Data

- **Universes:** the committed `universe_cache.csv` snapshots from git history
  (2026-04-17 → 2026-09-14). Each formation date uses the latest snapshot committed on or
  before it — the universe the screener actually saw.
- **Formation dates:** every Friday session from 2026-04-17 onward that has enough forward
  data for the horizon measured.
- **Prices:** yfinance daily bars, split- and dividend-adjusted, from 2025-11-03.
- **Eligible at formation:** price ≥ $1; 20-session average dollar volume ≥ $500K; snapshot
  market cap ≤ $5B; the security-type and prohibited-business exclusions in `screener.py`.
- **Population ranked:** eligible stocks passing the **price-computable Phase 1 gates** —
  deal-pinned (ATR(14) < 0.75%), >40% above the 50-day SMA, >20% above the 20-day SMA, days
  1–3 of a >10% breakout. The shrinking-revenue and post-earnings gates cannot be replicated
  historically (no point-in-time fundamentals or earnings dates) — the one known departure from
  the live screener.

### Measurement

- **Rank IC:** per formation date, the Spearman correlation between a signal and the forward
  total return over **5, 10 and 20 sessions**.
- Reported per signal and horizon: **mean IC**, **hit rate** (share of dates with IC > 0), and a
  **Newey-West t-statistic** with lags 0 / 1 / 3 (weekly formation dates overlap the 10- and
  20-session horizons, so a naive t-statistic would overstate confidence).
- Also: the forward return of the **top 50 by composite** minus the population median; and the
  forward return of the **watchlists actually delivered** (committed `watchlist.csv` snapshots)
  minus the universe median.

### Decision rules — fixed before results

1. **Primary horizon: 10 sessions** — the default catalyst-timing directive, and it covers the
   5-session minimum hold. 5 and 20 sessions are secondary.
2. **Keep or add a signal:** mean IC > 0, Newey-West t ≥ 2.0, and IC > 0 on ≥ 60% of dates — all
   at 10 sessions.
3. **Drop a signal:** mean IC ≤ 0, or t < 1.0, at 10 sessions.
4. **Unproven (1.0 ≤ t < 2.0):** a current component keeps *at most* its current weight; a
   candidate is not added.
5. **Weights:** the new composite **equal-weights** the ranks of every signal passing rule 2.
   No IC-optimized weights — about twenty formation dates cannot support them without overfitting.
6. **If no signal passes rule 2:** no signal has demonstrated ranking skill. Rank survivors by
   the highest-t signal that was not dropped, say plainly that the ordering is weak evidence, and
   keep collecting data (the screener now saves every screen for Phase 4).
7. **Timing modes (option 3):** build them only if some signal passes rule 2 at one of the 5- or
   20-session horizons **and** has mean IC ≤ 0 at the other. Otherwise one mode.
8. **Regime split** (IWM above / below its 50-day SMA) is **descriptive only** — too few dates
   in each regime to set regime-specific weights.
9. **Gates are rules, not tuning parameters.** The study reports how each gated group performed;
   a gate changes only if its group clearly outperformed the survivors, and then only as a
   proposal to the user.

### Limitations — stated before results

- About twenty formation dates, all within five months of 2026. One market environment.
- **Survivorship:** stocks delisted or acquired since formation may be missing forward prices.
  Coverage is reported per date.
- The live screener computes signals on unadjusted closes; this study uses adjusted prices so
  splits and dividends do not create false returns. Signal values can differ slightly.
- **Fundamentals cannot be tested** — no point-in-time history exists before 2026-09-14.

---

## Part 2 — Results

*Run 2026-09-14. Pre-registration committed as `6c28378` (22:54 ET); the study script and its
raw output committed as `02ddb4d` before any interpretation. Everything under "Post-hoc
diagnostics" was added after seeing the results and is labelled as such.*

### Data as run

- **21 Friday formation dates** (2026-04-17 → 2026-09-04): 21 with 5-session forward data, 20 with
  10, 18 with 20. **19 universe snapshots** from git history.
- **2,364 of 2,415 tickers priced.** Forward returns exist for 99.6–100% of survivors on every
  date, so survivorship bias is negligible.
- **636–1,121 survivors per date.** The universe definition changed twice during the window:
  about −25% around 2026-07-21, then +55% on 2026-08-16 when the ceiling rose to $5B.
- **Regime at formation:** 18 dates RISK-ON, 3 RISK-OFF — only 2 of them with 10-session data.

### Pre-registered results — 10 sessions (primary horizon)

| Signal | Mean IC | Newey-West t | IC > 0 | Verdict |
|---|---|---|---|---|
| `vol_ratio` | 0.052 | 3.22 | 80% | **PASS** |
| `low_vol` | 0.113 | 2.88 | 80% | **PASS** |
| `near_high` | 0.097 | 2.74 | 80% | **PASS** |
| `vol_5_50` | 0.048 | 2.51 | 70% | **PASS** |
| `squeeze` | 0.086 | 2.45 | 75% | **PASS** |
| `vs_sma50` | 0.052 | 2.21 | 80% | **PASS** |
| `composite` (current) | 0.072 | 2.15 | 85% | PASS |
| `vs_sma20` | 0.042 | 1.50 | 65% | UNPROVEN |
| `mom20` (40% of today's composite) | 0.031 | 1.24 | 70% | UNPROVEN |
| `mom5` | 0.017 | 0.69 | 55% | DROP |
| `mom60_skip5` | 0.011 | 0.42 | 60% | DROP |
| `mom60_riskadj` | 0.010 | 0.40 | 55% | DROP |
| `squeeze_own` | −0.007 | −0.62 | 35% | DROP |

At 5 sessions the same signals lead (`low_vol` t 3.42, `near_high` 2.96, `squeeze` 2.92,
`vol_ratio` 2.82; `mom20` 0.58). At 20 sessions: `near_high` 3.42, `vol_5_50` 2.91, `vol_ratio`
2.78, `low_vol` 2.71, `vs_sma50` 2.68; both 60-session momentum signals turn negative.

**The rules applied:**

- **Rule 5 — new composite:** equal-weight ranks of `low_vol`, `near_high`, `squeeze`, `vol_5_50`,
  `vol_ratio` and `vs_sma50`.
- **Rule 4 — `mom20`:** unproven, so it may keep at most its current 40%. It is removed: it adds
  nothing once low volatility is controlled for (partial IC 0.016, t 0.8), and it is 0.77
  correlated with `vs_sma50`, which is in.
- **Rule 7 — timing modes:** no signal passes at one horizon while ≤ 0 at the other. **One screener
  mode; option 3 is not supported by the evidence.**
- **Rule 8 — regime (descriptive):** the current composite's 10-session IC was −0.10 on the 2
  RISK-OFF dates and +0.09 on the 18 RISK-ON dates. Two dates cannot support a conclusion; the
  regime filter stays.
- **Rule 9 — gates:** every price-computable gate removed names that did worse than survivors.
  Median 10-session excess: **>40% above the 50-day −3.2pp, >20% above the 20-day −2.3pp, fresh
  breakout −2.1pp.** Deal-pinned stocks earned **0.00%** mean and median; they beat the survivors'
  median only in weeks it fell. **No gate changes.**

**A flaw in the pre-registration.** Two metrics — "top 50 minus the survivor median" and
"delivered watchlist minus the universe median" — compared a group *mean* with a population
*median*. Small-cap returns are right-skewed, so that comparison flatters any group. The unbiased
versions are reported below.

### Post-hoc diagnostics — not pre-registered

**1. The typical stock fell; the average did not.** Survivors' median 10-session return was
−0.54% (negative on 13 of 20 dates), while the mean was +0.47%. At 20 sessions: median −0.98%,
mean +1.10%. A few large winners carry the average.

**2. The signals mainly identify losers.** The worst-ranked quintile's median 10-session return was
−3.6% for `low_vol`, −2.9% for `near_high`, −2.6% for `squeeze` and −2.5% for the current
composite; the best-ranked quintile sat near zero. **On mean returns the quintile spread was about
zero or negative** — `low_vol` −0.5pp, `squeeze` −0.9pp, current composite −0.4pp — with
`near_high` (+0.3pp) and rule 5 (+0.25pp) the exceptions. The extreme right tail lives in volatile,
low-ranked names.

**3. With the most extreme 1% of returns clipped, the skill survives.** IC against winsorized
returns at 10 sessions: `vol_ratio` 0.052 (t 3.30), `vs_sma50` 0.052 (2.58), `near_high` 0.081
(2.55), `vol_5_50` 0.045 (2.52), `low_vol` 0.082 (2.05). A book of 3–5 stop-managed positions
rarely captures those lottery outcomes, so this is the more relevant measure.

**4. Low volatility explains most of the rest.** Partial IC after controlling for `low_vol`, at 10
sessions:

| Signal | Partial IC | t |
|---|---|---|
| `vol_5_50` | 0.033 | 2.38 |
| `vol_ratio` | 0.023 | 1.90 |
| `vs_sma50` | 0.033 | 1.72 |
| `near_high` | 0.039 | 1.61 |
| `mom20` | 0.016 | 0.80 |
| current composite | 0.011 | 0.61 |
| `squeeze` | −0.001 | −0.07 |
| `squeeze_own` | −0.029 | −2.47 |

**`squeeze` carries no information beyond low volatility, and neither — nearly — does the current
composite.** Volume is the one ingredient with clearly independent information (`vol_5_50` t 2.67
and `vol_ratio` t 2.43 at 20 sessions). A true squeeze (`squeeze_own`) predicted slightly *worse*
outcomes once calmness is accounted for.

**5. Most of the skill appears in weeks when the median stock fell.** 10-session IC when the
survivors' median fell (13 dates) vs rose (7 dates): `near_high` 0.164 vs −0.026; `low_vol` 0.172 vs
0.004; current composite 0.108 vs 0.005; `vs_sma50` 0.091 vs −0.022. **The exception is
`vol_ratio`: 0.048 vs 0.058** — it works either way.

**6. Redundancy.** Mean cross-sectional rank correlation: `low_vol` ~ `squeeze` **0.80**,
`near_high` ~ `vs_sma50` **0.75**, `vs_sma50` ~ `mom20` 0.77, `vol_ratio` ~ `vol_5_50` 0.49. Rule 5
therefore counts calmness and trend twice each.

**7. Composites side by side** (all in-sample; the rule-5 signal set was chosen on this data).
`rule5_dedup` drops the weaker signal of each pair at ρ ≥ 0.70 by winsorized-return t, keeping
`low_vol`, `vol_5_50`, `vol_ratio`, `vs_sma50`. Its deciding step was a near-tie: `vs_sma50` over
`near_high` at t 2.58 vs 2.55.

| 10 sessions | Current | Rule 5 | Rule 5 + `mom20` | Rule 5 dedup |
|---|---|---|---|---|
| Rank IC (t) | 0.072 (2.15) | 0.105 (2.82) | 0.099 (2.70) | 0.096 (3.15) |
| Winsorized IC (t) | 0.064 (2.11) | 0.085 (2.44) | 0.083 (2.46) | 0.083 (2.88) |
| Q5−Q1 median / mean | +2.5 / −0.4pp | +3.4 / +0.3pp | +3.4 / +0.2pp | +3.3 / +0.3pp |
| Top 15: mean vs mean (dates +) | +0.50pp (60%) | +0.27pp (50%) | +0.34pp (65%) | +0.80pp (60%) |
| Top 15: median vs median (dates +) | +0.88pp (75%) | +0.91pp (70%) | +1.20pp (75%) | +0.94pp (75%) |
| IC, median falling / rising | 0.108 / 0.005 | 0.169 / 0.009 | 0.160 / 0.009 | 0.137 / 0.022 |

At 20 sessions the top 15 beat survivors on mean by +0.38pp (current), +1.12pp (rule 5), +0.86pp
(rule 5 + `mom20`) and +1.94pp (dedup). **None of these differences is distinguishable from noise**
— each top 15 beats the survivors' mean on only 50–67% of dates.

**8. The watchlists the book actually researched did better than the rebuilt composite.** The 18
delivered top-15 lists beat the eligible universe by **+2.25pp on mean (67% of lists) and +1.45pp
on median** over 10 sessions, and +4.08pp / +3.02pp over 20. The rebuilt current composite's top 15
managed +0.50pp / +0.88pp. The gap is unexplained: the delivered lists predate the gates and were
built from unadjusted, sometimes intraday, data. Open for Phase 4.

**9. Momentum.** 20-day momentum — today's largest weight — showed no skill after gates. 60-session
momentum's top quintile trailed its bottom quintile by **5.7pp** on mean over 20 sessions: reversal,
not continuation, in this universe and period.

### Decision

**Adopted — the pre-registered outcome:**

1. **Composite:** equal-weight ranks of `low_vol`, `near_high`, `squeeze`, `vol_5_50`, `vol_ratio`
   and `vs_sma50`. `mom20` leaves the composite.
2. **One screener mode** — option 3 (timing-based modes) is not built.
3. **Gates unchanged.** **Regime filter unchanged.**

**Not adopted, tracked instead:** the deduplicated composite scored best in-sample, but it was
chosen after seeing the data and its key step was a near-tie. Recording it and the old composite
alongside rule 5 in every screen's history file lets Phase 4 compare all three on data none of
them was chosen on.

**What this means for the book.** The ranking's practical edge is modest — under 1pp per 10
sessions for the top 15 on mean, about 1pp on median — and it comes mostly from steering away from
volatile losers, not from finding winners. Volume confirmation is the most robust ingredient.
Screener rank remains sourcing, not conviction: the PRV gate and research decide.

**Limits.** 20 formation dates in one five-month period; only 2 RISK-OFF dates, while the market is
RISK-OFF now; composite comparisons are in-sample; fundamentals untested; the universe definition
changed twice mid-sample.

**Next:** Phase 3 implements the adopted composite and the shadow scores. Phase 4 re-runs this
study on the saved screens — including the Finviz fundamentals saved since 2026-09-14 — after about
12 more weekly screens.
