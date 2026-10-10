# Capacity Rules Study — Phase 3.8

**Status: PRE-REGISTRATION ONLY. Nothing has been computed for these questions.** Parts 2 and 3
are deliberately empty. This file must be committed before the study script is written or run,
per `.claude/rules/research-methods.md` rule 1.

---

## Why this study exists, and why it supersedes the obvious one

Phase 3.75 was built to ask whether the RISK-OFF defensive profile's calm threshold was set too
strictly. While answering that, a **post-hoc** check asked a more basic question that had never
been run in this project's history: *do the weeks we label RISK-OFF actually go on to perform
worse?*

They do not. Over 178 RISK-OFF and 346 RISK-ON weeks, 2016–2026, survivorship-free:

| | Weeks | Mean 20-session forward return | Median week | % weeks positive | Worst week | Return ÷ risk |
|---|---|---|---|---|---|---|
| **RISK-OFF** | 178 | **+1.89%** | +1.94% | **66%** | **−27.1%** | **0.26** |
| **RISK-ON** | 346 | **+0.53%** | +0.49% | 60% | −38.7% | 0.09 |

**The weeks the rules restrict buying are the better weeks to buy, by +1.36pp per month.** It is
not a 2020 artifact (+1.29pp excluding 2020), it holds in both halves of the decade, RISK-OFF wins
in **8 of 11 years**, and the gap is *largest* during the 2022–23 bear market (**+2.35pp**) —
exactly when the filter is supposed to earn its keep. It does not even protect the downside: the
single worst month of the decade fell in a **RISK-ON** week.

**The mechanism, in hindsight.** RISK-OFF means *small caps have already fallen*, and the hard
gates only admit stocks **above their own 50-day average**. So the system buys relative strength
inside a weak index — a well-documented profitable pattern. **A trend-following rule on the index
was applied at a mean-reverting horizon.** A 50-day index average may say something about the next
year; it does not say this about the next month, and the book holds 40–60 sessions.

**So the question is no longer the calibration of a leg of the profile.** Phase 3.75's Q2 (is 0.90
too strict?) is now secondary: it tunes a gate that only exists because of a filter whose sign is
wrong. This study puts the capacity rules themselves on trial, because they are what has left the
book at **85% cash with the gap widening**.

**What is NOT in question here.** The **drawdown circuit breaker** and the **stop-loss discipline**
act on positions already held, not on a forecast of the next month, and the exit discipline is the
one part of this system with evidence behind it (stop exits are not implicated by any finding in
Phase 3.75). Neither is tested here and neither should be relaxed on this study's results.

---

## Part 1 — Pre-registration

### 1.1 The four questions

| # | Question | Rule it decides |
|---|---|---|
| **Q1** | Does **any** index-trend regime filter earn its place at a 40–60 session horizon? | The regime filter itself (`portfolio_rules.md` → *Risk Control*, *Capacity by regime*) |
| **Q2** | If a regime signal is worth having, **which** signal? Trend, volatility, breadth, or drawdown? | What replaces it, if anything |
| **Q3** | Does the **defensive profile** add anything once Q1 and Q2 are settled? | The two-leg profile and its 0.90 threshold |
| **Q4** | Is **5–6 positions at ~20% each** the right shape, or would more, smaller positions deploy cash better at equal or lower risk? | Position ceiling, risk-per-trade, and the 15% cash floor |

Q4 is included because it is the other arithmetic cause of idle cash: even with candidates
available, a 5–6 position ceiling at 2% risk cannot absorb 70% of equity quickly.

### 1.2 Data — reuses Phase 3.75's panel, unchanged

The survivorship-free panel built and verified in Phase 3.75: **~499 companies**, point-in-time
membership from Tiingo's dated ticker file, **weekly formation dates 2016–2026**, forward returns
at 5/10/20/40/60 sessions truncated at delisting, price-computable gates applied before ranking.
R12's gate-differential test passed at 2.5–3.8pp, so no gate selects on survival.

**No new data is required and no further API allowance is spent.** The regime variants in Q2 are
computed from IWM, SPY and ^VIX, all of which are still listed and come from yfinance.

### 1.3 Method — portfolio simulation, not cross-sectional spreads

Phase 3.75's Q2 failed because a *median spread* over 2 qualifying names is an unstable estimate
of a population parameter. **This study simulates portfolios instead**, which is well defined at
any group size: a book that buys the two names that qualify is exactly what the live book does.

For each rule set and each formation date: buy every qualifying name **equal-weighted**, hold for
the horizon, and record the return. **A date with no qualifying name holds cash and records 0.00%**
— that is not a missing observation, it is the rule's actual consequence, and excluding it would
hide the capacity cost entirely.

Reported for every rule set, always together:
1. **Mean and median** week, separately (never crossed — `research-methods.md`).
2. **% of weeks positive**, and **% of weeks forced to cash**.
3. **Worst week, 5th percentile, standard deviation, and mean ÷ standard deviation.** A capacity
   rule that trades return for tail protection must be allowed to show it.
4. **Average names held** — the capacity measure.
5. **Newey-West t-statistic with the lag cap**, and **effective n** beside it.
6. **Both halves of the decade separately**, and **by year**.

### 1.4 Horizons

**40 sessions is PRIMARY** — the midpoint of the book's stated 40–60 session hold, and the horizon
the capacity rules actually govern. 20 and 60 are reported. *Phase 3.75 used 20 as primary because
Phase 3.5 had set it for the factor work; that was the wrong choice for a question about capacity
rules, which bind over a whole holding period.* 5 and 10 are reported for continuity and cannot
decide anything.

Effective n = dates ÷ (h/5). At ~520 weekly dates: **~65 at 40 sessions, ~130 at 20, ~43 at 60**.
All clear the floor. Newey-West lags `min(ceil(h/5), floor(n/5))`, and the output says when the cap
binds. **Decision floor: effective n ≥ 10.** **Abort on any |t| > 5** — a t that large on
overlapping windows is a broken variance estimate, not a signal.

### 1.5 The rule sets to be simulated

**Q1 — regime filters, on trial:**
- `none` — always fully deployable, no regime concept at all
- `current` — IWM vs its 50-day SMA, ±1% band (the rule in force)
- `inverted` — the current rule with the sign flipped, as the direct test of the Phase 3.75 finding
- `current_200d` — IWM vs its 200-day SMA, to test whether the lookback rather than the concept is wrong
- `spy_50d` — the same rule on SPY rather than IWM

**Q2 — alternative regime concepts** (only if Q1 shows a filter of some kind helps):
- `vix_high` / `vix_low` — ^VIX above/below its own 1-year median
- `breadth` — the share of the eligible universe above its own 50-day SMA, high vs low
- `drawdown` — IWM more than 10% below its 1-year high

**Q3 — the defensive profile**, under whichever regime rule Q1 and Q2 select:
- profile applied (two legs: `rank_low_vol` ≥ 0.90, `vol_5_50` > 1.0)
- profile dropped
- the calm leg at 0.50 / 0.60 / 0.75 / 0.80 / 0.90

**Q4 — book shape**, under the selected rules: hold the top **3 / 5 / 8 / 12 / 20** qualifying
names by composite score, equal-weighted, and separately at a **15% / 5% / 0% cash floor**.

### 1.6 Decision rules — pre-committed, applied mechanically

**Q1 — the regime filter.**
- **R1.** The current filter is **retained** only if it beats `none` on **either** mean return
  **or** mean ÷ standard deviation at 40 sessions, with NW **t ≥ 2.0** on the paired difference.
- **R2.** If `none` beats `current` on **both** measures with t ≥ 2.0, **the regime filter is
  removed** from the capacity rules, and the capacity table reverts to a single set of limits
  that does not reference regime.
- **R3.** If `inverted` beats `current` on both measures with t ≥ 2.0, that is reported as
  **confirmation that the signal is inverted** — but **inverting the live rule is NOT adopted on
  this study alone.** An inverted trend filter is an unvalidated market-timing bet in the opposite
  direction, and the honest conclusion from a signal with the wrong sign is to stop using it, not
  to trade it backwards. Removal (R2) is the adoptable outcome; inversion is evidence, not a rule.
- **R4.** If neither direction clears t ≥ 2.0, the filter is **removed** regardless: a capacity
  rule that cannot be shown to help in either direction is unjustified complexity gating real
  capital.

**Q2 — what replaces it.**
- **R5.** A replacement signal is adopted only if it beats `none` on **both** mean return and
  mean ÷ standard deviation at 40 sessions with t ≥ 2.0, **and** does so in **both halves** of
  the decade. Otherwise no regime signal is adopted and capacity becomes regime-independent.
- **R6.** Where two replacements both qualify, the one with the **better worst week** is chosen,
  not the one with the higher mean. Capacity rules exist to bound bad outcomes.

**Q3 — the defensive profile.**
- **R7.** The profile is **retained** only if it beats "profile dropped" on mean ÷ standard
  deviation at 40 sessions with t ≥ 2.0. Its stated purpose is defensive, so return alone cannot
  justify it.
- **R8.** If retained, the calm threshold is set to the **loosest** value whose 95% interval on the
  paired difference against 0.90 **excludes a loss worse than 0.5pp** — a non-inferiority test,
  not "no significant harm found", which noise satisfies. It must additionally hold in both halves
  and **the verdict must be monotonic across the threshold curve**; a curve whose verdict flips
  more than once is noise and keeps 0.90.
- **R9.** Any rule set whose **median group size is below 10 names** is reported but cannot decide
  anything. A median over fewer names is the average of whichever handful qualified.

**Q4 — book shape.**
- **R10.** A different position count is adopted only if it beats 5 names on **both** mean return
  and mean ÷ standard deviation at 40 sessions with t ≥ 2.0, in both halves.
- **R11.** The **15% cash floor** is lowered only if a lower floor beats it on mean ÷ standard
  deviation with t ≥ 2.0. A floor that costs return and buys no stability is removed; a floor
  that buys stability stays, whatever it costs in return.
- **R12.** **Nothing in Q4 may raise risk-per-trade above 2%.** That is the one parameter this
  study cannot test, because the simulation is equal-weighted and carries no stops — see §1.7.

**Universal.**
- **R13.** Any horizon whose effective n < 10 is descriptive and decides nothing.
- **R14.** Every rule is evaluated at the **primary 40-session horizon**. Agreement at 20 and 60 is
  corroboration; disagreement does not override R1–R12 but must be stated prominently.
- **R15.** Raw output is committed **before** interpretation; every analysis not listed in this
  Part is labelled **post hoc**.
- **R16.** No rule in `portfolio_rules.md` is amended until this file's Part 2 is written and
  committed, and the amendment names the decision rule that produced it.

### 1.7 What this study cannot answer

- **Anything involving stops.** The simulation holds blind for the full horizon. The real book
  exits on a trailing stop, which truncates losses and changes every tail figure — most for the
  most volatile rule sets. **So this study systematically overstates the tail risk of loose rules
  and understates the value of the stop.** That is the correct direction of caution for a study
  being used to loosen capacity, but it means no tail figure here is the book's tail figure.
- **Risk-per-trade sizing.** The book sizes by 2% risk at the stop, which gives calm names larger
  positions. This simulation is equal-weighted and cannot represent that, which is why R12
  forbids touching it.
- **The 10-session pre-earnings guard**, which in practice blocks 23–39 of the top 50 in October
  and is therefore a major cause of candidate scarcity. **Historical earnings dates are not
  available on this data**, so the guard cannot be re-tested here. Its only evidence remains the
  one-season study of 2026-09-19. **This is the largest untested capacity constraint in the book
  and it needs its own study with an earnings-date source.**
- **Any fundamental gate** — revenue growth, EPS, forward P/E, analyst targets. No point-in-time
  fundamentals.
- **The catalyst lane.** No historical dataset of dated, non-binary catalysts exists here.
- **Absolute expected returns.** Every figure is a relative comparison inside one survivorship-free
  universe, on an equal-weighted basis with no costs, no slippage and no stops.

### 1.8 Deliverables

- `research/capacity_study.py` — written only after this file is committed.
- `research/output/capacity_study_*.csv` — raw output, committed before interpretation.
- **Part 2**: results, applying §1.6 mechanically.
- **Part 3**: post-hoc observations, explicitly labelled.

---

## Part 2 — Results

*Empty. To be filled only after Part 1 is committed and the script has been run.*

---

## Part 3 — Post-hoc observations

*Empty. Anything not pre-registered in Part 1 belongs here, labelled as post hoc.*
