# Research Methods Rules

Rules for any quantitative study of screener signals, gates or strategy performance
(`research/`). Origin: the Phase 2 screener factor study, 2026-09-14
(`Experiment Details/Screener Factor Study — Phase 2.md`).

---

- **Pre-register decision rules and commit them before computing any result.** Apply them as
  written. Commit the raw output before interpreting it, and label every later analysis as post
  hoc. (Phase 2: rules `6c28378`, raw results `02ddb4d`.)

- **Compare like with like: group mean with population mean, group median with population
  median — never a group mean against a population median.** Small-cap forward returns are
  right-skewed: in the Phase 2 sample the survivors' 10-session mean exceeded their median by
  1.0pp (2.1pp at 20 sessions), so a mean-vs-median comparison flatters any group by about that
  much. Two pre-registered Phase 2 metrics made this mistake.

- **Never judge a signal on rank IC alone.** Report, beside it, the quintile spread on **mean**
  returns and an IC on winsorized (1%/99%) returns. In a universe whose median stock falls, rank
  IC rewards calm stocks sitting near zero: `low_vol` had the highest rank IC (0.113) while its
  calmest quintile earned *less* on average than its most volatile one.

- **Check redundancy before combining signals.** Report the mean cross-sectional rank correlation
  of every pair; at ρ ≥ 0.7 two signals are one exposure counted twice (`low_vol` ~ `squeeze`
  0.80).

- **Control for low volatility before crediting a signal** (partial IC). The screener's
  "squeeze" had no information beyond low volatility: partial IC −0.001.

- **Control for size too.** Low volatility and company size move together, so a volatility-
  flavoured ranking can quietly become a size tilt: on 2026-09-15 the top 50 had a median market
  cap of $2.4Bn against a universe median of $1.2Bn. Report partial IC on log market cap, and top-N
  results within size terciles, before crediting a signal with skill.

- **Set sample-size floors on EFFECTIVE observations, not raw formation dates.** With weekly
  formation dates and an h-session forward window, roughly `h/5` consecutive dates share the same
  forward period, so effective n ≈ dates ÷ (h/5). Phase 3.5 pre-registered a floor of 8 *dates*;
  at 40 sessions its 14 dates were **effective n 1.75**, and at 60 sessions its 10 dates were
  **0.83** — fewer than one independent observation — yet both cleared the floor. Report effective
  n beside every t-statistic.
  - **A Newey-West lag length approaching the sample size is not a conservative correction, it is
    an undefined one.** Phase 3.5's pre-registered `ceil(h/5)` rule put **12 lags on 10
    observations** at 60 sessions. Cap lags well below n, and say so when the cap binds.
  - **Symptom to watch for:** implausibly large t-statistics. Phase 3.5 printed t 14.77 and 15.30
    on 14 and 10 dates. A t above roughly 5 on a small sample of overlapping windows is evidence
    of a broken variance estimate, not of a strong signal.
  - This is the same class of error as the mean-vs-median comparison below: a pre-registered
    metric that looks rigorous and silently flatters the result.

- **Split results by the direction of the universe's median return.** Phase 2's defensive
  signals had most of their skill in falling-median weeks and almost none in rising ones —
  an average across both hides which kind of week a signal is useful in.
