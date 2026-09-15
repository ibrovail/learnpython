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

- **Split results by the direction of the universe's median return.** Phase 2's defensive
  signals had most of their skill in falling-median weeks and almost none in rising ones —
  an average across both hides which kind of week a signal is useful in.
