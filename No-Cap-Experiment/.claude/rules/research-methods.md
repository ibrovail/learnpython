# Research Methods Rules

Binding on any quantitative study in `research/`. Carried over from the predecessor
experiment, where each of these was learned by getting it wrong first.

---

- **Pre-register the decision rules and commit them before computing any result.** Apply them
  as written. Commit the raw output before interpreting it. Label every later analysis POST HOC.

- **Split the sample and honour the split.** Rank configurations on train only; run the
  survivors on test once. A result that holds in one period is not a result. The audited
  "winner rotation" thread would have been discarded on its own 1997–2011 data.

- **Print the number of configurations tried.** With N tries, the best train result is biased
  upward by roughly the spread of N draws, and that bias must be visible beside the number.

- **Never report a bare win count.** 19 of 30 years ahead of an index is p = 0.100 against a
  coin flip. Report the proportion with an exact binomial p-value, every time.

- **A parameter is only adopted on a plateau, never on a peak.** Sweep it and require at least
  three adjacent levels to work. A single optimum flanked by sharp falls is a fitted number —
  the audited 35% stop lost 61% of terminal wealth at 30% and 27% at 40%.

- **Count the events a parameter was estimated on.** Four is not an estimate.

- **Exclude incomplete periods.** A year-to-date figure is not a year. Including one open
  position inflated the audited result by roughly 6.7×.

- **Survivorship is the default state of financial data; absence of it must be demonstrated.**
  Report coverage of delisted and acquired names as a percentage of expected member-years,
  before any return is computed. Below 90%, report the study inconclusive rather than publish
  a number.

- **Compare like with like, and say what each figure is *per*** — per name, per row, per date,
  per dollar. A group mean against a population median flatters the group by the skew.

- **Set sample-size floors on EFFECTIVE observations, not raw dates.** With overlapping windows,
  effective n is far below the row count. Report it beside every t-statistic.
  - A t-statistic above roughly 5 on a small overlapping sample is evidence of a broken variance
    estimate, not of a strong signal.

- **State the prior before seeing the result**, so the outcome cannot be retrospectively claimed
  as expected.

- **Model frictions explicitly and pessimistically.** Name the optimistic assumptions that
  remain — selling exactly at a stop price, for instance, when real exits gap through it.
