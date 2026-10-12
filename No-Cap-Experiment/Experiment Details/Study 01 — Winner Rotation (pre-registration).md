# Study 01 — Winner Rotation: is the edge momentum, or concentration?

**Pre-registered 2026-10-12. Committed before any result was computed.**
Nothing below may be changed once data has been seen. Anything added later is labelled POST HOC.

Motivation: `Experiment Details/Thread Audit — Buying Last Year's Winner.md`

---

## 1. The question

The audited claim buys **one** stock — the prior calendar year's best S&P 500 performer — with
the full balance. It mixes two things that must be separated:

- **Momentum**: last year's winners keep winning, on average.
- **Concentration**: holding exactly one of them.

If the edge is momentum, a diversified basket of recent winners should also beat the index, and
is adoptable. If the edge only appears at N=1, it is a lottery ticket and is rejected.

> **Primary question: does a top-N momentum basket beat the S&P 500 in BOTH halves of the
> sample, and does the result survive as N rises?**

---

## 2. Hypotheses, stated before the data

- **H1.** Top-N baskets beat the index in both periods for some N ≥ 5. *(Adoptable.)*
- **H2.** Only N = 1–3 beat the index, and performance decays monotonically with N.
  *(Lottery. Rejected.)*
- **H3.** No N beats the index in both periods. *(Rejected; matches the predecessor
  experiment's small-cap finding.)*

**Prior belief, recorded now:** H2 is most likely, because the audit shows the headline is
carried by three years and a −97% drawdown. H1 is plausible at modest N because cross-sectional
momentum is well documented in large caps. Recording this so the result cannot be
retrospectively claimed as expected.

---

## 3. Design

**Universe.** S&P 500 constituents, point-in-time, as of each 31 December.

**Formation.** Rank members by total return over the completed calendar year.

**Holding.** Buy the top N on the first trading day of January, equal-weighted. Hold to the last
trading day of December. Rebalance annually. One trade cycle per year.

**N swept:** 1, 3, 5, 10, 20, 50.

**Periods — fixed now, never moved:**

| | Span | Role |
|---|---|---|
| **TRAIN** | 1997-01 → 2011-12 | Every configuration is ranked here and only here |
| **TEST** | 2012-01 → 2025-12 | The top few are then run on data the search never saw |

**2026 is excluded entirely.** It is an incomplete year, and including it is the single largest
distortion in the audited thread.

**Benchmark.** S&P 500 total return (SPY where available, index total return before it).
Reported in USD for the study; the live book's benchmark is the CAD series per
`portfolio_rules.md` §1. The study is about signal, not currency.

---

## 4. Decision rules — binding

1. **A configuration is reported as beating the index only if it clears the index's CAGR in
   BOTH train and test.** A win in one period is not a result.
2. **The full N curve is reported, always** — including the Ns that fail. If only N ≤ 3 passes,
   the family is **rejected as a lottery**, whatever N=1 returns.
3. **Adoption additionally requires the win to hold at N ≥ 5 with a maximum drawdown no worse
   than −50% in either period.** A result that requires surviving −97% is not adoptable at this
   book's horizon.
4. **Years-ahead-of-index is reported as a proportion with an exact binomial p-value**, never as
   a bare count. With ~29 annual observations, a count of 19 is not evidence.
5. **The number of configurations tried is printed with the results**, because the best of N
   tries is biased upward by roughly the spread of N draws.

**Stop-loss, tested separately and under a stricter rule.** Sweep 20/25/30/35/40/45/50%.
A stop is adoptable **only if a contiguous plateau of at least three adjacent levels all beat the
no-stop variant in both periods.** A single peak is a fitted parameter and is rejected.
This rule exists because the audited sweep showed −61% and −27% at the two levels adjacent to its
own optimum.

**Exclusion list.** `portfolio_rules.md` §6 is applied to every pick. When a name is excluded the
**next-ranked eligible name substitutes**, and the substitution is counted and reported. Results
are reported **both** with and without the exclusion list, so its cost is visible.

---

## 5. Data, and what is known to be wrong with it

**Point-in-time membership is the hard part and the main threat to validity.** Using today's
S&P 500 list to pick 2003's best performer would be survivorship bias of the worst kind — the
bias the predecessor experiment spent months eliminating.

- **Membership** will be reconstructed from the public record of index additions and removals,
  rolled backward from a current list. The reconstruction will be **committed as data before any
  backtest runs**, and spot-checked against known historical members.
- **Prices must include delisted and acquired names.** A name that left the index because it
  collapsed must still be present. Coverage will be measured and reported as a percentage of
  expected member-years, **before** results.
- **Dividends.** Total return where available; price return otherwise, stated per period.
- **Stops require intraday lows.** Where only closes exist, the stop is evaluated on closes and
  **that year is flagged**. The audited thread had this gap and reported ranges up to 1.75× wide
  because of it.

**If coverage of delisted names falls below 90% of expected member-years, the study is reported
as inconclusive rather than published with a number.**

---

## 6. Stages, each committed before the next begins

| Stage | Output | Commit gate |
|---|---|---|
| **0** | This pre-registration | Committed before any data is fetched |
| **1** | Membership reconstruction + coverage report | Committed before any return is computed |
| **2** | Raw backtest output, all N, all stop levels | Committed **before** being interpreted |
| **3** | Verdict against §4, written against the committed raw output | — |

Stage 2's output is raw CSV. The verdict is written afterwards, and any analysis not named in
this document is labelled POST HOC in stage 3.

---

## 7. What this study cannot answer

- **Whether we can execute it.** A one-stock-a-year rule in a TFSA is a single annual trade,
  which suits §5's turnover budget — but a 100% single-name position violates every tier in §2.
  Even a passing result would enter the book as a **satellite sleeve**, never as the whole book.
- **Whether momentum keeps working.** Thirty annual observations cannot establish that. A pass
  here means "not rejected", not "proven".
- **Statistical proof of alpha.** At a concentrated book's tracking error, demonstrating 5% of
  annual alpha at conventional confidence would take roughly 46 years. This study is a
  disqualification filter, not a confirmation.

---

## 8. Status

- [x] Stage 0 — pre-registered and committed
- [ ] Stage 1 — **blocked: this environment's network policy denies external data hosts.**
      Needs Wikipedia (membership) and a price source reachable from the session.
- [ ] Stage 2
- [ ] Stage 3

The engine is written and unit-tested against synthetic prices with known answers
(`research/test_engine.py`), so only the data acquisition is outstanding.
