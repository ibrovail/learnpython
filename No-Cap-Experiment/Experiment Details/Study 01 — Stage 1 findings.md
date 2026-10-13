# Study 01 — Stage 1: membership reconstruction and its quality

**Committed before any return was computed**, per the pre-registration's staging gate.
Raw output: `research/output/membership_quality.csv`, `membership_sizes.csv`,
`membership_validation.csv`.

---

## Headline

**The pre-registered train period (1997–2011) cannot be built and is abandoned.** Two
independent tests agree, and neither was a judgement call.

The usable span is **formation years 2011–2024, holding years 2012–2025** — fourteen years.
That is, uncomfortably, the window in which the audited strategy worked.

---

## Test 1 — completeness of the change record

Membership is reconstructed by rolling the current constituent list backward through the
published record of index changes. Rolling backward is the only safe direction: rolling an
old list forward drops every company later acquired or delisted.

The reconstruction is therefore only as good as the change record. The real index replaces
roughly 20–25 names a year. The record holds:

| Period | Changes recorded per year | Verdict |
|---|---|---|
| 1996–2006 | **0 to 7** (median 1) | Unusable — ~90% of changes missing |
| 2007–2010 | 8 to 13 | Marginal |
| 2011–2025 | 16 to 30 | Usable |

Eleven of the twelve years before 2007 fail a 50%-completeness bar. 2001, 2002 and 2004
record **zero** changes.

**Set size is not a validity test and was rejected as one.** Every reconstructed year holds
503–513 names, which looks correct and is not: an unrecorded addition and an unrecorded
removal cancel in the count while leaving the composition wrong.

## Test 2 — composition against known listing dates

Thirty companies with known listing, spin-off or ticker-start years were checked against each
snapshot. A company present before it could possibly have been a member is a direct measure of
reconstruction error. Three distinct failure modes appeared:

| Mode | What it means | Where it bites |
|---|---|---|
| **Survivorship** | A modern winner the backward roll failed to remove | 1996–2003: GOOG/GOOGL, MA, V |
| **Ticker reuse** | The ticker was a member, but of a *different* company | Through 2018: DELL, DOW, FOXA, CEG, SNDK, LIN, GM |
| **Ticker rename** | Same company, later ticker | META (was FB until 2022) |

Snapshots are clean of all three only from **2022**. But the three modes are not equally
damaging, and the distinction decides the study:

- **Survivorship contamination effectively disappears after 2003.** From 2007 onward the only
  flagged names are reuse and rename artifacts. The backward roll is not leaking modern winners
  into the recent past.
- **Ticker reuse is a price-keying problem, not a membership problem.** It is fixable by mapping
  a ticker to the right company-period, and it must be fixed: a price series keyed on `DELL`
  alone silently splices Dell Computer onto Dell Technologies across a 2013 going-private and a
  2016 relisting.

---

## Consequence for the study

**The 1997–2011 half is gone.** It cannot be reconstructed from a free source, and it is the
half that mattered most: it contains the −97% drawdown, the twelve consecutive losing years,
and two of the four years the stock fell more than 50% (2000 QCOM −53%, 2002 NVDA −83%).

**What remains is a test on the strategy's favourable window.** This is worth running anyway,
because of an asymmetry worth stating before the result is known:

> **A failure in 2012–2025 is conclusive. A pass is weak evidence.**

The thread already established that N=1 won in this window — that is the window it was built
from. So the informative question is not whether N=1 won; it is **whether the edge survives
diversification**. If a top-10 basket does not beat the index even here, the momentum story is
dead and only the lottery remains, and that conclusion holds regardless of the missing years.

---

## Amendment to the pre-registration

Made **before any return was computed**, and driven by data availability rather than by a
result. Recorded so the change is auditable:

| | Pre-registered | Amended |
|---|---|---|
| TRAIN | 1997-01 → 2011-12 | **2012-01 → 2018-12** |
| TEST | 2012-01 → 2025-12 | **2019-01 → 2025-12** |
| Holding years | 29 | **14** |

Everything else in the pre-registration stands unchanged, including the rule that a family
passing only at N ≤ 3 is rejected as a lottery, and that a stop is adopted only on a plateau.

**Added to Stage 1's remaining work:** a ticker reuse/rename map, applied before prices are
keyed. Without it the price panel is wrong for at least seven names in the study window.

**Power warning, recorded now:** seven observations per half is very little. The N curve and
the stop surface — shapes across many configurations — carry more information than any single
CAGR comparison, and the verdict should lean on them.
