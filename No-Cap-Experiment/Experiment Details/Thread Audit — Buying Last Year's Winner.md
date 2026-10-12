# Thread Audit — "Buy last year's best S&P 500 stock"

**Date:** 2026-10-12 · **Source:** a public thread by @gurgavin, Exhibits 1–9
**Status:** audit of someone else's result. No data of our own was used. Every figure below is
derived from the thread's own published numbers.

**Claim.** Each January, move the full balance into the prior calendar year's best-performing
S&P 500 stock; hold for the year; repeat.

| Window | Strategy | S&P 500 |
|---|---|---|
| 2012 → Oct 2026 | $100 → $176,114 (65.9%/yr) | $100 → $800 (15.1%/yr) |
| 1997 → Oct 2026 | $100 → $89,418 (25.6%/yr) | $100 → $1,752 (10.1%/yr) |
| 1997 → Oct 2026, with a 35% stop-loss | $100 → $1,514,581 (38.2%/yr) | — |

---

## Verdict

**The arithmetic is sound. The conclusion is not.** Every headline figure reconciles against its
own stated inputs. Nothing is fabricated. The strategy nonetheless fails on four counts, three of
which are visible in the thread's own exhibits.

---

## 1. It fails a split-sample test, using the thread's own numbers

The two published windows overlap, so the first half can be recovered by division:
$89,418 ÷ ($176,114/$100) = $51, which matches Exhibit 5's stated "$51 at end of 2011".
The same division on the S&P gives $1,752 ÷ 8 = $219.

| Period | Strategy | S&P 500 |
|---|---|---|
| **1997–2011** (15 yrs) | $100 → **$51** (**−4.4%/yr**) | $100 → **$219** (**+5.4%/yr**) |
| **2012–2026** (14.8 yrs) | $100 → **$176,114** (+65.9%/yr) | $100 → **$800** (+15.1%/yr) |

Two consecutive fifteen-year samples. In the first, the strategy **lost half its capital while
the index more than doubled** — a gap of roughly **10 percentage points a year for fifteen
years**. Evaluated in 2011 on fifteen years of real data, it would have been discarded.

Corroborating tell: the 30-year annualised figure (25.6%) is **less than half** the 15-year
figure (65.9%). Adding older data halves the result. Durable edges do not behave this way.

---

## 2. Beating the index 19 times in 30 years is not evidence of skill

Exhibit 6 reports **19 of 30 years ahead of the S&P 500**. Against a null of a coin flip:

- **19/30, no stop** → exact one-sided binomial **p = 0.100**. Not significant.
- **20/30, with the 35% stop** → exact one-sided **p = 0.049**. Nominally marginal —
  **but that p is conditional on having selected 35% from a sweep of nine stop levels.**
  Correcting for nine looks requires p < 0.006. It does not survive.

Thirty annual observations cannot distinguish this strategy from luck.

---

## 3. The stop-loss is fitted to four events, and the thread says so

Exhibit 9 sweeps the stop level. The surface is a **spike, not a plateau**:

| Stop | Terminal value | vs the 35% setting |
|---|---|---|
| 15% | $4,741 – $21,384 | −99% |
| 20% | $109,357 – $270,233 | −87% |
| 25% | $390,593 – $682,756 | −66% |
| 30% | $589,266 | **−61%** |
| **35%** | **$1,514,581** | — |
| 40% | $1,099,624 | **−27%** |
| 45% | $776,408 | −49% |
| 50% | $530,297 | −65% |

Moving the parameter five points in either direction destroys 27% to 61% of terminal wealth.
A robust parameter shows a flat region; this shows a peak.

Three further facts, all from the thread's own notes:

- **The stop fired exactly four times in thirty years** (2000, 2002, 2008, 2011). A parameter
  estimated on four observations is not estimated.
- **The optimum is pinned to one stock's intraday low.** The note states the single best level
  was 33%, *"less than a point past Reebok's 2001 dip (32.3%)"*. The entire parameter choice is
  determined by one company's worst day in one year. The author concedes it is
  *"picked with hindsight."*
- **The stop contributed nothing to the modern sample.** Exhibit 9: *"$100 from Jan 2012 with a
  35% stop = $176,114. Same as no stop: no pick since 2012 fell 35%."* Its entire 17× advantage
  comes from the 1997–2011 window — the same window in which the strategy lost half its money.

Several rows carry a **range** rather than a value because the intraday low is unconfirmed —
at 25% the answer is uncertain by a factor of 1.75×. That is an estimate, not a backtest.

---

## 4. The headline year is not a finished year

$176,114 includes Sandisk up **566% through October 9, 2026** — a position still open.
That single incomplete year is roughly **6.7× of the total**; without it the figure is $26,428.

The thread's own footnote adds that **Moderna overtook Sandisk for the 2026 lead in October**,
so the rule cannot stably identify its own winner inside the measurement window.

---

## 5. It would have been unimplementable here

The 2025 pick was **Palantir**, which fails `portfolio_rules.md` §6 on defence and intelligence
contracting. One of the three years carrying the entire headline result is a prohibited name,
and a single-holding rule has no fallback when its pick is excluded.

This produced a standing rule: **any mechanical rule must define its substitute before it runs.**

---

## 6. Concentration of the result

Reading Exhibit 1: end-2023 ≈ $4,400 → $26,428 at end-2025 → $176,114 at Oct 2026.
**Three consecutive years — 2024 (NVDA +171%), 2025 (PLTR +135%), 2026 (SNDK +566% YTD) —
produce roughly 40× of the 1,761× total.** Those three years are one thematic bull market in
AI and semiconductors. That is a regime, not a rule.

---

## What survives

Stripped of presentation, this is **cross-sectional momentum** — buying recent winners — in its
most extreme possible form: one stock, full capital, annual rebalance.

Momentum is among the better-documented anomalies in finance. The thread has not discovered it;
it has levered it to the maximum. **Momentum supplies whatever edge exists. Concentration
supplies the lottery ticket, the −97% drawdown, and the twelve consecutive years of trailing the
index from 2008 to 2019.**

Two caveats carried into the study design:

- The predecessor experiment tested 12-1 momentum and 52-week-high proximity among 15 signals;
  none cleared its bar in both train and test. **That was a small-cap universe.** This is
  large-cap. The prior null does not automatically transfer, which is why the question stays open.
- **Momentum crashes.** Its failure mode is rare, violent, and clustered in sharp reversals.
  The −97% and the 23-year recovery are that failure mode, and it is poorly matched to a
  horizon measured in years with a possible project withdrawal.

**This audit motivates Study 01, which asks the one question the thread does not: is the edge in
the momentum, or in the concentration?**
