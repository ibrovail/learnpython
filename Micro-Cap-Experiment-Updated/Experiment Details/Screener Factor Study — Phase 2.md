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

*(Appended after the analysis runs.)*
