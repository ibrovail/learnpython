# Regime Profile Backtest — Phase 3.75

**Status: PRE-REGISTRATION ONLY. Nothing has been computed.** Parts 2 and 3 are deliberately
empty. This file must be committed before the study script is written or run, per
`.claude/rules/research-methods.md` rule 1. *(Phase 2 precedent: rules `6c28378`, raw results
`02ddb4d`.)*

**Why this exists.** The RISK-OFF screener allowance carries a sunset at Phase 4 (December 2026),
and the 0.90 calm threshold is recorded in `portfolio_rules.md` as "chosen by judgment, not by the
study." Waiting for Phase 4 means waiting for *forward* formation dates to accumulate. The signals
in question are **price- and volume-only**, so their history already exists.

**This is a survivorship-free study.** Earlier drafts of this file accepted a surviving-universe
design and built elaborate rules around the resulting bias. That compromise is gone — see §1.2. The
decision rules below are correspondingly simpler and stronger: a result can now be acted on in
either direction.

**What this is not.** It does not replace Phase 4. It cannot test a single fundamental gate.

---

## Part 1 — Pre-registration

### 1.1 The three questions

Each is tied to a rule that a result would change. Nothing else will be tested.

| # | Question | Rule it decides |
|---|---|---|
| **Q1** | Conditional on **RISK-OFF**, does the two-leg profile (`rank_low_vol` ≥ 0.90 ∧ `vol_5_50` > 1.0) plus the above-50-day trend rule beat the gate-survivor universe at the primary horizon? | The RISK-OFF screener allowance and its Phase 4 sunset |
| **Q2** | Is **0.90** the right calm threshold, or does a looser one do as well? Test 0.50 / 0.60 / 0.75 / 0.80 / 0.90. | The threshold itself — the lever that takes the book from 2–3 candidates a week to 10–11 |
| **Q3** | Was removing **`near_high`** right? Three-leg vs two-leg profile on identical dates. | Whether `near_high` returns, as a percentile rather than the absolute −5% |

### 1.2 The universe — point-in-time and survivorship-free

**Membership comes from a dated snapshot of Tiingo's `supported_tickers.csv`**, downloaded
2026-10-06, keyless, 16,538 US common stocks with a `startDate` **and** an `endDate` for every one.
A ticker is in the universe on formation date *d* iff `startDate ≤ d ≤ endDate`. That single rule
removes both biases at once:

- **Survivorship bias** — names that later delisted are present on the dates they traded.
- **Listing bias** — names that IPO'd mid-span are absent before they listed, rather than
  back-filled.

**Prices come from Tiingo's EOD API**, which serves delisted tickers. Key in `.env`
(gitignored; `TIINGO_API_KEY`), read only by `research/` scripts.

**Why this replaces the previous design.** Every other source tested returns nothing for a delisted
ticker, so a surviving-universe study was the only option until this file was found:

| Source | Delisted prices | Evidence |
|---|---|---|
| yfinance, and Yahoo's own `v8/finance/chart` | **None** | `"No data found, symbol may be delisted"` for TWTR, ATVI, VMW over dates they traded freely |
| Stooq bulk archive / CSV endpoint | **None** | HTTP 401; CSV endpoint behind a JS proof-of-work |
| `api.nasdaq.com` historical | **None** | `"Symbol not exists"` |
| Alpaca | **None** | Documented: no history for inactive assets |
| Polygon (free) | Yes, but **2-year** history cap | Free-tier limit |
| **Tiingo** | **Yes**, 50+ years depth | Chosen |

**The bias this removes is large, and now measured exactly rather than sampled.** Of 7,703 US
common stocks listed on 2021-10-01, **2,770 (36.0%) had delisted by 2026-09** — annualised **8.7%**.
Over the full ten-year span the 2016 cohort lost **46.8%**. A surviving-universe study would have
been blind to roughly half its own sample, and `research-methods.md` would not have been able to
say by how much.

| Cohort start | Listed then | Delisted by 2026-09 | Per year |
|---|---|---|---|
| 2016-10-01 | 5,949 | **46.8%** | 6.2% |
| 2019-10-01 | 6,021 | 34.9% | 6.0% |
| 2021-10-01 | 7,703 | 36.0% | **8.7%** |

**Reproducibility.** The ticker snapshot is a point-in-time artifact: re-downloading it later gives
a different file. Its download date, byte count and SHA-256 are recorded with the raw output, and
the snapshot itself is cached under `research/data/` (gitignored, rebuildable).

### 1.3 Sampling — the one real compromise, and it is noise rather than bias

Tiingo's free tier allows **500 unique symbols per month** (50 requests/hour, full history per
call). The universe is far larger, so the study runs on a **stratified random sample of 500
tickers**, drawn once and fixed before any result is computed.

**The frame, measured 2026-10-07.** US common stocks (NYSE / NASDAQ / AMEX / NYSE MKT, USD) whose
listed interval overlaps the span, after two exclusions found while building the download:
**9,934 names**, of which **37.9% delisted** before 2026-09. Strata: at-start/survived **2,889**,
at-start/delisted **1,953**, mid-span/survived **3,282**, mid-span/delisted **1,810**.

A proportional 500-name draw leaves **244–311 names listed per formation date (median 276)** and
**no date below `MIN_NAMES` = 100** before the liquidity gate — which would have to remove more
than 60% of names on a date to breach the floor.

**Exclusion 1 — non-common share classes, 28.4% of the raw frame.** Tiingo's `assetType == "Stock"`
includes **951 preferred shares, 1,840 warrants, 1,487 SPAC units and 94 rights**. Leaving them in
would have invalidated *this* study in particular: preferreds are bond-like and SPAC units sit near
$10 until their deal, so both are **calm by construction** and would have flooded
`rank_low_vol` ≥ 0.90 — the exact leg under test. The headline would have read "the calm decile
outperforms" when what it actually said was "the calm decile is full of preferred shares."
*Found because the download's first four tickers happened to include `ABLLW` (a warrant) and
`ABR-P-E` (Arbor Realty preferred series E).* The filter is anchored on Nasdaq's fifth-letter
convention — a **five-character** ticker ending W/U/R is a warrant, unit or right, while a 3–4
character one usually is not — and was validated on 24 known ordinary commons (none excluded,
including `LOW`, `FLOW`, `PLOW`, `TWOU`, `U`) and 10 known non-commons (all excluded). Dual-class
common (`BRK-B`, `HEI-A`, `LEN-B`) is common equity and is kept.

**Exclusion 2 — recycled symbols, 695 of them (4.5%).** The same ticker names different companies
in different eras: `AAC` is three of them. Tiingo's price endpoint is keyed by symbol alone, so a
recycled symbol's history cannot be attributed to the right company, and joining it to one listing
interval would splice two businesses into a single series. **10.8% of the first drawn sample were
recycled symbols**, so this was not a rounding error. They are dropped rather than guessed at.

Both exclusions are **administrative rather than performance-related** — share class and symbol
reuse are facts about listing mechanics, not about a stock's returns — so removing them does not
select on the outcome. They also move the frame closer to the live screener's universe, which holds
common equity only.

**Stratification**, so the sample is not quietly a survivor sample again:

- Strata are **(delisting status × listing era)**: names that survived the span vs names that
  delisted during it, crossed with whether they were listed at the span start or IPO'd mid-span.
- Each stratum is sampled **in proportion to its true size** in the point-in-time universe. The
  delisted share of the sample must therefore match the ~36% measured in §1.2, not the 0% a
  surviving universe would give.
- The draw uses a **fixed seed, recorded in this file before the run: `seed = 20261007`.**

**Why 500 names is enough, stated as a claim that can be checked.** The test statistic is computed
across **formation dates**, not across names. Per-date cross-sectional noise averages out over
~180 RISK-OFF dates; sampling 500 of ~7,000 names widens the per-date spread's standard error by
roughly √14 ≈ 3.7×, but divides into the across-date t-statistic by √180. **Sample the
cross-section, not the time series.** The pre-committed check on this reasoning: §1.6 metric 9
reports the per-date spread's own standard error, and if the median per-date standard error exceeds
the full-sample spread being measured, the sample is declared too small and the study stops rather
than reporting a result.

**Floor:** any formation date with fewer than **`MIN_NAMES` = 100** usable sampled names yields no
observation. With 500 names and staggered listing dates, early dates are the ones at risk; the
count per date is reported (§1.6 metric 9).

**If the sample proves too thin**, the options in order are: a second month's free allowance for
another 500 names (the strata and seed extend, they are not redrawn), or EODHD "HISTORIAN"
($19.99, one month, 100k calls/day) for the full universe. Neither is assumed here.

### 1.4 Span and formation dates

**Ten years.** The previous draft chose five to limit attrition. With delisted prices in hand
attrition is no longer a cost, so span buys sample for free:

| Span | RISK-OFF weekly dates | Effective n @ 20s | @ 40s | @ 60s |
|---|---|---|---|---|
| 5 years | ~104 | 26 | 13 | 9 — **below floor** |
| **10 years** | **~182** | **46** | **23** | **15** |

Ten years is the first span at which **all three decision horizons clear the effective-n floor**,
which makes the 40- and 60-session results decidable rather than descriptive.

| | |
|---|---|
| Price history | 2015-10-01 → 2026-09-18 (the extra year pre-loads the 60-day and 50-day windows) |
| Formation dates | **Weekly, Fridays**, 2016-01-01 → the last date with a full 60-session forward window |
| Expected dates | ~520 total, **~182 RISK-OFF** (35% long-run share, measured from IWM 2016–2026) |
| Regime | IWM close vs its 50-session SMA with the **±1% band** in force since 2026-09-19 — the same path-dependent rule the live system uses, recomputed over the whole span |

The regime is the one input that is **exactly** reconstructable: pure IWM arithmetic, verified
against `regime_history.csv`. RISK-OFF share by year, measured: 2016 14%, 2017 22%, 2018 45%,
2019 35%, 2020 25%, 2021 38%, **2022 67%**, 2023 47%, 2024 22%, 2025 38%, 2026 26%.

### 1.5 Signals — price and volume only

Reconstructed from **split- and dividend-adjusted** daily OHLCV, by the same formulas as
`screener.py`:

> **Corrected before the run, 2026-10-07.** This section first said *unadjusted*, to match
> `screener.py`. That is right for the live screener, which looks at a 60-day window on current
> data where splits are rare, and **wrong for a ten-year study**. In an unadjusted series a 2:1
> split is a −50% daily return, which corrupts `low_vol` (the standard deviation of returns) and
> `near_high` (distance from the 60-day high) for every name that ever split, and corrupts forward
> returns outright. Adjusted prices are used for both signals and returns; the unadjusted series is
> stored alongside but not used. Found while writing the download stage, before any result existed.


| Signal | Definition |
|---|---|
| `low_vol` | −(20-day standard deviation of daily returns) × 100 |
| `near_high` | (close ÷ 60-day high − 1) × 100 |
| `vol_5_50` | 5-day mean volume ÷ 50-day mean volume |
| `vol_ratio` | last volume ÷ 50-day mean volume |
| `pct_vs_sma50` / `above_sma50` | distance from the 50-day SMA; above it or not |
| `bb_width` | Bollinger width (the "squeeze" input) |

Percentile ranks (`rank_*`) are computed **among that date's gate survivors**, matching
`screener.py` exactly — verified: the stored `rank_low_vol` reproduces as
`low_vol.rank(pct=True)` to 0.0000.

**Price-computable gates re-applied at each date:** deal-pinned (ATR < 0.75% of price), >40% above
the 50-day SMA, >20% above the 20-day SMA, days 1–3 of a >10% breakout, and a **liquidity floor of
$1M median 20-day dollar volume** (the live screener's own gate, and computable from price × volume).

**Gates that cannot be applied, and are therefore absent:** shrinking revenue (Sales Q/Q), negative
TTM EPS, forward P/E, analyst rating and target, prohibited business, and **the $5Bn market-cap
ceiling** — Tiingo's free tier carries no shares outstanding, so market cap is not reconstructable.
**Median dollar volume is used as the size/liquidity proxy** wherever §1.6 calls for size controls;
it is explicitly a liquidity measure, not market cap, and is labelled as such in the output. Their
absence means **this study tests the price profile, not the strategy.**

### 1.6 Horizons, lags and effective sample size

| Horizon (sessions) | Role |
|---|---|
| **20** | **PRIMARY** — Phase 3.5 set it as Phase 4's primary horizon |
| 40, 60 | Secondary; decidable at this span (effective n 23 and 15) |
| 5, 10 | Continuity with Phase 2; cannot trigger a decision |

**Effective n = dates ÷ (h/5)**, since weekly dates with an h-session forward window overlap
roughly h/5 deep. **Effective n is reported beside every t-statistic.** No exceptions.

**Newey-West lags: `min(ceil(h/5), floor(n_dates / 5))`.** The cap is the Phase 3.5 lesson — it
pre-registered `ceil(h/5)` unconditionally and put **12 lags on 10 observations** at 60 sessions,
producing t-statistics up to 15.3. A lag length approaching n is not conservative, it is undefined.
**When the cap binds, the output must say so.**

**Decision floor: effective n ≥ 10.** Below that a horizon is descriptive only.

**Abort condition.** Any |t| > 5 is treated as evidence of a broken variance estimate, not a strong
signal. The run stops, the cause is found, and the finding is reported as a defect.

### 1.7 Metrics — every one required, none optional

Per `.claude/rules/research-methods.md`:

1. **Rank IC** (Spearman) per date, with the NW t-statistic and effective n.
2. **Quintile spread on MEAN returns** — never rank IC alone. In a universe whose median stock
   falls, rank IC rewards calm stocks sitting near zero.
3. **IC on winsorized (1%/99%) returns**, beside the raw IC.
4. **Group vs universe, like for like: mean-vs-mean AND median-vs-median, reported separately.**
   Never a group mean against a universe median. *(That error cost about half the claimed edge in
   the first Phase 3.5 write-up.)*
5. **Pairwise mean cross-sectional rank correlation** of every signal pair; ρ ≥ 0.7 is flagged as
   one exposure counted twice.
6. **Partial IC controlling for `low_vol`** — the test that found the squeeze had no information of
   its own (partial IC −0.001).
7. **Partial IC on log median dollar volume**, plus group results **within dollar-volume terciles**
   (the market-cap substitute of §1.5). Low volatility and size move together; a volatility tilt can
   quietly become a size tilt.
8. **Split by the direction of the universe's median return** that week, and **separately by
   regime**. Phase 2 found the defensive signals' skill concentrated in falling-median weeks —
   and "falling-median week" is *not* the same measurement as "RISK-OFF week". Reporting both is
   the point of this study.
9. **Per-date diagnostics:** usable sampled names, dates dropped by `MIN_NAMES`, the delisted share
   of usable names, and **the per-date spread's standard error** (the §1.3 adequacy check).
10. **Delisted-name returns are included to the delisting date and not beyond.** A name that
    delists mid-forward-window contributes its realised partial return, and the count of such
    truncated windows is reported per horizon. This is the mechanism by which survivorship bias is
    actually removed, so it is reported rather than assumed.

### 1.8 Decision rules — applied mechanically, written before any result

**Q1 — the RISK-OFF allowance. Now symmetric, because the bias that broke the symmetry is gone.**
- **R1.** If, conditional on RISK-OFF at 20 sessions, the two-leg-plus-trend group beats the gate
  survivors on **both** mean-vs-mean and median-vs-median with NW **t ≥ 2.0**: the allowance is
  **confirmed** and survives Phase 4 on this evidence.
- **R2.** If the group fails to beat the survivors on either measure, or t < 2.0: the RISK-OFF
  screener allowance **reverts to a freeze** at Phase 4, which is the default the sunset clause
  already specifies.

**Q2 — the calm threshold.** Three design choices, each with its reason.

*Choice 1 — the comparison is **paired**.* Every threshold is evaluated on the **same formation
dates and the same names**, and the test statistic is the NW t on the *difference* series. Paired
differences remove the common market component and have far lower variance.

*Choice 2 — **median-vs-median is primary**, mean-vs-mean corroborating.* Small-cap forward returns
are right-skewed, so the mean is driven by outliers the book will not own: it holds **4–6
positions**. The typical outcome is the decision-relevant one.

*Choice 3 — the **burden of proof sits on keeping 0.90**.* `portfolio_rules.md` records that 0.90
was "chosen by judgment, not by the study", while the cost of keeping it is measured and ongoing —
2–3 candidates a week against 10–11, with the book at 85% cash.

- **R3.** Adopt the **loosest** threshold (0.50 / 0.60 / 0.75 / 0.80 / 0.90) whose paired difference
  against 0.90 at 20 sessions is **both** (a) not significantly negative — NW t > −2.0 — **and**
  (b) not worse by more than **0.5pp** on median-vs-median. Report the full threshold curve.
- **R4.** Keep **0.90** only where its advantage over the adopted threshold is **both** significant
  (paired NW t ≥ 2.0) **and** material (**≥ 0.5pp** median-vs-median).
- **R5.** If median-vs-median and mean-vs-mean **disagree in sign**, keep **0.90**.
- **R5b — stability, and it can veto.** The span is split at its midpoint (≈2021-06) and the
  threshold chosen under R3 must **win or tie in both halves** on median-vs-median. With 2018 at 45%
  and 2022 at 67% RISK-OFF against 2016–17 at 14–22%, the halves are genuinely different regimes.
  If it fails, keep 0.90 and report the instability as the finding.

*Why 0.5pp is the materiality floor.* The observed scale of edge in this universe is the
top-50-vs-survivors spread, **+1.29pp** median-vs-median at 20 sessions (`entry-discipline.md`).
0.5pp is ~40% of that — large enough that a real difference clears it, small enough not to demand an
implausible effect. 1.0pp would be nearly the whole observed edge and make R4 unsatisfiable; 0.2pp
sits inside the noise.

**Q3 — `near_high`.**
- **R6.** If the three-leg profile beats the two-leg profile by more than **0.5pp** on both measures
  at 20 sessions in RISK-OFF, `near_high` is reinstated **as `rank_near_high` ≥ 0.90**, never as the
  absolute −5%. The three structural faults recorded on 2026-10-05 — anti-correlation with the
  regime, an absolute cut among relative ones, and double-counting a composite input — are not
  repaired by a favourable return result.
- **R7.** Otherwise the removal stands.

**Universal.**
- **R8.** Any horizon whose effective n < 10 is descriptive and decides nothing.
- **R9.** Every rule is evaluated at the **primary 20-session horizon**. Agreement at 40 and 60 is
  corroboration; disagreement does not override R1–R7 but must be stated prominently.
- **R10.** Raw output is committed **before** interpretation; every analysis not listed in this Part
  is labelled **post hoc**.
- **R11.** If the sample-adequacy check in §1.3 fails — median per-date standard error exceeding the
  spread being measured — **the study stops and reports that**, rather than reporting a result at
  the measured precision.
- **R12 — do the gates select on survival?** *(Restated 2026-10-07, before any result. See the
  note below: this rule was wrong three times, always the same way.)*
  - **R12a.** For **every** gate, the share of rows it rejects must differ by **no more than 10pp**
    between delisted and survived names. A gate that rejects delisted names disproportionately
    reintroduces exactly the survivorship bias this design exists to remove, and would do so
    invisibly. *Measured on the first 100 downloaded names: the liquidity gate rejects **24.5%** of
    delisted rows against **26.1%** of survived rows — a −1.6pp differential, so it does not select
    on survival.*
  - **R12b.** The **name-weighted** delisted share of usable names must be materially above zero
    (**≥ 15%**). *Measured: 26.5% against the frame's 37.9%.* Near zero would mean the
    point-in-time join had silently failed.
  - **Row-weighted shares are not comparable to name-weighted ones and must not be checked against
    each other.** A delisted name is listed for less of the span by construction, so it appears on
    fewer formation dates: survived names contribute a median of **421** formation-date rows,
    delisted names **133**. The row-weighted share is therefore *correctly* lower — 13.9% against a
    31.9% name-weighted share — and that gap is arithmetic, not bias.

> **Three wrong versions of R12, all the same error.** It first read "~36%", the *five-year* cohort
> attrition, against a *ten-year* frame. Corrected to 47.7%, it was then measured against a frame
> that still contained preferred shares and recycled symbols; the clean frame is 37.9%. Corrected
> again, it compared a **row-weighted** panel share against a **name-weighted** frame share.
>
> Every version compared two numbers measured on different bases — which is precisely what
> `.claude/rules/research-methods.md` forbids for means and medians, in a guise the rule did not
> name. The lesson is generalised there now. It is also the reason this rule is stated as a
> **differential between two groups under the same gate** rather than as a level against a
> reference: a differential cannot be wrong about its own units.

### 1.9 What this study cannot answer

Stated now so no later reader mistakes its scope:

- **Any fundamental gate** — revenue growth, TTM EPS, forward P/E, analyst rating or target. Point-in-time
  fundamentals are not available here. **They remain Phase 4 questions.**
- **The $5Bn market-cap band.** No shares outstanding on this tier; dollar volume substitutes as a
  liquidity proxy and is labelled as one.
- **The research layer.** Stage-2 judgment, thesis quality, driver freshness and conviction are
  human decisions, not reconstructable signals.
- **The thin trend-gate question** from the HOPE post-mortem. It needs `research_log.csv`'s
  `pct_vs_sma50` against realised outcomes, which accumulates **forward only**.
- **The catalyst lane** (added 2026-10-05 b). No historical dataset of dated, non-binary catalysts
  exists here, so the lane cannot be backtested at all.
- **Precision beyond a 500-name sample.** The design trades cross-sectional precision for the
  removal of a 36% survivorship hole. That is the right trade — noise shrinks with more dates,
  bias does not shrink with anything — but it is a trade, and §1.3 and R11 are how it is policed.

### 1.10 Deliverables

- `research/regime_backtest.py` — written only after this file is committed.
- `research/data/` — the dated ticker snapshot and the price cache (gitignored, rebuildable).
- `research/output/regime_backtest_*.csv` — raw output, committed before interpretation.
- **Part 2** of this file: results, applying §1.8 mechanically.
- **Part 3**: post-hoc observations, explicitly labelled.

---

## Part 2 — Results

*Empty. To be filled only after Part 1 is committed and the script has been run.*

---

## Part 3 — Post-hoc observations

*Empty. Anything not pre-registered in Part 1 belongs here, labelled as post hoc.*
