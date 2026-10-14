# Study 01 — Stage 1b: the price panel, and the coverage gate

**Committed before any return was computed**, per the pre-registration's staging gate.
Raw output: `research/output/coverage.csv`, `research/output/ticker_map_applied.csv`.

---

## Headline

**The price panel cannot be built from the sources reachable here, and Study 01 is
reported INCONCLUSIVE.** Coverage of expected member-years is **0.5%** as measured, against
a pre-registered bar of 90%.

That 0.5% is a *floor* from a sweep still rate-limited at 50 requests an hour, and it is
not the reason for the verdict. **The reason is structural and does not depend on how
long the sweep runs:** the source permits 500 unique symbols per calendar month, the
universe is 754, and the month's allowance was already spent before this session began.
754 cannot be fetched from a 500-symbol allowance, so no amount of patience inside this
month reaches 90%. The measured figure is reported because the methodology rules require
coverage to be measured and committed rather than asserted — not because the verdict
turns on it.

This is not a result about the strategy. It is a result about the data, and the
pre-registration is explicit about which one this is:

> If coverage of delisted names falls below 90% of expected member-years, the study is
> reported as inconclusive rather than published with a number.

The gate is now enforced in code, not in prose: `research/run_study.py` reads the
committed coverage report and refuses to write any result file below the bar. A rule
that lives only in a document is a rule a later run forgets at exactly the moment it is
inconvenient.

**Inconclusive is a third outcome, and it is worth being precise about it.** Stage 1
recorded that a failure in 2012–2025 would be conclusive and a pass would be weak
evidence. Inconclusive is neither. Nothing below licenses any statement about whether
buying last year's winner works.

---

## What was built, and what it does

Stage 1 left two items outstanding. Both are done and tested.

### 1. The ticker reuse/rename map (`research/ticker_map.py`)

Stage 1 found that a price series keyed on the bare ticker string silently splices
unrelated companies together. The two failure modes are **not symmetric**, and the map
treats them differently because the cost of getting each one wrong is different.

**Reuse — one ticker, two companies — is the dangerous direction.** Unmapped, it
manufactures a return out of a discontinuity, and the output is a plausible number
rather than an error. All seven cases Stage 1 named are recorded with the **effective
date** of the event rather than the year:

| Ticker | First company | Ends | Second company | Resumes |
|---|---|---|---|---|
| DELL | Dell Inc (Dell Computer) | 2013-10-29 | Dell Technologies | 2018-12-28 |
| DOW | The Dow Chemical Company | 2017-08-31 | Dow Inc | 2019-04-02 |
| FOXA | Twenty-First Century Fox | 2019-03-19 | Fox Corporation | 2019-03-19 |
| CEG | Constellation Energy Group | 2012-03-12 | Constellation Energy Corp | 2022-02-02 |
| SNDK | SanDisk Corporation | 2016-05-12 | Sandisk Corporation | 2025-02-21 |
| LIN | LIN TV / LIN Media | 2014-12-19 | Linde plc | 2018-10-31 |
| GM | General Motors Corporation | 2009-06-01 | General Motors Company | 2010-11-18 |

Dates, not years, because **FOXA's two companies meet inside 2019** — a year-granularity
map would have missed the one case that falls in the TEST half. The panel builder blanks
a reused ticker outside its member company's era, and blanks the whole year when two eras
meet inside it. A gap year (DELL 2015, DOW 2018, LIN 2016, SNDK 2017) is blanked too.

GM's first era ends before the 2011 window, so that row is inert here. It is recorded
anyway, because "we checked and it does not bite" and "we did not check" are different
states and only one of them is auditable.

**Rename — one company, two tickers — fails safe.** Unmapped, the old ticker finds no
series, the name is counted MISSING, and the coverage gate sees it. So the rename list
does not have to be exhaustive to be correct: an omission costs coverage, not accuracy.
Seventeen are mapped (FB→META, PCLN→BKNG, UTX→RTX, CBS/VIAC→PARA, SYMC/NLOK→GEN,
CTL→LUMN, ANTM→ELV, FISV→FI, ABC→COR, PKI→RVTY, HFC→DINO, WLTW→WTW, BBT→TFC, COG→CTRA,
TSO→ANDV).

**RTN is in the map specifically so that its absence is deliberate.** Raytheon Company
was *absorbed* in the 2020 merger, not renamed: RTX's price series is United
Technologies' history. Mapping RTN onto RTX would splice two companies exactly as the
reuse cases do, so RTN points at a sentinel with no file and resolves to nothing.

### 2. Partial formation years (`research/winners.py`)

The engine ranked any name with two prices in the formation year — including a company
that listed in July. The pre-registration already forbade this ("a company added mid-year
has no full prior-year return and must be excluded from that formation year, not scored
on a stub"), and the thread audit records it as the Sandisk error: +566% through October,
scored as though it were a year.

`_calendar_return` now returns NaN unless the series spans the formation year, with five
sessions of slack at each end measured on the panel's own trading calendar. The guard
applies to **formation only** — a holding acquired in June is correctly worth its price
at the acquisition, and forcing it to a full year would be the opposite error.

This narrows the universe and can only lower a result. The 22 original synthetic tests
pass untouched; five more cover the guard, including the case that matters — a stub
tripling in half a year is the largest "return" in the test panel and must stay invisible
to the ranking.

### 3. The §6 exclusion list (`research/exclusions.py`)

Rule 6 requires that "any mechanical rule must define its substitute before it is run",
and its origin note is this study. The list is therefore committed before the backtest.

Only the **industry-screenable** part is encoded: 20 defence and weapons names. Prisons
and payday lending are empty because no operator was an S&P 500 member inside the window
— a checked finding, not an omission.

**Israeli affiliation is not encoded and not approximated.** Rule 6 says it cannot be
screened by industry and must be checked by hand, and a list invented here would appear
in the output indistinguishable from one the owner had actually checked. The same applies
to Security & Protection Services and Credit Services, which rule 6 never auto-clears.
`COMPLETE` is `False`, and every exclusion-applied row in Stage 2 carries that flag.

**This is outstanding work for the owner, and it is not optional.** The pre-registration
reports results both with and without the list; until the hand check is done, only the
"without" arm is meaningful.

---

## Why the panel could not be built

### Yahoo stays rejected, on measurement

Of the 755 tickers that were ever members in the window, roughly a third return no
history at all — overwhelmingly the acquired and bankrupt names. That is the survivorship
bias the study exists to avoid, and it biases in the **flattering** direction: a
prior-year winner that later collapsed and delisted vanishes from the panel instead of
recording its loss. Yahoo also serves broken history for some *live* members.

Yahoo's ~67% coverage would fail this gate regardless. The gate protects the study from
both sources, which is the point of having one.

### Tiingo's free tier, measured today

Tiingo retains delisted tickers, which is why it was chosen. Two of its free-tier
limits were hit against this key on 2026-10-14, and both are quoted here from the
responses themselves rather than from the vendor's documentation:

| Limit | Value | How it is served |
|---|---|---|
| Requests per hour | **50** | HTTP 429, "over your hourly request allocation" |
| **Unique symbols per month** | **500** | **HTTP 200 with a JSON object**, not an array |

A daily cap is also documented by the vendor; it was never reached here, so no figure for
it is claimed. The monthly symbol cap is the binding one. **Even with unlimited
time, the free tier cannot cover this study's universe in a single calendar month**: 500
symbols is two thirds of what the study needs.

The arithmetic of the universe, for the record: **755 distinct membership tickers** in
formation years 2011–2024; **754 symbols to fetch** once the rename map collapses old
names onto current ones, RTN's sentinel is dropped, and SPY is added.

The monthly allowance was already spent when this session began: the first probe of the
session succeeded and the next was refused, so the counter stood at its limit within one
request. Symbols already looked up earlier in the month still serve and cost nothing,
which is why a handful of names are present.

**The 200-with-an-object is the dangerous part**, and it is worth recording why. The
obvious handling — treat an unparseable response as "no history" — would have recorded an
API limit as a fact about the market. The universe would have quietly shrunk to whatever
the sweep reached first, the coverage report would have shown a plausible number, and the
names dropped would have been determined by alphabetical order. Quota refusals are now a
distinct outcome, logged separately, and counted MISSING against the gate. The sweep also
runs in a fixed-seed random order, so a partial run samples the universe rather than the
start of the alphabet.

### Other sources

Stooq serves free daily CSV for US tickers and retains many delisted names, so it was
tested. It resets the connection on most requests from this container and returns an
interstitial page rather than CSV on the rest. It is not a dependable bulk source for 754
symbols. The container's egress policy is open — every failure here is the vendor's rate
limiting, not a blocked host.

---

## Coverage, measured

The sweep is incomplete and still running; this is the state at the time of commit, and
the committed CSV is `research/output/coverage.csv`.

| Formation year | Members | Priced, full year | Coverage |
|---|---|---|---|
| 2011 | 506 | 3 | 0.6% |
| 2012 | 505 | 3 | 0.6% |
| 2013 | 503 | 3 | 0.6% |
| 2014 | 504 | 3 | 0.6% |
| 2015 | 507 | 3 | 0.6% |
| 2016 | 508 | 3 | 0.6% |
| 2017 | 507 | 3 | 0.6% |
| 2018–2024 | 504–506 | 2 | 0.4% |
| **Overall** | **7 076** | **35** | **0.5%** |

Four symbols were recoverable: AA, AAPL, AEP and AET — names already looked up earlier in
the month, which cost nothing against the allowance. Everything else is refused.

Two points about how this table is built, both of which exist to stop a number being
flattered:

- **"Priced" means the series spans the formation year**, not merely that a column
  exists. A name that listed or was acquired mid-year is not rankable on that year, so
  counting it would overstate what the study can actually use. `data.py.coverage_report`
  computes the looser version and is explicitly labelled not the gate.
- **Absences are attributed, not lumped.** A quota refusal, a genuine absence of history
  at the source, and an unexplained gap are three different facts. Collapsing them is how
  an API limit comes to read as a property of the market.

---

## The gate, honoured

Coverage is 0.5% against a 90% bar. **The study is reported INCONCLUSIVE. No return was
computed and no result file was written.**

The gate is enforced mechanically. `research/run_study.py` reads the committed coverage
report before doing anything else and exits 3 without writing. Both guards were exercised
against the real sparse panel, not only against synthetic data: the gate refused, and
behind it the benchmark check also refused, because SPY is absent too. A study with no
benchmark has nothing to compare against, and that now fails loudly rather than
producing a strategy curve with nothing beside it.

What the pre-registration asked for in Stage 2 — the full N curve, the stop surface, both
exclusion arms, years-ahead as a proportion with an exact binomial p, and the
configuration count beside the results — is built, tested and unrun. The machinery is
verified on 83 synthetic checks with the answers worked out by hand, and on the four real
series that do exist end to end. **The only missing input is the data.**

A note on what the configuration count would have been, since it is the kind of number
that is easy to quote wrongly later: the search space is **96** — six values of N, eight
stop levels including no-stop, two exclusion arms — evaluated 192 times across two
periods. 96 is the multiple-comparisons burden; 192 is not, because evaluating one
configuration on both halves is a single look. A nominal p of 0.05 among 96 looks needs
roughly p < 0.0005 to carry the same meaning.

---

## What would change the answer

The study is not dead; it is waiting on one input. In rough order of cost:

1. **Wait for the monthly allowance to reset.** The 500-unique-symbol counter is monthly.
   Two clean months of 500 symbols each would cover 754, if the fetcher's resume log is
   preserved between them — note that `research/data.nosync/` is gitignored, so a fresh
   container starts from zero. That is the cheapest path and the slowest.
2. **An upgraded Tiingo key**, added as an environment secret under the same variable
   name `TIINGO_API_KEY`. A new session picks it up and `python3 -I
   research/fetch_prices_tiingo.py` completes in one run. Nothing else changes.
3. **Any source that retains delisted names** and allows 754 symbols. The panel builder
   takes per-ticker CSVs with `date,adjclose,low,close`; only `fetch_*.py` would change.

Everything downstream is built and tested and needs no further work:
`build_panel.py` → `coverage.csv` → `run_study.py` → Stage 2 CSVs. The gate in
`run_study.py` opens by itself once coverage clears 90%.

---

## What this does NOT mean

- **It is not evidence for the strategy.** No return was computed. The pre-registered
  decision rules in §4 remain unapplied.
- **It is not evidence against it.** A data failure is not a finding.
- **It does not retire the question.** The informative test — whether the edge survives
  diversification, measured on the N curve rather than on N=1 — is unchanged and still
  worth running.
- **It does not change the audit.** The thread audit stands on the thread's own published
  numbers and needed no data of ours.

---

## Status

- [x] Stage 0 — pre-registered and committed
- [x] Stage 1a — membership reconstructed; train period amended to 2012–2018 / 2019–2025
- [x] Stage 1b — ticker map, partial-year guard, exclusion list, panel builder, coverage
      report. **Coverage fails the 90% gate.**
- [ ] Stage 2 — blocked by the gate. `run_study.py` refuses to write results and exits 3.
- [ ] Stage 3 — not written. A verdict is written *against committed raw output*, and
      there is none. Writing one would imply Stage 2 had run.
