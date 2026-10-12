# Week 58 — Catalyst sourcing audit (2026-10-12)

**Not a report.** This is the sourcing input for the Week 58 weekend run, written on Monday
2026-10-12 because the rules session asked for *what was searched*, not only what was found.
Filename deliberately avoids `Week N Full.md` / `Week N Monitor.md` so the 30-session backstop
(`trading_script.py:1692`) does not count it.

Prices are the **2026-10-09 settled close** unless marked. Nothing here is a recommendation: the
market was open while this was written, no daily had run, and every name below is already dead.

---

## Why this exists

Week 57 logged 5 catalyst rows across **3 of the 6 qualifying categories** in
`portfolio_rules.md` → *Catalyst plays* → *What qualifies*:

| Category | Week 57 rows |
|---|---|
| index inclusion announced, effective date ahead | 2 (WK, FRSH) |
| guidance already raised, next print dated | 2 (ZIM, NNBR) |
| spin-off / separation with announced date | 1 (TORO) |
| **contract or backlog awarded, dated conversion** | **0 — no record of a search** |
| **cost or margin inflection already underway** | **0 — no record of a search** |
| **refinancing or maturity resolved on a stated date** | **0 — no record of a search** |

The rules session asked whether the empty weeks mean (a) the categories are too narrow, (b) the
search did not reach the places catalysts are published, or (c) nothing qualified.

**Answer: (b) for three categories, (c) for two, and a size constraint nobody had named.**
The three unsearched categories produced **4 of the 6 names** found today.

---

## Surfaces searched, 2026-10-12

### 1. Index inclusion — GENUINELY EMPTY (c), verified at source

Checked `press.spglobal.com` release archive directly in the browser, not via search summary.
Every S&P DJI release since 2026-09-28:

| Release | Effective | SmallCap 600 change |
|---|---|---|
| Oct 1 | Oct 1 / **Oct 6** | WK replaces FORM; **OLN** replaces QRVO |
| Oct 5 | **Oct 8** | FRSH replaces BLFS |

All effective dates have passed. An index add counts only **before** its effective date, so the
surface is spent. FTSE Russell reconstitution is late June — nothing live. The December quarterly
announcement lands **~Dec 4** (two weeks before the third Friday, Dec 18).

**Process finding — this category cannot be sourced on a weekend cadence.** Both October adds were
**M&A-driven intra-quarter replacements**, announced with 3–5 days' notice (Oct 1 → effective Oct 6;
Oct 5 → effective Oct 8). A Saturday check sees only adds already past their effective date, which
is exactly how Week 57 found WK and FRSH and killed both on timing. The surface is productive; the
cadence is wrong. *Also noted: `OLN` was a **demotion** — deleted from MidCap 400 the same morning
it joined SmallCap 600. The larger index's selling dominates; a demotion is not the same event as
a promotion and should not be treated as one.*

### 2. Contract / backlog awarded with dated conversion — NOT searched in W57, PRODUCES NAMES (b)

| Name | Catalyst (dated) | Verdict |
|---|---|---|
| **AISP** Airship AI, $83M | $29.0M firm-fixed-price DHS awards (Oct 1); backlog record $30.6M, "all installations scheduled for execution over the next two fiscal quarters" (Oct 8) | **KILL — extended entry** |
| **VAL** Valaris | $220M of awards (Oct 7); Suriname start Q4'26, North Sea extension Feb 2027 | **KILL — out of universe** |
| **BORR** Borr Drilling, $1.56Bn | multi-year jackup awards into 2026–27 (Oct 9) | **KILL — negative earnings** |
| Flint | $451M awards/renewals | **KILL — TSX-listed, not a permitted exchange** |

- **AISP** passes more gates than anything else found: revenue TTM **$18.15M +32.2%**, beta 0.63,
  next print **Nov 16** (~24 sessions, clears the 10-session guard), ATR 9.12% of price (not
  deal-pinned), **+19.78% above its 50-day SMA**. It dies on **distance-from-base**:
  **+21.11% above its 20-day SMA** against a ≤20% limit, and it closed at a **new 20-day high after
  +14.15% in five sessions** — days 1–3 of a >+10% breakout, which `entry-discipline.md` says to
  avoid. Two independent legs of the same rule, failed by 1.11pp and outright.
  *Also: TTM net income reads $26.24M on $18.15M of revenue — net income above revenue, which for a
  de-SPAC is a warrant/derivative fair-value gain, not operating profit. "Credible earnings path"
  cannot be assessed from the headline; Q2 was $4.12M revenue at 75% gross margin.*
- **VAL** is decisive on size: **market cap $5.81Bn** against `MAX_MARKET_CAP = 5e9`. Independently
  it would also fail on **TTM revenue −13.2%** and an analyst PT of **$68.29, −18.4% below market**.
- **BORR** TTM net income **−$243.6M**, EPS −$0.83, PT $4.97 (−1.5%, below market).

### 3. Guidance already raised — covered in W57, no new names (c)

Re-searched; the same two names surface (ZIM prohibited, NNBR negative earnings −$47.92M). The
surface is genuinely thin, not unsearched.

### 4. Cost or margin inflection — NOT searched in W57, PRODUCED THE BEST NAME (b)

**TREX** Trex Company, **$4.56Bn — in universe**, revenue TTM **$1.21Bn +7.0%**, net income
**+$177.35M**, forward P/E **23.25 vs 26.57 trailing** (earnings expected up), analysts **Buy**,
PT **$55.26 = +23.5%**, next print **Nov 4**. Catalyst: FY26 guidance raised **2026-07-13**
(net sales $1.215–1.250Bn, adj. EBITDA $335–350M), reaffirmed in the Q2 release, adjusted gross
margin lifted to ~38% on **Little Rock starting production in Q3 with the ramp pulled forward six
months**. On the written test this is a resolved fact — the plant is running and the guide is out.

**KILL, two ways:**
1. **Below the 50-day SMA: −1.77%.** The hard trend gate requires ≥0%. This alone is a stage-1 kill
   (`below-sma50`).
2. The margin accretion management actually points to is **2027** ("half the lines operational by
   year-end to support 2027 margin accretion") — **outside the 90-day catalyst window**. The
   July raise is a three-month-old fact whose forward impact is dated beyond the horizon the rule
   allows.

> **Corrected 2026-10-12, after the rules session checked it.** This section originally claimed a
> third kill — that TREX "cannot be sized inside the 30% ceiling" — and drew a structural finding
> from it. **Both were wrong**, and the rules session was right to reject them (`ab22c01`,
> `ce87857`).
>
> - **Nothing is excluded.** `portfolio_rules.md` → *Position sizing* says actual size is the
>   **minimum** of the risk-based figure, the 30% ceiling and available cash. So a tight-stop name
>   is buildable at any stop width; the ceiling simply binds instead of the risk budget and the
>   position carries **less** than 2% risk. TREX would have been a 30%-of-equity position at
>   $216 carrying **~1.8% risk**, not an impossible one.
> - **It is not set by book size.** The 6.67% figure is `risk_per_trade ÷ name_cap` = 2% ÷ 30%, a
>   **ratio of two rule constants**. Both scale with equity, so equity cancels: it is identical at
>   $721 and at $100,000. Computing it as "$14.42 ÷ $216.30" made a scale-invariant ratio look
>   like a small-book constraint.
>
> **TREX's no-buy stands on kill #1,** which is a hard gate and was found independently.

**What the arithmetic actually says, and it is the opposite shape.** The 6.67% threshold is real,
but it is the point where the **ceiling takes over from the risk budget** — not an exclusion. At a
1.75×ATR stop that corresponds to **ATR ≥ 3.81% of price** to deploy the full 2%. Measured on this
week's own lists:

| | n | median ATR | below 3.81% | can carry full 2% risk |
|---|---|---|---|---|
| watchlist top 50 | 50 | 2.71% | **90%** | 10% |
| extended, ranks 51–100 | 50 | 3.01% | 82% | 18% |
| **combined** | 100 | 2.83% | **86%** | 14% |

So for **86% of live candidates the ceiling binds, not the risk budget**, and the median name at a
1.75×ATR stop carries **1.49% risk — 74% of the 2% budget.** The exposure is therefore *under*-use
of the risk budget, not exclusion of names. The rules session traced the cause to the **selector**:
the backtest ranks by `pct_vs_sma50`, which picks extended and so more volatile names (only 5% of
its selections fall below the threshold), while the watchlist is composite-ranked and the composite
is **built** to favour low volatility. Their separate finding is that composite ranking tests
**4.9pp worse** than `pct_vs_sma50` on the test period. Two results converge on the same selector —
weigh it when choosing which stage-1 names to promote.

### 5. Spin-off / separation with announced date — EMPTY for a qualifying name (c)

Nothing small-cap with a set completion date. Resideo/ADI is mid-cap; Archer SpinCo (Flex) has a
**blank record date** in its Form DRS; Vylor/Corteva completed Oct 1. **Modine/Gentherm** is the
instructive one: a Reverse Morris Trust slated to close by end-CY2026 **but conditioned on Gentherm
shareholder approval and an IRS determination letter** — an unresolved event with a dated
announcement, i.e. the prohibited binary case. Correctly killed by the "if it succeeds" test.

### 6. Refinancing / maturity resolved on a stated date — NOT searched in W57, thin (b)

| Name | Catalyst | Verdict |
|---|---|---|
| **FIRY** Firy Inc. (formerly Skillz), $201M | notice delivered to redeem remaining **$49.7M of 10.25% notes, redemption date Oct 15**, completing full repayment of $300M; ~$0.8M cash interest saved ($3.6M with the August redemption) | **KILL — negative earnings** |
| RC Ready Capital | retiring $350M 4.50% notes with a smaller, **higher-coupon** borrowing | **KILL — raises interest cost; a negative catalyst** |
| OpenText, BXP | completed refinancings | **KILL — out of universe on size** |

**FIRY** TTM net income **−$70.76M**, EPS **−$4.53**, losses widened 50.5% in 2025, **beta 4.63**,
52-week range $2.23–$20.00. Revenue +31.3% is the only passing leg.

---

## Outcome

**Six of six categories searched. Six names produced. Zero survive stage 1.**

**The reason is not sourcing, and not the catalysts.** By binding reason code: two **prohibited**
(AISP, FIRY), one out of **universe** on size (VAL), one **loss-making** (BORR), one **below its
50-day SMA** (TREX), one **binary** (MOD). Not one died because its catalyst was weak or undated.

Underneath the codes, the **earnings-quality** observation still holds and is the one to carry
forward: of the six names, **three were loss-making** — BORR −$243.6M, FIRY −$70.8M, and NNBR
−$47.9M from Week 57. (FIRY's *binding* code is now `prohibited`, but it was loss-making
independently of that.) Dated-catalyst small caps skew heavily loss-making, and "growing TTM
revenue with a credible earnings path" is the gate that kills them. The one profitable, growing,
in-universe, Buy-rated name found — TREX — died on the **trend gate**.

So: the catalyst lane is not short of events. It is short of events attached to **profitable**
companies trading **above their 50-day SMA**. *(An earlier version of this paragraph added "and
whose stop is wide enough to size in a $721 book" — withdrawn; see the correction above. Stop
width excludes nothing.)*

## Both exclusion questions are DECIDED — AISP and FIRY are prohibited

**The user's determination, 2026-10-12.** Both are in `screener.py`'s `_PROHIBITED_TICKERS`
(`dbd51c9`), which covers **both lanes**, so neither can return by any route. **AISP will not come
back after it consolidates** — the earlier note in this file saying it would is void.

1. **AISP — prohibited, and it widened the category.** Its $29.0M of "brand-name-only" DHS awards
   are public-safety video/data platforms; it operates no facility. The detention bullet in
   `portfolio_rules.md` read **"operators"**, under which AISP was not prohibited — which is
   exactly why nothing coded caught a company whose industry reads *Software — Infrastructure*.
   The rule now covers **the vendors that supply them**.
2. **FIRY — prohibited (Israeli-affiliated).** "Operates a mobile game platform in the United
   States, **Israel**, China, Malta, Hong Kong, Cyprus" — same basis as PLX: affiliation read off
   the operating footprint, not the domicile field.

Reasoning and the honest accounting: `Rules Amendment History.md` (2026-10-12).

## For the Week 58 run

1. **Re-verify every price at the PRV gate** — these are 10-09 closes read on 10-12.
2. **Log these six rows** with the codes below. They are not yet in `research_log.csv`; this file
   is the evidence, the log is the record, and the report is what writes it.
   **AISP and FIRY are `prohibited`, not `extended` or `negative-earnings`** — the exclusion is the
   binding reason and it outranks every other gate (same handling as ZIM in Week 57). The log is
   **append-only**, so these two rows are the permanent precedent for the widened category.
   ```
   AISP  catalyst:contract-ramp      PASS  prohibited        (DHS detention-agency supplier; user's determination 10/12, blocklisted)
   VAL   catalyst:contract-ramp      PASS  market-cap        ($5.81Bn > $5Bn; also revenue -13.2%, PT -18.4%)
   BORR  catalyst:contract-ramp      PASS  negative-earnings (TTM net income -$243.6M, EPS -$0.83)
   TREX  catalyst:margin-inflection  PASS  below-sma50       (-1.77% vs 50d; accretion dated 2027, outside the 90-day window)
   FIRY  catalyst:refinancing        PASS  prohibited        (Israeli-affiliated; user's determination 10/12, blocklisted)
   MOD   catalyst:spin-off           PASS  binary-thesis     (RMT conditioned on shareholder vote + IRS letter)
   ```
3. **Do not re-search the index category** before December unless an intra-quarter replacement is
   announced in the interim — check `press.spglobal.com` for any release dated after 2026-10-12
   with an effective date still ahead.
4. Blackouts expire this week: **HOPE ~10/13, ATRC ~10/15**. CON's $35.36 raise stays declined.
