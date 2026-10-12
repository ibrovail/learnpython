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

**KILL, three ways:**
1. **Below the 50-day SMA: −1.77%.** The hard trend gate requires ≥0%. This alone is a stage-1 kill
   (`below-sma50`).
2. **Cannot be sized inside the 30% ceiling.** Widest stop candidate is $42.05 (1.75×ATR; the
   10-session low $42.10 and the 50-day SMA $45.59 both sit above it), a $2.73 risk per share.
   2% of $721 = $14.42 → **5.3 shares = $236 = 32.8% of equity**, over the 30% ceiling.
3. The margin accretion management actually points to is **2027** ("half the lines operational by
   year-end to support 2027 margin accretion") — **outside the 90-day catalyst window**. The
   July raise is a three-month-old fact whose forward impact is dated beyond the horizon the rule
   allows.

**New structural constraint, worth the rules session's attention.** Kill #2 is not about TREX. At
$721 of equity, 2% risk and a 30% position ceiling, the stop must be at least
**risk ÷ (0.30 × equity) = $14.42 ÷ $216.30 = 6.67% below entry** or the position cannot be built.
TREX's widest stop is **6.1%** — it fails by 0.57pp. So the book's size imposes a **minimum
stop-width of ~6.7%**, which systematically excludes **low-volatility names** — precisely what the
screener composite is built to favour. This is an interaction between the 2% rule, the 30% ceiling
and book size that has not been written down anywhere, and it will relax as equity grows.

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

The reason is not sourcing. **Four of the six died on earnings quality or universe size, not on
the catalyst** — and three of the last six catalyst names across W57–W58 died on the *same* code:
NNBR −$47.9M, BORR −$243.6M, FIRY −$70.8M. Dated-catalyst small caps skew heavily loss-making,
and "growing TTM revenue with a credible earnings path" is the gate that kills them. The one
profitable, growing, in-universe, Buy-rated name found (TREX) died on the trend gate and on book
size, not on its catalyst.

So: the catalyst lane is not short of events. It is short of events attached to **profitable**
companies whose price is **above its 50-day SMA** and whose stop is **wide enough to size** in a
$721 book.

## Two exclusion questions for the user — not mine to decide

Both names below are **already dead on other gates**, so nothing turns on them this week. Flagged
now because each would otherwise pass the coded checks and recur:

1. **AISP — Department of Homeland Security exposure.** Its $29.0M award is "brand-name-only"
   contracts from a DHS agency for public-safety video/data platforms. Industry reads
   *Software — Infrastructure*, so no `REVIEW` flag fires. DHS contains components that operate
   detention. The exclusion covers **prisons/detention** operators; AISP is a software vendor, not
   an operator. Precedent cuts both ways: CXW was dropped for being a prison operator, and PLX was
   blocked on **affiliation rather than domicile**. Its backlog converts over two quarters, so the
   catalyst persists and the name will come back once it consolidates. **Needs a determination.**
2. **FIRY — Israel among its operating countries.** Profile: "operates a mobile game platform in
   the United States, **Israel**, China, Malta, Hong Kong, Cyprus". Same shape as the PLX call,
   which was made on operational base, not domicile. **Needs a determination.**

## For the Week 58 run

1. **Re-verify every price at the PRV gate** — these are 10-09 closes read on 10-12.
2. **Log these six rows** with the codes below. They are not yet in `research_log.csv`; this file
   is the evidence, the log is the record, and the report is what writes it.
   ```
   AISP  catalyst:contract-ramp      PASS  extended          (+21.11% vs 20d SMA, limit 20%; new 20d high after +14.15%/5d)
   VAL   catalyst:contract-ramp      PASS  market-cap        ($5.81Bn > $5Bn; also revenue -13.2%, PT -18.4%)
   BORR  catalyst:contract-ramp      PASS  negative-earnings (TTM net income -$243.6M, EPS -$0.83)
   TREX  catalyst:margin-inflection  PASS  below-sma50       (-1.77% vs 50d; also 32.8% of equity > 30% ceiling; accretion dated 2027)
   FIRY  catalyst:refinancing        PASS  negative-earnings (TTM net income -$70.76M, EPS -$4.53, beta 4.63)
   MOD   catalyst:spin-off           PASS  binary-thesis     (RMT conditioned on shareholder vote + IRS letter)
   ```
3. **Do not re-search the index category** before December unless an intra-quarter replacement is
   announced in the interim — check `press.spglobal.com` for any release dated after 2026-10-12
   with an effective date still ahead.
4. Blackouts expire this week: **HOPE ~10/13, ATRC ~10/15**. CON's $35.36 raise stays declined.
