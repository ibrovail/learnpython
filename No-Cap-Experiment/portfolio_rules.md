# No-Cap Experiment — Portfolio Rules

**Standing rules. Read before every session.** Amendments go in
`Experiment Details/Rules Amendment History.md` with a date and a reason.

- **Opened:** 2026-10-12 · **Starting capital:** CAD $200 · **Account:** TFSA
- **Goal:** beat the S&P 500 after all costs, in Canadian dollars.
- **Horizon:** years, not decades. Capital may be withdrawn for project investment.

---

## 1. The benchmark

**A CAD-denominated S&P 500 total-return series, unhedged.** Not SPY in USD.

The book's returns are in Canadian dollars, so the yardstick must be too — otherwise a move
in the loonie decides the "alpha". The benchmark is **measured, never held**: it is unscreened,
while §6 governs everything actually bought.

**Returns are measured injection-neutral (time-weighted).** Deposits are not performance.
The balance will rise for years because of deposits; only the injection-neutral series says
whether any decision added value. See `ledger/`.

---

## 2. Capital tiers

The account's size sets its shape. **The tier is read off the balance, not chosen.**
This rule exists because the predecessor experiment wrote rules for a 10–20 position book
and ran them on a 3–4 position account, which cost an estimated 9–14 percentage points a year.

| Balance (CAD) | Satellite | Max names | Note |
|---|---|---|---|
| **< $1,000** | 0% | — | Core only. Too small for anything else. |
| **$1,000 – $5,000** | 25% | 2 | Satellite opens. |
| **$5,000 – $25,000** | 30% | 3–5 | |
| **$25,000 +** | 35% | 5–8 | Returns now outweigh deposits. |

**Why the top tier is where it changes:** returns only begin to dominate deposits once the
balance passes roughly **10× the annual deposit**. Below that, the deposit rate decides the
outcome and the strategy is nearly irrelevant to it.

**Tier changes act on new deposits only.** Crossing a threshold never triggers a sale.

---

## 3. Structure

**Core (65–75%)** — broad, diversified, **Canadian-listed** equity funds, so that no currency
is converted for the core. Bought and held. The core may carry a deliberate tilt away from
US mega-cap concentration; that tilt is sized as a lean, not a thesis, because valuation
spreads can take longer to close than this book's horizon.

**Satellite (25–35%, per §2)** — direct holdings. Multi-year holds. This is where alpha is
attempted and where it may fail.

---

## 4. Deposits and rebalancing

- **Deploy in full on arrival.** No averaging in, no waiting for a dip.
- **New cash goes to whatever sits furthest below its target weight.** This is the *only*
  rebalancing mechanism.
- **Never sell to rebalance.** Selling creates turnover, which §5 forbids.
- **As a project withdrawal comes into view, new deposits go entirely to the core.** Money
  already invested keeps its horizon; money arriving last gets the shortest one.

---

## 5. Turnover — a hard budget

> **Target: 6 or fewer trades per year. Hard ceiling: 10.**

Three independent reasons, all pointing the same way:

1. **Tax.** A TFSA that is found to be *carrying on a business* loses its exemption and its
   gains become fully taxable as business income. CRA weighs transaction frequency, holding
   period, speculative character and intent. Low frequency and multi-year holds are the defence.
2. **Evidence.** In the predecessor experiment's daily simulation, the lowest-turnover variant
   beat the mechanical-exit variant by roughly 2.8 percentage points a year on identical rules.
3. **Friction.** Spread and FX cost are paid per trade and compound against you.

**No mechanical stop-losses.** Exits happen on written thesis failure, never on a price trigger.

**Write the research log in investment language, not trading language.** "Revenue growth,
competitive position, five-year view" — not "breakout, stop level, 40-session hold." The log is
a record of intent, and intent is what §5.1 is judged on.

---

## 6. Exclusions

**Prohibited businesses — never bought, regardless of rank, screen or thesis.**
Carried over verbatim from the predecessor experiment.

- **Prisons and detention centres** — private corrections and immigration-detention operators
- **Weapons, defence and firearms** — defence contractors, arms makers, gun and ammunition companies
- **Predatory lending** — payday lenders, pawn lenders, high-cost subprime instalment credit
- **Israeli-affiliated companies**

*Not prohibited:* fossil fuels, tobacco, gambling, alcohol, cannabis, adult entertainment.

**Scope.** The list governs **direct holdings**. Broad funds are treated as market exposure and
are exempt — the alternative is losing diversification at a size that cannot afford it.

**Enforcement.**
- Checked **per name, before any buy is written** — not after.
- **Defence** can be partly screened by industry; **Israeli affiliation cannot** and must be
  checked by hand. A company's listed country of domicile is not the test; affiliation is.
- Two industries mix prohibited and permitted businesses and are never auto-cleared:
  **Security & Protection Services**, **Credit Services**.
- **Any mechanical rule must define its substitute** before it is run. A rule that selects a
  prohibited name and has no fallback is an incomplete rule. *(Origin: the "buy last year's
  winner" study — its 2025 pick was Palantir, which fails this list on defence and intelligence
  contracting, in one of the three years carrying its entire headline result.)*

---

## 7. Drawdown control — without selling

The book carries no stop-losses, so drawdown is controlled through **where new money goes**:

| Drawdown from peak (injection-neutral) | Action |
|---|---|
| **−20%** | Satellite's share of new deposits drops to **zero**. All new cash to the core. |
| **Recovered to −10%** | Normal deposit split resumes. |

Nothing is sold at any level. No realised losses, no turnover, no tax exposure.

---

## 8. What must never be adopted without evidence

A rule enters this file only after a study that was **pre-registered before any result was
computed**, with a **train/test split**, and whose **raw output was committed before being
interpreted**. See `.claude/rules/research-methods.md`.

**A backtest that passes only on recent data is rejected**, however large its headline.
