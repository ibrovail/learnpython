# No-Cap Experiment

A real-money portfolio run as a research project. **Goal: beat the S&P 500 after all costs,
in Canadian dollars.**

Opened 2026-10-12 with **CAD $200** in a **TFSA**. Deposits are irregular: roughly $200/month
baseline, possibly $500/month, occasionally $2,000 or more.

The name is the point: no market-cap mandate. The predecessor experiment
(`../Micro-Cap-Experiment-Updated`) capped its universe at $5Bn and spent a year establishing
that the cap was not the edge. This one starts with no universe constraint and adopts rules
only on evidence.

---

## What is here

| Path | Contents |
|---|---|
| `portfolio_rules.md` | **Standing rules. Read before every session.** |
| `Experiment Details/` | Pre-registrations, audits, amendment history |
| `research/` | Backtest engine and data acquisition |
| `ledger/` | Holdings, deposits, and the injection-neutral return series |
| `.claude/rules/` | Methodology rules binding on every study |

## The three things this project knows before it starts

Carried over from a completed rules audit of the predecessor experiment, so they do not
have to be relearned at cost:

1. **Capital is usually the binding constraint, not strategy.** Rules written for a 10–20
   position book, run on a 3–4 position account, cost an estimated 9–14 percentage points a
   year. `portfolio_rules.md` §2 sizes the book to the money, in tiers fixed in advance.
2. **Returns are nearly irrelevant below roughly 10× the annual deposit.** At CAD $200, five
   years of beating the index by 12 points a year is worth about $204. One extra deposit beats
   it. The system's early job is to be correct and ready, not to produce returns it cannot yet
   produce.
3. **Alpha cannot be proven at this horizon.** At a concentrated book's tracking error,
   demonstrating 5% of annual alpha at conventional confidence takes roughly 46 years. So the
   design is justified by advantages that can be *audited directly* — tax wrapper, FX cost,
   turnover, position count — and the return series is treated as weak evidence indefinitely.

## Studies

| # | Question | Status |
|---|---|---|
| 01 | Winner rotation: is the edge momentum, or concentration? | Pre-registered. Stage 1 blocked on data access. |

## Running

```bash
python3 -I research/test_engine.py     # engine self-test, no network needed
```

The engine is verified against synthetic prices with known answers. The real study needs
network egress to an index-membership source and a price source covering delisted names;
this environment currently denies both.
