# Rules Amendment History

Every change to `portfolio_rules.md`, with the reasoning that produced it. Append only.

---

## 2026-10-12 — Rules v1

Opened the book. CAD $200, TFSA, benchmarked against a CAD-denominated S&P 500.

The three decisions that shaped v1, all carried from a completed audit of the predecessor
experiment rather than relearned at cost:

- **Capital tiers (§2).** The predecessor's largest measured failure was writing rules for a
  10–20 position book and running them on a 3–4 position account, costing an estimated 9–14
  percentage points a year. Tiers are fixed in advance and read off the balance.
- **Turnover budget of 6 trades a year (§5).** Set independently by three constraints that all
  point the same way: TFSA business-income risk, trading friction, and the predecessor's own
  daily simulation, where the lowest-turnover variant beat the mechanical-exit variant by about
  2.8 percentage points a year on identical rules.
- **Drawdown control without selling (§7).** Deposits are redirected rather than positions sold,
  so the book can de-risk with no realised losses, no turnover and no tax exposure.

---

## 2026-10-13 — Satellite universe, yield tiebreak, broker (§3.1, §3.2, §9)

Three operational decisions that were left open in v1.

**§3.1 — Satellite universe: US primary, Canadian permitted.**
The FX argument favours Canadian listings and is outweighed. Financials, energy and materials
are roughly 60% of the S&P/TSX Composite and healthcare is about 0.3%, so a 3–5 name Canadian
satellite is structurally a bet on a commodity economy rather than a stock selection. The book's
income, savings and currency are already Canadian and the loonie tracks oil, so concentrating
the alpha sleeve there stacks one exposure on another. The benchmark is also the S&P 500, and
competing with it from a universe missing most of technology is a self-imposed handicap.
Canadian Depositary Receipts were considered and rejected: currency-hedged, so they discard the
USD diversification, with a hedging cost up to ~0.60% a year that exceeds the FX cost saved.

**§3.2 — Yield is a tiebreak in the satellite.**
US dividend withholding in a TFSA is 15% and unrecoverable; unlike an RRSP there is no treaty
relief. The drag is a function of yield — about 0.075% a year at 0.5%, about 0.45% at 3% — so
preferring low-yield growth names reduces the TFSA's only structural penalty to near zero at no
cost to anything the book wants. Written as a tiebreak rather than a gate, because a yield
screen would reject good businesses for a reason worth 40 basis points.

**§9 — Broker: Wealthsimple, revisit at $25,000.**
*This reverses an earlier claim in the same discussion that FX cost would matter more than the
first few years of returns.* The arithmetic does not support it: the core is Canadian-listed, so
only the satellite converts, and 1.5% on 30% of a $5,000 year is about $22.50 — roughly 0.45% of
deposits. The binding constraint below $5,000 is **fractional shares**, without which a $60
satellite cannot buy a position at all. Wealthsimple is chosen for fractional shares, zero
commission and low operational friction; Interactive Brokers' near-free conversion wins later,
when the FX gap is larger and there is operating experience to spend on a harder interface.
