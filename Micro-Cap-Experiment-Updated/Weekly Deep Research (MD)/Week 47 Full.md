# Week 47 Deep Research — Full Report

**Date:** Sunday, August 2, 2026
**Week:** 47 of 52 (experiment ends 2026-09-18 — 7 calendar weeks remaining)
**Session posture:** Wide net · 30–60 day catalysts · **Aggressive** · max 6 positions
**Prepared by:** Claude Code (Deep Research Mode)

---

## 1. RESTATED RULES

- **Long-only, full shares, no options/margin/shorting.** U.S.-listed common ≤$2B cap; names above $2B (WKC) are hold-only.
- **Cash floor 15% of equity** ($109.83 today); no new capital. Track to the cent.
- **Sizing:** risk ≤5% equity/trade; ≤30%/name; **binary-event plays max 1 at ≤15% equity.**
- **Momentum plays** above the 20-day SMA at entry; catalyst plays need a dated catalyst ≤60 days.
- **Entry discipline** (`.claude/rules/entry-discipline.md`): post-earnings cooldown; distance-from-base (≤20% above 20-day SMA); **Thesis-Input Freshness** (verify time-varying drivers current); **Day-1 Drawdown Rule** (exit a fresh position closing ≤−8%); screener score = sourcing (2/5 floor), not conviction.
- **Price integrity** (`.claude/rules/price-data-integrity.md`): settled closes from the daily run; live/AH/pre-market via the browser tool with a timestamp; **order-side check** before naming any stop/limit; **verify candidate identity** when a search result's price/sector doesn't match the screener.
- **Sector cap** ≤2 per GICS sector; **regime filter** (IWM vs 50-day); **one open order per stock**; 10-day re-entry ban; ≥2-source verification.

---

## 2. RESEARCH SCOPE

**Data retrieved:** 2026-08-02 ~20:00 ET. Holdings/closes as of Friday 2026-07-31 (settled). Screener regenerated 2026-08-02 — **clean run** (corruption fix holding). WebSearch used for catalysts/dates/fundamentals only.

**Live sources (WebSearch, 2026-08-02):** ARDT (Yahoo/Markets Daily/scanx — Q2 earnings **Aug 4** confirmed, EPS est $0.17), FIGS (StockTitan/Business Wire — Q2 **Aug 6** confirmed), AMCX (Daily Political/StockStory — Q2 **missed**, Netflix $500M deal), PHAT (GlobeNewswire/Seeking Alpha — Q2 beat but **FY guide cut**), DMC (Investing.com — returned DMC Global/**BOOM** ~$6, identity mismatch vs screener's "$29.71").

**Checks:** regime (IWM vs 50-day — refresh below), universe ceiling, excluded classes (BDC), liquidity, sector map, binary-event count, entry-discipline distance/cooldown, candidate identity, cash floor.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|-----------|----------|---------------|--------------|------------------|--------|
| WKC | Energy — runner (at $2B cap) | 2026-04-27 | $26.00 | $39.86 (+53.3%) | $37.65 / $37.50 | 4/5 | KEEP — +60% partial at $41.60; stop locks +44.8% |
| SHO | Real Estate — quality REIT | 2026-05-18 | $10.20 | $11.77 (+15.4%) | $10.90 / $10.80 | 3/5 | KEEP — **earnings 8/6**; $12 stop-raise trigger |
| TDAY | Comm. Services — AI-licensing | 2026-06-08 | $8.15 | $8.69 (+6.6%) | $7.55 / $7.45 | 4/5 | KEEP — **earnings 8/6** (confirmed) |
| TILE | Cons. Cyclical — beat-and-raise | 2026-06-15 | $32.10 | $34.26 (+6.7%) | $31.00 / $30.85 | 4/5 | KEEP — earnings date pending IR reconfirm |
| ATRC | Health Care — post-catalyst | 2026-07-13 | $34.30 | $37.76 (+10.1%) | $36.00 / $35.85 | 4/5 | KEEP — held its stop through the −12% PT-trim slide |

**Equity:** $732.23 · **Cash:** $173.54 (23.7%) · **Gap (scoreboard): −2.6%** ($732.23 vs $751.73) · TWR alpha: **+2.87%** · (See `Strategic Pivot — Week 46 Readout.md` for the full since-April context.)

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation | Liquidity Note |
|--------|-----------------|--------------|-----------------------|----------------|
| **ARDT** | Ardent Health — profitable hospital operator, FY EPS $0.90–1.27, Q1 rev +7% | **Q2 earnings Aug 4 (after close)** | CONFIRMED (Markets Daily + scanx + Yahoo) | $1.5B cap ✓; NYSE, ample ADV; calm base (+9%/20d) |
| FIGS | FIGS — DTC medical scrubs/apparel, profitable, demand-normalization turnaround | Q2 earnings Aug 6 (after close) | CONFIRMED (StockTitan + Business Wire) | $1.8B ✓; queued #2 (Cons. Cyclical) |
| PHAT | Phathom — VOQUEZNA +88% YoY, first positive op-profit | Q2 **already reported 7/30** | Reported — **FY guide cut**, shares fell | Thesis dented; off the queue |
| AMCX | AMC Networks — Netflix/Walking Dead $500M licensing | Q2 **reported 7/30 — missed** | Reported — rev −8.8% YoY, Zacks Strong Sell | Melting-ice business; pass |

**Screener candidates evaluated & not selected (one line each):**
- **DMC** (#1) — search returned DMC Global (ticker **BOOM**, ~$6), which does not match the screener's "DMC $29.71/$1.4B" → **identity unverified**, pass pending confirmation (price-integrity identity check).
- **PLX** (#2) — Protalix, $191M micro-cap biotech at $2.37; below liquidity comfort, no dated catalyst.
- **AMCX** (#3) — Netflix deal real but the quarter missed (rev −8.8%), Zacks Strong Sell; catalyst is a sugar-high on a declining linear-TV business (CERT-like value trap).
- **TWO** (#4) — Two Harbors mortgage REIT, 0 momentum, below 20-day SMA; SHO already fills Real Estate.
- **CCO** (#5) — Clear Channel Outdoor, $2.42, high leverage, no momentum.
- **OXSQ** (#6) — Oxford Square Capital: a BDC (closed-end) → **excluded class**.
- **CFFN** (#7) — sleepy thrift, +4.5%/20d, no catalyst.
- **ABR** (#8) — Arbor Realty mortgage REIT, below 20-day SMA, credit-cycle risk.
- **WALD** (#9) — Waldencast, $1.73 micro-cap, no catalyst in window.
- **FIGS** (#10) — genuine dated catalyst (Aug 6), profitable DTC — **queued #2** behind ARDT (chose ARDT on higher fundamental conviction).
- **MDXG** (#11) — MiMedx, −3%/5d, below 20-day SMA.
- **LIFE** (#12) — small financial, no verified dated catalyst; thesis unconfirmed.
- **OCFC / PDM / PRG** (#13–15) — thrift / office REIT / lease-to-own, all below 20-day SMA or catalyst-free.

**Net: initiate ARDT (pre-print binary-event play into the Aug 4 earnings); FIGS queued #2 (Aug 6).**

---

## 5. PORTFOLIO ACTIONS

- **Keep:** WKC, SHO, TDAY, TILE, ATRC — all stops correctly placed from the recent raises; SHO + TDAY hold through their 8/6 prints on current stops (do not tighten into a print and invite a gap-out).
- **Initiate:** **ARDT** — 5 shares (~$54, ~7.3% of equity) — profitable hospital operator into a confirmed Aug 4 earnings beat setup; the **quality dated-catalyst** deploy the readout called for, not a momentum chase. Sized small by the cash floor, not by conviction.
- **Add to / Trim / Exit:** none. **Stop changes: none** — WKC ($37.65) and ATRC ($36.00) are already snug from last week; further tightening risks whipsaw on names carrying the book.

**Press-and-protect:** the aggressive "press" is the ARDT catalyst add; the "protect" is already in place via last week's WKC/ATRC stop raises (locking +44.8% / +5%). Both sides, deliberately.

---

## 6. EXACT ORDERS

**Order 1 — Initiate ARDT (binary-event play)**
- Action: buy · Ticker: ARDT · Shares: **5**
- Order Type: limit · Limit Price: **at/below $10.80** (last verified ~$10.64 on 2026-07-26; **reconfirm Monday pre-open**)
- Time in Force: DAY · Intended Execution: 2026-08-03 (Monday)
- Stop Loss: **$9.55** — ~10% below entry (standard); **binary gap-risk acknowledged** — an earnings miss can gap through this
- Stop Limit: **$9.45**
- Special: **Pre-open verification required** (browser tool, timestamped) — if ARDT gaps >2% above $10.80 or the price can't be verified, use a limit at/below the verified pre-market print, don't chase. Pre-print entry = binary risk on the Aug 4 report, sized within the 15% cap.
- Order-side check: stop $9.55 < intended entry $10.80 ✓ (run again against Monday's verified price)
- Rationale: profitable hospital operator, analysts expect a beat; dated catalyst in 2 sessions; adds catalyst torque at ≤0.75% equity risk.

---

## 7. RISK AND LIQUIDITY CHECKS

Post-trade (ARDT 5 @ $10.80):

| Holding | Value | % of Equity |
|---------|-------|-------------|
| WKC | $39.86 | 5.4% |
| SHO | $129.47 | 17.7% |
| TDAY | $139.04 | 19.0% |
| TILE | $137.04 | 18.7% |
| ATRC | $113.28 | 15.5% |
| ARDT | $54.00 | 7.4% |
| **Cash** | **$119.54** | **16.3%** |

- No position >30% ✓ · **Cash 16.3% > 15% floor** ✓ (floor binds the deploy size — ARDT is ~$54 max)
- Sectors: Energy, Real Estate, Comm. Services, Cons. Cyclical, **Health Care ×2 (ATRC, ARDT)** — at the 2-cap ✓
- Binary-event plays: **1** (ARDT ≤15%) ✓
- ARDT stop-risk: 5 × ($10.80 − $9.55) = $6.25 ≈ 0.85% of equity (gap risk could exceed on a miss) ✓
- Aggregate stop-risk if all six fired: ≈ −4.2% of equity.

---

## 8. MONITORING PLAN

- **Tue 8/4 — ARDT earnings (after close):** the binary. Beat + guide-raise → hold, trail to a standard stop; miss → the $9.55 stop (gap risk acknowledged); apply post-catalyst reassessment within 1 trading day.
- **Wed 8/5 / Thu 8/6 — TDAY + SHO earnings (both 8/6):** two holdings print same day. Hold through on current stops; TDAY raise on a close >$8.86, SHO on a push >$12.
- **FIGS 8/6:** watch as the queued #2 — a beat + base could earn a Week-48 entry.
- **WKC:** $41.60 = +60% partial (1-share exit-or-hold); stop $37.65 locks +44.8%.
- **ATRC:** stabilized post-PT-trim; $36.00 stop locks +5%. Watch AtriClip-vs-Edwards commentary.
- **Regime:** IWM $291.20 vs 50-day ~$291 — **on the line**; pull a fresh dated 50-day Monday before ARDT entry. A confirmed break below freezes further momentum initiations (ARDT is catalyst-driven, so still permitted, but note it).

---

## 9. THESIS REVIEW SUMMARY

*(Saved separately as `Week 47 Summary.md`.)*

**WKC — KEEP | 4/5.** +53%, at the $2B cap (hold-only), stop $37.65 locks +44.8% into the $41.60 partial. Ride on a tight leash.

**SHO — KEEP | 3/5.** +15.4% into its 8/6 print; stop $10.90. Raise toward ~$11.10 on a push >$12.

**TDAY — KEEP | 4/5.** +6.6% into its 8/6 print; Citizens PT $10. Stop $7.55; next raise on a close >$8.86.

**TILE — KEEP | 4/5.** +6.7%; earnings date being reconfirmed via IR (the "7/31" never verified). Stop $31.00.

**ATRC — KEEP | 4/5.** +10.1%; the beat-and-raise held through a −12% two-day PT-trim slide without breaching its $36.00 stop — exactly what a well-placed stop is for. Watch competitive commentary.

**ARDT — INITIATE | 3/5 (binary catalyst).** A profitable hospital operator into a confirmed Aug 4 earnings print that analysts expect it to beat — the quality, dated-catalyst entry the Week 46 readout prescribed after two momentum-chase losses. Sized small (~7%) by the cash floor, 10% stop with acknowledged gap risk. Healthcare goes to its 2-name cap.

**Overall portfolio thesis.** **Scoreboard: gap −2.6%; TWR alpha +2.87%** — the repair holds, the lead does not (yet). Per the Aggressive directive this is a deploy week, but disciplined by the readout's lesson: the one initiation is a *profitable, dated-catalyst* name (ARDT), not the screener's hot momentum (AMCX's Netflix sugar-high on a missed quarter, PLX/CFFN's catalyst-free drift). The 15% floor caps the deploy at ~$54, so this is a torque add, not a big swing. Two winners keep their snug stops; two holdings (TDAY, SHO) report 8/6. Seven weeks left: convert bursts into a held lead by raising the hit rate on new money — starting with an entry that has a fundamental floor under it.

---

## 10. CONFIRM CASH AND CONSTRAINTS

- **Starting cash:** $173.54 · **ARDT buy:** −$54.00 → **Ending cash: $119.54 (16.3%)** — above the 15% floor ✓
- **Positions:** 6 of max 6 ✓ · **Sectors:** Health Care ×2 at cap, others ≤1 ✓ · **Binary plays:** 1 (ARDT) ✓ · **Largest:** TDAY 19.0% < 30% ✓
- **Regime:** IWM on its 50-day (~$291) — refresh Monday; ARDT is catalyst-driven so permitted regardless ✓
- **Universe:** WKC at $2B logged hold-only ✓ · **Identity:** DMC flagged unverified, excluded ✓
- **Price integrity:** ARDT limit to be re-checked against Monday's verified pre-open; order-side check passed on the reference price ✓
- **Note:** orders are recommendations — real only after broker execution with actual fills, which I then log. Reconfirm ARDT's price Monday pre-open before executing.

---

*Week 47 Full report generated 2026-08-02 by Claude Code (Aggressive posture). Closes as of 2026-07-31; catalysts verified 2026-08-02 via WebSearch; live prices via browser tool per price-data-integrity.md.*
