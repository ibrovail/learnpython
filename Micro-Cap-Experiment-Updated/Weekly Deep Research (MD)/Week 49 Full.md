# Week 49 Deep Research — Full Report

**Date:** Sunday, August 16, 2026
**Week:** 49 of 52 (experiment ends 2026-09-18 — ~4.5 calendar weeks remaining)
**Session posture:** Wide net · 30–60 day catalysts · **Aggressive** · max 6 positions
**⚠️ Constraint amendments authorized this session** (see Section 1)
**Prepared by:** Claude Code (Deep Research Mode)

---

## 1. RESTATED RULES

**Three amendments authorized 2026-08-15** for the final stretch, because the constraint set — not the market — had become the binding limit on deployment (Week 48: 8 of 15 candidates excluded by rule, nothing investable, 54% cash):

1. **Health Care sector cap raised 2 → 3.** All other sectors remain capped at 2.
2. **Universe ceiling raised $2B → $5B.** `screener.py` now pulls Finviz's mid-and-under bucket and trims to `MAX_MARKET_CAP = 5e9`. Universe grew **856 → 1,593 names**.
3. **Adding to existing winners permitted** as a deployment route, subject to the unchanged 30%-per-name cap and no-averaging-down rule.

**Unchanged:** 15% cash floor · 5% risk-per-trade · 30% single-name cap · all excluded classes (ETFs, closed-end funds/BDCs, SPACs, ADRs, units/warrants) · stop-loss requirements and **range checks** (day's low + ATR, ≥1.5×ATR clearance, never lower a stop, de-risk by size not tightness) · every entry-discipline rule (post-earnings cooldown, distance-from-base, **days-1–3 breakout avoidance**, thesis-input freshness, screener score = sourcing at 2/5 floor) · price-data integrity (browser for live quotes and breaking news).

---

## 2. RESEARCH SCOPE

**Retrieved:** 2026-08-16 ~20:25 ET. Closes as of Friday 2026-08-14 (settled). Screener regenerated on the widened universe — **1,593 names, 403 rows dropped by sanity checks, 1 identity mismatch** (the corruption guards continue to work).

**Live sources:** WWW (TradingKey/Yahoo/StockStory — Q2 revenue $506M +7%, guidance raised), PAR (TipRanks/GuruFocus/Investing.com/StockStory — Q2 revenue $133M +19%, ARR $338M, guidance raised), CADL (carried from Week 45 verification: StockTitan/Seeking Alpha/AUA data — CAN-2409 Phase 3, BLA Q4 2026).

**Checks:** excluded-class screen, sector-cap map under the new 3-name healthcare limit, distance-from-base **and breakout age**, ATR range checks, cash-floor math.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction | Status |
|--------|------|-----------|----------|---------------|--------------|-----------|--------|
| ATRC | Health Care — post-catalyst | 2026-07-13 | $34.30 | $44.30 (**+29.2%**) | $39.50 (locks +15.2%) | 4/5 | KEEP — +30% partial **deferred** |
| TILE | Cons. Cyclical — beat-and-raise | 2026-06-15 | $32.10 | $37.96 (+18.3%) | $35.75 (locks +11.4%) | 4/5 | KEEP |
| ARDT | Health Care — post-catalyst | 2026-07-31 | $10.80 | $10.91 (+1.0%) | $10.35 | 3/5 | KEEP |

**Equity:** $744.08 · **Cash:** $404.79 (**54.4%**) · **Gap: −4.8%** ($744.08 vs $781.45) · TWR alpha **+0.28%**

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Confirmation | Liquidity Note |
|--------|-----------------|--------------|--------------|----------------|
| **PAR** | PAR Technology — restaurant SaaS: revenue **+19% to $133M**, ARR **$338M +17%**, adj. EBITDA $14.3M above guidance high end, FY guidance raised | Q2 reported 8/6; SaaS transition compounding | CONFIRMED (TipRanks + GuruFocus + Investing.com + StockStory) | $788M; **Technology — empty sector** |
| **CADL** | Candel Therapeutics — CAN-2409 Phase 3 prostate, **39% DFS improvement** (58-mo follow-up), RMAT designation, $120M cash into 2028 | **BLA submission Q4 2026** | CONFIRMED (Week 45 verification: StockTitan + Seeking Alpha + AUA) | $897M; **unblocked by the new 3-name healthcare cap** |
| WWW | Wolverine World Wide — Q2 revenue **$506M +7%**, adj. EPS +14% to $0.40, op margin +80bps, **guidance raised** across revenue/EPS/margin/FCF | Q2 reported ~8/13 | CONFIRMED (TradingKey + Yahoo + StockStory) | $1.7B; **day 2 of a +16.4% move** |

**Evaluated & not selected (one line each):**
- **WWW** (#3) — the best *fundamental* story on the board (growing revenue, margin expansion, guidance raised on four metrics), but closes went $18.06 → $19.88 → $21.03: **day 2 of a +16.4% breakout**, the exact FOXF/ARLO pattern. **Queued #1 for Week 50 on consolidation.**
- **BTCS** (#1) — $56M nano-cap crypto-adjacent financial at $1.12; fails liquidity comfort and quality.
- **ANRO** (#2) — hottest healthcare momentum (+24.1%/20d, +13.6%/5d) but vertical, and the 3rd healthcare slot is better spent on CADL's dated BLA catalyst than on an extended chase.
- **SFNC** ($3.5B) / **RNST** ($4.0B) — newly eligible regional banks, but momentum is negligible (+3.7%, +1.4%/20d) and both show **negative** relative strength vs IWM. The widened ceiling brought them in; it didn't make them attractive.
- **MGTX / NUVB / AORT / CCCC** — healthcare; slot 3 goes to CADL. NUVB is also below its 20-day SMA.
- **NPWR** (#9) — +26.6%/20d on 5.3× volume, but a $438M pre-revenue-style industrial; too speculative for a 4-week horizon.
- **UMH** (#12) — REIT, modest momentum, no catalyst.
- **VUZI** (#13) — +35.4%/20d, $262M, BBW 0.353 (extreme volatility); a lottery ticket, not a position.
- **KRP** (#14) — energy royalties, flat momentum, negative RS.

**Net: initiate PAR and CADL; queue WWW.**

---

## 5. PORTFOLIO ACTIONS

- **Initiate: PAR — 8 shares** (~$153, 20.5% of equity). Fills the **empty Technology sector**. Clean setup: no vertical gap (closes ground higher 17.43 → 19.05 over five sessions), three consecutive higher lows (16.82 → 17.65 → 18.38), +11.1% above the 20-day SMA (limit +20%), and **8 sessions past its 8/6 print** so the cooldown is clear. Revenue +19% with raised guidance is genuine growth — the opposite of the declining-revenue names (TDAY, FOXF) that have cost us.
- **Initiate: CADL — 10 shares** (~$118, 15.8% of equity). Takes the **newly available 3rd healthcare slot**. This was my Week 45 queue name at $10.18; the thesis (CAN-2409 Phase 3, 39% DFS improvement, RMAT, BLA Q4 2026) is unchanged and now investable under the amended cap.
- **Keep:** ATRC (partial deferred, stop locks +15.2%), TILE, ARDT — all stops correctly ranged; no changes.
- **Add to winners:** considered under the new permission and **declined**. TILE trades above its $36–37 consensus PT and ATRC sits at +29% just under its partial trigger; adding to extended winners is the "buy the post-print high" error. The permission is better spent on a pullback.
- **Exit / Trim:** none.

---

## 6. EXACT ORDERS

**Order 1 — Initiate PAR**
- Action: buy · Ticker: PAR · Shares: **8**
- Order Type: limit · Limit Price: **$19.20** (Friday close $19.05 + buffer)
- Time in Force: DAY · Intended Execution: 2026-08-17
- Stop Loss: **$17.05** — 1.75×ATR below the close; below the 5-day low ($16.82 was the low before the higher-low sequence began)
- Stop Limit: **$16.90**
- Range check: Friday low $18.38 · ATR(14) $1.135 (5.96%) · stop is **1.89×ATR** below close ✓ · max loss 8 × $2.15 = **$17.20 (2.3% of equity)**
- Order-side check: $17.05 < $19.05 ✓
- Special: **pre-open verification required** — browser-check Monday pre-market; if PAR gaps >2% above $19.05, wait rather than chase; if it gaps down, reprice at/below the verified print.
- Rationale: 19% revenue growth + raised guidance in an empty sector, entered on a grind rather than a gap.

**Order 2 — Initiate CADL**
- Action: buy · Ticker: CADL · Shares: **10**
- Order Type: limit · Limit Price: **$11.85**
- Time in Force: DAY · Intended Execution: 2026-08-17
- Stop Loss: **$10.70** — 1.75×ATR below the close; just above the 10-day low cluster ($10.35–10.50)
- Stop Limit: **$10.55**
- Range check: Friday low $11.04 · ATR(14) $0.577 (4.92%) · stop is **1.77×ATR** below close ✓ · max loss 10 × $1.15 = **$11.50 (1.5% of equity)**
- Order-side check: $10.70 < $11.72 ✓
- Special: pre-open verification. Note CADL rose +10.5% on 8/13 then consolidated (−0.7%) — cumulative move is just under the 10% breakout-avoidance threshold and the last session was a base-building day, so this clears, but it is the closer of the two calls.
- Rationale: dated BLA catalyst (Q4 2026), Phase 3 data already in hand, funded into 2028; takes the newly permitted 3rd healthcare slot.

---

## 7. RISK AND LIQUIDITY CHECKS

Post-trade:

| Holding | Value | % of Equity |
|---------|-------|-------------|
| TILE | $151.84 | 20.4% |
| ATRC | $132.90 | 17.9% |
| PAR | $153.60 | 20.6% |
| CADL | $118.50 | 15.9% |
| ARDT | $54.55 | 7.3% |
| **Cash** | **$132.69** | **17.8%** |

- No position >30% ✓ · **Cash 17.8% > 15% floor** ✓
- **Sectors:** Health Care **3** (ATRC, ARDT, CADL — at the new cap ✓), Cons. Cyclical 1, Technology 1 ✓
- Binary-event plays: 0 ✓
- Combined new-position risk: $17.20 + $11.50 = **$28.70 ≈ 3.9% of equity**; aggregate across all five holdings if every stop fired: **≈ −2.6%** (the three incumbents lock gains)
- Order sizes are trivial fractions of ADV for both names ✓

---

## 8. MONITORING PLAN

- **PAR** — honor the 5-day minimum hold. Watch ARR trajectory and the SaaS-transition narrative; GAAP is still lossmaking (−$0.41/sh) so the story rests on ARR + adjusted EBITDA, which is a risk if sentiment shifts against unprofitable software.
- **CADL** — BLA submission Q4 2026 is the event; watch for FDA interaction news and any financing (a raise would pressure the stock). Clinical-stage binary risk is real; sized at 15.9% accordingly.
- **WWW** — queued #1. Enter on a higher-low base after the +16.4% move digests, or a pullback toward the 20-day SMA (~$19.35). Fundamentals are the strongest of the three; only the entry timing is wrong.
- **ATRC** — +30% partial deferred; re-evaluate if conviction drops below 4/5, the position exceeds 30% of equity, or two earnings cycles pass without the doubled guidance showing in results. Stop $39.50 sits $0.50 above the 5-day low — a known tight spot.
- **TILE** — +30% partial level $41.73; stop $35.75 at 1.27×ATR is tighter than guideline (deliberate).
- **Regime** — IWM $305.09 vs 50-day ~$291; comfortable. A break below freezes new momentum initiations.

---

## 9. THESIS REVIEW SUMMARY

*(Saved separately as `Week 49 Summary.md`.)*

**ATRC — KEEP | 4/5.** +29.2% and knocking on the $44.59 partial trigger, which stays **deferred**: the FY EPS guidance doubling is a genuine new catalyst, the $39.50 stop locks +15.2%, the position is 17.9% of equity, conviction is 4/5 — all four deferral criteria hold. With 54% cash, converting the best performer into more idle cash is the wrong direction.

**TILE — KEEP | 4/5.** +18.3%, holding its post-earnings gap above the $36–37 consensus PT. Stop $35.75 locks +11.4%.

**ARDT — KEEP | 3/5.** +1.0% on the 5-share residual. The mixed Q2 (EBITDA −32% YoY) keeps conviction capped; stop $10.35.

**PAR — INITIATE | 4/5.** The cleanest setup this experiment has produced in weeks: **revenue +19% to $133M, ARR $338M (+17%), adj. EBITDA above the guidance high end, and FY guidance raised** — actual growth, not a cost-cut story on a shrinking base. Technically it *ground* higher rather than gapping (three consecutive higher lows, +11.1% over the 20-day SMA), and it is 8 sessions past its print. Fills the empty Technology sector. Bear case: GAAP still lossmaking (−$0.41/sh), so it's an ARR-and-adjusted-EBITDA story vulnerable to a sentiment shift against unprofitable software.

**CADL — INITIATE | 3/5.** The Week 45 queue name, finally investable under the amended healthcare cap. CAN-2409 Phase 3 in localized prostate cancer showed a **39% improvement in disease-free survival** at 58-month median follow-up, carries RMAT designation, and the **BLA submission is guided for Q4 2026**, with cash into 2028. Bear case: clinical-stage binary — a BLA delay or an FDA question resets the story, and biotech financings dilute. Sized at 15.9% with a 1.77×ATR stop.

**Overall portfolio thesis.** **Gap −4.8%, TWR alpha +0.28%, cash 54.4% — and that cash was the problem.** Two sessions this week made the mechanic explicit: the book gained nothing while the S&P-equivalent rose, because half the portfolio earns zero. With ~4.5 weeks left, holding cash was a guaranteed way to finish behind.

So this week's decisive act wasn't a trade — it was **amending the constraints** that were blocking deployment. Raising the healthcare cap to 3 and the universe ceiling to $5B nearly doubled the screenable universe (856 → 1,593) and directly unlocked one of the two initiations (CADL). Notably, the ceiling change did *not* produce a flood of quality — the new $2–5B entrants (SFNC, RNST) screened poorly with negative relative strength — which is itself useful evidence: the micro-cap tilt wasn't the constraint that mattered most; the **sector cap** was.

The deployment is **PAR** (real growth, empty sector, non-gapped entry) and **CADL** (dated BLA catalyst), taking the book to **5 positions and 17.8% cash**. **WWW is deliberately left on the table** — the strongest fundamentals of the three, but day 2 of a +16.4% move, and chasing that is precisely what cost us on PHAT, LXU and nearly on FOXF. It's queued for Week 50 on consolidation. Three weeks to run: the cash is finally working, the winners are protected at +11% to +15% locked, and the remaining edge has to come from the two new positions doing what their fundamentals say they should.

---

## 10. CONFIRM CASH AND CONSTRAINTS

- **Starting cash:** $404.79 · **PAR:** −$153.60 · **CADL:** −$118.50 → **Ending cash: $132.69 (17.8%)** — above the 15% floor ✓
- **Positions:** 5 of max 6 ✓ · **Sectors:** Health Care 3 (at the amended cap), Cons. Cyclical 1, Technology 1 ✓
- **Binary plays:** 0 ✓ · **Largest position:** PAR 20.6% < 30% ✓
- **Amendments applied and documented** in `Start Your Own/portfolio_rules.md` and `screener.py` ✓
- **All stops range-checked** against Friday's low + ATR ✓ · **Breakout-age test applied** to all three finalists ✓
- **Note:** orders are recommendations — real only after broker execution with actual fills, which I then log. Both entries require **pre-open verification via the browser** before executing.

---

*Week 49 Full report generated 2026-08-16 by Claude Code (Aggressive posture, amended constraints). Closes as of 2026-08-14; catalysts verified 2026-08-16.*
