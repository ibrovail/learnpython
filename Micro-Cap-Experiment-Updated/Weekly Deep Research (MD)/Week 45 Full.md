# Week 45 Deep Research — Full Report (Rev. 2 — corrected screener)

**Date:** Sunday, July 19, 2026 (revised Monday, July 20 with the repaired screener watchlist)
**Week:** 45 of 52 (experiment ends 2026-09-18 — 9 calendar weeks remaining)
**Session posture:** Wide net · Within 10 days · **Neutral** · max 6 positions
**Prepared by:** Claude Code (Deep Research Mode)

> **Revision note:** the original Week 45 run invoked the No-Candidates rule because the screener universe was corrupted (every ticker's first character doubled by a Finviz page change — TotalEnergies masquerading as a $1.6B small-cap). `screener.py` now auto-repairs the corruption, validates ticker identity against a second price source, and excludes CEFs/ADRs/units at the source. This revision re-evaluates the **clean** watchlist. The portfolio decision is unchanged — but now for per-candidate reasons, not data failure.

---

## 1. RESTATED RULES

- **Long-only, full shares, no options/margin/shorting.** U.S.-listed common stock ≤$2B market cap on NYSE/NASDAQ/NYSE American. No OTC, ETFs/ETNs/closed-end funds (incl. BDCs), SPACs, ADRs, warrants/units/preferreds, defence, or Israeli-affiliated names.
- **Cash floor 15% of equity**; no new capital beyond snapshot ($177.29). Track to the cent.
- **Sizing:** risk ≤5% of equity per trade; ≤30% per name; binary-event plays max **1** at ≤15% of equity (ATRC currently holds this slot).
- **Stops on everything**; binary override to technical support allowed, auto-expiring at event resolution.
- **Momentum plays** above the 20-day SMA at entry; 5-day minimum hold. Catalyst plays need a confirmed catalyst within 60 days.
- **Entry discipline** (`.claude/rules/entry-discipline.md`): no buys in days 1–3 of a >+10% breakout; ≤40% above 50-day SMA / ≤20% above 20-day SMA; post-earnings cooldown; screener score is sourcing, not conviction (2/5 floor).
- **Sector cap:** ≤2 per GICS sector. **Regime filter:** IWM below 50-day SMA freezes momentum initiations.
- **Re-entry ban:** 10 trading days post-stop-out (PHAT through ~7/27).
- **One open order per stock**; **No-Candidates rule**; ≥2-source verification for every ticker and catalyst.

---

## 2. RESEARCH SCOPE

**Data retrieved:** 2026-07-19 ~18:30 ET (holdings/context) and 2026-07-20 ~09:15 ET (corrected screener + candidate verification). Prices as of Friday 2026-07-17 close; screener regenerated 2026-07-20 after the corruption fix (856-stock universe, 15 candidates, all verified genuine).

**Live sources consulted (WebSearch):**
- Regime: IWM 50-day SMA $290.80 (Barchart / Markets Daily 7/18); IWM $294.04 → above by ~1.1%.
- Candidates: IKT (SEC 8-K, ChartMill, company IR — Phase 3 PAH, RA Capital $50M), HTLD (SEC 8-K, TipRanks, public.com — Q2 earnings 7/23), CADL (StockTitan, Seeking Alpha, businesswire — CAN-2409 BLA Q4 2026), XNCR (GuruFocus, MarketBeat — ESMO pop 7/17), GPRE (StockStory, Simply Wall St, timothysykes — UBS PT hike 7/17), CCRN (StockTitan SEC filings, businesswire — Knox Lane $13.25 take-private, vote passed 7/16).
- Holdings (carried from 7/19 run): ATRC 7/23 + consensus (MarketBeat/StockTitan/WallStreetZen), WKC 7/23 (SEC 8-K), TDAY 7/30 (MarketBeat/Investing.com), TILE 7/31, SHO 8/6.

**Checks performed:** regime, universe ceiling, excluded classes (CEF/BDC/ADR), liquidity, sector map, binary count, re-entry blackout, entry-discipline distance/breakout-age tests, trailing-stop floors, cash floor.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|-----------|----------|---------------|--------------|------------------|--------|
| WKC | Energy — runner post-partial | 2026-05 | $26.00 | $36.50 (+40.4%) | $32.50 / $32.35 | 4/5 | KEEP — earnings 7/23; stop locks +25% |
| SHO | Real Estate — quality REIT | 2026-06 | $10.20 | $11.73 (+15.0%) | $10.90 / $10.80 | 3/5 | KEEP — 2.3% from $12 raise trigger; earnings 8/6 |
| TDAY | Comm. Services — AI-licensing | 2026-06 | $8.15 | $8.50 (+4.3%) | $7.55 / $7.45 | 4/5 | KEEP — earnings 7/30; Citizens PT $10 |
| TILE | Cons. Disc. — beat-and-raise | 2026-06 | $32.10 | $33.06 (+3.0%) | $31.00 / $30.85 | 4/5 | KEEP — earnings 7/31; Strong Buy |
| ATRC | Health Care — binary event | 2026-07-13 | $34.30 | $34.95 (+1.9%) | $30.50 / $30.30 (override) | 3/5 | KEEP — earnings Thu 7/23 AC; playbook armed |

**Equity:** $715.91 · **Cash:** $177.29 (24.8%) · **Gap (scoreboard): −4.4%** ($715.91 vs $748.52 S&P-equivalent) · TWR alpha (process check): **+0.77%**.

---

## 4. CANDIDATE SET (corrected screener, all tickers verified genuine)

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |
|--------|-----------------|--------------|------------------------------|----------------|
| **CADL** | Candel Therapeutics — CAN-2409 Phase 3 prostate: 39% DFS improvement (58-mo follow-up), RMAT status | **BLA submission Q4 2026**; NSCLC pivotal underway | CONFIRMED (StockTitan + Seeking Alpha + AUA data) | $737M ✓; +22.5%/20d but not a days-1–3 spike |
| **HTLD** | Heartland Express — freight-cycle turn; Q1 loss narrowed to −$0.06 vs −$0.128 est | **Q2 earnings Thu 7/23** | CONFIRMED (TipRanks + public.com) | $1.16B ✓; 1.7x volume accumulation |
| IKT | Inhibikase — Phase 3 PAH (IMPROVE-PAH); RA Capital $50M backing 7/14 | Topline data **years out** (Part B ~346 pts) | Confirmed but outside any timing window | $283M; $2.11 price; fresh 25M-share dilution |
| XNCR | Xencor — XmAb819 ccRCC (25% ORR), ESMO oral | ESMO presentation (dated but H2); pivotal 2027 | CONFIRMED — but +17% pop was 7/17 | $1.3B ✓; +38%/20d — breakout day 2 |
| GPRE | Green Plains — ethanol EBITDA inflection; UBS PT $12→$20 | UBS estimate reset 7/17; Q2 earnings ~early Aug | CONFIRMED — but +11.9% pop was 7/17 | $1.3B ✓; 52-wk high; breakout day 2 |
| CCRN | Cross Country Healthcare — Knox Lane $13.25 cash take-private | Merger close Q3 2026 (vote passed 7/16) | CONFIRMED | Arb spread ≈ $0.01 (0.08%) — no edge |

**Evaluated & not selected (one line each):**
- **CADL** — the genuine new candidate, but it competes for the **single open Health Care slot** whose fate ATRC's Thursday print decides; deferring to Week 46 costs ~3 sessions and buys resolution of the book's biggest event. Queued #1.
- **HTLD** — entering 2 sessions before its print under a Neutral posture repeats the pattern the rules were built to prevent; post-print entry per the cooldown rule is the cleaner shape. Queued #2.
- **IKT** (#1 by score) — no catalyst inside any window, $2 stock, fresh 25M-share dilution; screener-floor conviction 2/5.
- **XNCR** (#5) — day 2 of a +17% breakout: entry-discipline disqualification (days 1–3 of >+10% move); re-eligible after ≥1 week of consolidation.
- **GPRE** (#15) — day 2 of a +11.9% breakout at a 52-week high on an analyst note (UBS kept it *Neutral* while doubling the PT): same disqualification; watch for a higher-low base.
- **CCRN** (#14) — merger arb with a closed vote and a 1-cent spread; the squeeze signal (BBW 0.006) is the deal price pinning it, not a setup.
- **CSWC / OCSL** (#4, #12) — BDCs: closed-end investment companies → **excluded class**.
- **JBGS** (#6) — DC office REIT, no catalyst in window, weakest structural story on the list.
- **RLJ** (#9) — hotel REIT: would double the book's lodging exposure (SHO) within the Real Estate cap — concentration without diversification.
- **TNDM** (#8) — diabetes-tech turnaround, +11.6%/20d, but Health Care slot contention (behind CADL) and earnings ~8/5 unverified this session.
- **AMCX** (#10) — melting-ice cable equity; momentum without a durable catalyst; pass.
- **CFFN** (#13) — sleepy thrift, +6.7%/20d, no catalyst; pass.

**Net: two credible candidates (CADL, HTLD) — both better entered AFTER Thursday resolves.** This is a timing deferral, not a No-Candidates finding.

---

## 5. PORTFOLIO ACTIONS

- **Keep:** WKC, SHO, TDAY, TILE, ATRC — unchanged from Rev. 1; every stop at/above its trailing floor; no thesis weakened.
- **Initiate:** none this week — the two credible candidates (CADL, HTLD) are deliberately **queued for Week 46**, gated on Thursday's ATRC/HTLD prints:
  - **CADL** — takes the Health Care slot at Week 46 *if* ATRC stops out on a miss (slot opens) or *if* ATRC beats and the sleeve cap still allows (it would — HC would be at 2 only if both held). Entry only on a non-extended base per distance rules.
  - **HTLD** — post-print entry per the cooldown rule: either pre-open Monday 7/27 after ≥1 week... no — after the 7/23 print, wait for the post-print close; enter only if not >+5% above it, or after a higher-low base forms.
- **Add to / Trim / Exit:** none. **Stop changes: none** (all verified against trailing floors — see Rev. 1 detail).

---

## 6. EXACT ORDERS

**None this week** — no buys (candidates queued behind the 7/23 prints), no sells, no stop modifications. The five resting GTC stop-limits remain the only open orders:

- **WKC:** stop $32.50 / limit $32.35 (GTC)
- **SHO:** stop $10.90 / limit $10.80 (GTC)
- **TDAY:** stop $7.55 / limit $7.45 (GTC)
- **TILE:** stop $31.00 / limit $30.85 (GTC)
- **ATRC:** stop $30.50 / limit $30.30 (GTC — binary override, expires at the 7/23 print)

---

## 7. RISK AND LIQUIDITY CHECKS

| Holding | Value | % of Equity |
|---------|-------|-------------|
| WKC | $36.50 | 5.1% |
| SHO | $129.03 | 18.0% |
| TDAY | $136.00 | 19.0% |
| TILE | $132.24 | 18.5% |
| ATRC | $104.85 | 14.6% |
| **Cash** | **$177.29** | **24.8%** |

- No position >30% ✓ · Cash 24.8% > 15% floor ✓ · Binary plays 1/1 (ATRC) ✓ · Sector caps ✓ · No orders → no ADV impact ✓
- Aggregate stop-risk if every stop fired from Friday's close: ~−$30.55 ≈ 4.3% of equity.

---

## 8. MONITORING PLAN

- **Thu 7/23:** ATRC Q2 after close (EPS consensus ~$0.06; AtriClip-vs-Edwards commentary is the swing factor) + WKC Q2 after close + **HTLD Q2** (watch as a candidate: loss-narrowing trajectory and freight-cycle commentary set up the Week 46 entry decision).
- **Fri 7/24:** post-catalyst reassessment (rule-mandated): strip ATRC's override stop, recalc trailing, re-rate conviction; log WKC's reaction; record HTLD's post-print close as the cooldown reference price.
- **CADL:** track daily into Week 46 — entry requires a non-extended base (≤20% above 20-day SMA) and the HC-slot decision post-ATRC.
- **XNCR / GPRE:** consolidation watch — re-eligible after ≥1 week of higher-low base-building per the cooldown rule.
- **TDAY 7/30, TILE 7/31:** pre-print checks in the dailies; TDAY stop-raise re-arms on a close >$8.86; SHO raise on a push through $12.
- **PHAT blackout ends ~7/27** — reconsider only on fresh verification.
- **Regime:** IWM cushion ~1.1% — a close below ~$290.80 freezes momentum initiations automatically.

---

## 9. THESIS REVIEW SUMMARY

*(Saved separately as `Week 45 Summary.md`.)*

**WKC — KEEP | Conviction 4/5.** The +40% runner meets its 7/23 print fully de-risked: the $32.50 stop banks +25% on any fade; a beat extends it. No action improves this shape.

**SHO — KEEP | Conviction 3/5.** +15.0%, 2.3% below the $12 stop-raise trigger, earnings 8/6. The $10.90 stop manages the tail until $12 prints.

**TDAY — KEEP | Conviction 4/5.** Citizens PT $10; earnings 7/30. Stop $7.55 at the trailing floor; next raise on a close above $8.86.

**TILE — KEEP | Conviction 4/5.** Strong Buy into the 7/31 print; stop $31.00 keeps it loss-free.

**ATRC — KEEP | Conviction 3/5 (binary).** Thursday decides it. Sized at the 15% cap, override stop $30.50, max loss 1.9% of equity vs +30%+ on a beat-and-raise. Post-print: strip the override within 1 trading day.

**Overall portfolio thesis.** **Scoreboard: gap −4.4%; process check: TWR alpha +0.77% and improving.** This revision replaces the corruption-forced No-Candidates finding with a real evaluation of a clean watchlist — and the answer is the same *hold*, but earned: the two credible new names (**CADL**, queued for the Health Care slot; **HTLD**, queued post-print) are both better entered after Thursday's double print resolves, the two hottest tapes (XNCR, GPRE) are day-2 breakouts the entry rules explicitly refuse to chase, and the rest of the list fails on excluded classes (BDCs), dead arb spreads (CCRN), or catalyst-free momentum. Week 44's PHAT stop-out is the fresh reminder of what buying heat costs. The book enters the gauntlet unchanged: 5 positions, every stop verified, 24.8% cash, one written playbook per print — with a **ranked redeployment queue (CADL → HTLD → XNCR/GPRE on consolidation)** ready for Week 46.

---

## 10. CONFIRM CASH AND CONSTRAINTS

- **Cash:** $177.29 (24.8%) — unchanged, above the 15% floor ✓
- **Positions:** 5 of max 6 ✓ · Sector caps ✓ · Binary plays 1/1 ✓ · Largest position 19.0% < 30% ✓
- **Regime:** GREEN (IWM $294.04 > 50-day $290.80) ✓ · PHAT blackout respected ✓
- **Entry-discipline tests documented** for every rejected candidate ✓ · All five stops live at the broker ✓

---

*Week 45 Full report Rev. 2 generated 2026-07-20 by Claude Code (Neutral posture). Prices as of 2026-07-17 close; screener corrected and candidates verified 2026-07-20.*
