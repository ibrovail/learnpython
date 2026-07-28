# Week 46 Deep Research — Full Report

**Date:** Sunday, July 26, 2026
**Week:** 46 of 52 (experiment ends 2026-09-18 — 8 calendar weeks remaining)
**Session posture:** Wide net · 30–60 day catalysts · **Aggressive** · max 6 positions
**Prepared by:** Claude Code (Deep Research Mode)

---

## 1. RESTATED RULES

- **Long-only, full shares, no options/margin/shorting.** U.S.-listed common stock ≤$2B market cap; existing names above $2B may be held or sold but **no new shares added**.
- **Cash floor 15% of equity** ($107.40 today); no new capital. Track to the cent.
- **Sizing:** risk ≤5% of equity per trade; ≤30% per name; binary-event plays max 1 at ≤15%.
- **Momentum plays** (screener-sourced) must be above the 20-day SMA at entry; ≤20% above the 20-day SMA and ≤40% above the 50-day; 5-day minimum hold. Catalyst plays need a dated catalyst within 60 days.
- **Entry discipline** (`.claude/rules/entry-discipline.md`): no buys in days 1–3 of a >+10% breakout; screener score is sourcing (2/5 floor), not conviction; post-earnings cooldown.
- **Partial profits:** +30% / +60%; deferral needs a new catalyst, conviction ≥4/5, and a trailing stop locking ≥15% (at +30%) / ≥40% (at +60%).
- **Price integrity** (`.claude/rules/price-data-integrity.md`): settled closes from the daily run; live/AH/pre-market only via the browser tool with a timestamp; order-side check before naming any stop/limit.
- **Sector cap** ≤2 per GICS sector; **regime filter** (IWM vs 50-day SMA); **one open order per stock**; **10-day re-entry ban** (PHAT clears ~7/27; HTLD not banned but thesis-dead).

---

## 2. RESEARCH SCOPE

**Data retrieved:** 2026-07-26 ~22:50 ET. Holdings/closes as of Friday 2026-07-24 (settled, yfinance via the daily run). Screener regenerated 2026-07-26 — **clean run, corruption fix holding** (sensible sectors/caps, no `:::` artifacts). WebSearch used only for catalysts/fundamentals/dates, never for live prices.

**Live sources (WebSearch, 2026-07-26):**
- Candidates: ORIC (Yahoo/StockTitan/MarketBeat — Himalayas-1 Phase 3 + Bayer, PT $19–21), ARDT (Simply Wall St/MarketBeat — profitable, FY EPS $0.90–1.27), CERT (StockTitan/Simply Wall St — deteriorating guide, −58% YoY), LXU (Motley Fool/StockTitan/Simply Wall St — nitrogen tailwind, Q1 NI $19.7M), HTLD (FreightWaves/Nasdaq — Q2 miss confirmed).
- Holdings: ATRC + WKC prints resolved (SEC 8-Ks + Markets Daily, verified 7/23); browser tool used 7/23 for live AH quotes (ATRC $31.47, WKC $38.60).

**Checks:** regime, universe ceiling (WKC now at $2.0B — hold-only), excluded classes, liquidity (screener $500K ADV floor), sector map, entry-discipline distance/breakout tests, cash floor, partial-profit thresholds.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|-----------|----------|---------------|--------------|------------------|--------|
| WKC | Energy — runner (at $2B cap) | 2026-05 | $26.00 | $38.14 (+46.7%) | $35.00 / $34.85 | 4/5 | KEEP — 72% EPS beat; raise stop; +60% partial at $41.60 |
| SHO | Real Estate — quality REIT | 2026-06 | $10.20 | $11.74 (+15.1%) | $10.90 / $10.80 | 3/5 | KEEP — earnings 8/6; $12 stop-raise trigger |
| TDAY | Comm. Services — AI-licensing | 2026-06 | $8.15 | $8.40 (+3.1%) | $7.55 / $7.45 | 4/5 | KEEP — **earnings 7/30** |
| TILE | Cons. Disc. — beat-and-raise | 2026-06 | $32.10 | $32.98 (+2.7%) | $31.00 / $30.85 | 4/5 | KEEP — **earnings 7/31** |
| ATRC | Health Care — post-catalyst | 2026-07-13 | $34.30 | $35.04 (+2.2%) | $30.50 / $30.30 | **4/5 ↑** | KEEP — beat-and-raise **held**; retire override, raise stop |

**Equity:** $716.01 · **Cash:** $177.29 (24.8%) · **Gap (scoreboard): −3.8%** ($716.01 vs $743.93 S&P-equivalent) · TWR alpha (process): **+1.48%** — recovery high.

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |
|--------|-----------------|--------------|------------------------------|----------------|
| **LXU** | LSB Industries — nitrogen/fertilizer cyclical; Q2 adj. EBITDA guided "meaningfully higher"; El Dorado CCS optionality | Q2 earnings ~early Aug; sustained nitrogen pricing into 2027 | CONFIRMED (Motley Fool transcript + StockTitan) | $881M cap ✓; screener vol ratio 1.33; passed $500K ADV floor |
| ORIC | ORIC Pharma — mCRPC + EGFRex20 oncology; Bayer collab; PT $19–21 (+70%) | Himalayas-1 Phase 3 *initiated*; data "2H 2026" (vague) | CONFIRMED but forward catalyst undated | $1.2B ✓; **+16%/5d vertical move** |
| ARDT | Ardent Health — profitable hospital operator; FY EPS $0.90–1.27 | Q2 earnings ~mid-Aug | Confirmed (approx. date) | $1.5B ✓; rank 1; calm base (+9%/20d) |

**Evaluated & not selected (one line each):**
- **ORIC** (#3) — best upside on the board, but +16% in 5 days is a vertical spike; entry-discipline + the fresh PHAT day-1 stop-out say don't chase heat. **Queued #1 on a consolidation/higher-low base.**
- **ARDT** (#1) — quality and cleanly-based, but it's a second Health Care name (ATRC holds the first) and a low-beta hospital operator — weakest fit for an *aggressive, widen-the-lead* mandate. **Queued #2 (the defensive alternative).**
- **AVNS** (#2) — BB-width 0.012 squeeze but zero momentum (+0.2%/20d); no catalyst in window.
- **KREF** (#4) — mortgage REIT; would sit alongside SHO in Real Estate and adds credit-cycle risk for little torque.
- **FTRE** (#5, HC) — Fortrea CRO turnaround near $1.9B cap; HC-slot contention behind ORIC/ARDT and no near dated catalyst.
- **CFFN** (#6) — sleepy thrift, +2%/20d; no catalyst.
- **AZTA** (#7, HC) — HC-slot contention; modest setup.
- **WKC** (#9) — already held; at the $2B cap, so **no new shares** regardless.
- **BLFS** (#10, HC) — the old Week-43 queue name; only +4.6%/20d now and HC-slot contention — superseded.
- **CERT** (#11) — momentum is a dead-cat bounce: −58% YoY, guidance cut to a loss, CFO resigned. Value trap; pass.
- **ARLO** (#12) — the Week-34 disaster name; below-average confidence flag, no catalyst; hard pass.
- **PAYS / ELME / HNST** (#13–15) — sub-scale or catalyst-free; pass.

**Net: initiate LXU (the diversifying materials cyclical); queue ORIC (torque, on consolidation) and ARDT (quality) behind it.**

---

## 5. PORTFOLIO ACTIONS

- **Keep:** SHO, TDAY, TILE — unchanged; catalysts 8/6, 7/30, 7/31 respectively.
- **Keep + raise stop:** **WKC** → $36.40 / $36.25 (locks +40.0%). The 72% EPS beat and ~20% guidance raise justify riding the runner, but it trades well above the $28.75 analyst consensus and now sits at the $2B universe cap, so protection tightens. This stop also makes the **+60% partial ($41.60) deferrable** (trailing lock ≥40% + conviction 4/5) if it gets there.
- **Keep + re-rate + raise stop:** **ATRC** → $32.50 / $32.35, conviction **3/5 → 4/5**. The beat-and-raise (revenue beat, first profitable quarter, FY EPS guide roughly doubled to $0.24–0.32) **held through the AH selloff** and closed +6.2% / above entry. Per the post-catalyst rule the $30.50 binary override is retired; $32.50 locks the worst case at −5.2% (from −11%) while leaving room for post-print volatility (AH ranged $31.47–$35.59).
- **Initiate:** **LXU** — 5 shares (~$61, ~8.6% of equity) — diversifying Basic Materials cyclical with a nitrogen-price tailwind and CCS optionality; the 15% cash floor caps the size, not conviction.
- **Trim / Exit:** none.

---

## 6. EXACT ORDERS

**Order 1 — Raise WKC stop**
- Action: modify stop (GTC stop-limit) · Ticker: WKC · Shares: 1
- Stop Loss: **$36.40** — locks +40.0%; ~4.6% below the $38.14 close, above the pre-breakout base
- Stop Limit: **$36.25**
- Order-side check: $36.40 < $38.14 settled close ✓
- Special: whipsaw risk acknowledged (≈1.3×ATR below close) — this is a deliberate protect-the-runner choice on a 1-share position at the universe cap. Looser alternative $35.80 (+37.7%) if you prefer retest room.
- Rationale: bank the outsized gain and unlock partial-deferral optionality into any push toward $41.60.

**Order 2 — Raise ATRC stop (retire binary override)**
- Action: modify stop (GTC stop-limit) · Ticker: ATRC · Shares: 3
- Stop Loss: **$32.50** — post-catalyst reset; ~7.2% below the $35.04 close (slightly looser than the ~$33.24 trailing-formula level, to absorb post-print volatility)
- Stop Limit: **$32.35**
- Order-side check: $32.50 < $35.04 settled close ✓
- Rationale: event resolved favorably; tighten from the override while respecting AH volatility.

**Order 3 — Initiate LXU**
- Action: buy · Ticker: LXU · Shares: **5**
- Order Type: limit · Limit Price: **$12.35** (≈ Friday close $12.24 + small buffer)
- Time in Force: DAY · Intended Execution: 2026-07-27 (Monday)
- Stop Loss: **$11.00** — ~10% below entry (wider of 1.5×ATR / 10%, momentum-play default)
- Stop Limit: **$10.90**
- Special: **pre-open verification** — check Monday pre-market via the browser tool; if LXU gaps >2% below $12.24, drop the limit to the pre-market price or pass. 5-day minimum hold applies.
- Order-side check: stop $11.00 < intended entry $12.35 ✓
- Rationale: nitrogen-price cyclical with a confirmed EBITDA inflection and CCS optionality; diversifies into a fresh sector at ≤1% equity risk.

---

## 7. RISK AND LIQUIDITY CHECKS

Post-trade (assuming LXU fills at $12.35):

| Holding | Value | % of Equity |
|---------|-------|-------------|
| WKC | $38.14 | 5.3% |
| SHO | $129.14 | 18.0% |
| TDAY | $134.40 | 18.8% |
| TILE | $131.92 | 18.4% |
| ATRC | $105.12 | 14.7% |
| LXU | $61.75 | 8.6% |
| **Cash** | **$115.54** | **16.1%** |

- No position >30% ✓ · **Cash 16.1% > 15% floor** ✓ (floor is the binding constraint on deployment size)
- Sectors: Energy, Real Estate, Comm. Services, Cons. Disc., Health Care, Basic Materials — **all ≤1 per sector** ✓
- Binary-event plays: 0 (ATRC's resolved) ✓
- LXU order $61.75 is a negligible fraction of its daily dollar volume (screener-confirmed >$500K ADV) — no slippage concern ✓
- LXU stop-risk: 5 × ($12.35 − $11.00) = $6.75 ≈ 0.94% of equity ✓
- Aggregate stop-risk if all six stops fired: ≈ −4.0% of equity.

---

## 8. MONITORING PLAN

- **TDAY — Thu 7/30 earnings:** the next print. Citizens PT $10; a beat + close >$8.86 re-arms the stop-raise. Pre-print check in Wednesday's daily.
- **TILE — Fri 7/31 earnings:** Strong Buy, PT $36–37; trail above $36 on a beat.
- **WKC:** price alert at **$41.60** (+60% partial). At 1 share this is an exit-or-hold decision, not a divisible partial — decide before it triggers; my lean is defer-and-trail (guidance just raised) given the $36.40 stop now locks +40%.
- **ATRC:** post-print drift; the $32.50 stop is the line. Watch AtriClip-vs-Edwards share commentary in sell-side notes.
- **LXU:** confirm Monday pre-market before entry; then watch the Q2 date announcement and nitrogen spot pricing. Honor the 5-day min hold.
- **SHO:** raise stop toward ~$11.10 on a push through $12 (1.7% away); earnings 8/6.
- **ORIC / ARDT:** consolidation watch for a Week-47 entry if a slot opens or cash builds.
- **Regime:** IWM $291.17 vs 50-day ~$290.80 — cushion back to a razor-thin ~0.1%. A close below $290.80 freezes new momentum initiations; LXU should be entered Monday before any such freeze, or re-evaluated if IWM gaps down.

---

## 9. THESIS REVIEW SUMMARY

*(Saved separately as `Week 46 Summary.md`.)*

**WKC — KEEP | 4/5.** The near-stop-out that became the book's champion: a 72% EPS beat and ~20% guidance raise took it to a 52-week high (+46.7%). Now at the $2B cap (hold-only) and far above consensus, so the stop climbs to $36.40 (+40% locked), which also makes the $41.60 partial deferrable. Ride it on a tight leash.

**SHO — KEEP | 3/5.** +15.1%, grinding toward the $12 stop-raise trigger; earnings 8/6. The $10.90 stop manages the tail.

**TDAY — KEEP | 4/5.** Book leader into its 7/30 print; Citizens PT $10. Stop $7.55; next raise on a close >$8.86.

**TILE — KEEP | 4/5.** Beat-and-raise breakout into 7/31; stop $31.00 keeps it loss-free.

**ATRC — KEEP | 4/5 ↑.** The thesis printed: revenue beat, first profitable quarter, FY EPS guide roughly doubled — and it *held* through an after-hours head-fake that briefly showed −4.6% before closing +6.2%. That AH quote (which a stale search snippet nearly turned into a bad stop) is exactly why price integrity now runs through the browser tool. Override retired, stop up to $32.50, conviction to 4/5.

**LXU — INITIATE | 3/5 (momentum + macro).** LSB Industries brings a nitrogen-price tailwind (supply disruption sustaining pricing into 2027), a guided Q2 EBITDA inflection, CCS optionality, and — the tiebreaker — a *sixth sector*, keeping the book diversified rather than doubling into healthcare. Entered on a calm base (+12.7%/20d, not vertical), 10% stop, ~0.9% equity risk.

**Overall portfolio thesis.** **Scoreboard: gap −3.8%, the best of the recovery; process check: TWR alpha +1.48%, a new high.** The catalyst gauntlet the last three weeks were built around is through, and it broke our way — WKC to a 52-week high, ATRC's profitability inflection confirmed and held, zero stops fired. Per the Aggressive directive this is a deploy week, but two facts discipline the aggression: the 15% cash floor leaves only ~$70 of true dry powder, and the screener's hottest names are either binary biotech (ORIC) or value traps (CERT). So the one initiation is the *quality-torque* compromise — LXU's real-earnings cyclical over ORIC's vertical biotech chase, with ORIC queued for a consolidation entry. Two winners get tighter stops; the book goes to the full 6 positions across 6 sectors. Eight weeks left: the lead is nearly even and the process is finally ahead — now compound it on a full-sized base without giving back the catalyst gains.

---

## 10. CONFIRM CASH AND CONSTRAINTS

- **Starting cash:** $177.29 · **LXU buy:** −$61.75 → **Ending cash: $115.54 (16.1%)** — above the 15% floor ✓
- **Positions:** 6 of max 6 ✓ · **Sectors:** all ≤1 (no cap breach) ✓ · **Binary plays:** 0 ✓ · **Largest position:** TDAY 18.8% < 30% ✓
- **Regime:** GREEN but razor-thin (IWM $291.17 > 50-day ~$290.80 by ~0.1%) — flagged; enter LXU Monday before any freeze ✓
- **Universe:** WKC at $2.0B logged as hold-only (no adds) ✓ · **Re-entry:** PHAT clears ~7/27; HTLD thesis-dead (confirmed Q2 miss) ✓
- **Price integrity:** all order levels order-side-checked against Friday's settled closes ✓
- **Note:** orders are recommendations — they become portfolio reality only after broker execution with actual fills, which I then log. Report Monday's LXU fill and confirm the two stop raises.

---

*Week 46 Full report generated 2026-07-26 by Claude Code (Aggressive posture). Closes as of 2026-07-24; catalysts verified 2026-07-26 via WebSearch; live prices via browser tool per price-data-integrity.md.*
