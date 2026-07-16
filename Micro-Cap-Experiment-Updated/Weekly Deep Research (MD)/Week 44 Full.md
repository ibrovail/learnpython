# Week 44 Deep Research — Full Report

**Date:** Sunday, July 12, 2026
**Week:** 44 of 52 (experiment ends 2026-09-18 — 10 calendar weeks remaining)
**Session posture:** Wide net · 30–60 day catalysts · **Aggressive** · max 6 positions
**Prepared by:** Claude Code (Deep Research Mode)

---

## 1. RESTATED RULES

- **Long-only, full shares, no options/margin/shorting.** U.S.-listed common stock, nano- to small-cap (≤$2B market cap). No OTC, ETFs, SPACs, ADRs, defence, or Israeli-affiliated names.
- **Cash reserve floor: 15% of equity.** No new capital beyond the snapshot ($285.74 cash). Track to the cent.
- **Position sizing:** risk-per-trade ≤5% of equity (`shares = equity×0.05 / (entry − stop)`); no single name >30% of equity.
- **Binary event plays:** max **1** at a time, ≤15% of equity; stop may use the binary-event override (technical support instead of ATR), auto-expiring within 1 trading day of the event.
- **Momentum plays** (from screener): no dated catalyst required, must be **above the 20-day SMA at entry**, standard stop `max(1.5×ATR, 10% below entry)`, minimum 5-day hold. Max 3–4 momentum positions.
- **Sector cap:** no more than 2 positions in the same GICS sector.
- **Market regime filter:** if IWM < 50-day SMA, freeze new momentum initiations. **Status this week: GREEN** (IWM 295.99 vs 50-day SMA ≈279.06 → new initiations permitted).
- **Entry discipline:** post-earnings cooldown, distance-from-base limits, ATR-based stops, pre-open verification, screener score = sourcing not conviction (start 2/5).
- **One open order per stock:** the protective GTC stop-limit occupies the single order slot; profit/exit targets are alerts, not resting orders.
- **Verification:** every ticker/catalyst confirmed by ≥2 independent sources or flagged INSUFFICIENT CONFIRMATION.

---

## 2. RESEARCH SCOPE

**Data retrieved:** 2026-07-12, ~16:20 ET (weekend session).
**Portfolio/market data:** `make weekend` snapshot (prices as of Friday 2026-07-10 close), screener watchlist generated 2026-07-12 (1,134-name Finviz universe → yfinance signals → 15 ranked candidates).

**Live sources consulted (WebSearch, 2026-07-12):**
- **Regime:** IWM 50-day SMA (Investing.com / Barchart technicals) — 279.06; IWM close 295.99 → above.
- **Holdings:** WKC (SEC 8-K, rutlandherald, MarketBeat), SHO (SEC 8-K, StockTitan, TradingView), TDAY (SEC 8-K, stockanalysis, Insider Monkey), TILE (SEC 8-K, Motley Fool transcript, WallStreetZen).
- **Candidates:** ATRC (StockTitan, MarketBeat, WallStreetZen, Investing.com), PHAT (SEC 8-K, StockTitan, ainvest, kavout), OPRT (SEC 8-K, TipRanks, MarketBeat), BLFS (public.com, MarketBeat, stockanalysis), TARA (SEC 8-K, StockTitan, Protara IR).
- **Screener rejects:** TOI, SVC, CNMD, ESPR, MLTX, CDZI, IART, MXCT, EYPT, HNST, PDO, NCI (watchlist metrics + sector-cap logic).

**Checks performed:** regime filter, sector-cap map, market-cap ceiling (ATRC $1.7B ✓), liquidity (ADV), binary-event count, 15% floor, per-position 30% cap, entry-discipline cooldown/distance.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction (1-5) | Status |
|--------|------|-----------|----------|---------------|--------------|------------------|--------|
| WKC | Energy — deep-value runner | 2026-05 | $26.00 | $35.32 (+35.8%) | $29.20 / $29.05 | 4/5 | KEEP — 1-sh runner post +30% partial; earnings 7/23; **raise stop** |
| SHO | Real Estate — quality REIT | 2026-06 | $10.20 | $11.22 (+10.0%) | $10.90 / $10.80 | 3/5 | KEEP — at fair value; earnings 8/6; stop does the work |
| TDAY | Comm. Services — AI-licensing | 2026-06 | $8.15 | $8.51 (+4.4%) | $7.20 / $7.10 | 4/5 | KEEP — book leader; **raise stop**; Q2 late-July |
| TILE | Cons. Disc. — beat-and-raise | 2026-06 | $32.10 | $33.26 (+3.6%) | $31.00 / $30.85 | 4/5 | KEEP — earnings 7/31; Buy, PT $36–37 |

**Portfolio equity:** $713.68 · **Cash:** $285.74 (40.0%) · **S&P-equivalent:** $760.33 · **Gap:** −6.1% · **TWR alpha (cum):** −1.35% (now trailing).

---

## 4. CANDIDATE SET

| Ticker | One-Line Thesis | Key Catalyst | Catalyst Confirmation Status | Liquidity Note |
|--------|-----------------|--------------|------------------------------|----------------|
| **ATRC** | AtriCure — cardiac ablation/AtriClip leader guiding first-ever profitable year; PT $49.60 vs ~$34 | **Q2 earnings 2026-07-23** (date-certain binary) | CONFIRMED (StockTitan + MarketBeat + company IR) | $1.7B cap ✓; ADV ~$17M/day; order <0.01% ADV |
| **PHAT** | Phathom — VOQUEZNA revenue +104% YoY, operating profitability guided Q3 2026; PT $24 vs $12.61 (+90%) | Voquezna ramp → Q3 profitability; Q2 print (date TBA) | Ramp CONFIRMED (SEC 8-K + StockTitan + ainvest); **Q2 date INSUFFICIENT CONFIRMATION** → treated as momentum play | ADV ~$15–25M/day; vol ratio 2.1x; order <0.01% ADV |
| OPRT | Oportun — consumer lender, reiterated Q2 guide, PT $8.50 (+48%) | Q2 earnings 2026-08-12 | CONFIRMED (SEC 8-K + TipRanks) | Alternate; subprime credit-cycle risk (12.2% NCO) |
| BLFS | BioLife — bioprod./cryo, Strong Buy, PT $30.86 (+19%) | Q2 earnings ~early Aug | Partial (date unconfirmed) | Superseded by PHAT (lower upside, off this week's screener) |
| TARA | Protara — TARA-002 68% CR NMIBC; strong data | ADVANCED-2 / THRIVE-3 readouts "2H 2026" | Vague timeline — **no date-certain catalyst** | Momentum-eligible but lower conviction; blocked by HC cap |

**Screener candidates evaluated & not selected (one line each):**
- **TARA** (#1) — best score but all catalysts vaguely dated "2H 2026"; healthcare-cap blocked once ATRC+PHAT taken.
- **TOI** (#4, HC) — thin-margin oncology services; blocked by 2-name healthcare cap.
- **SVC** (#5, Real Estate) — hotel/net-lease REIT with dividend-cut history and high leverage; SHO already fills Real Estate.
- **CNMD** (#6, HC) — solid medtech but healthcare-cap blocked; less upside than ATRC.
- **ESPR** (#7) — squeeze setup (BBW 0.016) but flat momentum, RS negative; no near catalyst.
- **MLTX / EYPT / MXCT / IART** (HC) — healthcare-cap blocked.
- **CDZI** (Utilities), **HNST** (Cons. Def.), **PDO** (closed-end fund → **excluded by rules**), **NCI** ($28M cap, thin) — lower conviction / rule-excluded.

---

## 5. PORTFOLIO ACTIONS

- **Keep:** WKC — 1-share runner, de-risked by the +30% partial; hold through the 7/23 print on a raised stop.
- **Keep:** SHO — quality operator at fair value; the $10.90 stop is the risk manager into 8/6 earnings.
- **Keep:** TDAY — book leader, AI-licensing inflection (Q1 adj. EBITDA +45%); raise stop to trailing level.
- **Keep:** TILE — beat-and-raise breakout into 7/31 earnings; Buy, PT $36–37; stop $31 stays.
- **Raise stop:** WKC $29.20 → **$32.50 / $32.35** — locks +25% ahead of the binary 7/23 print (trades 23% above the highest analyst PT of $33). *Re-place at broker* — this slot was cancelled to execute the partial.
- **Raise stop:** TDAY $7.20 → **$7.55 / $7.45** — trailing rule (15% below the 20-day high of $8.86).
- **Initiate:** **ATRC** — 3 shares (~14.5% of equity) — the queued #1 name; Strong Buy, first profitable year, PT $49.60; binary-event play into 7/23 earnings.
- **Initiate:** **PHAT** — 5 shares (~9.0% of equity) — screener rank 2 momentum play; Voquezna +104% YoY, Q3 profitability; PT +90%.

This takes the book from 4 → **6 positions** (the aggressive-posture target), deploying ~$168 of the 40% cash while holding the reserve at 16.5%.

---

## 6. EXACT ORDERS

**Order 1 — Raise WKC protective stop (re-place; no buy/sell)**
- **Action:** modify stop (GTC stop-limit)
- **Ticker:** WKC
- **Shares:** 1 (runner)
- **Stop Loss:** $32.50 — locks +25% ahead of the 7/23 binary print; WKC trades 23% above the top analyst PT
- **Stop Limit:** $32.35
- **Time in Force:** GTC
- **Special Instructions:** This is a re-placement — the WKC stop was cancelled to execute the 7/10 +30% partial and is currently NOT live. Re-arm at the new level.
- **Rationale:** Protect the runner's gain into an earnings event where the name is already well above consensus.

**Order 2 — Raise TDAY protective stop**
- **Action:** modify stop (GTC stop-limit)
- **Ticker:** TDAY
- **Shares:** 16
- **Stop Loss:** $7.55 — 15% below the 20-day high ($8.86), standard trailing rule
- **Stop Limit:** $7.45
- **Time in Force:** GTC
- **Rationale:** Ratchet protection on the book leader without whipsaw-tight placement.

**Order 3 — Initiate ATRC (binary-event play)**
- **Action:** buy
- **Ticker:** ATRC
- **Shares:** 3
- **Order Type:** limit
- **Limit Price:** $34.50
- **Time in Force:** DAY
- **Intended Execution:** 2026-07-13 (Monday)
- **Stop Loss:** $30.50 — **binary-event override**: recent-range support (~$30) / below the $32 consolidation floor, not ATR. Auto-expires within 1 trading day of the 7/23 print (recalculate to standard trailing stop).
- **Stop Limit:** $30.30
- **Special Instructions:** **Pre-open verification required** — check Monday pre-market before executing. If ATRC is down >2% pre-market, drop the limit to the pre-market price or pass. Hold-through-earnings binary risk is acknowledged and sized within the 15% cap.
- **Rationale:** Queued #1 catalyst name; Strong Buy, first profitable year, PT $49.60 (+44%) into a dated 7/23 print.

**Order 4 — Initiate PHAT (momentum play)**
- **Action:** buy
- **Ticker:** PHAT
- **Shares:** 5
- **Order Type:** limit
- **Limit Price:** $12.85
- **Time in Force:** DAY
- **Intended Execution:** 2026-07-13 (Monday)
- **Stop Loss:** $11.55 — ~10% below entry (standard momentum stop; wider of 1.5×ATR / 10%)
- **Stop Limit:** $11.45
- **Special Instructions:** Pre-open check; if PHAT gaps >2% below $12.61 pre-market, reduce to a limit at the pre-market price. Minimum 5-day hold applies.
- **Rationale:** Screener rank 2; VOQUEZNA revenue +104% YoY, Q3 operating profitability guided, PT $24 (+90%).

---

## 7. RISK AND LIQUIDITY CHECKS

**Post-trade concentration (% of $713.68 equity):**

| Holding | Value | % of Equity |
|---------|-------|-------------|
| SHO | $123.42 | 17.3% |
| TDAY | $136.16 | 19.1% |
| TILE | $133.04 | 18.6% |
| ATRC | $103.50 | 14.5% |
| PHAT | $64.25 | 9.0% |
| WKC (runner) | $35.32 | 4.9% |
| **Cash** | **$117.99** | **16.5%** |

- No single position exceeds 30%. ✓
- **Cash after trades: $117.99 = 16.5%** — above the 15% floor. ✓
- **Binary-event plays: 1** (ATRC only; PHAT is a momentum play). ✓
- **Sector map:** Energy (WKC), Real Estate (SHO), Comm. Services (TDAY), Cons. Disc. (TILE), **Health Care ×2 (ATRC, PHAT)** — at the 2-name cap, not exceeded. ✓
- **Per-order ADV:** ATRC $103.50 and PHAT $64.25 are each <0.01% of daily dollar volume — no slippage concern (guard is 5% of ADV). ✓

**Sizing math shown:**
- ATRC risk: 3 × ($34.50 − $30.50) = **$12.00 = 1.68% of equity** (binary cap $107.05 → 3 sh binds before the 5% risk budget).
- PHAT risk: 5 × ($12.85 − $11.55) = **$6.50 = 0.91% of equity** (cash/floor constraint binds; full 5%-risk sizing would be 27 sh, far above the reserve).

---

## 8. MONITORING PLAN

- **ATRC** — 7/23 earnings is the event. Watch for pre-print drift toward the $30.50 stop; on the print, remove the binary override within 1 trading day and reset to a trailing stop. Watch competitive-risk headlines (analysts flagged new competition).
- **PHAT** — confirm the Q2 earnings date once announced (likely early Aug); watch VOQUEZNA prescription trend and any payer-coverage news. Honor the 5-day minimum hold; stop $11.55.
- **WKC** — 7/23 earnings; the raised $32.50 stop is the plan. If it rips post-print, trail up; if it fades, the stop harvests +25%. Confirm the re-placed stop is live at the broker.
- **TDAY** — Q2 print late July; watch for a hold above $8.80 to justify a further stop-raise toward ~$7.80. AI-licensing deal-flow is the thesis driver.
- **SHO** — 8/6 earnings; watch RevPAR/AFFO vs the raised guidance. Stop $10.90 locks +6.9%.
- **TILE** — 7/31 earnings; breakout intact above $33. Trail up on new highs above $36; stop $31 until then.
- **Regime** — re-check IWM vs 50-day SMA next weekend; a break below freezes new momentum adds.

---

## 9. THESIS REVIEW SUMMARY

*(This section is saved separately as `Week 44 Summary.md`.)*

**WKC — KEEP | Conviction 4/5.** A 1-share runner after the rule-mandated +30% partial, now +35.8%. Trades 23% above the highest analyst PT ($33) on a Hold consensus, so the raised **$32.50 stop** (locking +25%) — re-placed at the broker after the partial cancelled it — is the right posture into the 7/23 print. Let the print prove itself; the stop harvests either way.

**SHO — KEEP | Conviction 3/5.** Quality operator at fair value ($11.08 PT vs $11.22). No new catalyst until 8/6 earnings; the $10.90 stop (locking +6.9%) does the risk work. Raise only on a decisive push through $12.

**TDAY — KEEP | Conviction 4/5.** The book's leader — Q1 adjusted EBITDA +45%, AI-licensing deals accretive, digital-only subscription growth returning. Stop raised to $7.55 (trailing). A hold above $8.80 into the late-July print unlocks a further raise toward $7.80.

**TILE — KEEP | Conviction 4/5.** Beat-and-raise breakout holding near highs into 7/31 earnings; Buy consensus, PT $36–37 (+9%). Stop $31 keeps it loss-free; trail above $36.

**ATRC — INITIATE | Conviction 3/5 (catalyst play).** The long-queued #1 name finally funded. AtriCure guides its first profitable year; Strong Buy, average PT $49.60 (+44%) into a date-certain 7/23 earnings print. Sized at the 15% binary cap (3 sh) with a support-based override stop ($30.50) that auto-expires post-print. Pre-open verification Monday. Bear case: pre-print binary risk and newly-flagged competitive pressure — hence the capped size and the wide-but-defined stop.

**PHAT — INITIATE | Conviction 3/5 (momentum play).** Screener rank 2, and the fundamentals back the tape: VOQUEZNA revenue +104% YoY, 1.1M scripts filled, operating profitability guided for Q3 2026, PT $24 (+90%). Entered as a momentum play (earnings date not yet dated) above its 20-day SMA with a standard 10% stop. Bear case: single-drug dependency, still pre-profit until Q3, and a hot 20-day move (+22.8%) that could mean-revert — sized modestly (9%) to respect that.

**Overall portfolio thesis.** Week 44 of 52, and for the first time since the pivot we are **trailing** (TWR alpha −1.35%, gap −6.1%). Per the aggressive directive, this is the week to **stop hoarding the 40% reserve and put it to work** — the S&P rebound that eroded our lead won't be out-defended from cash. The plan deploys ~$168 into the two highest-conviction queued/screened names (ATRC catalyst + PHAT momentum), taking the book to the full 6 positions while keeping the reserve at 16.5%. Two winners (WKC, TDAY) get raised stops to protect gains into their prints — aggression on offense, discipline on defense. Ten weeks remain: this is the redeployment the queue was built for.

---

## 10. CONFIRM CASH AND CONSTRAINTS

- **Starting cash:** $285.74
- **ATRC buy:** −$103.50 (3 × $34.50)
- **PHAT buy:** −$64.25 (5 × $12.85)
- **Ending cash:** **$117.99 (16.5% of equity)** — above the 15% floor ✓
- **Positions:** 6 of max 6 ✓
- **Sector cap:** Health Care ×2 (at cap, not exceeded); all others ≤1 ✓
- **Binary-event plays:** 1 (ATRC) ✓
- **Largest position:** TDAY 19.1% (< 30% cap) ✓
- **Regime filter:** GREEN (IWM > 50-day SMA) ✓
- **Liquidity:** all orders <0.01% of ADV ✓
- **All stops set/raised on every holding** ✓

**Note:** Stop changes and buys are recommendations. They become portfolio reality only after you execute them at the broker (with actual fills) — then I log them. ATRC/PHAT limits assume Monday execution with pre-open verification; report actual fills back.

---

*Week 44 Full report generated 2026-07-12 by Claude Code (Aggressive posture). Prices as of 2026-07-10 close; catalysts confirmed 2026-07-12 via WebSearch.*
