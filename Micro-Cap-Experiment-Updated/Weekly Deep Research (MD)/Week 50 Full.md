# Week 50 Deep Research — Micro-Cap Portfolio

**Date:** Monday, 2026-08-24 | **Week:** 50 of 52 | **Runway:** ends 2026-09-18 (~3.5 weeks)
**Session directives:** Sector = Wide net · Catalyst timing = Within 5 days · Risk posture = Tighten stops · Max positions = 5

---

## 1. RESTATED RULES

- **Universe:** US-listed common stock, market cap ≤ $5B (amended Week 49 from $2B). No ETFs, CEFs/BDCs, SPACs, ADRs, units/warrants.
- **Cash floor: 15% of equity — UNCHANGED and explicitly reaffirmed in the Week 49 amendments.** This is the binding constraint this week (see §7).
- **Risk-per-trade:** size so a stop-out costs ≤5% of equity. **Single-name ceiling: 30% of equity.**
- **Stops mandatory on every long**, default max(1.5×ATR(14), 10% below entry). Never lowered.
- **Range check before any stop level:** must sit below the most recent session's low AND ≥1.5×ATR below the reference price (target 1.75×). De-risk by cutting size, never by tightening into the noise band.
- **Sector cap:** max 2 positions per GICS sector; healthcare amended to 3 (Week 49).
- **PRV gate:** browser-fetch the live quote page + news feed before ANY buy/sell/trim/add. Applies equally to exits.
- **Entry discipline:** post-earnings cooldown, ≤40% above 50-day SMA, ≤20% above 20-day SMA, thesis-input freshness, screener score = sourcing not conviction (starts 2/5).
- **Partial profits:** ~1/3 at +30%, ~1/3 at +60%; deferrable when all four criteria hold.
- Full shares only. No averaging down >5% below entry without a confirmed new catalyst (≥2 sources).

---

## 2. RESEARCH SCOPE

**Data retrieved Monday 2026-08-24, 07:00–07:25 AM EDT (pre-market).**

| Check | Method | Result |
|---|---|---|
| Settled OHLC/volume | `trading_script.py` daily run (yfinance), through Fri 2026-08-21 | Complete |
| ATR(14), swing lows, SMA20/50 | Computed locally from 6-month price history | Complete — never estimated by eye |
| PRV quote pages — **all 4 holdings** | Browser → `stockanalysis.com` | TILE, ATRC, PAR, CADL |
| PRV quote pages — **6 candidates** | Browser → `stockanalysis.com` | WWW, SIBN, ABUS, BEAM, CYRX, PD |
| News-feed recency | Browser → `stocktitan.net` | ATRC, WWW |
| Screener universe | `screener.py` → Finviz + yfinance | 1,604 names; 395 dropped by validation; 15 ranked |

**Data-integrity notes:**
- The Finviz parse arrived corrupted again — **100% of tickers had a doubled first character.** `_repair_ticker_corruption()` detected and repaired it automatically. The screener output is sound.
- **My local 52-week ranges were computed over a 6-month window and are wrong as "52-week" figures.** Every 52-week range quoted below comes from the browser quote page, which is authoritative. (WWW: local calc said $14.00–21.41; the real range is **$13.47–32.80** — a material difference that changes how the setup reads.)
- A staleness bug in the `make weekend` gate blocked the first run: it demanded data for *today's* session while the market was still closed. Fixed in the script (§ Addendum).

**The PRV gate produced one decision-changing find this morning** — see ATRC below.

---

## 3. CURRENT PORTFOLIO ASSESSMENT

Equity **$781.68** · Cash **$191.44 (24.5%)** · S&P-equivalent **$770.27** · **Gap +1.48%** · TWR alpha +7.91%

| Ticker | Role | Entry Date | Avg Cost | Current Price | Current Stop | Conviction | Status |
|---|---|---|---|---|---|---|---|
| ATRC | Core winner | 2026-07-15 | $34.30 | $48.98 (+42.8%) | $44.20 | 5/5 | **KEEP — raise stop** |
| TILE | Core | 2026-06-30 | $32.10 | $38.94 (+21.3%) | $37.10 | 4/5 | **KEEP — stop already at limit** |
| PAR | Growth | 2026-08-17 | $18.85 | $19.53 (+3.6%) | $16.50 | 4/5 | **KEEP — raise stop** |
| CADL | Catalyst | 2026-08-17 | $11.43 | $13.13 (+14.9%) | $12.00 | 3/5 | **KEEP — stop at ATR floor** |

### ATRC — AtriCure | Conviction 5/5 (raised from 4/5)

**PRV, 8/24 07:00 AM EDT:** $48.98 close, pre-market $48.98 (0.00%, no meaningful volume). Market cap $2.49B. Revenue TTM **$569.62M +13.9%**. Net income $10.55M, EPS $0.21. **PE 232.32 / forward PE 199.35.** 52-wk range **$25.36–49.20 — the stock closed at its 52-week high.** Beta 1.23. Consensus **Buy, PT $47.33 (−3.37%, i.e. BELOW the current price)**.

**The find:** posted **18 minutes before my fetch** — *"AtriCure price target raised to $55 from $45 at BTIG,"* Buy maintained, on the reasoning that a **new STS (Society of Thoracic Surgeons) Quality Metric could increase use** of concomitant ablation. This is a Monday pre-market upgrade on a *structural* adoption driver, and it was invisible in Friday's settled data. Consensus PT $47.33 predates it and will pull upward as it refreshes.

Seven consecutive up sessions ($42.95 → $48.98, +14%) with perfectly stair-stepped rising lows (41.65 → 46.99). News feed clean since 7/27 — no adverse company release.

**Partial-profit deferral (+30% trigger $44.59) — re-affirmed.** All four criteria hold: (1) new catalyst — FY EPS guidance doubling from the Q2 print, now reinforced by the STS metric; (2) trailing stop locks **+33.7%** at the new level, far above the +15% requirement; (3) position 18.8% of equity, under the 30% cap; (4) conviction 5/5. Per the rule, *the trailing stop replaces the partial sell as the primary risk control.* Selling a third of a three-share position into a fresh upgrade would be the exact error this book keeps flagging in reverse.

**Honest counterweight:** PE 232 and a consensus PT *below* the price are real. The stop, not the story, is what protects this gain.

### TILE — Interface | Conviction 4/5

**PRV, 8/24:** $38.94 close, pre-market **$39.20 (+0.67%)** 4:00 AM. Revenue TTM $1.44B +6.5%; net income $145.55M **+52.0%**; EPS $2.47 **+52.5%**. PE 15.77, forward PE 16.20. 52-wk **$24.40–40.50**. Beta 1.91. **Strong Buy, PT $45.25 (+16.2%).** Sector: **Consumer Discretionary** (not Industrials — corrected from prior weeks).

Thesis intact: earnings growing five times faster than revenue, i.e. genuine margin expansion. Forward PE slightly above trailing is the one soft spot — the market expects earnings growth to slow from here.

**Stop stays $37.10.** See §5 for why tightening is rejected.

### PAR — PAR Technology | Conviction 4/5

**PRV, 8/24:** $19.53 close, pre-market $19.53 (4:01 AM — session open, no meaningful volume; not a usable quote). Revenue TTM **$496.67M +18.8%** — the strongest top-line growth in the book. Net income −$72.14M, EPS −$1.77 (still GAAP-lossmaking). **Forward PE 17.82.** 52-wk **$11.59–54.62** — 64% below its high. Beta 1.31. **Buy, PT $25.31 (+29.6%)** — the largest verified upside among holdings. No adverse news; latest item is a 5-day-old marketing survey release.

The Week 49 thesis is delivering: +3.6% in four sessions, closed Friday +3.06% at $19.53.

### CADL — Candel Therapeutics | Conviction 3/5

**PRV, 8/24 07:12 AM EDT:** $13.13 close (+6.32% Friday), **pre-market $12.85 (−2.17%)** — giving back part of Friday's pop. Market cap $1.00B. No revenue; net income −$88.53M, EPS −$1.44. 52-wk **$4.35–13.82** — near the high. **Beta −0.50.** **Strong Buy, PT $20.88 (+59.0%)** — the largest upside on the board.

**A caution the headline hides:** the most recent analyst action is **BofA raising its PT to $12 from $9 — but keeping a Neutral rating, and $12 sits BELOW Friday's $13.13 close.** The Strong Buy consensus is real, but the marginal analyst is not bullish at this price. Conviction stays 3/5.

Clinical-stage binary: CAN-2409 BLA submission guided Q4 2026, cash into 2028. **This is the most likely stop-out in the book this week** — the $12.00 stop sits only 1.13×ATR below the $12.85 pre-market print.

---

## 4. CANDIDATE SET

All six evaluated through the full PRV gate — live quote page fetched for every name, not merely searched.

| Ticker | One-Line Thesis | Key Catalyst | Confirmation Status | Liquidity |
|---|---|---|---|---|
| **WWW** | Profitable footwear recovery: Merrell +11.1%, Saucony +9.9%, guidance raised on revenue, EPS, margin AND FCF | Post-print drift on a raised FY guide; KeyBanc PT $25 (Overweight) | **CONFIRMED** — Q2 release 8/13 verified on StockTitan; quote page verified 8/24 | ~$14M/day ADV — ample |
| SIBN | SI-Bone: sacroiliac medtech, revenue +15.3%, beta 0.70, fresh breakout on 2.9× volume | No dated catalyst inside window (earnings were 8/3) | Verified, but **no near-term catalyst** | ~$25M/day — ample |
| CYRX | Cryoport: cold-chain logistics for cell/gene therapy, revenue +12.1% | Q2 beat (8/6), Needham PT $19 | Verified | ~$17M/day — ample |
| ABUS | Arbutus: $230M Dutch auction tender = 22% of market cap | Tender in progress | Verified but **unrateable** | ~$94M/day — ample |
| BEAM | Beam Therapeutics: base-editing platform, PT $52.15 (+75%) | Pipeline readouts, no dated event in window | Verified | ~$73M/day — ample |
| PD | PagerDuty: incident-management software | **Earnings Aug 27 (3 days)** | Verified — and **disqualifying** | ~$27M/day — ample |

### Why each non-selected candidate was rejected

- **PD — DISQUALIFIED on fundamentals, despite being the only name with a catalyst inside the 5-day window.** Consensus **Hold**, PT **$10.14 = 17.7% BELOW** the $12.32 price; revenue TTM **+3.7%** (near-stagnant); **forward PE 9.41 vs trailing 5.82** — forward *above* trailing means earnings are expected to *fall*, and the $190.6M net income looks like a one-off tax item, not operations. Buying a Hold-rated stock trading 21% above consensus PT three days before it prints is a negative-skew coin flip. The screener ranked it #5 on momentum; momentum is sourcing, not conviction.
- **SIBN — the runner-up, and genuinely close.** Beta 0.70, Strong Buy, PT $25.00 (+25.3%), revenue +15.3%, clean rising-lows base, Friday breakout on 2.9× average volume. Rejected on two grounds: it is **lossmaking** (EPS −$0.33) where WWW earns $1.28, and it would make **healthcare 3 of 5 positions** alongside ATRC — another medical-device name, i.e. directly correlated exposure — which is the opposite of what a protect-the-lead posture wants in the final three weeks.
- **CYRX — rejected on risk/reward.** PT $19.33 is only **+12.8%** upside, and the stock closed **at its 52-week high ($17.20 = the day's high)**. Lossmaking (EPS −$0.91), beta 1.89. Paying a 52-week high for 12.8% of analyst headroom is a poor trade.
- **ABUS — rejected as unrateable.** **No analyst coverage at all** (rating n/a, PT n/a), so there is no verifiable upside target and conviction cannot rise above 2/5 under the screener rule. The +1,078.8% "revenue growth" is a litigation windfall, not operations. The $230M Dutch auction is a special situation whose risk I cannot size without the tender range — and post-tender, the technical support disappears.
- **BEAM — rejected on portfolio shape.** **Beta 2.21**, the highest on the board, clinical-stage, and directly correlated with CADL's XBI exposure. The +75% PT is attractive, but stacking a second high-beta clinical binary while the directive is to tighten stops is incoherent risk management.
- **ELME, FBRT, BKKT, NVAX, HNI, GEO, ANY, RXRX, IRD, VIR** (ranks 6–15) — not advanced: ELME/FBRT are yield-driven real estate with ~0% relative strength; ANY is a $24M nano-cap below any workable liquidity floor for a real position; the remainder are additional healthcare names blocked by the same concentration logic that rejected SIBN and BEAM.

### Selected: WWW — Wolverine World Wide | Conviction 4/5

**PRV, 8/24 07:15 AM EDT:** $20.96 at close 8/21; **after-hours $21.04 (+0.38%) at 7:30 PM EDT 8/21**. No Monday pre-market print yet at fetch time. Market cap $1.72B. Day range $20.32–21.08. Volume 662,720.

**The fundamentals — the only candidate that is profitable AND growing on every line:**
- Revenue TTM **$1.95B, +7.2%**
- Net income **$105.60M, +28.3%**
- EPS **$1.28, +25.3%**
- **PE 16.40 → forward PE 11.91** — forward materially *below* trailing, the signature of expected earnings growth (the mirror image of PD's profile)
- 52-wk **$13.47–32.80** — trading **36% below its 52-week high**: a recovery, not an extension
- Beta 1.74 · Dividend $0.40 (1.91%) · **Buy, PT $24.30 (+15.9%)**

**Q2 print (8/13), verified on StockTitan:** revenue $506.4M +6.8%; Merrell **+11.1%**, Saucony **+9.9%**; gross margin 46.5% (−70bps on US tariffs) but **operating margin improved to 9.3%**; adjusted EPS $0.40 vs $0.38 consensus; inventory **−17%**, net debt **−22%**. **FY guidance raised** to $1.98–2.00B revenue, 9.5% operating margin, EPS $1.48–1.58. Analyst response: **KeyBanc PT $25 from $20 (Overweight)**; Williams Trading $20 from $17 (Hold).

**The gate condition set in Week 49 has been met.** WWW was deliberately passed over last week at day 2 of a +16.4% move and queued #1 "on a higher-low base or a pullback toward the ~$19.35 20-day SMA." Five sessions later:

| Date | Low | Close | Volume |
|---|---|---|---|
| 8/14 | 19.93 | 21.03 | 1,304,600 |
| 8/17 | **19.38** | 19.83 | 1,145,500 |
| 8/18 | 19.88 | 19.99 | 1,122,600 |
| 8/19 | 20.08 | 20.85 | 725,600 |
| 8/20 | 19.85 | 20.45 | 700,100 |
| 8/21 | **20.32** | 20.96 | 662,700 |

Lows stepped 19.38 → 19.88 → 20.08 → 19.85 → 20.32 (one shallow dip), price consolidated in a 19.38–21.08 range for five sessions, and **volume declined the whole way through the base** (1.30M → 0.66M) before Friday's close reclaimed the range top and still sits below the 8/14 intraday high of 21.26. That is textbook constructive consolidation, not distribution. **The patience was rewarded: the stock is 0.4% below where I declined to chase it, after building a base underneath.**

**Entry-discipline checks — all pass:**
- Post-earnings cooldown: **6 sessions** past the 8/13 print (rule requires >3, prefers ≥1 week) ✓
- Distance from 20-day SMA ($19.74): **+6.2%** (limit ≤20%) ✓
- Distance from 50-day SMA ($18.47): **+13.5%** (limit ≤40%) ✓
- Not days 1–3 of a breakout ✓
- Liquidity: ~$14M/day ADV, far above the $1M full-sizing threshold ✓

**Thesis-input freshness:** the live external driver is **US footwear tariffs**, which cost 70bps of gross margin in Q2. Critically, the thesis does **not require tariff relief** — management raised full-year guidance *with the tariff drag already in the numbers*. The thesis rests on brand momentum (Merrell/Saucony ~+10%) and balance-sheet repair (inventory −17%, net debt −22%), both dated 8/13, inside the 10-trading-day window. This is the structural test the LXU failure taught: a thesis is void when it *depends* on a driver that is reversing. WWW's does not.

**Bear case:** Beta 1.74 — it will fall harder than the book in a market drawdown, which matters with three weeks left. Tariffs remain a live margin headwind. Sweaty Betty (−2.4%) and Work Group (−1.6%) are shrinking; growth is carried by two brands. At 36% below its 52-week high, this is a repair story, not a leader — the same shape as PAR. And, honestly, **there is no dated catalyst inside the 5-day window** — this is post-earnings drift on a raised guide, which is a weaker reason to act than a scheduled event.

---

## 5. PORTFOLIO ACTIONS

**Keep:**
- **ATRC** — 5/5. Fresh BTIG PT $55 (+12%) on the new STS Quality Metric, landed pre-market today. Partial-profit deferral re-affirmed on all four criteria; the raised stop is the risk control.
- **TILE** — 4/5. Strong Buy, PT $45.25 (+16.2%), EPS +52.5%. Thesis intact.
- **PAR** — 4/5. Revenue +18.8%, PT $25.31 (+29.6%). Highest verified upside in the book.
- **CADL** — 3/5. Strong Buy, PT $20.88, but the marginal analyst (BofA) is Neutral at $12, below the market. Riding a locked +5.0% into the Q4 BLA catalyst.

**Trim:** none. The ATRC +30% partial stays deferred (criteria in §3); a 1-of-3-share trim into a fresh upgrade is high-friction and rule-inconsistent.

**Exit:** none. No stop breached, no thesis broken.

**Initiate:**
- **WWW — 3 shares @ limit $21.10** — the only candidate that is profitable, growing, and cheap on forward earnings; fills the Week 49 queue on a met gate condition. **Size is capped at 3 shares by the 15% cash floor, not by conviction** (see §7).

### Stop changes — and two deliberate refusals

The "tighten stops" directive means *tighten by one ATR*. Applied mechanically to all four holdings, the **range check rejects two of them** — a stop above the most recent session's low is inside normal noise and must be refused. This is the ARDT lesson, applied prospectively rather than after the fact.

| Ticker | Current | One-ATR tighten | Friday's low | Verdict |
|---|---|---|---|---|
| **ATRC** | $44.20 (2.84×ATR) | **$45.85** (1.86×ATR) | $46.99 | ✅ **APPLY** — clears the low, locks +33.7% |
| **PAR** | $16.50 (2.97×ATR) | **$17.50** (1.99×ATR) | $18.80 | ✅ **APPLY** — clears the low, cuts max loss from −12.5% to −7.2% |
| **TILE** | $37.10 (1.24×ATR) | $38.58 (0.24×ATR) | $38.55 | ❌ **REFUSED** — sits *above* Friday's low; and $37.10 is **already tighter than the 1.5×ATR minimum** |
| **CADL** | $12.00 (1.51×ATR) | $12.75 (0.51×ATR) | $12.38 | ❌ **REFUSED** — sits *above* Friday's low; $12.00 is **already exactly at the 1.5×ATR floor** |

TILE and CADL cannot be tightened because they are already at or inside the ATR guideline. Forcing the directive onto them would convert two working thesis-stops into coin flips — which is precisely how ARDT was lost at the exact low of the day on 8/21 before recovering 6.2% into the close.

---

## 6. EXACT ORDERS

### Order 1 — BUY WWW

- **Action:** BUY
- **Ticker:** WWW (Wolverine World Wide, NYSE)
- **Shares:** 3
- **Order Type:** LIMIT
- **Limit Price:** **$21.10**
- **Time in Force:** DAY
- **Intended Execution:** 2026-08-24 (Monday)
- **Stop Loss:** **$19.30** — below the $19.38 swing low of 8/17 (the base floor) and 1.88×ATR below the limit, satisfying the 1.75×ATR target
- **Stop Limit:** **$19.15**
- **Special Instructions:** **Skip condition — do not buy if WWW opens below $19.74 (the 20-day SMA)**, which would void the higher-low base the entry depends on. No Monday pre-market print existed at 07:15 AM; the limit is set $0.06 above the **last verified price of $21.04 (after-hours, Aug 21, 7:30 PM EDT)** and $0.14 above the $20.96 close, allowing for a modest gap. If WWW gaps above $21.30, do not chase — stand down and re-evaluate Tuesday.
- **Rationale:** Only candidate profitable and growing on every line (EPS +25.3%, forward PE 11.91 vs trailing 16.40) with a raised FY guide, entered on a five-session consolidation with declining volume, 36% below its 52-week high.

### Order 2 — RAISE STOP, ATRC

- **Action:** MODIFY existing stop (no share change)
- **Ticker:** ATRC
- **Shares covered:** 3 (full position)
- **Stop Loss:** **$45.85** (from $44.20)
- **Stop Limit:** **$45.70** (from $44.05)
- **Time in Force:** GTC
- **Order-side check:** run against $48.98 (close, Aug 21 4:00 PM EDT; pre-market $48.98 unchanged at 07:00 AM Aug 24). Stop is **6.4% below** the last verified price ✓
- **Range check:** 1.86×ATR(14) below price (ATR $1.681 = 3.43%); **below Friday's low of $46.99** ✓
- **Rationale:** Locks **+33.7%** from the $34.30 entry, up from +28.9%, while preserving nearly two ATRs of room for a name that just received a $55 price target.

### Order 3 — RAISE STOP, PAR

- **Action:** MODIFY existing stop (no share change)
- **Ticker:** PAR
- **Shares covered:** 8 (full position)
- **Stop Loss:** **$17.50** (from $16.50)
- **Stop Limit:** **$17.35** (from $16.35)
- **Time in Force:** GTC
- **Order-side check:** run against $19.53 (close, Aug 21 4:00 PM EDT). Stop is **10.4% below** ✓
- **Range check:** 1.99×ATR(14) below price (ATR $1.021 = 5.23%); **below Friday's low of $18.80 and the 5-day low of $18.18** ✓
- **Rationale:** Cuts the worst case from **−12.5% to −7.2%** of the position, removing ~$10 of tail risk with a stop that still sits two ATRs away.

**No stop change for TILE ($37.10/$36.95) or CADL ($12.00/$11.85)** — both already at or inside the 1.5×ATR floor; tightening is refused by the range check.

---

## 7. RISK AND LIQUIDITY CHECKS

### The binding constraint this week is the cash floor, not conviction

| | |
|---|---|
| Equity | $781.68 |
| Cash | $191.44 (24.49%) |
| **15% cash floor** | **$117.25** |
| **Deployable** | **$74.19** |

At a $21.10 limit that funds **3 shares ($63.30)**. Four shares would leave cash at 13.69% — a floor violation. Meanwhile **risk-per-trade would permit 21 shares** — so the position is roughly one-seventh the size the risk budget allows.

**This is worth naming plainly:** with $74 deployable, a new position is a stub that will move the final gap by well under a percentage point. The 15% cash floor was explicitly reaffirmed as *unchanged* in the Week 49 amendments, so I am not going to quietly reinterpret it — but if you want fuller deployment for the last three weeks, amending the floor is a decision available to you, exactly as the healthcare cap and market-cap ceiling were amended in Week 49. **I am not recommending it:** at Week 50 with a lead to protect, cash is a legitimate defensive asset, and the floor is doing real work.

### Post-trade concentration

| Holding | Value | % Equity |
|---|---|---|
| TILE | $155.76 | 19.93% |
| PAR | $156.24 | 19.99% |
| ATRC | $146.94 | 18.80% |
| CADL | $131.30 | 16.80% |
| **WWW (new)** | **$63.30** | **8.10%** |
| **Cash** | **$128.14** | **16.39%** ✓ |
| **Total** | **$781.68** | 100% |

All five positions under the 30% single-name ceiling ✓ · Cash above the 15% floor ✓

### Sector cap (GICS, per quote pages)

| Sector | Positions | Cap | Status |
|---|---|---|---|
| Consumer Discretionary | TILE, **WWW** | 2 | ✅ at cap |
| Healthcare | ATRC, CADL | 3 (amended) | ✅ under |
| Technology | PAR | 2 | ✅ under |

**Correction carried forward:** TILE is **Consumer Discretionary**, not Industrials as earlier reports assumed. Adding WWW therefore consumes the Consumer Discretionary cap — which is a further reason SIBN's healthcare concentration was the wrong direction, but also means no third Consumer Discretionary name can be added in Week 51.

### Risk per position (loss if stop fires)

| Holding | Shares | Price − Stop | Max loss | % Equity |
|---|---|---|---|---|
| ATRC | 3 | $48.98 − $45.85 | $9.39 | 1.20% ✓ |
| TILE | 4 | $38.94 − $37.10 | $7.36 | 0.94% ✓ |
| PAR | 8 | $19.53 − $17.50 | $16.24 | 2.08% ✓ |
| CADL | 10 | $13.13 − $12.00 | $11.30 | 1.45% ✓ |
| WWW | 3 | $21.10 − $19.30 | $5.40 | 0.69% ✓ |
| **Aggregate** | | | **$49.69** | **6.36%** |

Every position within the 5% risk-per-trade limit. Aggregate simultaneous-stop-out risk is 6.36% of equity — acceptable, and the positions are not tightly correlated.

### Liquidity

WWW ADV ≈ 662K–1.3M shares (~$14–27M/day). A 3-share order is **~0.0004% of ADV** — immaterial. Every holding trades well above the $1M/day full-sizing threshold.

---

## 8. MONITORING PLAN

| Holding | Watch for | Trigger |
|---|---|---|
| **CADL** | **Highest stop-out probability in the book.** Pre-market $12.85 is only 1.13×ATR above the $12.00 stop. No company news since the 8/13 Q2 release; its swings are XBI beta. | If stopped at $12.00, that is +5.0% realized — accept it, do not re-enter. Watch XBI. |
| **ATRC** | Whether the BTIG $55 target draws follow-on upgrades and lifts the $47.33 consensus above the market price. Also the **+60% partial trigger at $54.88**. | If $54.88 is reached, re-run the four deferral criteria fresh — at that level the stop must lock ≥+40%. |
| **PAR** | Holding the $18.38 higher-low. Any move toward the $25.31 PT. | Below $18.18 (5-day low) on volume = thesis-crack warning, well before the $17.50 stop. |
| **TILE** | Ex-dividend **Sep 4**. Forward PE (16.20) above trailing (15.77) signals decelerating earnings. | A close below $38.55 would be the first lower low since the run began. |
| **WWW** | Whether the base holds. Tariff commentary from peer footwear/apparel names. | **Day-1 drawdown rule applies:** if WWW closes −8% or worse on entry day with no positive catalyst, exit at the next open regardless of the stop. |
| **Book** | IWM ($299.96) and XBI ($165.73) — both rose Friday. Broad risk-on supports four of five holdings. | A market break hits WWW (β 1.74) and TILE (β 1.91) hardest. |

**Calendar:** PD reports 8/27 (not held — relevant only as a read on software sentiment). No holding reports earnings before the experiment ends; all four printed in August. **This means the remaining three weeks are drift-and-stop management, not event trading** — which is why the stop levels in §6 are the most consequential decisions in this report.

---

## 9. THESIS REVIEW SUMMARY

*(Reproduced verbatim as `Week 50 Summary.md`)*

---

## 10. CONFIRM CASH AND CONSTRAINTS

**Cash:** $191.44 → **$128.14** after the WWW purchase ($63.30). **16.39% of equity — above the 15% floor.**

| Constraint | Status |
|---|---|
| Cash floor ≥15% | ✅ 16.39% (binding — capped the WWW size at 3 shares) |
| Single-name ≤30% | ✅ largest is PAR at 19.99% |
| Max 5 concurrent positions | ✅ exactly 5 after the initiation |
| Risk-per-trade ≤5% | ✅ largest is PAR at 2.08% |
| Sector cap (2, healthcare 3) | ✅ Cons Disc 2, Healthcare 2, Tech 1 |
| Market cap ≤$5B | ✅ WWW $1.72B; largest holding ATRC $2.49B |
| Excluded security classes | ✅ all common stock |
| Stops on all longs | ✅ five stops, all order-side and range checked |
| Full shares only | ✅ |
| No averaging down | ✅ no adds to losing positions |
| PRV gate | ✅ all 4 holdings + all 6 candidates browser-verified 8/24 07:00–07:25 EDT |
| Post-earnings cooldown | ✅ WWW 6 sessions past its print |
| Distance-from-base | ✅ WWW +6.2% over SMA20, +13.5% over SMA50 |
| Thesis-input freshness | ✅ WWW's tariff driver verified as a headwind already absorbed into raised guidance |

**All constraints satisfied. Three orders for execution at the broker on 2026-08-24.**

---

## Addendum — Script fix applied this session

The first `make weekend` run aborted with *"Portfolio data is not current for the last trading day"* even though Friday 8/21 data was present. The gate computed the required session as the last NYSE session **including today**, so running Monday at 07:18 AM demanded data for a session that had not opened yet — the workflow was structurally unrunnable before any Monday close.

Fixed in the script rather than worked around: added `last_completed_session()` to `trading_script.py`, which returns the most recent session whose 16:00 ET close has actually passed, plus a `--check-data-current` flag. The Makefile now calls that instead of an inline shell-Python one-liner that had an `exec()`-inside-a-`-c`-string fallback. Verified across five timestamps, including both sides of the 4 PM boundary.

---

*Week 50 Full Report — generated 2026-08-24 by Claude Code. Posture: Tighten stops · Wide net · 5-day catalyst window · max 5 positions. All prices browser-verified pre-market 2026-08-24; all ATR and range figures computed from price history.*
