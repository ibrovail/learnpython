# Week 51 — Thesis Review Summary

**Date:** 2026-08-31 | **Week:** 51 of 52 | **Posture:** Aggressive — trailing benchmark
**14 sessions remain (Sept 7 is Labor Day).** All prices live intraday, ~10:00 AM EDT.

---

## Per-Position Thesis

**ATRC (AtriCure) — KEEP | Conviction 5/5**
**+36.9%**, and the most consequential position on the board. The fundamental case has **strengthened sharply while the price fell**: **Piper Sandler raised its target to $60 from $50 (Overweight)** four days ago, following BTIG's **$55 from $45 (Buy)** on 8/24 — both citing the new STS quality metric driving adoption of concomitant ablation. **Consensus PT has risen $47.33 → $49.56 and now sits above the market.** Revenue TTM +13.9%. No adverse news since 7/27.
**And the stop is $0.66 away — 0.38×ATR.** ATRC will most likely be stopped out at $46.30 for **+35.0%** exactly as two banks put $55–60 on it. There is no lever; stops cannot be lowered.
*On my own decision:* I raised this stop $45.85 → $46.30 last Thursday at $49.27, near the 52-week high. It gained **$1.35** of locked profit and materially raised the stop-out probability. It passed every check at the time. **In hindsight it was poor value** — a marginal gain bought with real optionality. Trailing at 1.75×ATR on a 3.5%-ATR name leaves nothing when the name gives back one week.

**PAR (PAR Technology) — KEEP | Conviction 4/5**
−3.5% at $18.39 after a −3.97% session, on **no news** (latest item is 13 days old). Revenue **+18.8%**, forward PE 16.82, **Buy with PT $25.31 — +35.1%, the largest upside of any holding.** Now the biggest position at 26.9% of equity, with $23 of headroom to the 30% cap.
The 3 shares added Friday at $19.60 are **−6.2%** — a poor fill in hindsight, since the $19.60 limit caught near the top of that day's $19.15–19.92 range.

**CADL (Candel Therapeutics) — KEEP | Conviction 3/5**
+11.5%. **PT raised to $21.00 (+65.75%), Strong Buy** — the largest upside on the board — and beta −0.50. But XBI fell −3.48% Friday and −1.27% again today, and CADL is **0.44×ATR from its $12.35 stop**. A stop-out realises +8.0%.

**TILE (Interface) — KEEP | Conviction 4/5**
+18.6%. Strong Buy, PT $45.25. Stop $37.10 locks **+15.6%** and is 0.81×ATR away — it can be neither raised (the trailing floor sits below it) nor lowered.

**WWW (Wolverine World Wide) — KEEP | Conviction 3/5**
−5.5%. Forward PE ~11.2, Buy, PT $24.30 (+22%). Bounced +2.94% Friday and gave part back today. 0.68×ATR from its stop.

---

## Overall Portfolio Thesis

**Gap: +1.52% (Aug 25) → −0.31% (Aug 28) → −2.47% (live).** Five sessions turned the widest lead of the experiment into the largest deficit since July. Today alone the book is **−2.53% against SPY's −0.37%** — and not one of the five declines is company-specific. Every holding was browser-checked; there is no news, no downgrade, no guidance change anywhere in the book.

**The defining fact of Week 51: all five positions are now inside 1.0×ATR of their stops** — ATRC 0.38×, CADL 0.44×, WWW 0.68×, TILE 0.81×, PAR 0.94×. None is at the 1.5× minimum the rules require for a stop to function as anything other than a coin flip. I did not tighten them into this state; the market walked the prices down to fixed lines that cannot be lowered.

This has one benign consequence and one severe one. Benign: **every stop firing costs only −2.87% of equity** — the book's remaining downside is capped under 3%. Severe: **no position can absorb new capital**, because the one-open-order-per-stock rule means new shares inherit a stop inside a single day's noise. That single test blocked every add this week.

**On the Aggressive directive.** It could not be expressed through deployment, and I would rather say so than quietly ignore it. Deployable cash is **$9.17**. Freeing more means selling into a −2.5% intraday drawdown on no news, to buy a candidate that is not better than what it replaces: **BOX** offers +6.4% upside on a Hold rating against WWW's +22% on a Buy; **RCKT** offers a remarkable +135% PT but is pre-revenue with **no dated catalyst inside a 14-session runway**, and would make healthcare 3 of 6 while XBI is selling off. So the posture is expressed the only legitimate way available: **by refusing to tighten a single stop.** With every position already inside 1.0×ATR, declining to trail is not passivity — it is the difference between letting five theses run and guaranteeing they are stopped out.

**The week's real finding was a data-integrity one.** The first screener run produced a watchlist where **every candidate showed a volume ratio of 0.10–0.38×** and **four BDCs — an excluded security class — ranked in the top seven.** The cause was `volume.iloc[-1]` picking up **today's partial-day bar** during market hours and measuring twenty minutes of trading against a 20-day average, which collapsed the volume factor for every real momentum name and floated low-volatility yield vehicles to the top. The list did not look broken; it looked like a defensible set of value names. Fixed by dropping unsettled bars at the fetch boundary, reusing `last_completed_session()`. The corrected screen shares **only one name** with the broken one.

**Also corrected:** the script's reported **beta of 1.5653** is computed over the full experiment history and is dominated by the pre-April strategy. Measured over the last 39 sessions the book's beta is **0.37 with R² of 0.03** — this portfolio is very nearly uncorrelated with the index, which is why cash never functioned as a market hedge here and why the gap moves on stock selection rather than market direction.

**Next:** 13 sessions after today. No holding reports earnings before the end, no catalyst is dated inside the window, and $9 is deployable. The result now rests entirely on whether ATRC, TILE, CADL, PAR and WWW hold lines that sit less than one average day's move beneath them.

---

*Week 51 Summary generated 2026-08-31 by Claude Code. All prices are live intraday quotes browser-verified 09:53–10:05 AM EDT; ATR and range figures computed from settled bars through 2026-08-28.*
