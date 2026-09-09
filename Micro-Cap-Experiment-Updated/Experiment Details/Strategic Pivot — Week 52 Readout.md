# Strategic Pivot — Week 52 Readout

**Date:** September 9, 2026 · **7 trading sessions remain** (experiment ends 2026-09-18)
**Companion documents:** `Strategic Pivot — Week 30 Readout.md` (the pivot), `Week 41 Readout.md` (the gap-closed bookend), `Week 46 Readout.md` (the honest interim)

---

## Executive Summary

**Two verdicts, and they point opposite ways.**

Measured from **inception**, this experiment has lost to a passive S&P position: the book is worth **$722.00** against an S&P-equivalent of **$767.90** — a gap of **−5.98% (−$45.90)** — with injection-neutral alpha of **−0.68%**.

Measured from the **April 13 pivot**, it has beaten that benchmark decisively: **TWR +25.94% against the S&P's +11.12%, alpha +14.82%**, with the gap repaired from **−19.99% to −5.98% (+14.01 points)** and maximum drawdown cut from −37.26% to **−7.70%**.

**Both are true. The pivot did not fail — it ran out of runway.** It repaired 14 of the 20 points it inherited, and needed roughly six more weeks at its realised rate to finish level. It has seven sessions.

---

## 1. The Three Windows

| | **Pre-pivot** | **Post-pivot** | **Full experiment** |
|---|---|---|---|
| Window | 2025-09-19 → 2026-04-13 | 2026-04-13 → 2026-09-09 | 2025-09-19 → 2026-09-09 |
| Sessions | 136 | 103 | 239 |
| Equity | $142.13 → $387.97 | $387.97 → **$722.00** | $142.13 → **$722.00** |
| Capital added | $331.10 | $216.54 | $547.64 |
| S&P-equivalent | $142.13 → $484.89 | $484.89 → $767.90 | $142.13 → **$767.90** |
| **Gap (start → end)** | 0.00% → **−19.99%** | −19.99% → **−5.98%** | 0.00% → **−5.98%** |
| **TWR** | **−9.33%** | **+25.94%** | **+14.19%** |
| S&P (same window) | +3.38% | +11.12% | +14.87% |
| **Alpha** | **−12.70%** | **+14.82%** | **−0.68%** |
| Max drawdown | **−37.26%** | **−7.70%** | −37.26% |
| Sharpe (ann.) | **−0.02** | **2.29** | 0.53 |
| Sortino (ann.) | — | **4.60** | 0.71 |
| Beta vs SPY | — | **0.56** | 0.87 |
| Win rate | 50.7% | 53.4% | 51.9% |

**The comparison that matters is the Sharpe.** Pre-pivot: **−0.02** — the book took a −37% drawdown and was compensated with nothing. Post-pivot: **2.29**, with a fifth of the drawdown. That is not the same strategy performing better; it is a different strategy.

---

## 2. What the Pivot Actually Changed

The Week 30 pivot went to 100% cash at a −21.1% gap and rebuilt around explicit process rules. The measurable consequences over 103 sessions:

- **Drawdown control.** Maximum drawdown fell from **−37.26% to −7.70%**. Worst single session improved from −15.04% to −3.70%. This is the ATR-based stop discipline working — losses were bounded before they compounded.
- **Beta discipline.** 0.87 → **0.56**. The book stopped being a leveraged proxy for small-caps and became a set of idiosyncratic positions (R² of 0.066 — essentially uncorrelated).
- **Realised trade quality.** Across the full experiment, **82 closed trades, a 50% win rate, and −$3.84 of net realised P&L** — a coin flip that paid nothing. **Since the pivot: 25 closed trades, 56% win rate, +$83.59 realised.**

The rules that produced this were written *in response to specific failures*, and the record is in `.claude/rules/`: the ATR range check (ARDT), thesis-input freshness (LXU), the PRV browser gate (ARDT/FOXF/PAR), the anti-ratchet stop minimum (ATRC), and the four-category unexplained-move check (ATRC/PAR).

---

## 3. The Gap Path — Month by Month

| Month | Gap | |
|---|---|---|
| 2025-09 | −7.14% | |
| 2025-10 | **+14.95%** | early lead |
| 2025-11 | **+16.96%** | ← **peak** |
| 2025-12 | +16.72% | |
| 2026-01 | −0.46% | lead gone in one month |
| 2026-02 | +6.55% | |
| 2026-03 | −8.10% | |
| 2026-04 | −10.75% | ← **pivot, Apr 13, at −19.99% intra-month** |
| 2026-05 | −11.75% | ← **trough** |
| 2026-06 | −2.05% | repair |
| 2026-07 | −2.86% | |
| 2026-08 | **−1.27%** | closest to level |
| 2026-09 | **−5.98%** | ← **the last nine sessions** |

**The single most expensive fact in this table is the November peak.** The book was **+16.96% ahead** and gave back 34 points over the following six months. Nothing in the post-pivot recovery has recouped that — the pivot repaired the *deficit*, not the *forfeited lead*.

---

## 4. The Last Nine Sessions

The gap went **−1.27% (Aug 31) → −5.98%**, giving back roughly a third of the entire post-pivot repair in under two weeks:

| Cause | Cost |
|---|---|
| **TYRA** — binary Phase 2 readout, entered 9/08, exited 9/09 | **−$29.04** |
| PAR — three-session slide, −7.3% unrealised on 27% of equity | ~−$15 |
| TILE — stopped out 9/03 at +12.9% after the restoration delayed the exit | −$3.44 vs stopping earlier |
| Cash drag — 23–36% idle against a rising benchmark | structural |

**TYRA deserves the plain accounting.** It was the only candidate on the Week 52 screen with a dated catalyst inside the runway. The weekend report recommended **against** it on the grounds that *a stop cannot protect a binary* — the position was taken as an explicit aggressive override, the SURF302 data disappointed (79% ORR and 64% "best overall" CR headlined, but the durable 3-month endpoint fell short of the 70–80% analysts expected, and the company simultaneously disclosed it is exploring a **70 mg cohort** — i.e. the dose it had chosen may be insufficient), and the stock **gapped through the stop at the open**, exactly as forecast. Cost: **$29.04, and about 3.8 points of gap in two sessions.**

That single decision is roughly two-thirds of the difference between finishing near level and finishing at −6%.

---

## 5. What the Experiment Actually Demonstrated

**On the headline question — can this process beat a passive index?** Over the full period: **no, narrowly.** −0.68% alpha, −5.98% gap. The honest answer is that a year of active micro-cap management underperformed simply owning SPY.

**On the more useful question — did the process improve when it was disciplined?** Emphatically yes, and the evidence is not subtle: **alpha −12.70% → +14.82%, Sharpe −0.02 → 2.29, max drawdown −37.26% → −7.70%.** The same manager, the same universe, the same market. The difference was rules.

**Three findings worth carrying forward:**

1. **Losses are governed by exit discipline, not entry selection.** Win rate barely moved (50.7% → 53.4%). What changed was the *size* of losses — the pre-pivot book had single days of −15.04%; the post-pivot worst was −3.70%.
2. **Shrinking revenue is the highest-value single filter this book found.** It correctly rejected TDAY, FOXF, NAVI and PD, and correctly selected PAR.
3. **Binary events are not compatible with a stop-based risk framework.** TYRA proved it in one session. A stop cannot bound a gap, which means the position size *is* the entire risk control — and the framework was never designed for that.

---

## 6. Standing, and the Final Seven Sessions

**Equity $722.00 · Cash $259.33 (35.9%) · Gap −5.98% · TWR alpha −0.68%**

| Holding | Value | % | P&L | Stop | Locks |
|---|---|---|---|---|---|
| PAR | $194.26 | 26.9% | −7.3% | $17.05 | −10.5% |
| ATRC | $159.09 | 22.0% | **+54.6%** | $50.30 | **+46.6%** |
| VTS | $109.32 | 15.1% | +2.1% | $16.65 | −6.7% |

Closing a −5.98% gap in seven sessions requires roughly **+$46 on $463 of holdings — about +10%**. Only ATRC has demonstrated moves of that size, and its consensus target now sits *below* its price.

**The realistic outcome is finishing 4–7% behind the benchmark.** The task for the remaining sessions is not to manufacture a recovery — attempting exactly that is what put $29 into TYRA — but to protect ATRC's **+54.6%**, whose stop locking **+46.6%** is now the single most consequential line in the book.

---

*Prepared 2026-09-09 by Claude Code. Equity and injection data from `chatgpt_portfolio_update.csv` and `capital_injections.csv`; S&P-equivalent is a shadow portfolio receiving identical cash flows into SPY; TWR uses the start-of-day injection convention in `trading_script.py` and reconciles exactly with the daily reports.*
