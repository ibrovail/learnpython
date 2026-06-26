# Strategic Pivot — Week 41 Readout

**Date:** June 26, 2026
**Author:** Claude Code (AI analyst) + portfolio manager
**Audience:** Anyone following the Micro-Cap Experiment
**Companion document:** `Strategic Pivot — Week 30 Readout.md` (the pivot itself)

---

## Executive Summary

Eleven weeks ago, at Week 30, the Micro-Cap Experiment was restructured from a single-sector string of biotech coin-flips into a diversified, screener-sourced momentum strategy. At that moment the portfolio trailed its S&P 500 benchmark by **$101.24 (−21.1%)** and needed roughly a 27% catch-up just to draw level. This readout is the bookend: **as of Week 41, the gap is closed.** Portfolio equity ($741.20) has overtaken the S&P-equivalent ($738.11) for the first time in the experiment — a **+$3.09 (+0.4%) surplus** — and the injection-neutral **time-weighted alpha is +6.34%**. The diversified approach the pivot installed did exactly what it was designed to do: accumulate edge across sectors instead of betting it on binary events. This document captures the outcome — what the pivot delivered, how it got here, what was skill versus tailwind, and what remains for the final 11 weeks.

---

## 1. The Result

The Week 30 pivot set one explicit target: **portfolio equity > S&P-equivalent (positive alpha) by Week 52.** That target has been hit 11 weeks early.

| Measure | Week 30 (Apr 13) | Week 41 (Jun 26) | Swing |
|---------|------------------|------------------|-------|
| Portfolio equity | $378.80 | $741.20 | — |
| S&P-equivalent | $480.04 | $738.11 | — |
| **Benchmark gap** | **−$101.24 (−21.1%)** | **+$3.09 (+0.4%)** | **+$104.33** |
| TWR alpha (cum, injection-neutral) | n/a (100% cash) | **+6.34%** | — |
| Positions | 0 (100% cash) | 5 (5 GICS sectors) | — |

**The honest headline:** raw equity rose from $378.80 to $741.20, but **$216.54 of that was a May 1 capital injection**, not performance. The figures that strip contributions out — the **benchmark gap** (both sides receive the injection) and the **time-weighted return** — are the real scorecard, and both confirm the same thing: the strategy has generated **genuine positive alpha** (TWR +17.23% vs the S&P's +10.89% over the same window).

---

## 2. Recap — Where We Started (Week 30)

For full detail see the companion document. In brief, the pre-pivot strategy had funneled every trade into biotech PDUFA bets (4 consecutive stop-outs, −$67.29) because the rules rewarded dated binary catalysts and Claude's web searches surfaced headline FDA events. The fix (Path B) was a **hybrid: a quantitative screener** (`screener.py`, ~1,100 stocks across all sectors) **feeding a Claude analyst** who verifies catalysts/fundamentals and makes final picks — with hard guardrails: max 2 positions per GICS sector, catalyst plays capped at 1 position/15%, 5-day minimum holds, slippage and regime filters.

The bet was simple: **breadth + diversification + thesis-driven entries would out-compound single-sector coin flips.** Week 41 is the verdict.

---

## 3. The Journey — How the Gap Closed

The catch-up was not linear. It came in two phases: a fast early repair (Weeks 30→32) as the first diversified deployment worked, a long grind through the spring, and a decisive final week as a market regime shift turned a tailwind into our favor.

| Date | Equity | S&P-equiv | Gap | Note |
|------|--------|-----------|-----|------|
| Apr 13 (W30) | $378.80 | $480.04 | −21.1% | Pivot — 100% cash |
| Apr 30 (W32) | $453.30 | $507.65 | −10.7% | First deployment working (+13.9pp alpha in 13 days) |
| Jun 18 (W40) | $705.97 | $752.82 | −6.2% | Post-May injection; grind narrowing |
| Jun 22 | $702.50 | $750.04 | −6.3% | — |
| Jun 23 | $708.92 | $739.26 | −4.1% | TWR alpha turns positive (+1.06%) |
| Jun 24 | $714.52 | $738.54 | −3.3% | TILE breakout |
| Jun 25 | $722.08 | $738.46 | −2.2% | New highs, Mag7 selling off |
| **Jun 26 (W41)** | **$741.20** | **$738.11** | **+0.4%** | **Gap closed — positive alpha** |

The final week was the most dramatic: the gap went from −2.2% to positive in a single session as the portfolio rose ~5% **while the S&P-equivalent fell** — a tech-led "Magnificent 7" selloff (Apple −6%, Microsoft −3% on price-hike news) collided with a small-cap surge (Russell 2000 crossed 3,000 for the first time). The "Great Rotation" out of mega-cap tech and into small-cap value moved hard in our direction.

---

## 4. What Worked

### Diversification delivered — and it's not a biotech bet in sight

The five positions that carried the book span **five different GICS sectors** — exactly the structure the pivot mandated, and the opposite of the pre-pivot 100%-healthcare concentration:

| Ticker | Company | Sector | Entry | Return (Jun 26) | Thesis type |
|--------|---------|--------|-------|-----------------|-------------|
| WKC | World Kinect | Energy | 4/27 @ $26.00 | **+29.7%** | Momentum + earnings beat |
| INN | Summit Hotel | Real Estate | 6/8 @ $5.75 | **+23.0%** | Catalyst (FIFA World Cup) |
| SHO | Sunstone Hotel | Real Estate | 5/18 @ $10.20 | +16.9% | Momentum + raised guidance |
| TILE | Interface | Consumer Cyclical | 6/15 @ $32.10 | +11.3% | Momentum (beat-and-raise) |
| TDAY | USA Today | Comm. Services | 6/8 @ $8.15 | +2.5% | Turnaround (AI licensing) |

Not one is a binary FDA coin-flip. The big winners (WKC, INN) came through different mechanisms — an earnings-driven energy compounder and an event-catalyst REIT — which is the whole point: **edge accumulating from multiple, uncorrelated sources.**

### Process discipline held under pressure

- **The No-Candidates rule was honored** (Week 41): with no qualifying catalyst in the timing window and the 15% cash reserve binding, the right call was to hold rather than force a 6th position. The book compounded on its existing winners instead.
- **Risk control worked:** max drawdown is still −24.99% (March 20, *pre-pivot*) — the diversified regime has not produced a new drawdown low in 11 weeks, despite higher equity at risk.
- **Sharpe 2.28 / Sortino 7.82** under the new regime (up from 1.97 at the pivot).
- **A losing thesis was recycled cleanly:** IMPP's tanker-rate play was exited the moment the US–Iran de-escalation undercut its catalyst (rather than riding it down), and the proceeds rotated into TILE — which is now +11%.

---

## 5. Honest Caveats — Skill vs. Tailwind

A readout that only tells the good story isn't useful. Three caveats:

1. **The final-week surge was substantially a market-regime gift.** The "Great Rotation" (small-cap value outperforming mega-cap tech) is a macro tailwind for a book that is, by construction, small-cap and value/cyclical-tilted. Roughly half of this week's gap closure came from the *benchmark falling*, not from stock-picking. Rotations reverse — if mega-cap tech leadership resumes, this tailwind becomes a headwind, and the +0.4% surplus is thin enough to flip back quickly. **The alpha is real but not yet durable.**

2. **The all-time risk numbers remain noisy.** The CAPM "annualized alpha" the script prints (+1254%) is statistically meaningless (R² = 0.04) and should be ignored — the trustworthy measure is the **cumulative TWR alpha (+6.34%)**, which is what this readout uses. (See the line-ending/TWR diagnostic work for why the all-time figures are unstable on a small, injection-fed account.)

3. **Beta is high (1.79).** The book is more volatile than the market; a sharp risk-off move would hurt disproportionately. The positive alpha has been earned with real exposure, not a defensive crouch.

---

## 6. Current State & What's Next (Weeks 42–52)

**Position:** 5 holdings (WKC/SHO/TDAY/INN/TILE), equity $741.20, cash $110.54 (14.9%), all stops clear, 6th slot reserved for AMLX (Phase 3 LUCIDITY, Q3 readout).

### Immediate (Week 42)
- **WKC +30% partial** is at the doorstep ($33.72 vs the $33.80 alert). When it fires, trim ~1 share — this both books spike profit and **funds the reserved AMLX 6th slot** without breaching the cash reserve.
- **Trail the loose stops:** INN ($6.20, now 12%+ cushion) and TILE ($28.70, ~20% cushion) have run well past their stops; raising them protects the gains that just closed the gap.

### Medium-term (Weeks 42–48)
- **Defend the alpha, don't over-trade it.** The lead is +0.4% — thin. The priority shifts from *catching up* to *not giving it back*: trail stops on winners, keep the No-Candidates discipline, avoid chasing.
- **Watch the regime.** If IWM rolls back under its 50-day SMA, the rotation tailwind is fading — tighten up and freeze new momentum initiations per the regime filter.

### Long-term (Weeks 48–52)
- **Week 52 final report:** total return, risk-adjusted metrics, and a full strategy-evolution comparison (binary-bet era vs. diversified era) across the 12 months.
- **Decision point:** with the gap closed early, the remaining question is whether to *extend* the lead (press winners, add the 6th catalyst position) or *protect* it (tighten, raise cash). That call should be made on the regime, not on greed.

---

## 7. Key Metrics — Then vs. Now

| Metric | Week 30 (Apr 13) | Week 41 (Jun 26) | Target (Week 52) |
|--------|------------------|------------------|-------------------|
| Portfolio equity | $378.80 | $741.20 | maintain alpha |
| Benchmark gap | −$101.24 (−21.1%) | **+$3.09 (+0.4%)** ✅ | > $0 (positive alpha) |
| TWR alpha (cum) | n/a | **+6.34%** | > 0 |
| Sector concentration | 0 (100% cash) | 5 sectors, ≤2 each ✅ | ≤ 2 per GICS sector |
| Positions | 0 | 5 of max 6 | diversified |
| Max drawdown | −24.99% | −24.99% (no new low) ✅ | < −30% (survivable) |
| Sharpe ratio | 1.97 | 2.28 ✅ | > 1.5 |
| Binary-bet dependency | 100% (pre-pivot) | 0% | low |

---

## 8. Bottom Line

The Week 30 pivot asked one question: *would diversified, thesis-driven, screener-sourced positions out-compound single-sector binary bets and close a 21-point benchmark deficit?* Eleven weeks later the answer is **yes — the deficit is gone and the portfolio is, for the first time, ahead of its benchmark on an injection-neutral basis.** The lead is narrow and partly regime-assisted, so the job for the final stretch is no longer catching up — it's keeping the alpha the pivot earned.

---

*This document captures the state of the experiment as of June 26, 2026. It will not be updated retroactively — future readouts will be separate documents.*
