# Entry Discipline Rules

Hard rules for new-position selection in the weekend deep research and daily analysis. Every recommended BUY must pass ALL of these checks before being proposed.

---

## Post-Earnings Cooldown

- **Never recommend a buy within 3 trading days of an earnings print** at a price more than +5% above the post-print close.
- Reason: ARLO Week 34 (entered $15.25 Monday, three trading days after Wednesday close of $13.60 pre-print, post-print closed $15.32 — bought at the post-print high, gave back -10.9% on entry day).
- Either enter pre-print (binary risk acknowledged) or wait for ≥1 trading week of post-print consolidation showing a higher-low base.

## Distance-from-Base Limits

For every screener candidate considered for entry, compute and report:
- Distance from 50-day SMA (must be ≤ 40% above for a momentum entry)
- Distance from 20-day SMA (must be ≤ 20% above)
- Days since 20-day breakout (avoid days 1-3 of a new breakout if the move is >+10% cumulative)

If a candidate is >50% above its 50-day SMA, it is **disqualified** for a fresh buy regardless of catalyst strength. Mean reversion risk dominates within a 5-trading-day window.

## ATR-Based Stop Sizing

Stops must be set at the wider of:
- 1.75 × 14-day ATR below entry, OR
- The most recent swing low on the daily chart, OR
- A technical level (50-day SMA, post-breakout VWAP)

A stop within 1.5 × ATR is too tight for normal daily noise and will be triggered by a routine down day. If the wider stop would create a max-loss exceeding 2% of equity, **reduce position size**, do not tighten the stop.

## Pre-Open Verification

Before recommending an open-of-Monday buy at the close-of-Friday price:
1. Check Monday pre-market action via WebSearch.
2. If pre-market is down >2%, downgrade to a limit at the pre-market price or pass entirely.
3. If pre-market is flat-to-up, the Friday-close limit is acceptable.

## Screener Score is Sourcing, Not Conviction

Screener composite score (momentum + volume + volatility-squeeze) identifies *candidates* but does NOT confer fundamental conviction. Apply the full 4-step verification to every screener pick:
1. Fundamental quality (revenue growth, margin, balance sheet)
2. Catalyst durability over the chosen timing window
3. Distance-from-base and post-earnings cooldown checks
4. Liquidity (ADV >$1M for full sizing, $500K-$1M for half sizing)

Conviction rating starts at 2/5 for any screener pick and can only rise on the strength of independent web-research evidence, not the screener score itself.

## Stop-Placement Surfacing

Every BUY recommendation must include an explicit reminder: **"Place this stop with your broker before the next market open."** The script's CSV stop field is informational and does not auto-execute at the broker.

In the next daily analysis after a BUY, the analyst must ask whether the stop was placed. If not, mark the position as "STOP NOT LIVE" in the per-holding review and surface it every day until acknowledged.

---

## Day-1 Drawdown Rule

If a freshly opened position closes -8% or worse on entry day:
- **Default action: exit at next market open** unless an explicit positive-news catalyst surfaced after entry.
- Reason: A -8% same-day move on a momentum-screened name signals thesis break, not noise. Continuing to hold rationalizes a bad entry.
- This rule overrides the stop-loss field — exit even if the stop was not breached intraday.
