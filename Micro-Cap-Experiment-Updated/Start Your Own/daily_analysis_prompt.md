# Daily Analysis Format

Use this format when producing the daily portfolio analysis after `make daily`.

Portfolio rules are in `Start Your Own/portfolio_rules.md` — read that file first.

---

## Daily Output Format (6 Sections)

```
Daily Portfolio Review — [DATE]

---

1. Market Regime Check

   IWM: $[price] | 50-day SMA: $[value] | [±X.XX%] (source: the script's <market_regime> block —
   computed from IWM closes; do not look it up elsewhere)
   Regime: RISK-ON / RISK-OFF / BORDERLINE
   Rule applied: [state the specific restriction if RISK-OFF, or "No restrictions" if RISK-ON]

---

2. [TICKER] — Holding Review
   (Repeat this section once per position)

   | Item           | Detail                          |
   |----------------|---------------------------------|
   | Current Price  | $X.XX (+/-X.XX% today)          |
   | Entry          | $X.XX                           |
   | Unrealised P&L | +/-$X.XX (+/-X.X%)              |
   | Current Stop   | $X.XX trigger / $X.XX limit     |
   | Stop Status    | Not breached — X.X% above stop  |
   | Primary Driver | [time-varying input the thesis rests on — feeds the driver cap] |
   | Sessions Held  | N (from <position_limits>) — re-underwrite due at 60 |

   Catalyst Research:
   - [Key upcoming event with date, confirmed by ≥2 sources]
   - [Clinical/regulatory/financial facts relevant to thesis]
   - [Risk factors or negative signals]
   Sources: [Name — URL], [Name — URL]

   Stop-Loss Update (portfolio_rules.md → "Raising a stop — anti-ratchet minimum"):
   - ATR(14): $X.XX — computed from yfinance price history, never estimated or searched
   - Close / today's low / 10-day low: $X.XX / $X.XX / $X.XX
   - Current stop: $X.XX = X.XX×ATR below the close   (1.5×ATR floor applies when PLACING a stop;
     drift below it afterwards is expected — note it and check restoration eligibility)
   - Candidate level: close − 2.0×ATR = $X.XX         (raise target 2.0×ATR — never raise TO the floor)
   - Test 1, raise size:   (candidate − current stop) ÷ ATR = X.XX   ≥ 0.50 → PASS / FAIL
   - Test 2, room left:    (close − candidate) ÷ ATR = X.XX          ≥ 1.50 → PASS / FAIL
   - Range check: candidate below today's low → PASS / FAIL
   Action: RAISE to $X.XX / $X.XX only if ALL pass; otherwise HOLD. A failing raise is declined,
   not reduced. If the current stop already sits below 1.5×ATR through price movement alone,
   say so and state whether this entry's one restoration is still available.

   Add Shares?
   - Risk budget: $[equity] × 2% = $[amount]
   - Risk per share at $[entry] entry / $[new stop] stop: $[diff]
   - Formula: $[budget] / $[risk/share] = [N] shares
   - 30% cap: $[equity] × 30% = $[max_value] → [max_shares] shares max total
   - Current position: [N] shares → room for [N] additional
   Decision: ADD [N] shares / NO ADD
   Rationale: [one-line reason]

---

3. New Positions

   [RISK-OFF regime — capacity per portfolio_rules.md → Allocation Framework: up to 3 non-binary
   catalyst positions at standard 2% sizing, plus screener-sourced entries at half the risk
   budget (1%) on the defensive profile only (top-decile low_vol, near the 60-day high, volume
   confirmation; this allowance sunsets at Phase 4). Name the capacity used and what remains.]

   [If screening candidates:]
   | Ticker | Thesis | Catalyst (≥2 sources) | Liquidity | Bear Case | Sizing |
   |--------|--------|----------------------|-----------|-----------|--------|

---

4. Final Decisions

   (One block per action. Omit if no action needed.)

   ACTION:        [BUY / SELL / UPDATE STOP / HOLD]
   Ticker:        [symbol]
   Shares:        [integer — omit for stop updates]
   Limit Price:   $[price — omit for stop updates]
   New Stop-Loss: $[price]
   New Stop-Limit:$[price]
   Reason:        [one line]

---

5. Post-Event Playbook
   (Only include when a dated non-binary catalyst — or a scheduled earnings print on an
   existing holding — is ≤10 trading sessions away. Binary-thesis entries are prohibited,
   so this section never plans a pass/fail wager; it plans the response to a known event.)

   | Scenario      | Action                                                        |
   |---------------|---------------------------------------------------------------|
   | Positive      | [hold; recalculate the trailing stop under the standard rule]  |
   | Negative      | [the stop governs the exit; 10-session re-entry ban if hit]    |
   | Delay / Other | [post-catalyst reassessment within 1 session; re-rate thesis]  |

   Note: there are no mechanical partial sells. The trailing stop is the only exit
   (`portfolio_rules.md` → Position Management).

---

6. Portfolio State After Today

   | Holding | Shares | Price  | Value   | % of Equity |
   |---------|--------|--------|---------|-------------|
   | [TICKER]| [N]    | $X.XX  | $X.XX   | X.X%        |
   | Cash    | —      | —      | $X.XX   | X.X%        |
   | TOTAL   | —      | —      | $X.XX   | 100%        |

   Stop / Stop-Limit: [TICKER] → $X.XX / $X.XX

   Correlated-risk check: sector counts [from <position_limits>, cap 3]; primary drivers
   [list them — cap 2 per shared driver]
   Circuit breaker: current drawdown from peak [from <risk_metrics>] — clear / DE-RISK / CASH

---

Sources:
- [Source name] — [URL] — [what it confirmed]
- [Source name] — [URL] — [what it confirmed]
```

---

## Weekend Question (only when `make trigger` says DUE)

Ask the user **one** optional question, then pass the answer as `make weekend FOCUS="..."`:

> **Anything specific you want researched this weekend** — a ticker, a sector, or a question?
> *(Default: no — wide net across all permitted sectors.)*

Everything the four retired questions used to set is now fixed by `portfolio_rules.md`: holding
horizon 40–60 sessions; catalyst window 90 days, non-binary only; 2% risk per trade; a 5–6
position ceiling (about 4 fit at current sizing); risk posture governed by the regime filter and
the drawdown circuit breaker. Then proceed to the full deep-research report.
