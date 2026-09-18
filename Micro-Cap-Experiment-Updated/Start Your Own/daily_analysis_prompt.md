# Daily Analysis Format

Use this format when producing the daily portfolio analysis after `make daily`.

Portfolio rules are in `Start Your Own/portfolio_rules.md` — read that file first.

---

## Daily Output Format (6 Sections)

```
Daily Portfolio Review — [DATE]

---

1. Market Regime Check

   IWM: $[price] | 50-day SMA: $[value] (source: [WebSearch result])
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

   Stop-Loss Update (trailing stop rule: max(1.5×ATR(14), 15% below rolling high)):
   - Rolling high: $X.XX ([date])
   - 15% below rolling high: $X.XX
   - 1.5×ATR(14): $X.XX (source: [WebSearch])
   - Trailing stop floor: $X.XX (the higher of the two)
   - Current stop: $X.XX → [RAISE to $X.XX / HOLD — already above floor]
   Action: [Raise stop-loss to $X.XX, stop-limit to $X.XX / No change needed]

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

   [RISK-OFF regime: "Market regime is RISK-OFF (IWM below 50-day SMA). No new initiations
   unless high-conviction catalyst-driven. Holding $X.XX cash."]

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

## Weekend Session Directive Questions

Before running the weekend deep research analysis, ask the user these 4 questions and update the
`<session_directives>` block in `weekend_summary.md` with their answers:

**Q1 — Sector focus:**
> Wide net across all sectors (default) | Biotech | Energy | Tech | Industrials

**Q2 — Catalyst timing priority:**
> Within 5 trading days | Within 10 trading days | 30–60 days (medium-term, high conviction only)

**Q3 — Risk posture:**
> Neutral | Aggressive — we are trailing the benchmark | Defensive — protect recent gains | Tighten all stop-losses by one ATR

**Q4 — Max concurrent positions:**
> 5 | 6

*(A ceiling, not a target. At 2% risk with 1.75×ATR stops a typical position runs 19–29% of
equity, so an 85% deployable book fits roughly four.)*

Update `<session_directives>` to include the selected options as active directives (not as comments),
then proceed immediately to the full 10-section deep research report defined in `weekend_summary.md`.
