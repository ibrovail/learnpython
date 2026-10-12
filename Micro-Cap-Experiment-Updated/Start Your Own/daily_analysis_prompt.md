# Daily Analysis Format

Use this format when producing the daily portfolio analysis after `make daily`.

Portfolio rules are in `Start Your Own/portfolio_rules.md` — read that file first.

---

## Daily Output Format (6 Sections)

```
Daily Portfolio Review — [DATE]

---

1. Scoreboard and Regime

   **Gap vs S&P-equivalent: ±X.XX% since re-base (2026-09-11)** — equity $X.XX against
   S&P-equivalent $X.XX, both from <portfolio_snapshot>. This is the scoreboard; lead with it.
   *(Added 2026-10-11: the daily had no scoreboard line at all, so the one number the book is
   judged on appeared only in weekend reports. The 10/11 audit also established where the
   realistic edge is — the adopted rules trail SPY by ~1pp on return but run roughly HALF its
   drawdown — so a widening gap in a falling market is not automatically a process failure.
   Say which it is.)*
   Drawdown from peak: ±X.XX% [from <risk_metrics>] — clear / DE-RISK at −20% / CASH at −30%

   IWM: $[price] | 50-day SMA: $[value] | [±X.XX%] (source: the script's <market_regime> block —
   computed from IWM closes; do not look it up elsewhere)
   Regime: RISK-ON / RISK-OFF [add "held — inside the ±1% band" when <market_regime> says so]
   Rule applied: [the capacity this regime allows — see Section 3]

---

2. Holdings — by exception (review item R6, 2026-09-19)

   The script's <holding_review> block marks each holding FULL or LINE. Also treat a holding as
   FULL if the user's `Run daily:` message says "full review TICKER", or if the news check below
   turns up anything material.

   **Every holding, every day — the news check:** open its live news feed (browser) and state
   "nothing new since the last close [timestamp]" or what landed. The script cannot see news.

   **Also every day, when a catalyst slot is open — the index-change check.** One fetch of
   `press.spglobal.com` for S&P DJI index announcements. *(Added 2026-10-12.)* This lives in the
   **daily**, not the weekend, for a measured reason: both October 2026 additions were M&A-driven
   **intra-quarter replacements announced with 3–5 days' notice** (WK for FORM, effective Oct 6;
   FRSH for BLFS, effective Oct 8). A weekend-only check structurally sees those *after* the
   effective date, which is exactly how Week 57 found both and had to kill them on timing — the
   forced buying ends at the effective date. **The surface is productive; the cadence was wrong.**
   State "no new S&P DJI release since [date]" or name the add, its effective date, and whether
   there are sessions left before it.

   **LINE holdings — one line each:**
   [TICKER] — $X.XX (±X.XX%), P&L ±X.X%, stop $X.XX (X.XX×ATR room), driver unchanged, news: nothing new [time]

   **FULL holdings — the block below, headed by the flag that triggered it:**

   2a. [TICKER] — Full review ([flag from <holding_review>])

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

   Stop-Loss Update (portfolio_rules.md → "Raising a stop — the default is DO NOT"):
   **The default is HOLD. Do not raise.** Adopted 2026-10-11 and measured at **10.3pp** — raising
   whenever the tests pass takes the stop-out rate from 9% to 98% and the average hold from 57
   sessions to 14. The tests below are **necessary, not sufficient**: a passing raise is still
   declined unless you can name what changed about the thesis, in writing, in this review.
   Compute and report them anyway — eligibility is information — then default to HOLD.
   *(This block previously ended "RAISE only if ALL pass", which treated eligibility as
   sufficient. Three CON raises were recommended in five sessions on exactly that reading.)*
   - ATR(14): $X.XX — computed from yfinance price history, never estimated or searched
   - Close / today's low / 10-day low: $X.XX / $X.XX / $X.XX
   - Current stop: $X.XX = X.XX×ATR below the close   (1.5×ATR floor applies when PLACING a stop;
     drift below it afterwards is expected — note it and check restoration eligibility)
   - Candidate level: close − 2.0×ATR = $X.XX         (raise target 2.0×ATR — never raise TO the floor)
   - Test 1, raise size:   (candidate − current stop) ÷ ATR = X.XX   ≥ 0.50 → PASS / FAIL
   - Test 2, room left:    (close − candidate) ÷ ATR = X.XX          ≥ 1.50 → PASS / FAIL
   - Range check: candidate below today's low → PASS / FAIL
   Action: **HOLD** — the default, even when all three tests pass. Propose RAISE to $X.XX / $X.XX
   only with a named, written thesis change (not "price rose", not "the tests pass"). A failing
   raise is declined, not reduced. If the current stop already sits below 1.5×ATR through price
   movement alone, say so and state whether this entry's one restoration is still available —
   restoration is a separate, mechanical allowance and is unaffected by the never-raise default.

   Add Shares? (FULL reviews only — and only if a slot and deployable cash above the **5%** floor
   exist; routine
   deployment decisions belong to the research report)
   - Risk budget: $[equity] × 2% = $[amount]
   - Risk per share at $[entry] entry / $[new stop] stop: $[diff]
   - Formula: $[budget] / $[risk/share] = [N] shares
   - 30% cap: $[equity] × 30% = $[max_value] → [max_shares] shares max total
   - Current position: [N] shares → room for [N] additional
   Decision: ADD [N] shares / NO ADD
   Rationale: [one-line reason]

---

3. New Positions

   State the capacity this regime allows, copied from `<lanes>`, and what remains.

   **RISK-OFF (revised 2026-10-11):** up to **3 non-binary catalyst positions at the full 2%**,
   and the **screener lane is CLOSED** — `SCREENER_CAP["RISK-OFF"] = 0`. A screener-sourced buy
   recommended in RISK-OFF is a rule violation, not a judgement call. The RISK-OFF defensive
   profile is **removed** (it cost 2.8pp of test return and was not defensive: top-decile calm
   names had median beta 0.85 against 0.84 for everything else), and its sunset clause expired
   as written rather than being extended. A holding already open when the lane closed is
   **grandfathered** — `<lanes>` printing `screener 1/0` is expected, because capacity gates
   govern new capital only.
   **RISK-ON:** up to 3 catalyst and up to 7 screener positions, both at the full 2%.
   **Either regime:** ceiling 10 positions, ≤3 per GICS sector, ≤2 per primary driver.

   [If screening candidates:]
   | Ticker | Thesis | Catalyst (≥2 sources) | Liquidity | Bear Case | Sizing |
   |--------|--------|----------------------|-----------|-----------|--------|

---

4. Final Decisions

   (One block per action. Omit if no action needed.)

   ACTION:        [BUY / SELL / UPDATE STOP / HOLD]
   Ticker:        [symbol]
   Shares:        [fractional permitted since 2026-10-11 — round to 4 dp; omit for stop updates]
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
   | Positive      | [hold; do NOT raise the stop — price rising is not a thesis change] |
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
horizon 40–60 sessions; catalyst window 90 days, non-binary only; 2% risk per trade; a 10
position ceiling (about 4 fit at current sizing); risk posture governed by the regime filter and
the drawdown circuit breaker. Then proceed to the full deep-research report.
