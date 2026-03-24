# Portfolio Rules

These rules govern all analysis — daily and weekend. Read this file before beginning any analysis session.

---

## Budget

- No new capital beyond what is shown in the portfolio snapshot unless explicitly approved.
- Track cash to the cent after every proposed trade.
- Out-of-cycle capital injections may be declared in `<capital_injection>`. When `planned=true`, add `<amount>` to the `Cash Balance` shown in `<portfolio_snapshot>` and use the combined total as available capital for all sizing calculations.

---

## Execution Limits

- Long-only. Full shares only (no fractional).
- No options, shorting, leverage, margin, or derivatives.

---

## Universe

- U.S.-listed common stocks: nano-cap to small-cap (market cap up to $2Bn).
- Allow up to $2Bn market cap for plays.
- Allowed exchanges: NYSE, NASDAQ, NYSE American.
- Existing positions above $2Bn may be held or sold; no new shares may be added.

---

## Exclusions

- OTC / pink sheets
- ETFs, ETNs, closed-end funds, SPACs
- Rights, warrants, units, preferred shares, ADRs
- Bankrupt or halted issuers
- Defence companies
- Israeli-affiliated companies

---

## Risk Control

- Maintain or set stop-losses on ALL long positions (default: max(1.5×ATR(14), 10% below entry)).
- **Position sizing (risk-per-trade):** size so that hitting the stop costs no more than 5% of portfolio equity:
  ```
  shares = (portfolio_equity × 0.05) / (entry_price − stop_price)
  ```
  Absolute ceiling: no single name may exceed 30% of portfolio equity.
- **No averaging down:** once a position falls >5% from entry, do not add shares unless a material new positive catalyst is confirmed with ≥2 independent sources.
- **Partial profit-taking:** sell ~1/3 at +30% gain, ~1/3 at +60% gain; let the remaining third run with a trailing stop at max(1.5×ATR(14), 15% below rolling high).
- **Market regime filter:** if IWM is below its 50-day SMA, restrict new initiations to high-conviction catalyst-driven plays only. Flag the regime status in every report.
- Flag any stop breach or position sizing violation immediately.

---

## Order Defaults

- Standard limit DAY orders placed for the next trading session unless otherwise specified.
- Limit orders preferred. Market orders require explicit reasoning.

---

## Research Safeguards

### Verification
- Do NOT hallucinate tickers. Every ticker must be a verified, currently listed U.S. security on an allowed exchange.
- All market cap, float, liquidity, and catalyst data must come from reputable, up-to-date sources and must be confirmed by at least two of the sources.
- Provide citations for every holding and new candidate: source name, URL, and access timestamp.

### Catalyst Confirmation
- Any claim about catalysts (earnings dates, contract awards, regulatory decisions, etc.) must be confirmed by at least two independent sources.
- If confirmation is insufficient, explicitly state "INSUFFICIENT CONFIRMATION" and do not rely on it.

### Liquidity Filters
- Price ≥ $1.00
- 3-month average daily dollar volume ≥ $500,000
- Bid-ask spread ≤ 2% (or ≤ $0.05 if price < $5)
- Float ≥ 5M shares (unless justified with reasoning)
- Relative strength: stock price must be above its 20-day SMA at the time of initiation

### Entry Requirements
- **Catalyst within 60 days:** new initiations must have a confirmed near-term catalyst (earnings, FDA decision, contract award, etc.) within 60 days. No story stocks without an upcoming event.
- **No re-entry ban:** once a ticker is stopped out, it is banned from re-entry for 10 trading days. Flag any proposed re-entry that falls within the blackout window.

### No Candidates Rule
If no candidates pass all filters, hold cash and explain why. Do not force trades.
