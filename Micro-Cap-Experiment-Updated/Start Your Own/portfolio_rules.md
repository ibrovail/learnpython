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
- **Binary event stop override:** for positions held through a date-certain binary catalyst (see definition in Entry Requirements), the stop-loss may be set at the nearest major technical support level (200-day SMA, prior selloff floor, key horizontal support) rather than the standard ATR/percentage formula, provided: (a) the wider stop still results in ≤5% portfolio risk (or ≤3.75% if the SMA waiver was used for entry), (b) the override rationale is documented in the weekly report, and (c) the override automatically expires when the event resolves — see post-catalyst reassessment.
- **Position sizing (risk-per-trade):** size so that hitting the stop costs no more than 5% of portfolio equity:
  ```
  shares = (portfolio_equity × 0.05) / (entry_price − stop_price)
  ```
  Absolute ceiling: no single name may exceed 30% of portfolio equity.
- **No averaging down:** once a position falls >5% from entry, do not add shares unless a material new positive catalyst is confirmed with ≥2 independent sources.
- **Partial profit-taking:** sell ~1/3 at +30% gain, ~1/3 at +60% gain; let the remaining third run with a trailing stop at max(1.5×ATR(14), 15% below 20-day rolling high).
- **Pre-catalyst exit orders:** for any position held through a date-certain binary catalyst, set a price alert at +30% from entry at least 2 trading days before the event date. When the alert triggers, manually place a DAY limit sell for ~1/3 of the position at the alert price. This captures spike-and-reverse profit and reduces gap risk. If the sell fills before the event, do NOT replace the sold shares. *(Platform constraint: GTC limit sells are not available — only GTC stop-limit sells are supported. Use price alerts + DAY limit sells as the workaround.)*
- **Post-catalyst reassessment:** within 1 trading day of any date-certain binary catalyst resolving (approval/rejection, beat/miss, awarded/denied): (1) remove any binary event stop override and recalculate the stop using normal trailing stop rules, (2) re-evaluate conviction with documented rationale, (3) if the stock is trading below where the normal trailing stop would be, either document a specific time-bound reason to hold or exit at market, (4) log the assessment in the daily analysis.
- **Market regime filter:** if IWM is below its 50-day SMA, restrict new initiations to high-conviction catalyst-driven plays only. Freeze new momentum/technical initiations until the next weekend review. Existing momentum positions are held with current stops. Flag the regime status in every report.
- **Slippage guard:** if the intended order size exceeds 10% of the stock's average daily dollar volume, reduce the position to ≤5% of ADV.
- Flag any stop breach or position sizing violation immediately.

---

## Order Defaults

- Standard limit DAY orders placed for the next trading session unless otherwise specified.
- Limit orders preferred. Market orders require explicit reasoning.
- **Platform constraint:** GTC orders are only available for stop-limit sells. GTC limit sells are not supported. For profit-taking targets, use price alerts and place DAY limit sells when triggered.

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
- Relative strength: stock price must be above its 20-day SMA at the time of initiation, **UNLESS** all of the following are true: (1) the stock has a date-certain binary catalyst within 15 trading days, (2) the stock is within 5% of its 20-day SMA, and (3) the catalyst is confirmed by ≥2 independent sources. When this waiver is applied, reduce the risk budget from 5% to 3.75% of equity.

### Entry Requirements
- **Catalyst within 60 days (catalyst plays only):** catalyst-play initiations must have a confirmed near-term catalyst (earnings, FDA decision, contract award, etc.) within 60 days. Momentum/technical plays from the screener watchlist do not require a dated catalyst — see Allocation Framework below.
- **Date-certain binary catalyst (definition):** an event with a publicly announced date or regulatory deadline and a pass/fail outcome expected to move the stock ≥20% in either direction. Examples across sectors: FDA PDUFA date, earnings report date, government contract award deadline, permit ruling date, drill/assay result release date, patent ruling date, M&A close/termination date. Excluded: vague timelines ("H1 2026 data readout"), analyst day presentations, conference appearances.
- **No re-entry ban:** once a ticker is stopped out, it is banned from re-entry for 10 trading days. Flag any proposed re-entry that falls within the blackout window.

### No Candidates Rule
If no candidates pass all filters, hold cash and explain why. Do not force trades.

---

## Allocation Framework

The portfolio uses two complementary strategies:

### Catalyst Plays (existing rules above apply)
- Require confirmed catalyst within 60 days
- Binary event framework applies for date-certain events
- **Max allocation: 1 position, max 15% of equity per binary event play**
- All existing risk control rules apply

### Momentum/Technical Plays
- Sourced from the quantitative screener watchlist (`screener.py`)
- **No catalyst date required** — entry based on momentum, volume confirmation, and technical setup
- Must still pass all liquidity filters (price ≥$1, ADV ≥$500K, float ≥5M)
- Must be above 20-day SMA at entry (no SMA waiver for momentum plays)
- Standard stop-loss: max(1.5×ATR(14), 10% below entry)
- **Minimum hold: 5 trading days** unless stop is triggered (prevents overtrading)
- Max allocation: 3-4 positions

### Sector Diversification
- **No more than 2 of 5 positions may be in the same GICS sector.** If 2 positions are in the same sector, new candidates from that sector are blocked until one exits.
- The screener watchlist includes sector tags — check before initiating.
