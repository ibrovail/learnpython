# Architectural Patterns

This document describes the architectural patterns and design decisions used throughout the codebase.

## 1. Multi-Layer Data Fallback Pattern

**Location**: `trading_script.py:256-410`

The system uses a cascade fallback strategy for fetching market data:

```
Yahoo Finance (yfinance.download)
    ↓ if empty/error
Stooq CSV API (direct HTTP)
    ↓ if blocked
Stooq PDR (pandas_datareader)
    ↓ all fail
Empty DataFrame with source tracking
```

**Key Components**:
- `download_price_data()` at line 361 orchestrates the fallback
- `FetchResult` dataclass (line 218-221) tracks which source provided data
- `STOOQ_MAP` (line 203-211) handles symbol remapping (e.g., ^GSPC → ^SPX)
- `STOOQ_BLOCKLIST` excludes symbols unavailable on Stooq

**Design Rationale**: Decouples the system from any single data provider's availability, ensuring continuous operation during API outages.

## 2. CSV-Based State Management

**Locations**:
- State loading: `trading_script.py:1385-1451`
- Trade logging: `trading_script.py:742-776`, `778-887`, `889-985`
- Daily results: `Generate_Graph.py:95-130`

The system uses CSV files as an append-only immutable ledger:

| File | Purpose | Update Pattern |
|------|---------|---------------|
| `chatgpt_portfolio_update.csv` | Daily snapshot | Append new rows each day |
| `chatgpt_trade_log.csv` | Transaction ledger | Append on each trade |
| `capital_injections.csv` | Capital tracking | Manual append |

**Design Rationale**:
- Full auditability without database complexity
- Portability (CSVs are universally readable)
- Easy manual inspection and correction
- Git-friendly version control

## 3. Interactive Portfolio Processing Flow

**Location**: `trading_script.py:482-741` (`process_portfolio()`)

The main processing function follows a strict sequence:

1. Load latest portfolio state from CSV
2. Display current holdings and cash
3. Prompt for manual buy orders → validate and process
4. Prompt for manual sell orders → validate and process
5. Update stop-losses based on latest prices
6. Calculate daily P&L and metrics
7. Write updated CSVs
8. Display formatted daily results

**Design Rationale**: Interactive mode ensures human oversight for all trades while maintaining systematic record-keeping.

## 4. Limit Order Execution Model

**Location**: `trading_script.py:816-823`

All orders execute only if price conditions are met:

- **Buy orders**: Fill only if daily low ≤ limit price (with epsilon tolerance of $0.005)
- **Sell orders**: Fill only if daily high ≥ limit price

**Design Rationale**: Prevents lookahead bias that would occur if orders filled at any intraday price. Simulates real limit order execution.

## 5. Dollar-Weighted Benchmark Calculation

**Location**: `trading_script.py:1110-1159`

The S&P 500 benchmark calculation accounts for capital injections:

1. **Initial tranche**: Calculate SPY returns from portfolio start date
2. **Each injection**: Calculate SPY returns from injection date
3. **Sum all tranches**: Total represents what S&P would return with same cash flows

**Design Rationale**: Fair comparison against buy-and-hold strategy, accounting for the fact that capital enters at different times (dollar-cost averaging effect).

## 6. CAPM Performance Analytics

**Location**: `trading_script.py:1162-1235`

Risk-adjusted performance uses Capital Asset Pricing Model:

- **Beta**: Slope of portfolio returns vs S&P 500 (measures market sensitivity)
- **Alpha (annualized)**: Excess return unexplained by market movement
- **R²**: Goodness of fit (how much variance is explained by market)

**Implementation**: Uses `scipy.stats.linregress` for regression analysis against ^GSPC daily returns.

## 7. Defensive Error Handling

**Locations**: Throughout `trading_script.py` and `Generate_Graph.py`

Pattern of graceful degradation:

| Error Scenario | Handling |
|---------------|----------|
| Yahoo Finance fails | Fallback to Stooq (lines 361-410) |
| Missing DataFrame columns | Auto-fill with NaN (lines 248-250) |
| Date parsing errors | `errors="coerce"` (returns NaT) |
| JSON config missing | Use hardcoded defaults (lines 120-168) |
| CSV file missing | Create on first write |
| Invalid user input | Re-prompt with validation |

## 8. Date Handling for Trading Days

**Location**: `trading_script.py:175-196`

Automatic mapping of non-trading days:

- Saturday → Previous Friday
- Sunday → Previous Friday
- `ASOF_DATE` environment variable allows backtesting

**Design Rationale**: Prevents data fetch errors on weekends and enables historical simulation.

## 9. Configuration via External JSON

**Location**: `trading_script.py:120-168`

Optional `tickers.json` allows customizing:
- Benchmark tickers (default: IWO, XBI, SPY, IWM, QQQ, VIX, TLT, HYG)
- Can be extended for other configuration

**Design Rationale**: Separates configuration from code, allowing customization without code changes.

## 10. Output Formatting for LLM Consumption

**Location**: `trading_script.py:1196-1330` (`daily_results()`)

The output is specifically formatted for ChatGPT:

- Structured sections with headers (Price & Volume, Risk & Return, CAPM, Holdings)
- Fixed-width columns for tabular data
- Clear metric labels with units (%, ratios, dates)
- Copy-paste friendly format

**Design Rationale**: The daily output is designed to be copied directly into ChatGPT as context for trading decisions.
