# General Portfolio Support - Implementation Summary

## Overview
The three scripts have been updated to support general investment portfolios with fractional shares, no market cap restrictions, and comprehensive performance tracking against buy-and-hold and S&P 500 benchmarks.

---

## Key Changes

### 1. **trading_script.py** - Core Updates

#### Fractional Share Support
- **Removed all `int()` conversions** on share quantities
- **Added `.8f` formatting** for share display (8 decimal precision)
- **Updated all math operations** to use `float()` consistently
- **Modified comparisons** to use epsilon tolerance (`1e-8`) for "selling all shares" checks

#### Initial Portfolio Snapshot
- **New file**: `initial_portfolio_snapshot.csv`
- **New function**: `capture_initial_snapshot()`
  - Auto-detects first run (no snapshot file exists)
  - Prompts user to enter holdings if portfolio is empty
  - Accepts shares up to 8 decimals, cost basis to 2 decimals
  - Fetches current prices and calculates total value
  - Validates against first TOTAL row in portfolio CSV
  - Creates snapshot with columns: `Date`, `Ticker`, `Shares`, `Price`, `Value`, `Total_Portfolio_Value`

#### Updated Functions
- `process_portfolio()`: Removed integer share conversions
- `log_manual_buy()`: Accepts fractional shares with proper formatting
- `log_manual_sell()`: Allows partial position exits with fractional shares
- `log_sell()`: Handles fractional share stop-loss executions
- `daily_results()`: Updated display to show fractional shares with 8 decimals
- `update_stops_only()`: Updated display formatting for fractional shares

#### New Global Variable
```python
INITIAL_SNAPSHOT_CSV = DATA_DIR / "initial_portfolio_snapshot.csv"
```

---

### 2. **Generate_Graph.py** - Three-Line Comparison

#### New Features

**Three-Line Plot:**
1. **Active Portfolio** (with all trades) - solid line with circles
2. **Buy-and-Hold** (original holdings, no trades) - dotted line with triangles
3. **Dollar-Weighted S&P 500** (benchmark) - dashed line with squares

#### New Functions

**`load_initial_snapshot()`**
- Loads `initial_portfolio_snapshot.csv`
- Extracts total portfolio value and calculates cash balance
- Returns holdings DataFrame, total value, and cash

**`build_buy_and_hold_series()`**
- Reconstructs what original portfolio would be worth without any trades
- Downloads price history for each ticker in initial snapshot
- Calculates daily portfolio value: (original_shares × current_price) + cash
- Cash stays constant (no capital injection/withdrawal effects)

**`download_ticker_history()`**
- Downloads historical prices for individual tickers
- Used by buy-and-hold calculation
- Handles MultiIndex columns from yfinance

#### Updated Functions

**`build_dollar_weighted_benchmark()`**
- Unchanged functionality (already supports multiple capital injections)
- Calculates S&P 500 returns as if starting equity + injections were invested

**`plot_three_way_comparison()`**
- New name (was `plot_comparison`)
- Plots all three series
- Shows return percentages on final points
- Enhanced styling and labels

**`main()`**
- Uses initial snapshot value as default starting equity
- Builds all three series
- Prints comprehensive performance summary:
  - Active portfolio return
  - Buy-and-hold return
  - Alpha vs buy-and-hold
  - S&P 500 benchmark return
  - Alpha vs S&P 500

---

### 3. **ProcessPortfolio.py** - Simplified Wrapper

#### Changes
- Cleaner implementation using `set_data_dir()`
- Added docstring explaining general portfolio support
- Removed unnecessary CSV path argument
- More robust sys.path handling

---

## Usage Instructions

### First Run (Initial Setup)

1. **Prepare your portfolio data:**
   - If you have an empty portfolio, the script will prompt you
   - If you have existing holdings, prepare ticker symbols, shares (8 decimals), and average cost (2 decimals)

2. **Run the trading script:**
   ```bash
   python trading_script.py --data-dir /path/to/portfolio
   ```

3. **Initial snapshot capture:**
   ```
   ============================================================
   INITIAL PORTFOLIO SETUP
   ============================================================
   No initial snapshot found. This appears to be your first run.
   Please enter your CURRENT holdings to establish a baseline.

   Ticker: AAPL
   Shares (up to 8 decimals): 10.5
   Average cost per share: 150.25
   ✓ Added AAPL: 10.50000000 shares @ $150.25

   Ticker: [Enter to finish]

   Cash balance: 5000.00

   Fetching current prices...
   ✓ Initial snapshot captured: 1 positions, $51,577.63 total value
     Saved to: initial_portfolio_snapshot.csv
   ```

### Daily Operations

**Process portfolio and view results:**
```bash
python trading_script.py
```

**Update stop losses only:**
```bash
python trading_script.py --update-stops
```

**Backtest with historical date:**
```bash
python trading_script.py --asof 2025-01-15
```

### Generate Performance Chart

**Basic chart (auto-detects dates):**
```bash
python Generate_Graph.py
```

**Custom date range:**
```bash
python Generate_Graph.py --start-date 2025-01-01 --end-date 2025-10-19
```

**Save chart to file:**
```bash
python Generate_Graph.py --output portfolio_performance.png
```

**Override starting equity:**
```bash
python Generate_Graph.py --start-equity 50000
```

---

## File Structure

```
portfolio_directory/
├── trading_script.py                    # Core trading logic
├── Generate_Graph.py                    # Performance visualization
├── ProcessPortfolio.py                  # Wrapper script
├── chatgpt_portfolio_update.csv        # Daily portfolio snapshots
├── chatgpt_trade_log.csv               # Trade history
├── capital_injections.csv              # Capital injection log
├── initial_portfolio_snapshot.csv      # 🆕 Initial holdings baseline
└── tickers.json                        # (Optional) Custom benchmarks
```

---

## CSV Schemas

### initial_portfolio_snapshot.csv (NEW)
```csv
Date,Ticker,Shares,Price,Value,Total_Portfolio_Value
2025-01-15,AAPL,10.50000000,150.25,1577.63,51577.63
2025-01-15,MSFT,8.25000000,380.50,3139.13,51577.63
2025-01-15,GOOGL,15.00000000,140.75,2111.25,51577.63
```

### chatgpt_portfolio_update.csv (UPDATED - fractional shares)
```csv
Date,Ticker,Shares,Buy Price,Cost Basis,Stop Loss,Current Price,Total Value,PnL,Action,Cash Balance,Total Equity
2025-01-15,AAPL,10.50000000,150.25,1577.63,145.00,152.30,1599.15,21.52,HOLD,,,
2025-01-15,TOTAL,,,,,,,1599.15,21.52,,5000.00,6599.15
```

---

## Backward Compatibility

✅ **Micro-cap portfolio scripts still work**
- Integer shares are just floats with `.0`
- Existing CSV files are read correctly
- No breaking changes to CSV structure

✅ **No initial snapshot required for micro-cap**
- Script checks if file exists before prompting
- Micro-cap portfolios can skip this feature

---

## Validation & Error Handling

### Initial Snapshot Validation
- Cross-checks snapshot total against first portfolio TOTAL row
- Warns if values differ by more than $1 (rounding tolerance)
- Example output:
  ```
  ✓ Validation: First portfolio TOTAL row matches snapshot
  ```

### Fractional Share Precision
- Uses epsilon tolerance (`1e-8`) for "close to zero" checks
- Prevents floating-point precision issues
- Example: `0.00000001` shares treated as "empty position"

### Missing Data Handling
- Buy-and-hold: forward-fills missing price data
- S&P 500 benchmark: forward-fills missing dates
- User-friendly error messages for missing files

---

## Performance Metrics

### Daily Results Output
- **Risk & Return**: Max Drawdown, Sharpe Ratio, Sortino Ratio
- **CAPM Analysis**: Beta, Alpha, R² vs S&P 500
- **Portfolio Snapshot**: Latest equity, cash balance, holdings with fractional shares

### Chart Output
- **Three-line visualization** with return percentages
- **Alpha calculations** vs buy-and-hold and S&P 500
- **Clean formatting** with daily tick marks
- **Professional styling** suitable for reporting

---

## Example Workflow

### Day 1: Initial Setup
```bash
$ python trading_script.py
# Enter initial holdings when prompted
# Script creates initial_portfolio_snapshot.csv
```

### Day 2-N: Daily Operations
```bash
$ python trading_script.py
# Process trades, update stops, view metrics
# CSV files updated automatically
```

### Weekly: Performance Review
```bash
$ python Generate_Graph.py --output weekly_report.png
# Generates 3-line chart showing:
# - How portfolio performed with trades
# - How it would have performed without trades (buy-and-hold)
# - How S&P 500 performed (benchmark)
```

---

## Key Advantages

✅ **Fractional shares**: Track partial positions (e.g., DRIP, robo-advisors)  
✅ **Buy-and-hold baseline**: See if active management adds value  
✅ **Dollar-weighted S&P 500**: Fair benchmark accounting for cash flows  
✅ **One-time setup**: Initial snapshot captured automatically  
✅ **Backward compatible**: Works with existing micro-cap portfolios  
✅ **Professional reporting**: Publication-quality charts and metrics  

---

## Notes

- **Fractional precision**: Up to 8 decimals (covers most brokerage precision)
- **Cash handling**: Stays constant in buy-and-hold (no re-investment assumed)
- **Capital injections**: Properly weighted in S&P 500 benchmark
- **Weekend handling**: Auto-adjusts to Friday prices for Saturday/Sunday runs
- **Data sources**: Yahoo Finance primary, Stooq fallback, proxy indices for gaps