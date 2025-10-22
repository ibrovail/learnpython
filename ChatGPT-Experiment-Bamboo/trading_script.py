"""Utilities for maintaining investment portfolios with fractional share support.

This module supports:
- Fractional shares (up to 8 decimal places)
- General investment portfolios (no market cap restrictions)
- Initial portfolio snapshot for buy-and-hold comparison
- Centralized market data fetching with Yahoo->Stooq fallback
- Weekend handling and testable date logic
- Capital injection tracking for accurate benchmarking

Notes:
- Some tickers/indices are not available on Stooq (e.g., ^RUT). These stay on Yahoo.
- Stooq end date is exclusive; we add +1 day for ranges.
- "Adj Close" is set equal to "Close" for Stooq to match downstream expectations.
- Capital injections are logged separately to enable dollar-weighted S&P comparison.
- Initial snapshot captured on first run to establish buy-and-hold baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
import os
import warnings

import numpy as np # type: ignore
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
import json
import logging

# Optional pandas-datareader import for Stooq access
try:
    import pandas_datareader.data as pdr # type: ignore
    _HAS_PDR = True
except Exception:
    _HAS_PDR = False

# -------- AS-OF override --------
ASOF_DATE: pd.Timestamp | None = None

def set_asof(date: str | datetime | pd.Timestamp | None) -> None:
    """Set a global 'as of' date so the script treats that day as 'today'. Use 'YYYY-MM-DD' format."""
    global ASOF_DATE
    if date is None:
        print("No prior date passed. Using today's date...")
        ASOF_DATE = None
        return
    ASOF_DATE = pd.Timestamp(date).normalize()
    pure_date = ASOF_DATE.date()
    print(f"Setting date as {pure_date}.")

_env_asof = os.environ.get("ASOF_DATE")
if _env_asof:
    set_asof(_env_asof)

def _effective_now() -> datetime:
    return (ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.now())

# ------------------------------
# Globals / file locations
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
CAPITAL_INJECTIONS_CSV = DATA_DIR / "capital_injections.csv"
INITIAL_SNAPSHOT_CSV = DATA_DIR / "initial_portfolio_snapshot.csv"
DEFAULT_BENCHMARKS = ["IWO", "XBI", "SPY", "IWM"]

logger = logging.getLogger(__name__)

def _log_initial_state():
    """Log the initial global file path configuration."""
    logger.info("=== Trading Script Initial Configuration ===")
    logger.info("Script directory: %s", SCRIPT_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Portfolio CSV: %s", PORTFOLIO_CSV)
    logger.info("Trade log CSV: %s", TRADE_LOG_CSV)
    logger.info("Capital injections CSV: %s", CAPITAL_INJECTIONS_CSV)
    logger.info("Initial snapshot CSV: %s", INITIAL_SNAPSHOT_CSV)
    logger.info("Default benchmarks: %s", DEFAULT_BENCHMARKS)
    logger.info("==============================================")

# ------------------------------
# Configuration helpers
# ------------------------------

def _read_json_file(path: Path) -> Optional[Dict]:
    """Read and parse JSON from path."""
    try:
        logger.info("Reading JSON file: %s", path)
        with path.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
            logger.info("Successfully read JSON file: %s", path)
            return result
    except FileNotFoundError:
        logger.info("JSON file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("tickers.json malformed: %s. Falling back to defaults.", exc)
        return None
    except Exception as exc:
        logger.warning("Unable to read tickers.json: %s. Falling back to defaults.", exc)
        return None

def load_benchmarks(script_dir: Path | None = None) -> List[str]:
    """Return list of benchmark tickers."""
    base = Path(script_dir) if script_dir else SCRIPT_DIR
    candidates = [base, base.parent]
    
    cfg = None
    for c in candidates:
        p = (c / "tickers.json").resolve()
        data = _read_json_file(p)
        if data is not None:
            cfg = data
            break
    
    if not cfg:
        return DEFAULT_BENCHMARKS.copy()
    
    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, list):
        return DEFAULT_BENCHMARKS.copy()
    
    seen = set()
    result: list[str] = []
    for t in benchmarks:
        if not isinstance(t, str):
            continue
        up = t.strip().upper()
        if up and up not in seen:
            seen.add(up)
            result.append(up)
    
    return result if result else DEFAULT_BENCHMARKS.copy()

# ------------------------------
# Date helpers
# ------------------------------

def last_trading_date(today: datetime | None = None) -> pd.Timestamp:
    """Return last trading date (Mon–Fri)."""
    dt = pd.Timestamp(today or _effective_now())
    if dt.weekday() == 5:
        return (dt - pd.Timedelta(days=1)).normalize()
    if dt.weekday() == 6:
        return (dt - pd.Timedelta(days=2)).normalize()
    return dt.normalize()

def check_weekend() -> str:
    """Return ISO date string for last trading day."""
    return last_trading_date().date().isoformat()

def trading_day_window(target: datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) window for last trading day."""
    d = last_trading_date(target)
    return d, (d + pd.Timedelta(days=1))

# ------------------------------
# Data access layer
# ------------------------------

STOOQ_MAP = {"^GSPC": "^SPX", "^DJI": "^DJI", "^IXIC": "^IXIC"}
STOOQ_BLOCKLIST = {"^RUT"}

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if len(set(df.columns.get_level_values(1))) == 1:
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            else:
                df = df.copy()
                df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
        except Exception:
            df = df.copy()
            df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
    
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download silently."""
    import io
    from contextlib import redirect_stderr, redirect_stdout
    
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                df = cast(pd.DataFrame, yf.download(ticker, **kwargs))
        except Exception:
            return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _weekend_safe_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute [start, end) window."""
    if start or end:
        end_ts = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
        return start_ts.normalize(), pd.Timestamp(end_ts).normalize()
    
    days = int(period[:-1]) if isinstance(period, str) and period.endswith("d") else 1
    end_trading = last_trading_date()
    start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
    end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
    return start_ts, end_ts

def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """Robust OHLCV fetch with fallbacks."""
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)
    
    s, e = _weekend_safe_range(period, start, end)
    
    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")
    
    proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")
    
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return FetchResult(empty, "empty")

# ------------------------------
# File path configuration
# ------------------------------

def set_data_dir(data_dir: Path) -> None:
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV, CAPITAL_INJECTIONS_CSV, INITIAL_SNAPSHOT_CSV
    logger.info("Setting data directory: %s", data_dir)
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
    CAPITAL_INJECTIONS_CSV = DATA_DIR / "capital_injections.csv"
    INITIAL_SNAPSHOT_CSV = DATA_DIR / "initial_portfolio_snapshot.csv"

# ------------------------------
# Capital Injection Management
# ------------------------------

def log_capital_injection(amount: float) -> None:
    """Log capital injection with today's date."""
    today = check_weekend()
    log = {"Date": today, "Amount": amount}
    
    if CAPITAL_INJECTIONS_CSV.exists():
        df = pd.read_csv(CAPITAL_INJECTIONS_CSV)
        df = pd.DataFrame([log]) if df.empty else pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    
    df.to_csv(CAPITAL_INJECTIONS_CSV, index=False)

def load_capital_injections() -> pd.DataFrame:
    """Load all capital injections."""
    if not CAPITAL_INJECTIONS_CSV.exists():
        return pd.DataFrame(columns=["Date", "Amount"])
    
    df = pd.read_csv(CAPITAL_INJECTIONS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

# ------------------------------
# Initial Portfolio Snapshot
# ------------------------------

def capture_initial_snapshot(portfolio_df: pd.DataFrame, cash: float, interactive: bool = True) -> None:
    """One-time capture of initial portfolio state."""
    if INITIAL_SNAPSHOT_CSV.exists():
        return
    
    if not interactive:
        return
    
    print("\n" + "="*64)
    print("INITIAL PORTFOLIO SETUP")
    print("="*64)
    print("No initial snapshot found. This appears to be your first run.")
    print("Please enter your CURRENT holdings to establish a baseline.\n")
    
    if portfolio_df.empty or len(portfolio_df) == 0:
        print("No holdings detected. Please enter your positions:")
        print("(Enter shares up to 8 decimals, average cost to 2 decimals)\n")
        
        holdings = []
        while True:
            ticker = input("Ticker (or press Enter to finish): ").strip().upper()
            if not ticker:
                break
            
            try:
                shares = float(input(f"Shares for {ticker} (up to 8 decimals): "))
                if shares <= 0:
                    print("Shares must be positive. Try again.")
                    continue
                
                avg_cost = float(input(f"Average cost per share for {ticker}: $"))
                if avg_cost <= 0:
                    print("Cost must be positive. Try again.")
                    continue
                
                holdings.append({
                    "ticker": ticker,
                    "shares": round(shares, 8),
                    "buy_price": round(avg_cost, 2),
                    "cost_basis": round(shares * avg_cost, 2),
                    "stop_loss": 0.0
                })
                
                print(f"✓ Added {ticker}: {shares:.8f} shares @ ${avg_cost:.2f}\n")
            except ValueError:
                print("Invalid input. Try again.\n")
                continue
        
        if not holdings:
            print("No holdings entered. Skipping initial snapshot.")
            return
        
        portfolio_df = pd.DataFrame(holdings)
    
    capture = input("\nCapture current portfolio as initial snapshot? (y/n): ").strip().lower()
    if capture != 'y':
        print("Skipping initial snapshot.")
        return
    
    today = last_trading_date().date().isoformat()
    snapshot_rows = []
    s, e = trading_day_window()
    total_value = cash
    
    print("\nFetching current prices...")
    for _, holding in portfolio_df.iterrows():
        ticker = str(holding["ticker"]).upper()
        shares = float(holding["shares"])
        
        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        if fetch.df.empty:
            price = float(holding.get("buy_price", 0))
        else:
            price = float(fetch.df["Close"].iloc[-1])
        
        value = shares * price
        total_value += value
        
        snapshot_rows.append({
            "Date": today,
            "Ticker": ticker,
            "Shares": round(shares, 8),
            "Price": round(price, 2),
            "Value": round(value, 2),
            "Total_Portfolio_Value": 0
        })
    
    for row in snapshot_rows:
        row["Total_Portfolio_Value"] = round(total_value, 2)
    
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df.to_csv(INITIAL_SNAPSHOT_CSV, index=False)
    
    print(f"\n✓ Initial snapshot captured: {len(snapshot_rows)} positions, ${total_value:,.2f} total value")
    print(f"  Saved to: {INITIAL_SNAPSHOT_CSV}")
    
    # Also create initial portfolio CSV entry with these holdings
    print("\nCreating initial portfolio tracking entries...")
    portfolio_results = []
    for row in snapshot_rows:
        portfolio_results.append({
            "Date": today,
            "Ticker": row["Ticker"],
            "Shares": row["Shares"],
            "Buy Price": row["Price"],
            "Cost Basis": row["Value"],
            "Stop Loss": 0.0,
            "Current Price": row["Price"],
            "Total Value": row["Value"],
            "PnL": 0.0,
            "Action": "INITIAL HOLDINGS",
            "Cash Balance": "",
            "Total Equity": ""
        })
    
    # Add TOTAL row
    portfolio_results.append({
        "Date": today,
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value - cash, 2),
        "PnL": 0.0,
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value, 2)
    })
    
    portfolio_df = pd.DataFrame(portfolio_results)
    portfolio_df.to_csv(PORTFOLIO_CSV, index=False)
    print(f"✓ Initial portfolio entries created in {PORTFOLIO_CSV}")


# ------------------------------
# Portfolio operations
# ------------------------------

def _ensure_df(portfolio: pd.DataFrame | dict | list) -> pd.DataFrame:
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        df = pd.DataFrame(portfolio)
        if df.empty:
            df = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return df
    raise TypeError("portfolio must be DataFrame, dict, or list[dict]")

def process_portfolio(
    portfolio: pd.DataFrame | dict | list,
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    today_iso = last_trading_date().date().isoformat()
    portfolio_df = _ensure_df(portfolio)
    results: list[dict] = []
    total_value = 0.0
    total_pnl = 0.0

    if interactive:
        print("\n--- Capital Injection ---")
        inject_choice = input("Inject additional capital? ('y' or Enter to skip): ").strip().lower()
        if inject_choice == "y":
            try:
                inject_amount = float(input("Enter amount to inject: "))
                if inject_amount > 0:
                    log_capital_injection(inject_amount)
                    cash += inject_amount
                    print(f"Injected ${inject_amount:.2f}. New cash: ${cash:.2f}")
            except ValueError:
                print("Invalid input. Skipping injection.")

    if interactive:
        while True:
            if not portfolio_df.empty:
                display_df = portfolio_df.copy()
                display_df['shares'] = display_df['shares'].apply(lambda x: f"{float(x):.8f}")
                print(display_df)
            
            action = input(f"\nYou have ${cash:.2f} in cash.\nLog manual trade? 'b'=buy, 's'=sell, Enter=continue: ").strip().lower()

            if action == "b":
                ticker = input("Ticker: ").strip().upper()
                order_type = input("Order type? 'm'=market-on-open, 'l'=limit: ").strip().lower()

                try:
                    shares = float(input("Shares (fractional allowed): "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid shares. Cancelled.")
                    continue

                if order_type == "m":
                    try:
                        stop_loss = float(input("Stop loss (or 0): "))
                        if stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid stop loss. Cancelled.")
                        continue

                    s, e = trading_day_window()
                    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False)
                    if fetch.df.empty:
                        print(f"MOO buy failed: no data (source={fetch.source}).")
                        continue

                    exec_price = round(float(fetch.df["Open"].iloc[-1] if "Open" in fetch.df else fetch.df["Close"].iloc[-1]), 2)
                    notional = exec_price * shares
                    if notional > cash:
                        print(f"MOO buy failed: cost ${notional:.2f} > cash ${cash:.2f}.")
                        continue

                    log = {
                        "Date": today_iso, "Ticker": ticker, "Shares Bought": shares,
                        "Buy Price": exec_price, "Cost Basis": notional, "PnL": 0.0,
                        "Reason": "MANUAL BUY MOO - Filled"
                    }
                    if TRADE_LOG_CSV.exists():
                        df_log = pd.read_csv(TRADE_LOG_CSV)
                        df_log = pd.DataFrame([log]) if df_log.empty else pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
                    else:
                        df_log = pd.DataFrame([log])
                    df_log.to_csv(TRADE_LOG_CSV, index=False)

                    rows = portfolio_df.loc[portfolio_df["ticker"].astype(str).str.upper() == ticker]
                    if rows.empty:
                        new_trade = {"ticker": ticker, "shares": float(shares), "stop_loss": float(stop_loss),
                                   "buy_price": float(exec_price), "cost_basis": float(notional)}
                        portfolio_df = pd.DataFrame([new_trade]) if portfolio_df.empty else pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True)
                    else:
                        idx = rows.index[0]
                        cur_shares = float(portfolio_df.at[idx, "shares"])
                        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
                        new_shares = cur_shares + shares
                        new_cost = cur_cost + notional
                        portfolio_df.at[idx, "shares"] = new_shares
                        portfolio_df.at[idx, "cost_basis"] = new_cost
                        portfolio_df.at[idx, "buy_price"] = new_cost / new_shares
                        portfolio_df.at[idx, "stop_loss"] = stop_loss

                    cash -= notional
                    print(f"MOO buy filled at ${exec_price:.2f}.")
                    continue

                elif order_type == "l":
                    try:
                        buy_price = float(input("Buy LIMIT price: "))
                        stop_loss = float(input("Stop loss (or 0): "))
                        if buy_price <= 0 or stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid input. Cancelled.")
                        continue
                    cash, portfolio_df = log_manual_buy(buy_price, shares, ticker, stop_loss, cash, portfolio_df)
                    continue

            if action == "s":
                try:
                    ticker = input("Ticker: ").strip().upper()
                    shares = float(input("Shares to sell (fractional): "))
                    sell_price = float(input("Sell LIMIT price: "))
                    if shares <= 0 or sell_price <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Cancelled.")
                    continue
                cash, portfolio_df = log_manual_sell(sell_price, shares, ticker, cash, portfolio_df)
                continue

            break

    # Stop-loss updates removed from interactive mode
    # Use --update-stops flag for dedicated stop-loss update workflow

    s, e = trading_day_window()
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = float(stock["shares"]) if not pd.isna(stock["shares"]) else 0.0
        cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else cost * shares
        stop = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False)
        data = fetch.df

        if data.empty:
            print(f"No data for {ticker} (source={fetch.source}).")
            results.append({
                "Date": today_iso, "Ticker": ticker, "Shares": shares, "Buy Price": cost,
                "Cost Basis": cost_basis, "Stop Loss": stop, "Current Price": "", "Total Value": "",
                "PnL": "", "Action": "NO DATA", "Cash Balance": "", "Total Equity": ""
            })
            continue

        o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])
        if np.isnan(o):
            o = c

        if stop and l <= stop:
            exec_price = round(o if o <= stop else stop, 2)
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - cost) * shares, 2)
            cash += value
            portfolio_df = log_sell(ticker, shares, exec_price, cost, pnl, portfolio_df)
            results.append({
                "Date": today_iso, "Ticker": ticker, "Shares": shares, "Buy Price": cost,
                "Cost Basis": cost_basis, "Stop Loss": stop, "Current Price": exec_price,
                "Total Value": value, "PnL": pnl, "Action": "SELL - Stop Loss Triggered",
                "Cash Balance": "", "Total Equity": ""
            })
        else:
            price = round(c, 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)
            total_value += value
            total_pnl += pnl
            results.append({
                "Date": today_iso, "Ticker": ticker, "Shares": shares, "Buy Price": cost,
                "Cost Basis": cost_basis, "Stop Loss": stop, "Current Price": price,
                "Total Value": value, "PnL": pnl, "Action": "HOLD",
                "Cash Balance": "", "Total Equity": ""
            })

    results.append({
        "Date": today_iso, "Ticker": "TOTAL", "Shares": "", "Buy Price": "", "Cost Basis": "",
        "Stop Loss": "", "Current Price": "", "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2), "Action": "", "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2)
    })

    df_out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != str(today_iso)]
        df_out = pd.concat([existing, df_out], ignore_index=True)
    df_out.to_csv(PORTFOLIO_CSV, index=False)

    return portfolio_df, cash

# ------------------------------
# Trade logging
# ------------------------------

def log_sell(ticker: str, shares: float, price: float, cost: float, pnl: float, portfolio: pd.DataFrame) -> pd.DataFrame:
    today = check_weekend()
    log = {"Date": today, "Ticker": ticker, "Shares Sold": shares, "Sell Price": price, "Cost Basis": cost, "PnL": pnl, "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED"}
    print(f"{ticker} stop loss met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.DataFrame([log]) if df.empty else pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio

def log_manual_buy(buy_price: float, shares: float, ticker: str, stoploss: float, cash: float, portfolio: pd.DataFrame, interactive: bool = True) -> tuple[float, pd.DataFrame]:
    today = check_weekend()

    if interactive:
        check = input(f"BUY LIMIT {shares:.8f} {ticker} @ ${buy_price:.2f}. Type '1' to cancel or Enter: ")
        if check == "1":
            return cash, portfolio

    if not isinstance(portfolio, pd.DataFrame) or portfolio.empty:
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False)
    if fetch.df.empty:
        print(f"Manual buy failed: no data (source={fetch.source}).")
        return cash, portfolio

    l = float(fetch.df["Low"].iloc[-1])
    if l > buy_price + 0.005:
        print(f"Buy limit ${buy_price:.2f} not reached. Order not filled.")
        return cash, portfolio

    cost_amt = buy_price * shares
    if cost_amt > cash:
        print(f"Buy failed: cost ${cost_amt:.2f} > cash ${cash:.2f}.")
        return cash, portfolio

    log = {"Date": today, "Ticker": ticker, "Shares Bought": shares, "Buy Price": buy_price, "Cost Basis": cost_amt, "PnL": 0.0, "Reason": "MANUAL BUY LIMIT - Filled"}
    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.DataFrame([log]) if df.empty else pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    rows = portfolio.loc[portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        new_trade = {"ticker": ticker, "shares": float(shares), "stop_loss": float(stoploss), "buy_price": float(buy_price), "cost_basis": float(cost_amt)}
        portfolio = pd.DataFrame([new_trade]) if portfolio.empty else pd.concat([portfolio, pd.DataFrame([new_trade])], ignore_index=True)
    else:
        idx = rows.index[0]
        cur_shares = float(portfolio.at[idx, "shares"])
        cur_cost = float(portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + shares
        new_cost = cur_cost + cost_amt
        portfolio.at[idx, "shares"] = new_shares
        portfolio.at[idx, "cost_basis"] = new_cost
        portfolio.at[idx, "buy_price"] = new_cost / new_shares
        portfolio.at[idx, "stop_loss"] = stoploss

    cash -= cost_amt
    print(f"BUY LIMIT filled at ${buy_price:.2f}.")
    return cash, portfolio

def log_manual_sell(sell_price: float, shares_sold: float, ticker: str, cash: float, portfolio: pd.DataFrame, reason: str | None = None, interactive: bool = True) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    if interactive:
        reason = input(f"SELL LIMIT {shares_sold:.8f} {ticker} @ ${sell_price:.2f}. Type '1' to cancel or Enter: ")
    if reason == "1":
        return cash, portfolio
    elif reason is None:
        reason = ""

    if ticker not in portfolio["ticker"].values:
        print(f"Sell failed: {ticker} not in portfolio.")
        return cash, portfolio

    ticker_row = portfolio[portfolio["ticker"] == ticker]
    total_shares = float(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(f"Sell failed: {shares_sold:.8f} > owned {total_shares:.8f}.")
        return cash, portfolio

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False)
    if fetch.df.empty:
        print(f"Sell failed: no data (source={fetch.source}).")
        return cash, portfolio

    h = float(fetch.df["High"].iloc[-1])
    if h < sell_price - 0.005:
        print(f"Sell limit ${sell_price:.2f} not reached. Order not filled.")
        return cash, portfolio

    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = sell_price * shares_sold - cost_basis

    log = {"Date": today, "Ticker": ticker, "Shares Bought": "", "Buy Price": "", "Cost Basis": cost_basis, "PnL": pnl, "Reason": f"MANUAL SELL LIMIT - {reason}", "Shares Sold": shares_sold, "Sell Price": sell_price}
    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.DataFrame([log]) if df.empty else pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if abs(total_shares - shares_sold) < 1e-8:
        portfolio = portfolio[portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        portfolio.at[row_index, "shares"] = total_shares - shares_sold
        portfolio.at[row_index, "cost_basis"] = portfolio.at[row_index, "shares"] * portfolio.at[row_index, "buy_price"]

    cash += shares_sold * sell_price
    print(f"SELL LIMIT filled at ${sell_price:.2f}.")
    return cash, portfolio

# ------------------------------
# Reporting / Metrics
# ------------------------------

def daily_results(portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics."""
    portfolio_dict: list[dict] = portfolio.to_dict(orient="records")
    today = check_weekend()

    rows: list[list[str]] = []
    header = ["Ticker", "Close", "% Chg", "Volume"]

    end_d = last_trading_date()
    start_d = (end_d - pd.Timedelta(days=4)).normalize()
    
    benchmarks = load_benchmarks()
    benchmark_entries = [{"ticker": t} for t in benchmarks]

    for stock in portfolio_dict + benchmark_entries:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(ticker, start=start_d, end=(end_d + pd.Timedelta(days=1)))
            data = fetch.df
            if data.empty or len(data) < 2:
                rows.append([ticker, "—", "—", "—"])
                continue

            price = float(data["Close"].iloc[-1])
            last_price = float(data["Close"].iloc[-2])
            volume = float(data["Volume"].iloc[-1])
            percent_change = ((price - last_price) / last_price) * 100
            rows.append([ticker, f"{price:,.2f}", f"{percent_change:+.2f}%", f"{int(volume):,}"])
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e}")

    if not PORTFOLIO_CSV.exists():
        print("\n" + "=" * 64)
        print(f"Daily Results – {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        if not portfolio.empty:
            display_portfolio = portfolio.copy()
            display_portfolio['shares'] = display_portfolio['shares'].apply(lambda x: f"{float(x):.8f}")
            print(display_portfolio)
        else:
            print("No holdings.")
        print(f"Cash balance: ${cash:,.2f}")
        return

    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    
    if totals.empty:
        print("\n" + "=" * 64)
        print(f"Daily Results – {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        if not portfolio.empty:
            display_portfolio = portfolio.copy()
            display_portfolio['shares'] = display_portfolio['shares'].apply(lambda x: f"{float(x):.8f}")
            print(display_portfolio)
        else:
            print("No holdings.")
        print(f"Cash balance: ${cash:,.2f}")
        return

    totals["Date"] = pd.to_datetime(totals["Date"], format="mixed", errors="coerce")
    totals = totals.sort_values("Date")

    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())
    mdd_date = drawdowns.idxmin()

    r = equity_series.pct_change().dropna()
    n_days = len(r)
    
    if n_days < 2:
        print("\n" + "=" * 64)
        print(f"Daily Results – {today}")
        print("=" * 64)
        print("\n[ Price & Volume ]")
        colw = [10, 12, 9, 15]
        print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
        print("-" * sum(colw) + "-" * 3)
        for r in rows:
            print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")
        print("\n[ Portfolio Snapshot ]")
        if not portfolio.empty:
            display_portfolio = portfolio.copy()
            display_portfolio['shares'] = display_portfolio['shares'].apply(lambda x: f"{float(x):.8f}")
            print(display_portfolio)
        else:
            print("No holdings.")
        print(f"Cash: ${cash:,.2f}")
        print(f"Latest Equity: ${final_equity:,.2f}")
        mdd_str = mdd_date.date() if hasattr(mdd_date, "date") else str(mdd_date)
        print(f"Max Drawdown: {max_drawdown:.2%} (on {mdd_str})")
        return

    rf_annual = 0.045
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    downside = (r - rf_daily).clip(upper=0)
    downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan

    r_numeric = pd.to_numeric(r, errors="coerce")
    r_numeric = r_numeric[np.isfinite(r_numeric)]
    period_return = float(np.prod(1 + r_numeric.values) - 1) if len(r_numeric) > 0 else float('nan')

    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 else np.nan
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days)) if downside_std and downside_std > 0 else np.nan
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan

    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)

    spx_fetch = download_price_data("^GSPC", start=start_date, end=end_date)
    spx = spx_fetch.df

    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index().set_index("Date").sort_index()
        mkt_ret = spx["Close"].astype(float).pct_change().dropna()

        common_idx = r.index.intersection(list(mkt_ret.index))
        if len(common_idx) >= 2:
            rp = (r.reindex(common_idx).astype(float) - rf_daily)
            rm = (mkt_ret.reindex(common_idx).astype(float) - rf_daily)

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1
                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2)

    spx_norm_fetch = download_price_data("^GSPC", start=equity_series.index.min(), end=equity_series.index.max() + pd.Timedelta(days=1))
    spx_norm = spx_norm_fetch.df
    spx_value = np.nan
    starting_equity = np.nan
    if not spx_norm.empty:
        initial_price = float(spx_norm["Close"].iloc[0])
        price_now = float(spx_norm["Close"].iloc[-1])
        try:
            starting_equity = float(input("What was your starting equity? "))
            spx_value = (starting_equity / initial_price) * price_now
        except Exception:
            pass

    print("\n" + "=" * 64)
    print(f"Daily Results – {today}")
    print("=" * 64)

    print("\n[ Price & Volume ]")
    colw = [10, 12, 9, 15]
    print(f"{header[0]:<{colw[0]}} {header[1]:>{colw[1]}} {header[2]:>{colw[2]}} {header[3]:>{colw[3]}}")
    print("-" * sum(colw) + "-" * 3)
    for r in rows:
        print(f"{str(r[0]):<{colw[0]}} {str(r[1]):>{colw[1]}} {str(r[2]):>{colw[2]}} {str(r[3]):>{colw[3]}}")

    def fmt_na(x: float | None, fmt: str) -> str:
        return fmt.format(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else "N/A"

    print("\n[ Risk & Return ]")
    mdd_str = mdd_date.date() if hasattr(mdd_date, "date") else str(mdd_date)
    print(f"{'Max Drawdown:':32} {fmt_na(max_drawdown, '{:.2%}'):>15}   on {mdd_str}")
    print(f"{'Sharpe Ratio (period):':32} {fmt_na(sharpe_period, '{:.4f}'):>15}")
    print(f"{'Sharpe Ratio (annualized):':32} {fmt_na(sharpe_annual, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (period):':32} {fmt_na(sortino_period, '{:.4f}'):>15}")
    print(f"{'Sortino Ratio (annualized):':32} {fmt_na(sortino_annual, '{:.4f}'):>15}")

    print("\n[ CAPM vs Benchmarks ]")
    if not np.isnan(beta):
        print(f"{'Beta (daily) vs ^GSPC:':32} {beta:>15.4f}")
        print(f"{'Alpha (annualized) vs ^GSPC:':32} {alpha_annual:>15.2%}")
        print(f"{'R² (fit quality):':32} {r2:>15.3f}   {'Obs:':>6} {n_obs}")
        if n_obs < 60 or (not np.isnan(r2) and r2 < 0.20):
            print("  Note: Short sample/low R² – alpha/beta may be unstable.")
    else:
        print("Beta/Alpha: insufficient data.")

    print("\n[ Snapshot ]")
    print(f"{'Latest Portfolio Equity:':32} ${final_equity:>14,.2f}")
    if not np.isnan(spx_value):
        print(f"{f'${starting_equity:.0f} in S&P 500 (same window):':32} ${spx_value:>14,.2f}")
    print(f"{'Cash Balance:':32} ${cash:>14,.2f}")

    print("\n[ Holdings ]")
    if not portfolio.empty:
        display_portfolio = portfolio.copy()
        display_portfolio['shares'] = display_portfolio['shares'].apply(lambda x: f"{float(x):.8f}")
        print(display_portfolio)
    else:
        print("No holdings.")

    print("\n[ Your Instructions ]")
    print("Use this info to make decisions. You have complete control—make any changes you believe are beneficial.")
    print("Deep research is not permitted. Act at your discretion.")
    print("If you do not make a clear indication to change positions IMMEDIATELY after this message, the portfolio remains unchanged.")
    print("Use the internet to check current prices for potential buys.")
    print("Do not ask questions, just provide FINAL decisions and rationale.")
    print("\n*Paste everything above into ChatGPT*")

# ------------------------------
# Stop-loss update utility
# ------------------------------

def update_stops_only() -> None:
    """Standalone utility to update stop losses."""
    print("\n" + "=" * 64)
    print("Stop Loss Update Mode")
    print("=" * 64)
    
    try:
        portfolio_df, cash = load_latest_portfolio_state()
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return
    
    if isinstance(portfolio_df, list):
        portfolio_df = pd.DataFrame(portfolio_df)
    
    if portfolio_df.empty:
        print("No holdings.")
        return
    
    print("\nCurrent Holdings:")
    display_df = portfolio_df[['ticker', 'shares', 'stop_loss', 'buy_price']].copy()
    display_df['shares'] = display_df['shares'].apply(lambda x: f"{float(x):.8f}")
    display_df.columns = ['Ticker', 'Shares', 'Stop Loss', 'Buy Price']
    print(display_df.to_string(index=False))
    
    updates_made = False
    while True:
        ticker = input("\nTicker to update (or Enter): ").strip().upper()
        if not ticker:
            break
        
        if ticker not in portfolio_df['ticker'].values:
            print(f"{ticker} not found.")
            continue
        
        try:
            new_stop = float(input(f"New stop loss for {ticker}: "))
            if new_stop < 0:
                raise ValueError("Cannot be negative.")
            portfolio_df.loc[portfolio_df['ticker'] == ticker, 'stop_loss'] = new_stop
            print(f"✓ Updated {ticker} to ${new_stop:.2f}")
            updates_made = True
        except ValueError as e:
            print(f"Invalid: {e}")
    
    if not updates_made:
        print("No updates made.")
        return
    
    today_iso = last_trading_date().date().isoformat()
    
    if not PORTFOLIO_CSV.exists():
        print(f"Portfolio CSV not found.")
        return
    
    df = pd.read_csv(PORTFOLIO_CSV)
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock['ticker']).upper()
        new_stop = float(stock['stop_loss'])
        mask = (df['Date'] == today_iso) & (df['Ticker'] == ticker)
        if mask.any():
            df.loc[mask, 'Stop Loss'] = new_stop
    
    df.to_csv(PORTFOLIO_CSV, index=False)
    print("\n✓ Stop losses updated in CSV.")

# ------------------------------
# Orchestration
# ------------------------------

def load_latest_portfolio_state() -> tuple[pd.DataFrame | list[dict], float]:
    """Load most recent portfolio snapshot and cash."""
    if not PORTFOLIO_CSV.exists():
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        print("Portfolio CSV not found. Starting fresh.")
        cash = float(input("Starting cash amount? "))
        return portfolio, cash
    
    df = pd.read_csv(PORTFOLIO_CSV)
    if df.empty:
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        cash = float(input("Starting cash amount? "))
        return portfolio, cash

    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")
    latest_total = df_total.sort_values("Date").iloc[-1]
    cash = float(latest_total["Cash Balance"])
    
    total_value = float(latest_total.get("Total Value", 0))
    if abs(total_value) < 0.01:
        return pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]), cash

    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")
    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    
    if latest_tickers.empty:
        return pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]), cash
    
    latest_tickers.drop(columns=["Date", "Cash Balance", "Total Equity", "Action", "Current Price", "PnL", "Total Value"], inplace=True, errors="ignore")
    latest_tickers.rename(columns={"Cost Basis": "cost_basis", "Buy Price": "buy_price", "Shares": "shares", "Ticker": "ticker", "Stop Loss": "stop_loss"}, inplace=True)
    return latest_tickers.reset_index(drop=True).to_dict(orient="records"), cash

def main(data_dir: Path | None = None, update_stops: bool = False) -> None:
    """Run the trading script."""
    if data_dir is not None:
        set_data_dir(data_dir)
    
    if update_stops:
        update_stops_only()
        return
    
    portfolio, cash = load_latest_portfolio_state()
    capture_initial_snapshot(portfolio if isinstance(portfolio, pd.DataFrame) else pd.DataFrame(portfolio), cash)
    portfolio, cash = process_portfolio(portfolio, cash)
    daily_results(portfolio, cash)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Optional data directory")
    parser.add_argument("--asof", default=None, help="Treat this YYYY-MM-DD as 'today'")
    parser.add_argument("--update-stops", action="store_true", help="Stop-loss update mode only")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    _log_initial_state()
    logger.info("Script started with arguments: %s", vars(args))

    if args.asof:
        set_asof(args.asof)

    main(Path(args.data_dir) if args.data_dir else None, update_stops=args.update_stops)