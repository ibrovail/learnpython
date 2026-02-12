"""Utilities for maintaining the ChatGPT micro-cap portfolio.

This module rewrites the original script to:
- Centralize market data fetching with a robust Yahoo->Stooq fallback
- Ensure ALL price requests go through the same accessor
- Handle empty Yahoo frames (no exception) so fallback actually triggers
- Normalize Stooq output to Yahoo-like columns
- Make weekend handling consistent and testable
- Keep behavior and CSV formats compatible with prior runs
- Track capital injections separately for accurate benchmarking

Notes:
- Some tickers/indices are not available on Stooq (e.g., ^RUT). These stay on Yahoo.
- Stooq end date is exclusive; we add +1 day for ranges.
- "Adj Close" is set equal to "Close" for Stooq to match downstream expectations.
- Capital injections are logged separately to enable dollar-weighted S&P comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast,Dict, List, Optional
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

# Allow env var override:  ASOF_DATE=YYYY-MM-DD python trading_script.py
_env_asof = os.environ.get("ASOF_DATE")
if _env_asof:
    set_asof(_env_asof)

def _effective_now() -> datetime:
    return (ASOF_DATE.to_pydatetime() if ASOF_DATE is not None else datetime.now())

# ------------------------------
# Globals / file locations
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files alongside this script by default
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
CAPITAL_INJECTIONS_CSV = DATA_DIR / "capital_injections.csv"
DEFAULT_BENCHMARKS = ["IWO", "XBI", "SPY", "IWM", "QQQ", "VIX", "TLT", "HYG"]

# Set up logger for this module
logger = logging.getLogger(__name__)

# Log initial global state configuration (only when run as main script)
def _log_initial_state():
    """Log the initial global file path configuration."""
    logger.info("=== Trading Script Initial Configuration ===")
    logger.info("Script directory: %s", SCRIPT_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Portfolio CSV: %s", PORTFOLIO_CSV)
    logger.info("Trade log CSV: %s", TRADE_LOG_CSV)
    logger.info("Capital injections CSV: %s", CAPITAL_INJECTIONS_CSV)
    logger.info("Default benchmarks: %s", DEFAULT_BENCHMARKS)
    logger.info("==============================================")

# ------------------------------
# Configuration helpers – benchmark tickers (tickers.json)
# ------------------------------



logger = logging.getLogger(__name__)

def _read_json_file(path: Path) -> Optional[Dict]:
    """Read and parse JSON from `path`. Return dict on success, None if not found or invalid.

    - FileNotFoundError -> return None
    - JSON decode error -> log a warning and return None
    - Other IO errors -> log a warning and return None
    """
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
        logger.warning("tickers.json present but malformed: %s -> %s. Falling back to defaults.", path, exc)
        return None
    except Exception as exc:
        logger.warning("Unable to read tickers.json (%s): %s. Falling back to defaults.", path, exc)
        return None

def load_benchmarks(script_dir: Path | None = None) -> List[str]:
    """Return a list of benchmark tickers.

    Looks for a `tickers.json` file in either:
      - script_dir (if provided) OR the module SCRIPT_DIR, and then
      - script_dir.parent (project root candidate).

    Expected schema:
      {"benchmarks": ["IWO", "XBI", "SPY", "IWM"]}

    Behavior:
    - If file missing or malformed -> return DEFAULT_BENCHMARKS copy.
    - If 'benchmarks' key missing or not a list -> log warning and return defaults.
    - Normalizes tickers (strip, upper) and preserves order while removing duplicates.
    """
    base = Path(script_dir) if script_dir else SCRIPT_DIR
    candidates = [base, base.parent]

    cfg = None
    cfg_path = None
    for c in candidates:
        p = (c / "tickers.json").resolve()
        data = _read_json_file(p)
        if data is not None:
            cfg = data
            cfg_path = p
            break

    if not cfg:
        return DEFAULT_BENCHMARKS.copy()

    benchmarks = cfg.get("benchmarks")
    if not isinstance(benchmarks, list):
        logger.warning("tickers.json at %s missing 'benchmarks' array. Falling back to defaults.", cfg_path)
        return DEFAULT_BENCHMARKS.copy()

    seen = set()
    result: list[str] = []
    for t in benchmarks:
        if not isinstance(t, str):
            continue
        up = t.strip().upper()
        if not up:
            continue
        if up not in seen:
            seen.add(up)
            result.append(up)

    return result if result else DEFAULT_BENCHMARKS.copy()


# ------------------------------
# Date helpers
# ------------------------------

def last_trading_date(today: datetime | None = None) -> pd.Timestamp:
    """Return last trading date (Mon–Fri), mapping Sat/Sun -> Fri."""
    dt = pd.Timestamp(today or _effective_now())
    if dt.weekday() == 5:  # Sat -> Fri
        friday_date = (dt - pd.Timedelta(days=1)).normalize()
        ## logger.info("Script running on Saturday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    if dt.weekday() == 6:  # Sun -> Fri
        friday_date = (dt - pd.Timedelta(days=2)).normalize()
        ## logger.info("Script running on Sunday - using Friday's data (%s) instead of today's date", friday_date.date())
        return friday_date
    return dt.normalize()

def check_weekend() -> str:
    """Backwards-compatible wrapper returning ISO date string for last trading day."""
    return last_trading_date().date().isoformat()

def trading_day_window(target: datetime | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) window for the last trading day (Fri on weekends)."""
    d = last_trading_date(target)
    return d, (d + pd.Timedelta(days=1))


# ------------------------------
# Data access layer
# ------------------------------

# Known Stooq symbol remaps for common indices
STOOQ_MAP = {
    "^GSPC": "^SPX",  # S&P 500
    "^DJI": "^DJI",   # Dow Jones
    "^IXIC": "^IXIC", # Nasdaq Composite
    # "^RUT": not on Stooq; keep Yahoo
}

# Symbols we should *not* attempt on Stooq
STOOQ_BLOCKLIST = {"^RUT"}


# ------------------------------
# Data access layer (UPDATED)
# ------------------------------

@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yahoo" | "stooq-pdr" | "stooq-csv" | "yahoo:<proxy>-proxy" | "empty"

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten multiIndex frame so we can lazily lookup values by index.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # If the second level is the same ticker for all cols, drop it
            if len(set(df.columns.get_level_values(1))) == 1:
                df = df.copy()
                df.columns = df.columns.get_level_values(0)
            else:
                # multiple tickers: flatten with join
                df = df.copy()
                df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
        except Exception:
            df = df.copy()
            df.columns = ["_".join(map(str, t)).strip("_") for t in df.columns.to_flat_index()]
            
    # Ensure all expected columns exist
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]

def _yahoo_download(ticker: str, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download with a real UA and silence all chatter."""
    import io, logging
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

def _stooq_csv_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Stooq CSV endpoint (daily). Good for US tickers and many ETFs."""
    import requests, io # type: ignore
    if ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()
    t = STOOQ_MAP.get(ticker, ticker)

    # Stooq daily CSV: lowercase; equities/ETFs use .us, indices keep ^ prefix
    if not t.startswith("^"):
        sym = t.lower()
        if not sym.endswith(".us"):
            sym = f"{sym}.us"
    else:
        sym = t.lower()

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text.strip():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to [start, end) (Stooq end is exclusive)
        df = df.loc[(df.index >= start.normalize()) & (df.index < end.normalize())]

        # Normalize to Yahoo-like schema
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
        return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

def _stooq_download(
    ticker: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch OHLCV from Stooq via pandas-datareader; returns empty DF on failure."""
    if not _HAS_PDR or ticker in STOOQ_BLOCKLIST:
        return pd.DataFrame()

    t = STOOQ_MAP.get(ticker, ticker)
    if not t.startswith("^"):
        t = t.lower()

    try:
        # Ensure pdr is imported locally if not available globally
        if not _HAS_PDR:
            return pd.DataFrame()
        import pandas_datareader.data as pdr_local # type: ignore
        df = cast(pd.DataFrame, pdr_local.DataReader(t, "stooq", start=start, end=end))
        df.sort_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

def _weekend_safe_range(period: str | None, start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Compute a concrete [start, end) window.
    - If explicit start/end provided: use them (add +1 day to end to make it exclusive).
    - If period is '1d': use the last trading day's [Fri, Sat) window on weekends.
    - If period like '2d'/'5d': build a window ending at the last trading day.
    """
    if start or end:
        end_ts = pd.Timestamp(end) if end else last_trading_date() + pd.Timedelta(days=1)
        start_ts = pd.Timestamp(start) if start else (end_ts - pd.Timedelta(days=5))
        return start_ts.normalize(), pd.Timestamp(end_ts).normalize()

    # No explicit dates; derive from period
    if isinstance(period, str) and period.endswith("d"):
        days = int(period[:-1])
    else:
        days = 1

    # Anchor to last trading day (Fri on Sun/Sat)
    end_trading = last_trading_date()
    start_ts = (end_trading - pd.Timedelta(days=days)).normalize()
    end_ts = (end_trading + pd.Timedelta(days=1)).normalize()
    return start_ts, end_ts

def download_price_data(ticker: str, **kwargs: Any) -> FetchResult:
    """
    Robust OHLCV fetch with multi-stage fallbacks:

    Order:
      1) Yahoo Finance via yfinance
      2) Stooq via pandas-datareader
      3) Stooq direct CSV
      4) Index proxies (e.g., ^GSPC->SPY, ^RUT->IWM) via Yahoo
    Returns a DataFrame with columns [Open, High, Low, Close, Adj Close, Volume].
    """
    # Pull out range args, compute a weekend-safe window
    period = kwargs.pop("period", None)
    start = kwargs.pop("start", None)
    end = kwargs.pop("end", None)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("threads", False)

    s, e = _weekend_safe_range(period, start, end)

    # ---------- 1) Yahoo (date-bounded) ----------
    df_y = _yahoo_download(ticker, start=s, end=e, **kwargs)
    if isinstance(df_y, pd.DataFrame) and not df_y.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_y)), "yahoo")

    # ---------- 2) Stooq via pandas-datareader ----------
    df_s = _stooq_download(ticker, start=s, end=e)
    if isinstance(df_s, pd.DataFrame) and not df_s.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_s)), "stooq-pdr")

    # ---------- 3) Stooq direct CSV ----------
    df_csv = _stooq_csv_download(ticker, s, e)
    if isinstance(df_csv, pd.DataFrame) and not df_csv.empty:
        return FetchResult(_normalize_ohlcv(_to_datetime_index(df_csv)), "stooq-csv")

    # ---------- 4) Proxy indices if applicable ----------
    proxy_map = {"^GSPC": "SPY", "^RUT": "IWM"}
    proxy = proxy_map.get(ticker)
    if proxy:
        df_proxy = _yahoo_download(proxy, start=s, end=e, **kwargs)
        if isinstance(df_proxy, pd.DataFrame) and not df_proxy.empty:
            return FetchResult(_normalize_ohlcv(_to_datetime_index(df_proxy)), f"yahoo:{proxy}-proxy")

    # ---------- Nothing worked ----------
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    return FetchResult(empty, "empty")



# ------------------------------
# File path configuration
# ------------------------------

def set_data_dir(data_dir: Path) -> None:
    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV, CAPITAL_INJECTIONS_CSV
    logger.info("Setting data directory: %s", data_dir)
    DATA_DIR = Path(data_dir)
    logger.debug("Creating data directory if it doesn't exist: %s", DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"
    CAPITAL_INJECTIONS_CSV = DATA_DIR / "capital_injections.csv"
    logger.info("Data directory configured - Portfolio CSV: %s, Trade Log CSV: %s, Capital Injections CSV: %s", 
                PORTFOLIO_CSV, TRADE_LOG_CSV, CAPITAL_INJECTIONS_CSV)


# ------------------------------
# Capital Injection Management
# ------------------------------

def log_capital_injection(amount: float) -> None:
    """Log a capital injection with today's date."""
    today = check_weekend()
    log = {
        "Date": today,
        "Amount": amount,
    }
    
    if CAPITAL_INJECTIONS_CSV.exists():
        logger.info("Reading CSV file: %s", CAPITAL_INJECTIONS_CSV)
        df = pd.read_csv(CAPITAL_INJECTIONS_CSV)
        logger.info("Successfully read CSV file: %s", CAPITAL_INJECTIONS_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    
    logger.info("Writing CSV file: %s", CAPITAL_INJECTIONS_CSV)
    df.to_csv(CAPITAL_INJECTIONS_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", CAPITAL_INJECTIONS_CSV)

def load_capital_injections() -> pd.DataFrame:
    """Load all capital injections, returning empty DataFrame if file doesn't exist."""
    if not CAPITAL_INJECTIONS_CSV.exists():
        return pd.DataFrame(columns=["Date", "Amount"])
    
    logger.info("Reading CSV file: %s", CAPITAL_INJECTIONS_CSV)
    df = pd.read_csv(CAPITAL_INJECTIONS_CSV)
    logger.info("Successfully read CSV file: %s", CAPITAL_INJECTIONS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ------------------------------
# Portfolio operations
# ------------------------------

def _ensure_df(portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]]) -> pd.DataFrame:
    if isinstance(portfolio, pd.DataFrame):
        return portfolio.copy()
    if isinstance(portfolio, (dict, list)):
        df = pd.DataFrame(portfolio)
        # Ensure proper columns exist even for empty DataFrames
        if df.empty:
            logger.debug("Creating empty portfolio DataFrame with proper column structure")
            df = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return df
    raise TypeError("portfolio must be a DataFrame, dict, or list[dict]")

def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    today_iso = last_trading_date().date().isoformat()
    portfolio_df = _ensure_df(portfolio)

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    # ------- Capital injection option (supports manual capital adds) -------
    if interactive:
        print("\n--- Capital Injection ---")
        inject_choice = input("Would you like to inject additional capital? ('y' or press Enter to skip): ").strip().lower()
        if inject_choice == "y":
            try:
                inject_amount = float(input("Enter amount to inject (e.g., 100): "))
                if inject_amount > 0:
                    log_capital_injection(inject_amount)
                    cash += inject_amount
                    print(f"Injected ${inject_amount:.2f}. New cash balance: ${cash:.2f}")
                else:
                    print("Invalid amount. Skipping injection.")
            except ValueError:
                print("Invalid input. Skipping injection.")

    # ------- Interactive trade entry (supports MOO) -------
    if interactive:
        while True:
            print(portfolio_df)
            action = input(
                f""" You have {cash} in cash.
Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
            ).strip().lower()

            if action == "b":
                ticker = input("Enter ticker symbol: ").strip().upper()
                order_type = input("Order type? 'm' = market-on-open, 'l' = limit: ").strip().lower()

                try:
                    shares = float(input("Enter number of shares: "))
                    if shares <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid share amount. Buy cancelled.")
                    continue

                if order_type == "m":
                    try:
                        stop_loss = float(input("Enter stop loss (or 0 to skip): "))
                        if stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid stop loss. Buy cancelled.")
                        continue

                    s, e = trading_day_window()
                    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
                    data = fetch.df
                    if data.empty:
                        print(f"MOO buy for {ticker} failed: no market data available (source={fetch.source}).")
                        continue

                    o = float(data["Open"].iloc[-1]) if "Open" in data else float(data["Close"].iloc[-1])
                    exec_price = round(o, 2)
                    notional = exec_price * shares
                    if notional > cash:
                        print(f"MOO buy for {ticker} failed: cost {notional:.2f} exceeds cash {cash:.2f}.")
                        continue

                    log = {
                        "Date": today_iso,
                        "Ticker": ticker,
                        "Shares Bought": shares,
                        "Buy Price": exec_price,
                        "Cost Basis": notional,
                        "PnL": 0.0,
                        "Reason": "MANUAL BUY MOO - Filled",
                    }
                    # --- Manual BUY MOO logging ---
                    if os.path.exists(TRADE_LOG_CSV):
                        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
                        df_log = pd.read_csv(TRADE_LOG_CSV)
                        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
                        if df_log.empty:
                            df_log = pd.DataFrame([log])
                        else:
                            df_log = pd.concat([df_log, pd.DataFrame([log])], ignore_index=True)
                    else:
                        df_log = pd.DataFrame([log])
                    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
                    df_log.to_csv(TRADE_LOG_CSV, index=False)
                    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

                    rows = portfolio_df.loc[portfolio_df["ticker"].astype(str).str.upper() == ticker.upper()]
                    if rows.empty:
                        new_trade = {
                            "ticker": ticker,
                            "shares": float(shares),
                            "stop_loss": float(stop_loss),
                            "buy_price": float(exec_price),
                            "cost_basis": float(notional),
                        }
                        if portfolio_df.empty:
                            portfolio_df = pd.DataFrame([new_trade])
                        else:
                            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([new_trade])], ignore_index=True)
                    else:
                        idx = rows.index[0]
                        cur_shares = float(portfolio_df.at[idx, "shares"])
                        cur_cost = float(portfolio_df.at[idx, "cost_basis"])
                        new_shares = cur_shares + float(shares)
                        new_cost = cur_cost + float(notional)
                        avg_price = new_cost / new_shares if new_shares else 0.0
                        portfolio_df.at[idx, "shares"] = new_shares
                        portfolio_df.at[idx, "cost_basis"] = new_cost
                        portfolio_df.at[idx, "buy_price"] = avg_price
                        portfolio_df.at[idx, "stop_loss"] = float(stop_loss)

                    cash -= notional
                    print(f"Manual BUY MOO for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
                    continue

                elif order_type == "l":
                    try:
                        buy_price = float(input("Enter buy LIMIT price: "))
                        stop_loss = float(input("Enter stop loss (or 0 to skip): "))
                        if buy_price <= 0 or stop_loss < 0:
                            raise ValueError
                    except ValueError:
                        print("Invalid input. Limit buy cancelled.")
                        continue

                    cash, portfolio_df = log_manual_buy(
                        buy_price, shares, ticker, stop_loss, cash, portfolio_df
                    )
                    continue
                else:
                    print("Unknown order type. Use 'm' or 'l'.")
                    continue

            if action == "s":
                try:
                    ticker = input("Enter ticker symbol: ").strip().upper()
                    shares = float(input("Enter number of shares to sell (LIMIT): "))
                    sell_price = float(input("Enter sell LIMIT price: "))
                    if shares <= 0 or sell_price <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Manual sell cancelled.")
                    continue

                cash, portfolio_df = log_manual_sell(
                    sell_price, shares, ticker, cash, portfolio_df
                )
                continue

            break  # proceed to pricing

    # ------- Daily pricing + stop-loss execution -------
    s, e = trading_day_window()
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock["ticker"]).upper()
        shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
        cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
        cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else cost * shares
        stop = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

        fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
        data = fetch.df

        if data.empty:
            print(f"No data for {ticker} (source={fetch.source}).")
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": "", "Total Value": "", "PnL": "",
                "Action": "NO DATA", "Cash Balance": "", "Total Equity": "",
            }
            results.append(row)
            continue

        o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
        h = float(data["High"].iloc[-1])
        l = float(data["Low"].iloc[-1])
        c = float(data["Close"].iloc[-1])
        if np.isnan(o):
            o = c

        # FIXED: Stop-loss only triggers during regular trading hours
        stop_triggered = False
        if stop and stop > 0:
            if o >= stop and l <= stop:
                # Stock opened at/above stop, fell during trading hours
                stop_triggered = True
                exec_price = round(stop, 2)
            elif o < stop:
                # Stock gapped below stop in pre-market - stop didn't execute
                print(f"⚠️  WARNING: {ticker} gapped below stop-loss ${stop:.2f} (opened at ${o:.2f}). Stop not executed - position still held.")
                stop_triggered = False

        if stop_triggered:
            value = round(exec_price * shares, 2)
            pnl = round((exec_price - cost) * shares, 2)
            action = "SELL - Stop Loss Triggered"
            cash += value
            portfolio_df = log_sell(ticker, shares, exec_price, cost, pnl, portfolio_df)
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": exec_price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }
        else:
            price = round(c, 2)
            value = round(price * shares, 2)
            pnl = round((price - cost) * shares, 2)
            action = "HOLD"
            total_value += value
            total_pnl += pnl
            row = {
                "Date": today_iso, "Ticker": ticker, "Shares": shares,
                "Buy Price": cost, "Cost Basis": cost_basis, "Stop Loss": stop,
                "Current Price": price, "Total Value": value, "PnL": pnl,
                "Action": action, "Cash Balance": "", "Total Equity": "",
            }

        results.append(row)

    total_row = {
        "Date": today_iso, "Ticker": "TOTAL", "Shares": "", "Buy Price": "",
        "Cost Basis": "", "Stop Loss": "", "Current Price": "",
        "Total Value": round(total_value, 2), "PnL": round(total_pnl, 2),
        "Action": "", "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df_out = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
        existing = pd.read_csv(PORTFOLIO_CSV)
        logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)
        existing = existing[existing["Date"] != str(today_iso)]
        print("Saving results to CSV...")
        df_out = pd.concat([existing, df_out], ignore_index=True)
    logger.info("Writing CSV file: %s", PORTFOLIO_CSV)
    df_out.to_csv(PORTFOLIO_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", PORTFOLIO_CSV)

    return portfolio_df, cash



# ------------------------------
# Trade logging
# ------------------------------

def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    today = check_weekend()
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)
    return portfolio

def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()

    if interactive:
        check = input(
            f"You are placing a BUY LIMIT for {shares} {ticker} at ${buy_price:.2f}.\n"
            f"If this is a mistake, type '1' or, just hit Enter: "
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(
            columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"]
        )

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    o = float(data.get("Open", [np.nan])[-1])
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    # Check if buy limit was triggered (low <= buy_price means price touched our limit)
    # Allow for floating-point precision tolerance
    epsilon = 0.005  # Half a cent tolerance
    if l <= (buy_price + epsilon):
        exec_price = buy_price  # Fill at exact limit price
    else:
        print(f"Buy limit ${buy_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    cost_amt = exec_price * shares
    if cost_amt > cash:
        print(f"Manual buy for {ticker} failed: cost {cost_amt:.2f} exceeds cash balance {cash:.2f}.")
        return cash, chatgpt_portfolio

    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": exec_price,
        "Cost Basis": cost_amt,
        "PnL": 0.0,
        "Reason": "MANUAL BUY LIMIT - Filled",
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)

    rows = chatgpt_portfolio.loc[chatgpt_portfolio["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        if chatgpt_portfolio.empty:
            chatgpt_portfolio = pd.DataFrame([{
                "ticker": ticker,
                "shares": float(shares),
                "stop_loss": float(stoploss),
                "buy_price": float(exec_price),
                "cost_basis": float(cost_amt),
            }])
        else:
            chatgpt_portfolio = pd.concat(
                [chatgpt_portfolio, pd.DataFrame([{
                    "ticker": ticker,
                    "shares": float(shares),
                    "stop_loss": float(stoploss),
                    "buy_price": float(exec_price),
                    "cost_basis": float(cost_amt),
                }])],
                ignore_index=True
            )
    else:
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])
        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(cost_amt)
        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = new_cost / new_shares if new_shares else 0.0
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    cash -= cost_amt
    print(f"Manual BUY LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio

def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    today = check_weekend()
    if interactive:
        reason = input(
            f"""You are placing a SELL LIMIT for {shares_sold} {ticker} at ${sell_price:.2f}.
If this is a mistake, enter 1, or hit Enter."""
        )
    if reason == "1":
        print("Returning...")
        return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""

    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio

    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]
    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}.")
        return cash, chatgpt_portfolio

    s, e = trading_day_window()
    fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
    data = fetch.df
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available (source={fetch.source}).")
        return cash, chatgpt_portfolio

    o = float(data["Open"].iloc[-1]) if "Open" in data else np.nan
    h = float(data["High"].iloc[-1])
    l = float(data["Low"].iloc[-1])
    if np.isnan(o):
        o = float(data["Close"].iloc[-1])

    # Check if sell limit was triggered (high >= sell_price means price touched our limit)
    # Allow for floating-point precision tolerance
    epsilon = 0.005  # Half a cent tolerance
    if h >= (sell_price - epsilon):
        exec_price = sell_price  # Fill at exact limit price
    else:
        print(f"Sell limit ${sell_price:.2f} for {ticker} not reached today (range {l:.2f}-{h:.2f}). Order not filled.")
        return cash, chatgpt_portfolio

    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = exec_price * shares_sold - cost_basis

    log = {
        "Date": today, "Ticker": ticker,
        "Shares Bought": "", "Buy Price": "",
        "Cost Basis": cost_basis, "PnL": pnl,
        "Reason": f"MANUAL SELL LIMIT - {reason}", "Shares Sold": shares_sold,
        "Sell Price": exec_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        logger.info("Reading CSV file: %s", TRADE_LOG_CSV)
        df = pd.read_csv(TRADE_LOG_CSV)
        logger.info("Successfully read CSV file: %s", TRADE_LOG_CSV)
        if df.empty:
            df = pd.DataFrame([log])
        else:
            df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    logger.info("Writing CSV file: %s", TRADE_LOG_CSV)
    df.to_csv(TRADE_LOG_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", TRADE_LOG_CSV)


    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"] * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash += shares_sold * exec_price
    print(f"Manual SELL LIMIT for {ticker} filled at ${exec_price:.2f} ({fetch.source}).")
    return cash, chatgpt_portfolio



# ------------------------------
# Reporting / Metrics
# ------------------------------

# Role classification for tickers in daily summary
TICKER_ROLES: Dict[str, str] = {
    "SPY": "Benchmark",
    "IWM": "Benchmark",
    "QQQ": "Benchmark",
    "IWO": "Benchmark",
    "XBI": "Benchmark",
    "TLT": "Macro",
    "HYG": "Macro",
}

def _get_ticker_role(ticker: str, holdings_set: set[str]) -> str:
    """Determine the role of a ticker for display purposes."""
    ticker_upper = ticker.upper()
    if ticker_upper in holdings_set:
        return "Holding"
    return TICKER_ROLES.get(ticker_upper, "Benchmark")


def _fmt_pct(val: float) -> str:
    """Format a percentage value as +X.XX% or -X.XX%."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:+.2f}%"


def _fmt_currency(val: float) -> str:
    """Format a currency value as $X.XX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.2f}"


def _fmt_num(val: float, decimals: int = 2) -> str:
    """Format a numeric value with specified decimal places."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def print_weekend_summary(chatgpt_portfolio: pd.DataFrame | list[dict[str, Any]], cash: float) -> None:
    """Print weekend summary in XML format for deep research sessions.

    Output includes:
    - Portfolio snapshot with holdings as XML attributes
    - Cash balance
    - Placeholder for analyst thesis notes
    - Recent trades (Monday through Friday of current week)
    """
    friday_date = last_trading_date()
    friday_iso = friday_date.date().isoformat()

    # Calculate Monday of the same week (Friday - 4 days)
    monday_date = friday_date - pd.Timedelta(days=4)

    # Convert list to DataFrame if needed
    if isinstance(chatgpt_portfolio, list):
        portfolio_df = pd.DataFrame(chatgpt_portfolio) if chatgpt_portfolio else pd.DataFrame()
    else:
        portfolio_df = chatgpt_portfolio

    # -------- Portfolio Snapshot --------
    print(f'\n<portfolio_snapshot date="{friday_iso}">')

    if not portfolio_df.empty:
        s, e = trading_day_window()
        for _, stock in portfolio_df.iterrows():
            ticker = str(stock["ticker"]).upper()
            shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
            avg_cost = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
            stop_loss = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

            # Fetch current price
            fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
            data = fetch.df

            if not data.empty:
                current_price = float(data["Close"].iloc[-1])
            else:
                current_price = 0.0

            print(f'<holding ticker="{ticker}" shares="{shares}" avg_cost="{avg_cost:.2f}" '
                  f'current_price="{current_price:.2f}" stop_loss="{stop_loss:.2f}" />')

    print("</portfolio_snapshot>")
    print()

    # -------- Cash Balance --------
    print("<cash_balance>")
    print(f"${cash:,.2f}")
    print("</cash_balance>")
    print()

    # -------- Last Analyst Thesis (placeholder) --------
    print("<last_analyst_thesis>")
    print("<!-- UPDATE: Paste the most recent thesis notes for each holding -->")
    print("</last_analyst_thesis>")
    print()

    # -------- Recent Trades (Mon-Fri of current week) --------
    print("<recent_trades>")
    print("<!-- Trades from Monday through Friday of current week -->")

    try:
        trade_log = pd.read_csv(TRADE_LOG_CSV)
        trade_log["Date"] = pd.to_datetime(trade_log["Date"])

        # Filter to Monday-Friday of current week
        mask = (trade_log["Date"] >= monday_date) & (trade_log["Date"] <= friday_date)
        recent = trade_log[mask]

        if not recent.empty:
            # Print CSV header
            print("Date,Ticker,Shares Bought,Buy Price,Cost Basis,PnL,Reason,Shares Sold,Sell Price")
            for _, row in recent.iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                ticker = row.get("Ticker", "")
                shares_bought = row.get("Shares Bought", "")
                buy_price = row.get("Buy Price", "")
                cost_basis = row.get("Cost Basis", "")
                pnl = row.get("PnL", "")
                reason = row.get("Reason", "")
                shares_sold = row.get("Shares Sold", "")
                sell_price = row.get("Sell Price", "")

                # Format numeric values, leave empty if NaN
                shares_bought = f"{shares_bought}" if not pd.isna(shares_bought) else ""
                buy_price = f"{buy_price}" if not pd.isna(buy_price) else ""
                cost_basis = f"{cost_basis}" if not pd.isna(cost_basis) else ""
                pnl = f"{pnl}" if not pd.isna(pnl) else ""
                shares_sold = f"{shares_sold}" if not pd.isna(shares_sold) else ""
                sell_price = f"{sell_price}" if not pd.isna(sell_price) else ""

                print(f"{date_str},{ticker},{shares_bought},{buy_price},{cost_basis},{pnl},{reason},{shares_sold},{sell_price}")
        else:
            print("<!-- No trades this week -->")
    except FileNotFoundError:
        print("<!-- Trade log not found -->")

    print("</recent_trades>")


def _print_xml_summary(
    today: str,
    price_volume_rows: list[tuple[str, float, float, int, str]],
    max_drawdown: float,
    mdd_date: str,
    sharpe_annual: float,
    sortino_annual: float,
    beta: float,
    alpha_annual: float,
    r2: float,
    final_equity: float,
    dollar_weighted_spx: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
) -> None:
    """Print the daily summary in XML-structured format."""

    print(f'\n<daily_summary date="{today}">')
    print()

    # -------- Market Data --------
    print("<market_data>")

    # Price & Volume table
    print("<price_volume>")
    print("| Ticker | Close   | % Chg  | Volume      | Role       |")
    print("|--------|---------|--------|-------------|------------|")
    for ticker, close_price, pct_chg, volume, role in price_volume_rows:
        close_str = f"{close_price:,.2f}"
        pct_str = f"{pct_chg:+.2f}%"
        vol_str = f"{volume:,}"
        print(f"| {ticker:<6} | {close_str:>7} | {pct_str:>6} | {vol_str:>11} | {role:<10} |")
    print("</price_volume>")
    print()

    # Risk Metrics table
    print("<risk_metrics>")
    print("| Metric                        | Value     | Note                    |")
    print("|-------------------------------|-----------|-------------------------|")

    # Max Drawdown
    mdd_val = _fmt_pct(max_drawdown * 100) if not (max_drawdown is None or (isinstance(max_drawdown, float) and np.isnan(max_drawdown))) else "N/A"
    mdd_note = f"on {mdd_date}" if mdd_date and mdd_date != "N/A" else ""
    print(f"| {'Max Drawdown':<29} | {mdd_val:>9} | {mdd_note:<23} |")

    # Sharpe Ratio (annualized)
    sharpe_val = _fmt_num(sharpe_annual, 4) if not (sharpe_annual is None or (isinstance(sharpe_annual, float) and np.isnan(sharpe_annual))) else "N/A"
    print(f"| {'Sharpe Ratio (annualized)':<29} | {sharpe_val:>9} | {'':<23} |")

    # Sortino Ratio (annualized)
    sortino_val = _fmt_num(sortino_annual, 4) if not (sortino_annual is None or (isinstance(sortino_annual, float) and np.isnan(sortino_annual))) else "N/A"
    print(f"| {'Sortino Ratio (annualized)':<29} | {sortino_val:>9} | {'':<23} |")

    # Beta (daily) vs ^GSPC
    beta_val = _fmt_num(beta, 4) if not (beta is None or (isinstance(beta, float) and np.isnan(beta))) else "N/A"
    print(f"| {'Beta (daily) vs ^GSPC':<29} | {beta_val:>9} | {'':<23} |")

    # Alpha (annualized) vs ^GSPC
    alpha_val = _fmt_pct(alpha_annual * 100) if not (alpha_annual is None or (isinstance(alpha_annual, float) and np.isnan(alpha_annual))) else "N/A"
    print(f"| {'Alpha (annualized) vs ^GSPC':<29} | {alpha_val:>9} | {'':<23} |")

    # R²
    r2_val = _fmt_num(r2, 3) if not (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else "N/A"
    r2_note = "Low — alpha/beta unstable" if not (r2 is None or (isinstance(r2, float) and np.isnan(r2))) and r2 < 0.15 else ""
    print(f"| {'R²':<29} | {r2_val:>9} | {r2_note:<23} |")

    print("</risk_metrics>")
    print("</market_data>")
    print()

    # -------- Portfolio Snapshot --------
    print("<portfolio_snapshot>")
    print("| Metric              | Value     |")
    print("|---------------------|-----------|")
    print(f"| {'Portfolio Equity':<19} | {_fmt_currency(final_equity):>9} |")
    spx_val = _fmt_currency(dollar_weighted_spx) if not (dollar_weighted_spx is None or (isinstance(dollar_weighted_spx, float) and np.isnan(dollar_weighted_spx))) else "N/A"
    print(f"| {'S&P Equivalent':<19} | {spx_val:>9} |")
    print(f"| {'Cash Balance':<19} | {_fmt_currency(cash):>9} |")
    print("</portfolio_snapshot>")
    print()

    # -------- Holdings --------
    print("<holdings>")
    print("| Ticker | Shares | Avg Cost | Cost Basis | Unrealized P&L      | Stop Loss |")
    print("|--------|--------|----------|------------|---------------------|-----------|")

    if not chatgpt_portfolio.empty:
        s, e = trading_day_window()
        for _, stock in chatgpt_portfolio.iterrows():
            ticker = str(stock["ticker"]).upper()
            shares = int(stock["shares"]) if not pd.isna(stock["shares"]) else 0
            buy_price = float(stock["buy_price"]) if not pd.isna(stock["buy_price"]) else 0.0
            cost_basis = float(stock["cost_basis"]) if not pd.isna(stock["cost_basis"]) else 0.0
            stop_loss = float(stock["stop_loss"]) if not pd.isna(stock["stop_loss"]) else 0.0

            # Fetch current price for P&L calculation
            fetch = download_price_data(ticker, start=s, end=e, auto_adjust=False, progress=False)
            data = fetch.df

            if not data.empty:
                current_price = float(data["Close"].iloc[-1])
                pnl_dollars = (current_price - buy_price) * shares
                pnl_percent = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                pnl_str = f"${pnl_dollars:+,.2f} ({pnl_percent:+.1f}%)"
            else:
                pnl_str = "N/A"

            stop_str = f"${stop_loss:.2f}" if stop_loss > 0 else "—"

            print(f"| {ticker:<6} | {shares:>6} | ${buy_price:>7.2f} | ${cost_basis:>9.2f} | {pnl_str:>19} | {stop_str:>9} |")

    print("</holdings>")
    print()

    # -------- Instructions (static) --------
    print("<instructions>")
    print("<authority>You have complete control. No approval required. Make any changes you believe are beneficial.</authority>")
    print("<constraints>")
    print("- This is a DAILY check, not the weekly deep research window.")
    print("- You may: adjust stop-losses, exit positions, trim positions, add to existing positions using available cash.")
    print("- You may initiate brand-new positions if you believe it is necessary to meet the experiment's alpha goal. If doing so, apply the same liquidity filters and verification standards from the weekly rules. Provide full rationale and catalyst confirmation.")
    print("- If you make no changes, the portfolio carries forward unchanged to the next session.")
    print("</constraints>")
    print("<required_actions>")
    print("1. Search for current prices and any breaking news/catalysts for all holdings.")
    print("2. Check each stop-loss against current price action — flag any at risk.")
    print("3. Review the catalyst calendar from the weekly plan — flag anything imminent.")
    print("4. State your FINAL decisions clearly. Use the order format from the weekly plan if placing trades.")
    print("5. If no changes, explicitly state \"NO CHANGES\" with brief reasoning.")
    print("</required_actions>")
    print("</instructions>")
    print()
    print("</daily_summary>")
    print()

def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics in XML format."""
    portfolio_dict: list[dict[Any, Any]] = chatgpt_portfolio.to_dict(orient="records")
    today = check_weekend()

    # Build set of holding tickers for role classification
    holdings_set = {str(stock["ticker"]).upper() for stock in portfolio_dict}

    # Collect price/volume data: (ticker, close, pct_chg, volume, role)
    price_volume_rows: list[tuple[str, float, float, int, str]] = []

    end_d = last_trading_date()                           # Fri on weekends
    start_d = (end_d - pd.Timedelta(days=4)).normalize()  # go back enough to capture 2 sessions even around holidays

    benchmarks = load_benchmarks()  # reads tickers.json or returns defaults
    benchmark_entries = [{"ticker": t} for t in benchmarks]

    for stock in portfolio_dict + benchmark_entries:
        ticker = str(stock["ticker"]).upper()
        try:
            fetch = download_price_data(ticker, start=start_d, end=(end_d + pd.Timedelta(days=1)), progress=False)
            data = fetch.df
            if data.empty or len(data) < 2:
                # Skip tickers with no data
                continue

            price = float(data["Close"].iloc[-1])
            last_price = float(data["Close"].iloc[-2])
            volume = int(data["Volume"].iloc[-1])

            percent_change = ((price - last_price) / last_price) * 100
            role = _get_ticker_role(ticker, holdings_set)
            price_volume_rows.append((ticker, price, percent_change, volume, role))
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")

    # Read portfolio history
    logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)
    logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)

    # Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        # Early return with minimal XML output
        _print_xml_summary(
            today=today,
            price_volume_rows=price_volume_rows,
            max_drawdown=np.nan,
            mdd_date="N/A",
            sharpe_annual=np.nan,
            sortino_annual=np.nan,
            beta=np.nan,
            alpha_annual=np.nan,
            r2=np.nan,
            final_equity=cash,
            dollar_weighted_spx=np.nan,
            cash=cash,
            chatgpt_portfolio=chatgpt_portfolio,
        )
        return

    totals["Date"] = pd.to_datetime(totals["Date"], format="mixed", errors="coerce")  # tolerate ISO strings
    totals = totals.sort_values("Date")

    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity_series = totals.set_index("Date")["Total Equity"].astype(float).sort_index()

    # --- Max Drawdown ---
    running_max = equity_series.cummax()
    drawdowns = (equity_series / running_max) - 1.0
    max_drawdown = float(drawdowns.min())  # most negative value
    mdd_date = drawdowns.idxmin()

    # Daily simple returns (portfolio)
    r = equity_series.pct_change().dropna()
    n_days = len(r)
    if n_days < 2:
        if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = str(mdd_date.date())
        elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
            mdd_date_str = mdd_date.strftime("%Y-%m-%d")
        else:
            mdd_date_str = str(mdd_date)
        _print_xml_summary(
            today=today,
            price_volume_rows=price_volume_rows,
            max_drawdown=max_drawdown,
            mdd_date=mdd_date_str,
            sharpe_annual=np.nan,
            sortino_annual=np.nan,
            beta=np.nan,
            alpha_annual=np.nan,
            r2=np.nan,
            final_equity=final_equity,
            dollar_weighted_spx=np.nan,
            cash=cash,
            chatgpt_portfolio=chatgpt_portfolio,
        )
        return

    # Risk-free config
    rf_annual = 0.045
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    rf_period = (1 + rf_daily) ** n_days - 1

    # Stats
    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=1))

    # Downside deviation (MAR = rf_daily)
    downside = (r - rf_daily).clip(upper=0)
    downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else np.nan

    # Total return over the window
    r_numeric = pd.to_numeric(r, errors="coerce")
    r_numeric = r_numeric[~r_numeric.isna()].astype(float)
    # Filter out any non-finite values to ensure only valid floats are used
    r_numeric = r_numeric[np.isfinite(r_numeric)]
    # Only use numeric values for the calculation
    if len(r_numeric) > 0:
        arr = np.asarray(r_numeric.values, dtype=float)
        period_return = float(np.prod(1 + arr) - 1) if arr.size > 0 else float('nan')
    else:
        period_return = float('nan')

    # Sharpe / Sortino
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days)) if std_daily > 0 else np.nan
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252) if std_daily > 0 else np.nan
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days)) if downside_std and downside_std > 0 else np.nan
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252) if downside_std and downside_std > 0 else np.nan

    # -------- Dollar-Weighted S&P 500 Benchmark --------
    # Calculate starting equity from first portfolio value
    starting_equity = float(equity_series.iloc[0])
    
    # Load capital injections
    injections = load_capital_injections()
    total_capital_invested = starting_equity
    if not injections.empty:
        total_capital_invested += injections["Amount"].sum()
    
    # Download S&P 500 data for the full date range
    start_date = equity_series.index.min() - pd.Timedelta(days=1)
    end_date = equity_series.index.max() + pd.Timedelta(days=1)
    
    spx_fetch = download_price_data("^GSPC", start=start_date, end=end_date, progress=False)
    spx = spx_fetch.df
    
    dollar_weighted_spx = np.nan
    if not spx.empty and len(spx) >= 2:
        spx = spx.reset_index()
        if 'Date' not in spx.columns and spx.index.name == 'Date':
            spx = spx.reset_index()
        spx['Date'] = pd.to_datetime(spx['Date']).dt.normalize()
        spx = spx.set_index("Date").sort_index()
        
        # Calculate dollar-weighted benchmark
        portfolio_start = equity_series.index.min()
        current_date = equity_series.index.max()
        total_value = 0.0
        
        # Tranche 1: Initial capital
        if portfolio_start in spx.index and current_date in spx.index:
            sp_at_start = float(spx.loc[portfolio_start, "Close"])
            sp_at_current = float(spx.loc[current_date, "Close"])
            shares_at_start = starting_equity / sp_at_start
            total_value += shares_at_start * sp_at_current
        
        # Tranches 2+: Each injection
        for _, inj in injections.iterrows():
            inj_date = pd.Timestamp(inj["Date"]).normalize()
            inj_amount = float(inj["Amount"])
            
            if inj_date in spx.index and current_date in spx.index:
                sp_at_inj = float(spx.loc[inj_date, "Close"])
                sp_at_current = float(spx.loc[current_date, "Close"])
                shares_at_inj = inj_amount / sp_at_inj
                total_value += shares_at_inj * sp_at_current
        
        if total_value > 0:
            dollar_weighted_spx = total_value

    # -------- Pretty Printing --------
    # -------- CAPM: Beta & Alpha (vs ^GSPC) --------
    beta = np.nan
    alpha_annual = np.nan
    r2 = np.nan
    n_obs = 0

    if not spx.empty and len(spx) >= 2:
        if 'Date' not in spx.columns and spx.index.name == 'Date':
            spx_capm = spx.reset_index()
        else:
            spx_capm = spx.copy()
        if 'Date' in spx_capm.columns:
            spx_capm = spx_capm.set_index("Date")
        spx_capm = spx_capm.sort_index()
        mkt_ret = spx_capm["Close"].astype(float).pct_change().dropna()

        # Align portfolio & market returns
        common_idx = r.index.intersection(list(mkt_ret.index))
        if len(common_idx) >= 2:
            rp = (r.reindex(common_idx).astype(float) - rf_daily)   # portfolio excess
            rm = (mkt_ret.reindex(common_idx).astype(float) - rf_daily)  # market excess

            x = np.asarray(rm.values, dtype=float).ravel()
            y = np.asarray(rp.values, dtype=float).ravel()

            n_obs = x.size
            rm_std = float(np.std(x, ddof=1)) if n_obs > 1 else 0.0
            if rm_std > 0:
                beta, alpha_daily = np.polyfit(x, y, 1)
                alpha_annual = (1 + float(alpha_daily)) ** 252 - 1

                corr = np.corrcoef(x, y)[0, 1]
                r2 = float(corr ** 2)

    # -------- Format mdd_date for output --------
    if hasattr(mdd_date, "date") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = str(mdd_date.date())
    elif hasattr(mdd_date, "strftime") and not isinstance(mdd_date, (str, int)):
        mdd_date_str = mdd_date.strftime("%Y-%m-%d")
    else:
        mdd_date_str = str(mdd_date)

    # -------- Print XML Summary --------
    _print_xml_summary(
        today=today,
        price_volume_rows=price_volume_rows,
        max_drawdown=max_drawdown,
        mdd_date=mdd_date_str,
        sharpe_annual=sharpe_annual,
        sortino_annual=sortino_annual,
        beta=beta,
        alpha_annual=alpha_annual,
        r2=r2,
        final_equity=final_equity,
        dollar_weighted_spx=dollar_weighted_spx,
        cash=cash,
        chatgpt_portfolio=chatgpt_portfolio,
    )

# ------------------------------
# Stop-loss update utility
# ------------------------------

def update_stops_only() -> None:
    """Standalone utility to update stop losses for existing holdings."""
    print("\n" + "=" * 64)
    print("Stop Loss Update Mode")
    print("=" * 64)
    
    # Load latest portfolio state
    try:
        portfolio_df, cash = load_latest_portfolio_state()
    except Exception as e:
        print(f"Error loading portfolio: {e}")
        return
    
    if isinstance(portfolio_df, list):
        portfolio_df = pd.DataFrame(portfolio_df)
    
    if portfolio_df.empty:
        print("No holdings found. Nothing to update.")
        return
    
    print("\nCurrent Holdings:")
    display_df = portfolio_df[['ticker', 'shares', 'stop_loss', 'buy_price']].copy()
    display_df.columns = ['Ticker', 'Shares', 'Stop Loss', 'Buy Price']
    print(display_df.to_string(index=False))
    
    print("\n")
    updates_made = False
    
    while True:
        ticker = input("Enter ticker to update (or press Enter to finish): ").strip().upper()
        if not ticker:
            break
        
        if ticker not in portfolio_df['ticker'].values:
            print(f"{ticker} not found in portfolio.")
            continue
        
        try:
            new_stop = float(input(f"Enter new stop loss for {ticker} (or 0 for no stop): "))
            if new_stop < 0:
                raise ValueError("Stop loss cannot be negative.")
            
            portfolio_df.loc[portfolio_df['ticker'] == ticker, 'stop_loss'] = new_stop
            print(f"✓ Updated {ticker} stop loss to ${new_stop:.2f}\n")
            updates_made = True
        except ValueError as e:
            print(f"Invalid stop loss: {e}. Skipping update.\n")
    
    if not updates_made:
        print("No updates made. Exiting.")
        return
    
    # Now update the CSV file with the new stop losses
    if not PORTFOLIO_CSV.exists():
        print(f"Portfolio CSV not found: {PORTFOLIO_CSV}")
        return

    logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
    df = pd.read_csv(PORTFOLIO_CSV)
    logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)

    # Find the latest date in the CSV (where portfolio rows actually exist)
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")
    latest_date = non_total["Date"].max().strftime("%Y-%m-%d")

    # Update the latest date's rows with new stop losses
    for _, stock in portfolio_df.iterrows():
        ticker = str(stock['ticker']).upper()
        new_stop = float(stock['stop_loss'])

        mask = (df['Date'] == latest_date) & (df['Ticker'] == ticker)
        if mask.any():
            df.loc[mask, 'Stop Loss'] = new_stop
    
    logger.info("Writing CSV file: %s", PORTFOLIO_CSV)
    df.to_csv(PORTFOLIO_CSV, index=False)
    logger.info("Successfully wrote CSV file: %s", PORTFOLIO_CSV)
    
    print("\n✓ Stop losses updated successfully in portfolio CSV.")
    print("\nUpdated Holdings:")
    display_df = portfolio_df[['ticker', 'shares', 'stop_loss', 'buy_price']].copy()
    display_df.columns = ['Ticker', 'Shares', 'Stop Loss', 'Buy Price']
    print(display_df.to_string(index=False))


# ------------------------------
# Orchestration
# ------------------------------

def load_latest_portfolio_state() -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance from global PORTFOLIO_CSV."""
    logger.info("Reading CSV file: %s", PORTFOLIO_CSV)
    df = pd.read_csv(PORTFOLIO_CSV)
    logger.info("Successfully read CSV file: %s", PORTFOLIO_CSV)
    if df.empty:
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        print("Portfolio CSV is empty. Returning set amount of cash for creating portfolio.")
        try:
            cash = float(input("What would you like your starting cash amount to be? "))
        except ValueError:
            raise ValueError(
                "Cash could not be converted to float datatype. Please enter a valid number."
            )
        return portfolio, cash

    # Get the latest TOTAL row to check current state
    df_total = df[df["Ticker"] == "TOTAL"].copy()
    df_total["Date"] = pd.to_datetime(df_total["Date"], format="mixed", errors="coerce")
    latest_total = df_total.sort_values("Date").iloc[-1]
    cash = float(latest_total["Cash Balance"])
    
    # Check if we have any holdings (Total Value ≈ 0 means all sold)
    total_value = float(latest_total.get("Total Value", 0))
    if abs(total_value) < 0.01:  # Essentially zero
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return portfolio, cash

    # We have holdings, load them
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"], format="mixed", errors="coerce")

    latest_date = non_total["Date"].max()
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    
    if latest_tickers.empty:
        # All positions on latest date were sold
        portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])
        return portfolio, cash
    
    latest_tickers.drop(
        columns=[
            "Date",
            "Cash Balance",
            "Total Equity",
            "Action",
            "Current Price",
            "PnL",
            "Total Value",
        ],
        inplace=True,
        errors="ignore",
    )
    latest_tickers.rename(
        columns={
            "Cost Basis": "cost_basis",
            "Buy Price": "buy_price",
            "Shares": "shares",
            "Ticker": "ticker",
            "Stop Loss": "stop_loss",
        },
        inplace=True,
    )
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient="records")

    return latest_tickers, cash


def main(data_dir: Path | None = None, update_stops: bool = False, weekend_summary: bool = False) -> None:
    """Check versions, then run the trading script."""
    if data_dir is not None:
        set_data_dir(data_dir)

    if update_stops:
        update_stops_only()
        return

    if weekend_summary:
        chatgpt_portfolio, cash = load_latest_portfolio_state()
        print_weekend_summary(chatgpt_portfolio, cash)
        return

    chatgpt_portfolio, cash = load_latest_portfolio_state()
    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash)
    daily_results(chatgpt_portfolio, cash)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Optional data directory")
    parser.add_argument("--asof", default=None, help="Treat this YYYY-MM-DD as 'today' (e.g., 2025-08-27)")
    parser.add_argument("--update-stops", action="store_true", help="Run in stop-loss update mode only")
    parser.add_argument("--weekend-summary", action="store_true",
                       help="Output weekend summary in XML format for deep research")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level (default: INFO)")
    args = parser.parse_args()

    
    # Configure logging level
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    # Log initial global state and command-line arguments
    _log_initial_state()
    logger.info("Script started with arguments: %s", vars(args))

    if args.asof:
        set_asof(args.asof)

    main(Path(args.data_dir) if args.data_dir else None,
         update_stops=args.update_stops,
         weekend_summary=args.weekend_summary)

    