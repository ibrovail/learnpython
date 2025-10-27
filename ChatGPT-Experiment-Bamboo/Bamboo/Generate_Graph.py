"""
Plot portfolio performance with three-line comparison:
1. Buy-and-hold baseline (original portfolio left untouched)
2. Active portfolio (with all trades executed)
3. Dollar-weighted S&P 500 benchmark

Supports multiple capital injections tracked in capital_injections.csv.
- Normalizes all three series with dollar-weighting.
- For each capital injection date, calculates a separate S&P 500 benchmark tranche.
- Reconstructs buy-and-hold from initial snapshot.
- Aligns S&P data to portfolio dates with forward-fill.
- Strict daily ticks with tz-naive midnight normalization.
- Robust MultiIndex column handling from yfinance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
import matplotlib.ticker as mticker # type: ignore
import pandas as pd # type: ignore
import yfinance as yf # type: ignore
import numpy as np # type: ignore

DATA_DIR = Path(__file__).resolve().parent
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
CAPITAL_INJECTIONS_CSV = DATA_DIR / "capital_injections.csv"
INITIAL_SNAPSHOT_CSV = DATA_DIR / "initial_portfolio_snapshot.csv"


# ---------- Helpers ----------

def _date_only_series(x: pd.Series) -> pd.Series:
    """tz-naive, normalized to midnight (YYYY-MM-DD 00:00)."""
    dt = pd.to_datetime(x, errors="coerce")
    if hasattr(dt, "dt"):
        try:
            dt = dt.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        return dt.dt.normalize()
    if getattr(dt, "tzinfo", None) is not None:
        try:
            dt = dt.tz_convert(None)
        except Exception:
            dt = dt.tz_localize(None)
    return pd.to_datetime(dt).normalize()


def parse_date(date_str: str, label: str) -> pd.Timestamp:
    try:
        dt = pd.to_datetime(date_str)
        if getattr(dt, "tzinfo", None) is not None:
            try:
                dt = dt.tz_convert(None)
            except Exception:
                dt = dt.tz_localize(None)
        return pd.to_datetime(dt).normalize()
    except Exception as exc:
        raise SystemExit(f"Invalid {label} '{date_str}'. Use YYYY-MM-DD.") from exc


def _normalize_to_start(series: pd.Series, target_value: float) -> pd.Series:
    """Normalize a series to start at target_value."""
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return pd.Series(dtype=float)
    start_value = s.iloc[0]
    if start_value == 0:
        return s * 0
    return (s / start_value) * target_value


def _flatten_columns(cols) -> List[str]:
    """Flatten possible MultiIndex columns from yfinance."""
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(p) for p in tup if p is not None and str(p) != ""]
            flat.append("_".join(parts))
        return flat
    return list(map(str, cols))


def _align_to_dates(sp500_data: pd.DataFrame, portfolio_dates: pd.Series) -> pd.Series:
    """Align S&P 500 data to portfolio dates using forward fill."""
    portfolio_dates_norm = _date_only_series(portfolio_dates)
    aligned_df = pd.DataFrame({"Date": portfolio_dates_norm})
    merged = aligned_df.merge(sp500_data, on="Date", how="left")
    merged["Close"] = merged["Close"].ffill()
    return merged["Close"]


# ---------- Data loaders ----------

def load_portfolio_details(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    portfolio_csv: Path = PORTFOLIO_CSV,
) -> pd.DataFrame:
    """Return TOTAL rows (Date, Total Equity) filtered to [start_date, end_date]."""
    if not portfolio_csv.exists():
        raise SystemExit(f"Portfolio file '{portfolio_csv}' not found.")

    df = pd.read_csv(portfolio_csv)
    totals = df[df["Ticker"] == "TOTAL"].copy()
    if totals.empty:
        raise SystemExit("Portfolio CSV contains no TOTAL rows.")

    totals["Date"] = _date_only_series(totals["Date"])
    totals["Total Equity"] = pd.to_numeric(totals["Total Equity"], errors="coerce")
    totals = totals.dropna(subset=["Date", "Total Equity"]).sort_values("Date")

    min_date = totals["Date"].min()
    max_date = totals["Date"].max()
    if start_date is None or start_date < min_date:
        start_date = min_date
    if end_date is None or end_date > max_date:
        end_date = max_date
    if start_date > end_date:
        raise SystemExit("Start date must be on or before end date.")

    mask = (totals["Date"] >= start_date) & (totals["Date"] <= end_date)
    return totals.loc[mask, ["Date", "Total Equity"]].reset_index(drop=True)


def load_capital_injections(
    injections_csv: Path = CAPITAL_INJECTIONS_CSV,
) -> pd.DataFrame:
    """Load capital injections, returning empty DataFrame if file doesn't exist."""
    if not injections_csv.exists():
        return pd.DataFrame(columns=["Date", "Amount"])
    
    df = pd.read_csv(injections_csv)
    df["Date"] = _date_only_series(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_initial_snapshot(
    snapshot_csv: Path = INITIAL_SNAPSHOT_CSV,
) -> tuple[pd.DataFrame, float, float]:
    """
    Load the initial portfolio snapshot.
    Returns: (holdings_df, total_portfolio_value, cash_balance)
    """
    if not snapshot_csv.exists():
        raise SystemExit(f"Initial snapshot file not found: {snapshot_csv}\nRun trading_script.py first to create it.")
    
    df = pd.read_csv(snapshot_csv)
    df["Date"] = _date_only_series(df["Date"])
    
    total_value = float(df["Total_Portfolio_Value"].iloc[0])
    
    # Calculate cash: total_value - sum(position values)
    position_values = df["Value"].sum()
    cash = total_value - position_values
    
    return df, total_value, cash


def download_sp500(dates: pd.Series) -> pd.DataFrame:
    """Download S&P 500 data for the given date range."""
    dates_norm = _date_only_series(dates)
    if dates_norm.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    start_date = dates_norm.min()
    end_date = dates_norm.max()

    try:
        sp500 = yf.download(
            "^GSPC",
            start=start_date,
            end=end_date + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=False,
            group_by="column"
        )
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return pd.DataFrame(columns=["Date", "Close"])

    if sp500 is None or sp500.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    sp500 = sp500.reset_index()
    sp500.columns = _flatten_columns(sp500.columns)
    sp500["Date"] = _date_only_series(sp500["Date"])

    # Robustly find Close column
    close_candidates = [c for c in sp500.columns if c.lower().startswith("close")]
    if not close_candidates:
        close_candidates = [c for c in sp500.columns if c.lower().startswith("adj close")]
    if not close_candidates:
        print("Could not find a Close column in yfinance data.")
        return pd.DataFrame(columns=["Date", "Close"])

    close_col = close_candidates[0]
    result = sp500[["Date", close_col]].rename(columns={close_col: "Close"}).copy()
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return result.dropna(subset=["Close"])


def download_ticker_history(ticker: str, dates: pd.Series) -> pd.DataFrame:
    """Download historical price data for a specific ticker."""
    dates_norm = _date_only_series(dates)
    if dates_norm.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    start_date = dates_norm.min()
    end_date = dates_norm.max()

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date + pd.Timedelta(days=1),
            progress=False,
            auto_adjust=False,
            group_by="column"
        )
    except Exception as e:
        print(f"Error downloading {ticker} data: {e}")
        return pd.DataFrame(columns=["Date", "Close"])

    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    data = data.reset_index()
    data.columns = _flatten_columns(data.columns)
    data["Date"] = _date_only_series(data["Date"])

    # Find Close column
    close_candidates = [c for c in data.columns if c.lower().startswith("close")]
    if not close_candidates:
        close_candidates = [c for c in data.columns if c.lower().startswith("adj close")]
    if not close_candidates:
        return pd.DataFrame(columns=["Date", "Close"])

    close_col = close_candidates[0]
    result = data[["Date", close_col]].rename(columns={close_col: "Close"}).copy()
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return result.dropna(subset=["Close"])


def build_buy_and_hold_series(
    portfolio_dates: pd.Series,
    initial_snapshot: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """
    Build a buy-and-hold series based on the initial snapshot.
    
    For each date, calculate what the original holdings would be worth
    if no trades were executed (only market price changes).
    
    IMPORTANT: Uses initial_snapshot to determine which tickers to track,
    NOT the current portfolio (which may have had positions sold).
    """
    portfolio_dates_norm = _date_only_series(portfolio_dates)
    if portfolio_dates_norm.empty or initial_snapshot.empty:
        return pd.DataFrame({"Date": portfolio_dates_norm, "Value": np.nan})
    
    result_dates = sorted(portfolio_dates_norm.unique())
    buy_hold_values = []
    
    # Download price history for each ticker in INITIAL snapshot (not current portfolio)
    ticker_prices = {}
    print("\nBuilding buy-and-hold baseline from initial snapshot...")
    for ticker in initial_snapshot["Ticker"].unique():
        print(f"  Downloading price history for {ticker}...")
        ticker_prices[ticker] = download_ticker_history(ticker, portfolio_dates)
    
    # For each date, calculate portfolio value based on ORIGINAL holdings
    for date in result_dates:
        portfolio_value = initial_cash  # Cash stays constant in buy-and-hold
        
        # Use INITIAL snapshot holdings (not current portfolio)
        for _, holding in initial_snapshot.iterrows():
            ticker = holding["Ticker"]
            shares = float(holding["Shares"])
            
            # Get price for this ticker on this date
            if ticker in ticker_prices:
                price_data = ticker_prices[ticker]
                price_on_date = price_data[price_data["Date"] == date]["Close"]
                
                if not price_on_date.empty:
                    price = float(price_on_date.iloc[0])
                    portfolio_value += shares * price
                else:
                    # No price data for this date, use forward fill
                    prior_prices = price_data[price_data["Date"] < date]
                    if not prior_prices.empty:
                        price = float(prior_prices.iloc[-1]["Close"])
                        portfolio_value += shares * price
        
        buy_hold_values.append(portfolio_value)
    
    return pd.DataFrame({"Date": result_dates, "Value": buy_hold_values})


def build_dollar_weighted_benchmark(
    portfolio_dates: pd.Series,
    injections: pd.DataFrame,
    sp500_data: pd.DataFrame,
    starting_equity: float,
) -> pd.DataFrame:
    """
    Build a dollar-weighted S&P 500 benchmark.
    
    For each capital injection, calculate what that tranche would be worth
    in S&P 500 from injection date to each portfolio date, then sum all tranches.
    """
    portfolio_dates_norm = _date_only_series(portfolio_dates)
    if portfolio_dates_norm.empty or sp500_data.empty:
        return pd.DataFrame({"Date": portfolio_dates_norm, "Value": np.nan})
    
    portfolio_start = portfolio_dates_norm.min()
    sp500_data = sp500_data.sort_values("Date").reset_index(drop=True)
    injections = injections.sort_values("Date").reset_index(drop=True)
    
    result_dates = sorted(portfolio_dates_norm.unique())
    benchmark_values = []
    
    for date in result_dates:
        total_value = 0.0
        
        # Handle initial capital (invested at portfolio start date)
        sp_at_start = sp500_data[sp500_data["Date"] == portfolio_start]["Close"]
        sp_at_date = sp500_data[sp500_data["Date"] == date]["Close"]
        
        if not sp_at_start.empty and not sp_at_date.empty and date >= portfolio_start:
            price_ratio = float(sp_at_date.iloc[0]) / float(sp_at_start.iloc[0])
            total_value += starting_equity * price_ratio
        
        # Handle each capital injection
        for _, inj in injections.iterrows():
            inj_date = inj["Date"]
            inj_amount = float(inj["Amount"])
            
            if date >= inj_date:
                sp_at_inj = sp500_data[sp500_data["Date"] == inj_date]["Close"]
                sp_at_date_inj = sp500_data[sp500_data["Date"] == date]["Close"]
                
                if not sp_at_inj.empty and not sp_at_date_inj.empty:
                    price_ratio = float(sp_at_date_inj.iloc[0]) / float(sp_at_inj.iloc[0])
                    total_value += inj_amount * price_ratio
        
        benchmark_values.append(total_value)
    
    return pd.DataFrame({"Date": result_dates, "Value": benchmark_values})


# ---------- Plotting ----------

def plot_three_way_comparison(
    active_portfolio: pd.DataFrame,
    buy_hold: pd.DataFrame,
    benchmark: pd.DataFrame,
    starting_equity: float,
    title: str = "Portfolio Performance: Active vs Buy-and-Hold vs S&P 500",
) -> None:
    """Plot three-line comparison with clean formatting and strict daily ticks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Active portfolio
    a_dates = _date_only_series(active_portfolio["Date"])
    a_values = active_portfolio["Value"]
    ax.plot(a_dates, a_values, label="Active Portfolio", marker="o", linewidth=2)

    # Buy-and-hold
    if not buy_hold.empty:
        bh_dates = _date_only_series(buy_hold["Date"])
        bh_values = buy_hold["Value"]
        ax.plot(bh_dates, bh_values, label="Buy & Hold (No Trades)", marker="^", linestyle=":", linewidth=2)

    # S&P 500 benchmark
    if not benchmark.empty:
        b_dates = _date_only_series(benchmark["Date"])
        b_values = benchmark["Value"]
        ax.plot(b_dates, b_values, label="Dollar-Weighted S&P 500", marker="s", linestyle="--", linewidth=2)

    # Add return % labels on the final points
    a_last = float(a_values.iloc[-1])
    a_return_pct = ((a_last / starting_equity) - 1) * 100
    ax.text(a_dates.iloc[-1], a_last * 1.01, f"{a_return_pct:+.1f}%", fontsize=9, weight='bold')
    
    if not buy_hold.empty:
        bh_last = float(bh_values.iloc[-1])
        bh_return_pct = ((bh_last / starting_equity) - 1) * 100
        ax.text(bh_dates.iloc[-1], bh_last * 1.01, f"{bh_return_pct:+.1f}%", fontsize=9)
    
    if not benchmark.empty:
        b_last = float(b_values.iloc[-1])
        b_return_pct = ((b_last / starting_equity) - 1) * 100
        ax.text(b_dates.iloc[-1], b_last * 1.01, f"{b_return_pct:+.1f}%", fontsize=9)

    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Portfolio Value (start = ${starting_equity:,.0f})", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Strict daily ticks with tz-naive YYYY-MM-DD labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))  # Major ticks every 3 days
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Add 1-day padding on each side to prevent lines from touching frame
    pad = pd.Timedelta(days=1)
    ax.set_xlim(a_dates.min() - pad, a_dates.max() + pad)

    fig.autofmt_xdate(rotation=45, ha="right")
    plt.tight_layout()


# ---------- Main ----------

def main(
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    starting_equity: Optional[float],
    output: Optional[Path],
    portfolio_csv: Path = PORTFOLIO_CSV,
    injections_csv: Path = CAPITAL_INJECTIONS_CSV,
    snapshot_csv: Path = INITIAL_SNAPSHOT_CSV,
) -> None:
    """Main execution: load data, build all three series, and plot."""
    
    # Load initial snapshot
    print("Loading initial portfolio snapshot...")
    initial_snapshot, snapshot_total, initial_cash = load_initial_snapshot(snapshot_csv)
    
    # Use snapshot total as starting equity if not provided
    if starting_equity is None:
        starting_equity = snapshot_total
        print(f"Using initial snapshot value as starting equity: ${starting_equity:,.2f}")
    
    # Load active portfolio
    print("Loading active portfolio history...")
    totals = load_portfolio_details(start_date, end_date, portfolio_csv=portfolio_csv)
    
    # Normalize active portfolio to starting equity
    norm_port = totals.copy()
    norm_port["Value"] = _normalize_to_start(norm_port["Total Equity"], starting_equity)
    norm_port = norm_port[["Date", "Value"]]
    
    # Load S&P 500 data
    print("Downloading S&P 500 data...")
    spx = download_sp500(norm_port["Date"])
    
    # Load capital injections
    injections = load_capital_injections(injections_csv=injections_csv)
    
    # Build buy-and-hold series
    print("\nBuilding buy-and-hold baseline...")
    buy_hold = build_buy_and_hold_series(norm_port["Date"], initial_snapshot, initial_cash)
    
    # DON'T normalize buy-and-hold - it already starts at the correct initial total value
    # The initial snapshot already accounts for the actual starting portfolio value
    print(f"Buy-and-hold starts at: ${buy_hold['Value'].iloc[0]:,.2f}")
    print(f"Active portfolio normalized to: ${starting_equity:,.2f}")
    
    # Only normalize if they don't match (for display consistency)
    if abs(buy_hold['Value'].iloc[0] - starting_equity) > 1.0:
        print(f"Normalizing buy-and-hold from ${buy_hold['Value'].iloc[0]:,.2f} to ${starting_equity:,.2f}")
        buy_hold["Value"] = _normalize_to_start(buy_hold["Value"], starting_equity)
    else:
        print("Buy-and-hold already at correct starting value, no normalization needed.")

    
    # Build dollar-weighted S&P 500 benchmark
    if injections.empty:
        print("No capital injections found. Using starting equity as initial tranche.")
    else:
        print(f"Found {len(injections)} capital injection(s). Building dollar-weighted benchmark...")
        for _, row in injections.iterrows():
            print(f"  - {row['Date'].date()}: ${row['Amount']:.2f}")
    
    benchmark = build_dollar_weighted_benchmark(norm_port["Date"], injections, spx, starting_equity)
    
    # Plot three-way comparison
    print("\nGenerating chart...")
    plot_three_way_comparison(norm_port, buy_hold, benchmark, starting_equity,
                              title="Portfolio Performance: Active vs Buy-and-Hold vs S&P 500")
    
    if output:
        output = output if output.is_absolute() else DATA_DIR / output
        plt.savefig(output, bbox_inches="tight", dpi=150)
        print(f"✓ Chart saved to {output}")
    else:
        plt.show()
    plt.close()
    
    # Print summary stats
    print("\n" + "="*64)
    print("PERFORMANCE SUMMARY")
    print("="*64)
    
    a_start = float(norm_port["Value"].iloc[0])
    a_end = float(norm_port["Value"].iloc[-1])
    a_return = ((a_end - a_start) / a_start) * 100
    print(f"\n{'Active Portfolio:':25} ${a_start:>12,.2f} → ${a_end:>12,.2f} ({a_return:>+7.2f}%)")
    
    if not buy_hold.empty:
        bh_start = float(buy_hold["Value"].iloc[0])
        bh_end = float(buy_hold["Value"].iloc[-1])
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        print(f"{'Buy & Hold:':25} ${bh_start:>12,.2f} → ${bh_end:>12,.2f} ({bh_return:>+7.2f}%)")
        
        alpha_vs_bh = a_return - bh_return
        print(f"{'Alpha vs Buy & Hold:':25} {alpha_vs_bh:>+34.2f}%")
    
    if not benchmark.empty:
        b_start = float(benchmark["Value"].iloc[0])
        b_end = float(benchmark["Value"].iloc[-1])
        b_return = ((b_end - b_start) / b_start) * 100
        print(f"{'S&P 500 Benchmark:':25} ${b_start:>12,.2f} → ${b_end:>12,.2f} ({b_return:>+7.2f}%)")
        
        alpha_vs_sp = a_return - b_return
        print(f"{'Alpha vs S&P 500:':25} {alpha_vs_sp:>+34.2f}%")
    
    print("\n" + "="*64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot portfolio vs buy-and-hold vs S&P 500")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--start-equity", type=float, 
                       help="Initial capital baseline (default: use initial snapshot value)")
    parser.add_argument("--baseline-file", type=str, 
                       help="Path to a text file containing a single number for baseline")
    parser.add_argument("--output", type=str, help="Optional path to save the chart (.png/.jpg/.pdf)")

    args = parser.parse_args()
    start = parse_date(args.start_date, "start date") if args.start_date else None
    end = parse_date(args.end_date, "end date") if args.end_date else None

    baseline = args.start_equity
    if args.baseline_file:
        p = Path(args.baseline_file)
        if not p.exists():
            raise SystemExit(f"Baseline file not found: {p}")
        try:
            baseline = float(p.read_text().strip())
        except Exception as exc:
            raise SystemExit(f"Could not parse baseline from {p}") from exc

    out_path = Path(args.output) if args.output else None
    main(start, end, baseline, out_path)