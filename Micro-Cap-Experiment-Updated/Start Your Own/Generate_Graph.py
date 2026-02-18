"""
Plot portfolio performance vs. dollar-weighted S&P 500 benchmark.

Supports multiple capital injections tracked in capital_injections.csv.
- Normalizes both series (portfolio and S&P) with dollar-weighting.
- For each capital injection date, calculates a separate S&P 500 benchmark tranche.
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


def build_dollar_weighted_benchmark(
    portfolio_dates: pd.Series,
    injections: pd.DataFrame,
    sp500_data: pd.DataFrame,
    starting_equity: float,
) -> pd.DataFrame:
    """
    Build a dollar-weighted S&P 500 benchmark.
    
    Starting equity buys S&P at portfolio start date.
    Each injection buys additional S&P at the injection date's price.
    Returns the combined value of all tranches for each date.
    """
    portfolio_dates_norm = _date_only_series(portfolio_dates)
    if portfolio_dates_norm.empty or sp500_data.empty:
        return pd.DataFrame({"Date": portfolio_dates_norm, "Value": np.nan})
    
    portfolio_start = portfolio_dates_norm.min()
    sp500_data = sp500_data.sort_values("Date").reset_index(drop=True)
    sp500_data.set_index("Date", inplace=True)
    injections = injections.sort_values("Date").reset_index(drop=True)
    
    result_dates = sorted(portfolio_dates_norm.unique())
    benchmark_values = []
    
    # Helper function to get nearest S&P price (forward-fill if needed)
    def get_sp_price(target_date):
        if target_date in sp500_data.index:
            return float(sp500_data.loc[target_date, "Close"])
        # Find nearest date on or after target (forward-fill)
        future_dates = sp500_data.index[sp500_data.index >= target_date]
        if len(future_dates) > 0:
            return float(sp500_data.loc[future_dates[0], "Close"])
        # If no future dates, use last available
        return float(sp500_data["Close"].iloc[-1])
    
    for date in result_dates:
        total_value = 0.0
        
        # Tranche 1: Initial capital invested at portfolio start date
        try:
            sp_at_start = get_sp_price(portfolio_start)
            sp_at_date = get_sp_price(date)
            
            if date >= portfolio_start:
                shares_at_start = starting_equity / sp_at_start
                total_value += shares_at_start * sp_at_date
        except Exception:
            pass
        
        # Tranches 2+: Each capital injection buys S&P at injection date's price
        for _, inj in injections.iterrows():
            inj_date = inj["Date"]
            inj_amount = float(inj["Amount"])
            
            if date >= inj_date:
                try:
                    sp_at_inj = get_sp_price(inj_date)
                    sp_at_date_inj = get_sp_price(date)
                    
                    shares_at_inj = inj_amount / sp_at_inj
                    total_value += shares_at_inj * sp_at_date_inj
                except Exception:
                    pass
        
        benchmark_values.append(total_value)
    
    return pd.DataFrame({"Date": result_dates, "Value": benchmark_values})


# ---------- Plotting ----------

def plot_comparison(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    starting_equity: float,
    last_injection_date: Optional[pd.Timestamp],
    injections: pd.DataFrame,
    title: str = "Portfolio vs. Dollar-Weighted S&P 500",
) -> None:
    """Plot portfolio vs benchmark with clean formatting and strict daily ticks."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    p_dates = _date_only_series(portfolio["Date"])
    p_values = portfolio["Value"]
    ax.plot(p_dates, p_values, label="GPT/Claude", marker="o")

    if not benchmark.empty:
        b_dates = _date_only_series(benchmark["Date"])
        b_values = benchmark["Value"]
        ax.plot(b_dates, b_values, label="S&P 500", marker="s", linestyle="--")

    # Calculate total capital invested
    total_capital_invested = starting_equity
    if not injections.empty:
        total_capital_invested += injections["Amount"].sum()

    # Calculate returns since last injection (or start if no injections)
    if last_injection_date is not None:
        # Find value right after last injection
        p_baseline_idx = portfolio[portfolio["Date"] >= last_injection_date].index[0]
        p_baseline = float(portfolio.loc[p_baseline_idx, "Value"])
        
        if not benchmark.empty:
            b_baseline_idx = benchmark[benchmark["Date"] >= last_injection_date].index[0]
            b_baseline = float(benchmark.loc[b_baseline_idx, "Value"])
    else:
        # No injections, use total capital as baseline
        p_baseline = total_capital_invested
        b_baseline = total_capital_invested if not benchmark.empty else 0

    # Calculate both return metrics
    p_last = float(p_values.iloc[-1])
    p_return_total = ((p_last - total_capital_invested) / total_capital_invested) * 100
    p_return_since = ((p_last - p_baseline) / p_baseline) * 100
    
    # Add return % labels on the final points (both total and since injection)
    label_text = f"{p_return_total:.1f}%\n({p_return_since:+.1f}% period)"
    ax.text(p_dates.iloc[-1], p_last * 0.96, label_text, fontsize=8, ha='left')
    
    if not benchmark.empty:
        b_last = float(b_values.iloc[-1])
        b_return_total = ((b_last - total_capital_invested) / total_capital_invested) * 100
        b_return_since = ((b_last - b_baseline) / b_baseline) * 100
        label_text = f"{b_return_total:.1f}%\n({b_return_since:+.1f}% period)"
        ax.text(b_dates.iloc[-1], b_last * 1.02, label_text, fontsize=8, ha='left')

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Index (start = {starting_equity:g})")
    ax.legend()
    ax.grid(True)

    # Strict daily ticks with tz-naive YYYY-MM-DD labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Add 1-day padding on each side to prevent lines from touching frame
    pad = pd.Timedelta(days=1)
    ax.set_xlim(p_dates.min() - pad, p_dates.max() + pad)

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
) -> None:
    """Main execution: load data, build dollar-weighted benchmark, and plot."""
    # Load portfolio - use actual values, no normalization
    totals = load_portfolio_details(start_date, end_date, portfolio_csv=portfolio_csv)
    portfolio_data = totals[["Date", "Total Equity"]].copy()
    portfolio_data.rename(columns={"Total Equity": "Value"}, inplace=True)

    # Auto-detect starting equity from first portfolio value if not provided
    if starting_equity is None:
        starting_equity = float(portfolio_data["Value"].iloc[0])

    # Load S&P 500 data
    spx = download_sp500(portfolio_data["Date"])
    
    # Load capital injections
    injections = load_capital_injections(injections_csv=injections_csv)

    # Filter injections to portfolio date range (prevents IndexError when --end-date
    # precedes the last injection date in the CSV)
    portfolio_end = portfolio_data["Date"].max()
    portfolio_start = portfolio_data["Date"].min()
    injections = injections[
        (injections["Date"] >= portfolio_start) &
        (injections["Date"] <= portfolio_end)
    ].reset_index(drop=True)

    # Find last injection date
    last_injection_date = None
    if not injections.empty:
        last_injection_date = injections["Date"].max()
        print(f"Found {len(injections)} capital injection(s). Building dollar-weighted benchmark...")
        for _, row in injections.iterrows():
            print(f"  - {row['Date'].date()}: ${row['Amount']:.2f}")
    else:
        print("No capital injections found. Using starting equity as initial tranche.")
    
    # Build dollar-weighted benchmark
    benchmark = build_dollar_weighted_benchmark(portfolio_data["Date"], injections, spx, starting_equity)
    
    # Plot
    plot_comparison(portfolio_data, benchmark, starting_equity, last_injection_date, injections,
                   title="Portfolio vs. Dollar-Weighted S&P 500 Benchmark")
    
    if output:
        output = output if output.is_absolute() else DATA_DIR / output
        plt.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Chart saved to {output}")
    else:
        plt.show()
    plt.close()
    
    # Calculate returns
    p_start = float(portfolio_data["Value"].iloc[0])
    p_end = float(portfolio_data["Value"].iloc[-1])
    
    if not benchmark.empty:
        b_start = float(benchmark["Value"].iloc[0])
        b_end = float(benchmark["Value"].iloc[-1])
    
    # Total capital invested (starting equity + all injections)
    total_capital_invested = starting_equity
    if not injections.empty:
        total_capital_invested += injections["Amount"].sum()
    
    # Total returns (based on total capital invested, not starting value)
    p_total_return = ((p_end - total_capital_invested) / total_capital_invested) * 100
    b_total_return = ((b_end - total_capital_invested) / total_capital_invested) * 100 if not benchmark.empty else 0
    
    # Returns since last injection
    if last_injection_date is not None:
        p_baseline_idx = portfolio_data[portfolio_data["Date"] >= last_injection_date].index[0]
        p_baseline = float(portfolio_data.loc[p_baseline_idx, "Value"])
        p_since_injection = ((p_end - p_baseline) / p_baseline) * 100
        
        if not benchmark.empty:
            b_baseline_idx = benchmark[benchmark["Date"] >= last_injection_date].index[0]
            b_baseline = float(benchmark.loc[b_baseline_idx, "Value"])
            b_since_injection = ((b_end - b_baseline) / b_baseline) * 100
    else:
        p_since_injection = p_total_return
        b_since_injection = b_total_return
    
    # Print summary stats
    print("\nPerformance Summary")
    print("-" * 60)
    print("Total Returns (vs total capital invested):")
    print(f"  Portfolio: ${total_capital_invested:.2f} → ${p_end:.2f} ({p_total_return:+.2f}%)")
    if not benchmark.empty:
        print(f"  Benchmark: ${total_capital_invested:.2f} → ${b_end:.2f} ({b_total_return:+.2f}%)")
        print(f"  Outperformance: {(p_total_return - b_total_return):+.2f}%")
    
    if last_injection_date is not None:
        print(f"\nReturns Since Last Injection ({last_injection_date.date()}):")
        print(f"  Portfolio: ${p_baseline:.2f} → ${p_end:.2f} ({p_since_injection:+.2f}%)")
        if not benchmark.empty:
            print(f"  Benchmark: ${b_baseline:.2f} → ${b_end:.2f} ({b_since_injection:+.2f}%)")
            print(f"  Outperformance: {(p_since_injection - b_since_injection):+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot portfolio vs dollar-weighted S&P 500")
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--start-equity", type=float, default=None,
                       help="Initial capital baseline (auto-detected from CSV if omitted)")
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