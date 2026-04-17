"""Quantitative screener for the micro/small-cap universe.

Scans all sectors via Finviz, enriches with yfinance price/volume signals,
and ranks candidates by a composite momentum + volume + volatility score.
Output is a sector-tagged watchlist CSV for the weekend analysis workflow.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Universe fetching (Finviz → cache fallback)
# ---------------------------------------------------------------------------

# Exclusion keywords for security types we don't trade (portfolio_rules.md)
_EXCLUDED_TYPES = {"ETF", "ETN", "SPAC", "ADR"}


def get_universe(data_dir: Path) -> pd.DataFrame:
    """Pull filtered stock list from Finviz. Falls back to cached file."""
    cache_path = data_dir / "universe_cache.csv"

    try:
        df = _fetch_finviz_universe()
        if len(df) > 0:
            df.to_csv(cache_path, index=False)
            print(f"  Universe: {len(df)} stocks from Finviz (cached to {cache_path.name})")
            return df
    except Exception as e:
        print(f"  Finviz fetch failed: {e}")

    # Fallback to cache
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  Universe: {len(df)} stocks from cache ({cache_path.name})")
        return df

    print("  ERROR: No universe data available (Finviz down, no cache).", file=sys.stderr)
    sys.exit(1)


def _fetch_finviz_universe() -> pd.DataFrame:
    """Use finvizfinance to pull the screened universe."""
    from finvizfinance.screener.overview import Overview

    foverview = Overview()
    filters_dict = {
        "Market Cap.": "-Small (under $2bln)",
        "Average Volume": "Over 500K",
        "Price": "Over $1",
    }
    foverview.set_filter(filters_dict=filters_dict)
    df = foverview.screener_view()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Standardize columns
    col_map = {
        "Ticker": "ticker",
        "Sector": "sector",
        "Industry": "industry",
        "Market Cap": "market_cap_raw",
        "Price": "price",
        "Volume": "avg_volume",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=available)

    # Keep only columns we need
    keep = [c for c in ["ticker", "sector", "industry", "market_cap_raw", "price", "avg_volume"] if c in df.columns]
    df = df[keep].copy()

    # Parse market cap string to numeric (e.g., "1.5B" → 1500000000)
    if "market_cap_raw" in df.columns:
        df["market_cap"] = df["market_cap_raw"].apply(_parse_market_cap)
        df = df.drop(columns=["market_cap_raw"])
    else:
        df["market_cap"] = np.nan

    # Ensure numeric
    for col in ["price", "avg_volume", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without ticker
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].str.strip().str.upper()

    return df.reset_index(drop=True)


def _parse_market_cap(val) -> float:
    """Parse Finviz market cap strings like '1.5B', '200M', '50K'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * mult
            except ValueError:
                return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Signal enrichment (yfinance batch fetch + technical signals)
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 60  # Fetch 60 calendar days to get ~40 trading days of history
BATCH_SIZE = 20     # yfinance batch download size


def enrich_with_signals(universe: pd.DataFrame) -> pd.DataFrame:
    """Fetch 30-day price/volume history and calculate technical signals."""
    tickers = universe["ticker"].tolist()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)

    # Also fetch IWM for relative strength calculation
    all_tickers = tickers + ["IWM"]

    print(f"  Fetching price data for {len(tickers)} stocks...")
    price_data = _batch_download(all_tickers, start_dt, end_dt)

    # Calculate IWM benchmark return
    iwm_ret_20d = np.nan
    if "IWM" in price_data and len(price_data["IWM"]) >= 20:
        iwm_close = price_data["IWM"]["Close"]
        if len(iwm_close) >= 20:
            iwm_ret_20d = (iwm_close.iloc[-1] / iwm_close.iloc[-20] - 1) * 100

    # Calculate signals per ticker
    records = []
    for _, row in universe.iterrows():
        tk = row["ticker"]
        hist = price_data.get(tk)
        record = _calculate_signals(tk, hist, iwm_ret_20d)
        records.append(record)

    signals_df = pd.DataFrame(records)
    result = universe.merge(signals_df, on="ticker", how="left")
    return result


def _batch_download(tickers: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Download price data for multiple tickers in batches."""
    result = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_str = " ".join(batch)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    df = yf.download(
                        batch_str,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=False,
                        threads=True,
                    )
            if df is not None and not df.empty:
                # yfinance returns MultiIndex columns for multi-ticker downloads
                if isinstance(df.columns, pd.MultiIndex):
                    for tk in batch:
                        try:
                            tk_df = df.xs(tk, axis=1, level=1) if len(batch) > 1 else df
                            if not tk_df.empty and tk_df["Close"].notna().sum() > 0:
                                result[tk] = tk_df.dropna(subset=["Close"])
                        except (KeyError, TypeError):
                            pass
                else:
                    # Single ticker download — columns are not MultiIndex
                    if len(batch) == 1 and df["Close"].notna().sum() > 0:
                        result[batch[0]] = df.dropna(subset=["Close"])
        except Exception:
            pass

        # Brief status
        done = min(i + BATCH_SIZE, len(tickers))
        if done % 100 == 0 or done == len(tickers):
            print(f"    ... {done}/{len(tickers)} tickers fetched")

    return result


def _calculate_signals(ticker: str, hist: pd.DataFrame | None, iwm_ret_20d: float) -> dict:
    """Calculate technical signals for a single ticker."""
    base = {"ticker": ticker}

    if hist is None or len(hist) < 15:
        base["data_confidence"] = "LOW"
        return base

    close = hist["Close"]
    volume = hist["Volume"]
    n = len(close)

    # Data confidence
    last_date = close.index[-1]
    days_stale = (pd.Timestamp.now() - last_date).days
    vol_zeros = (volume.tail(5) == 0).sum()
    if n >= 20 and days_stale <= 3 and vol_zeros == 0:
        base["data_confidence"] = "HIGH"
    elif n >= 15 and days_stale <= 5:
        base["data_confidence"] = "MEDIUM"
    else:
        base["data_confidence"] = "LOW"

    # Momentum
    if n >= 20:
        base["momentum_20d"] = round((close.iloc[-1] / close.iloc[-20] - 1) * 100, 2)
    if n >= 5:
        base["momentum_5d"] = round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2)

    # Volume ratio (latest volume vs 20-day average)
    if n >= 20 and volume.tail(20).mean() > 0:
        base["volume_ratio"] = round(volume.iloc[-1] / volume.tail(20).mean(), 2)
    elif volume.mean() > 0:
        base["volume_ratio"] = round(volume.iloc[-1] / volume.mean(), 2)

    # Relative strength vs IWM
    if "momentum_20d" in base and not np.isnan(iwm_ret_20d):
        base["rs_vs_iwm"] = round(base["momentum_20d"] - iwm_ret_20d, 2)

    # Bollinger Band width (20-day, 2 std dev)
    if n >= 20:
        sma20 = close.tail(20).mean()
        std20 = close.tail(20).std()
        if sma20 > 0:
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            base["bb_width"] = round((upper - lower) / sma20, 4)

    # SMA checks
    if n >= 20:
        base["above_sma20"] = bool(close.iloc[-1] > close.tail(20).mean())
    if n >= 50:
        base["above_sma50"] = bool(close.iloc[-1] > close.tail(50).mean())

    # Latest price (may differ from Finviz due to timing)
    base["latest_price"] = round(float(close.iloc[-1]), 2)

    # Average daily dollar volume (20-day) for liquidity filter
    if n >= 20:
        dollar_vol = (close.tail(20) * volume.tail(20)).mean()
        base["avg_dollar_volume"] = round(float(dollar_vol), 0)

    return base


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------

def score_and_rank(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Composite scoring: 40% momentum + 30% volume breakout + 30% volatility squeeze."""
    # Filter out LOW confidence
    scored = df[df.get("data_confidence", pd.Series(dtype=str)) != "LOW"].copy()

    if len(scored) == 0:
        print("  WARNING: No stocks with sufficient data quality.", file=sys.stderr)
        return scored

    # Apply $500K average daily dollar volume floor (portfolio_rules.md liquidity filter)
    if "avg_dollar_volume" in scored.columns:
        before = len(scored)
        scored = scored[scored["avg_dollar_volume"] >= 500_000]
        dropped = before - len(scored)
        if dropped > 0:
            print(f"  Filtered {dropped} stocks below $500K avg daily dollar volume")

    # Require minimum signals
    required = ["momentum_20d", "volume_ratio", "bb_width"]
    scored = scored.dropna(subset=required)

    if len(scored) == 0:
        print("  WARNING: No stocks with complete signal data.", file=sys.stderr)
        return scored

    # Rank each factor (higher = better)
    scored["mom_rank"] = scored["momentum_20d"].rank(pct=True)
    scored["vol_rank"] = scored["volume_ratio"].rank(pct=True)
    # BB squeeze: LOWER width = TIGHTER = better setup (rank ascending, invert)
    scored["bb_rank"] = (1 - scored["bb_width"].rank(pct=True))

    # Composite: 40% momentum, 30% volume breakout, 30% volatility squeeze
    scored["composite_score"] = (
        0.40 * scored["mom_rank"]
        + 0.30 * scored["vol_rank"]
        + 0.30 * scored["bb_rank"]
    ).round(4)

    # Sort and take top N
    scored = scored.sort_values("composite_score", ascending=False).head(top_n)
    scored["rank"] = range(1, len(scored) + 1)

    return scored.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_watchlist(df: pd.DataFrame, data_dir: Path) -> str:
    """Write watchlist CSV and return formatted table for stdout."""
    csv_path = data_dir / "watchlist.csv"

    # Columns for output
    out_cols = [
        "rank", "ticker", "sector", "latest_price", "market_cap",
        "momentum_20d", "momentum_5d", "volume_ratio", "rs_vs_iwm",
        "bb_width", "above_sma20", "above_sma50", "data_confidence",
        "composite_score",
    ]
    available = [c for c in out_cols if c in df.columns]
    out = df[available].copy()

    # Format market cap for display
    if "market_cap" in out.columns:
        out["market_cap_display"] = out["market_cap"].apply(_fmt_market_cap)
    else:
        out["market_cap_display"] = "N/A"

    # Write full CSV
    out.to_csv(csv_path, index=False)
    print(f"\n  Watchlist saved to {csv_path}")

    # Build display table
    lines = []
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Candidates: {len(out)}")
    lines.append("")
    header = f"  {'Rk':>3} {'Ticker':<7} {'Sector':<16} {'Price':>8} {'Mkt Cap':>9} {'Mom20d':>7} {'Mom5d':>7} {'VolRat':>7} {'RS/IWM':>7} {'BBW':>7} {'SMA20':>5} {'SMA50':>5} {'Conf':>6} {'Score':>6}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for _, r in out.iterrows():
        sma20 = "Y" if r.get("above_sma20") else "N"
        sma50 = "Y" if r.get("above_sma50") else ("N" if "above_sma50" in r and pd.notna(r.get("above_sma50")) else "-")
        lines.append(
            f"  {int(r.get('rank', 0)):>3} "
            f"{r.get('ticker', '')::<7} "
            f"{str(r.get('sector', 'N/A'))[:15]:<16} "
            f"${r.get('latest_price', 0):>7.2f} "
            f"{r.get('market_cap_display', 'N/A'):>9} "
            f"{r.get('momentum_20d', 0):>+6.1f}% "
            f"{r.get('momentum_5d', 0):>+6.1f}% "
            f"{r.get('volume_ratio', 0):>6.1f}x "
            f"{r.get('rs_vs_iwm', 0):>+6.1f}% "
            f"{r.get('bb_width', 0):>6.3f} "
            f"{'  ' + sma20:>5} "
            f"{'  ' + sma50:>5} "
            f"{str(r.get('data_confidence', ''))[:4]:>6} "
            f"{r.get('composite_score', 0):>6.3f}"
        )

    return "\n".join(lines)


def _fmt_market_cap(val) -> str:
    if pd.isna(val):
        return "N/A"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Micro/small-cap quantitative screener")
    parser.add_argument("--data-dir", default="Start Your Own", help="Data directory (default: 'Start Your Own')")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top candidates to output (default: 15)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("Micro-Cap Screener")
    print("=" * 40)

    # Step 1: Get universe
    print("\n[1/4] Fetching universe...")
    universe = get_universe(data_dir)

    # Step 2: Enrich with signals
    print("\n[2/4] Enriching with technical signals...")
    enriched = enrich_with_signals(universe)

    # Step 3: Score and rank
    print("\n[3/4] Scoring and ranking...")
    ranked = score_and_rank(enriched, top_n=args.top_n)

    if len(ranked) == 0:
        print("\n  No candidates passed all filters.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Format and output
    print("\n[4/4] Generating watchlist...")
    table = format_watchlist(ranked, data_dir)
    print(table)

    # Sector distribution summary
    if "sector" in ranked.columns:
        print("\n  Sector Distribution:")
        for sector, count in ranked["sector"].value_counts().items():
            print(f"    {sector}: {count}")

    print(f"\n  Done. {len(ranked)} candidates ready for weekend analysis.")


if __name__ == "__main__":
    main()
