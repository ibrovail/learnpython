"""Quantitative screener for the micro/small-cap universe.

Pipeline: Finviz universe (identity + fundamentals) -> yfinance price/volume
signals -> hard gates encoding portfolio_rules.md / entry-discipline.md ->
composite momentum + volume + volatility rank of the gate survivors only -> a
sector-capped watchlist CSV for the weekend workflow, plus the full gated
universe saved to screener_history/ for factor research.
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

from trading_script import last_completed_session

# ---------------------------------------------------------------------------
# Universe fetching (Finviz → cache fallback)
# ---------------------------------------------------------------------------

# Exclusions for security types we don't trade (portfolio_rules.md):
# industry-text keywords and ticker suffixes marking units/warrants/rights
_EXCLUDED_TYPES = ("EXCHANGE TRADED FUND", "ETF", "ETN", "CLOSED-END", "CLOSED END", "SHELL COMPAN", "SPAC", "ADR")
_EXCLUDED_TICKER_SUFFIX = r"(?:-U|-UN|-WS|-R|\.U|\.WS)$"

# Prohibited businesses (portfolio_rules.md -> Exclusions). Whole industries are
# excluded only where the industry is itself prohibited; tickers catch prohibited
# names whose industry label also covers legitimate businesses.
# 2026-09-14: CXW, a private prison operator, ranked and was recommended because
# nothing here filtered it -- and the long-standing defence exclusion was never
# enforced in code either.
_PROHIBITED_INDUSTRIES = ("AEROSPACE & DEFENSE",)
_PROHIBITED_TICKERS = frozenset({
    # prisons & immigration detention
    "CXW", "GEO",
    # firearms & ammunition
    "RGR", "SWBI", "POWW", "AOUT",
    # payday, pawn and high-cost subprime consumer lenders
    # OPFI added 2026-09-14: OppLoans installment credit for consumers "turned away
    # by mainstream options" -- caught by the manual Credit Services review, which is
    # why that industry is flagged rather than trusted to this list.
    "CURO", "ENVA", "OPRT", "WRLD", "RM", "EZPW", "FCFS", "ELVT", "OPFI",
})
# Industries that contain prohibited businesses alongside legitimate ones. Not
# auto-excluded -- any name from these reaching the watchlist is flagged so it is
# checked by hand before a recommendation (analysis-workflow.md PRV gate).
_REVIEW_INDUSTRIES = ("SECURITY & PROTECTION", "CREDIT SERVICES")

# Universe ceiling (portfolio_rules.md). Raised 2026-08-15 from $2B to $5B for the
# experiment's final stretch — the $2B cap plus the sector caps were structurally
# blocking the field (Week 48: 8 of 15 candidates excluded, none investable).
MAX_MARKET_CAP = 5e9

# A fresh Finviz universe smaller than this share of the cached one is treated as a
# truncated fetch: the cache is kept and used instead.
MIN_UNIVERSE_SHARE = 0.5


def get_universe(data_dir: Path) -> pd.DataFrame:
    """Pull filtered stock list from Finviz. Falls back to cached file."""
    cache_path = data_dir / "universe_cache.csv"

    try:
        df = _fetch_finviz_universe()
        if len(df) > 0:
            df = _repair_ticker_corruption(df)
            cached_rows = len(pd.read_csv(cache_path, usecols=[0])) if cache_path.exists() else 0
            if cached_rows and len(df) < MIN_UNIVERSE_SHARE * cached_rows:
                # 2026-09-14: a one-page fetch (20 stocks) was accepted and overwrote a
                # 1,572-stock cache. A universe that shrinks by half overnight is a
                # broken fetch, not a market event.
                print(f"  WARNING: Finviz returned {len(df)} stocks against {cached_rows} in the "
                      "cache — treating the fetch as truncated and keeping the cache.")
            else:
                df.to_csv(cache_path, index=False)
                print(f"  Universe: {len(df)} stocks from Finviz (cached to {cache_path.name})")
                return df
    except Exception as e:
        print(f"  Finviz fetch failed: {e}")

    # Fallback to cache
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        df = _repair_ticker_corruption(df)
        print(f"  Universe: {len(df)} stocks from cache ({cache_path.name})")
        return df

    print("  ERROR: No universe data available (Finviz down, no cache).", file=sys.stderr)
    sys.exit(1)


# Finviz custom-view columns: finvizfinance column index -> (returned header, our name).
# One pull supplies identity, the fundamentals the hard gates need, and fields saved
# with every run so point-in-time fundamentals accumulate for factor research
# (yfinance keeps no point-in-time fundamentals history).
_FINVIZ_COLUMNS = {
    1: ("Ticker", "ticker"),
    2: ("Company", "company"),
    3: ("Sector", "sector"),
    4: ("Industry", "industry"),
    6: ("Market Cap", "market_cap_raw"),
    7: ("P/E", "pe"),
    8: ("Forward P/E", "fwd_pe"),
    22: ("EPS Q/Q", "eps_qq"),
    23: ("Sales Q/Q", "sales_qq"),
    27: ("Insider Trans", "insider_trans"),
    29: ("Inst Trans", "inst_trans"),
    30: ("Short Float", "short_float"),
    44: ("Perf Quart", "perf_quarter"),
    45: ("Perf Half", "perf_half"),
    48: ("Beta", "beta"),
    57: ("52W High", "pct_from_52w_high"),
    62: ("Recom", "recom"),
    63: ("Avg Volume", "avg_volume"),
    64: ("Rel Volume", "rel_volume"),
    65: ("Price", "price"),
    68: ("Earnings", "earnings"),       # "Aug 12/b" -- no year; /b before open, /a after close
    69: ("Target Price", "target_price"),
}
# Reported in percent units (Sales Q/Q +12.3 means +12.3% year over year)
_FINVIZ_PCT_COLUMNS = ("eps_qq", "sales_qq", "insider_trans", "inst_trans", "short_float",
                       "perf_quarter", "perf_half", "pct_from_52w_high")
_FINVIZ_NUM_COLUMNS = ("pe", "fwd_pe", "beta", "recom", "avg_volume", "rel_volume", "price",
                       "target_price")


def _finviz_number(s: pd.Series, percent: bool = False) -> pd.Series:
    """Finviz cells -> floats.

    finvizfinance converts most "12.3%" cells to fractions (0.123) but leaves some
    columns as strings ("6.95%"), and uses "-" for missing. With percent=True the
    result is in percent units either way.
    """
    if s.dtype == object:
        txt = s.astype(str).str.strip()
        had_pct = txt.str.endswith("%")
        num = pd.to_numeric(txt.str.rstrip("%"), errors="coerce")
        return num.where(had_pct, num * 100) if percent else num
    num = pd.to_numeric(s, errors="coerce")
    return num * 100 if percent else num


def _fetch_finviz_universe() -> pd.DataFrame:
    """Use finvizfinance to pull the screened universe with fundamentals."""
    from finvizfinance.screener.custom import Custom

    fcustom = Custom()
    filters_dict = {
        # Fetch mid-and-under, then trim to MAX_MARKET_CAP in _validate_enriched.
        # Finviz has no $5bln bucket, so the wider bucket is pulled and filtered.
        "Market Cap.": "-Mid (under $10bln)",
        "Average Volume": "Over 500K",
        "Price": "Over $1",
        # U.S.-domiciled only: excludes ADRs/foreign issuers at the source
        # (portfolio_rules.md exclusions)
        "Country": "USA",
    }
    fcustom.set_filter(filters_dict=filters_dict)
    # limit must be explicit: Custom.screener_view defaults to limit=-1, which the
    # base pager reads as already exhausted and stops after page 1 (20 rows).
    df = fcustom.screener_view(columns=list(_FINVIZ_COLUMNS), verbose=0, limit=100_000)

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Standardize columns. A header Finviz renames silently disables the gate that
    # reads it, so say so rather than carry on.
    col_map = dict(_FINVIZ_COLUMNS.values())
    missing = [h for h in col_map if h not in df.columns]
    if missing:
        print(f"  WARNING: Finviz did not return columns: {', '.join(missing)}")
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df[[c for c in col_map.values() if c in df.columns]].copy()

    for col in _FINVIZ_PCT_COLUMNS:
        if col in df.columns:
            df[col] = _finviz_number(df[col], percent=True)
    for col in _FINVIZ_NUM_COLUMNS:
        if col in df.columns:
            df[col] = _finviz_number(df[col])

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


def _repair_ticker_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and repair systematic first-character duplication in tickers.

    2026-07-19 incident: a Finviz page change made every ticker scrape with its
    first letter doubled (AAC-U -> AAAC-U, TE -> TTE, SG -> SSG), so yfinance
    signals were later fetched for entirely different securities (TTE =
    TotalEnergies) while sector/cap stayed with the real small-cap. The
    corruption is uniform when present: 100% of rows had a doubled first
    character vs a ~3% natural base rate. If the doubled-first-char share is
    far above natural, strip the duplicated character across the board.
    """
    t = df["ticker"].astype(str)
    eligible = t.str.len() >= 2
    if not eligible.any():
        return df
    dup_share = (t[eligible].str[0] == t[eligible].str[1]).mean()
    if dup_share > 0.60:
        print(
            f"  WARNING: {dup_share:.0%} of tickers have a doubled first character "
            "(natural rate ~3%) — repairing Finviz parse corruption by stripping it."
        )
        df = df.copy()
        df.loc[eligible, "ticker"] = t[eligible].str[1:]
        df = df.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return df


def _validate_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check enriched rows so corrupted data can't reach the watchlist.

    Guards (portfolio_rules.md universe + the 2026-07-19 corruption incident):
    - market cap must be within the $5B universe ceiling
    - Finviz price must be >= $1
    - excluded security types (ETF/ETN/SPAC/ADR keywords in sector/industry)
    - ticker identity: the yfinance-derived latest_price must be within 40% of
      the Finviz price — a larger gap means the two sources are describing
      different securities (the signature of ticker corruption).
    """
    before = len(df)

    if "market_cap" in df.columns:
        df = df[df["market_cap"].isna() | (df["market_cap"] <= MAX_MARKET_CAP)]
    if "price" in df.columns:
        df = df[df["price"].isna() | (df["price"] >= 1.0)]

    for col in ("sector", "industry"):
        if col in df.columns:
            text = df[col].astype(str).str.upper()
            df = df[~text.str.contains("|".join(_EXCLUDED_TYPES), na=False)]
    df = df[~df["ticker"].astype(str).str.upper().str.contains(_EXCLUDED_TICKER_SUFFIX, regex=True, na=False)]

    # Prohibited businesses: blocklisted tickers and wholly-prohibited industries.
    _tick = df["ticker"].astype(str).str.upper()
    _prohibited = _tick.isin(_PROHIBITED_TICKERS)
    if "industry" in df.columns:
        _ind = df["industry"].astype(str).str.upper()
        for _name in _PROHIBITED_INDUSTRIES:
            _prohibited |= _ind.str.contains(_name, regex=False, na=False)
    if _prohibited.any():
        print(f"  Prohibited businesses: removed {int(_prohibited.sum())} "
              f"(prisons/detention, defence & firearms, predatory lending)")
    df = df[~_prohibited]

    if {"price", "latest_price"}.issubset(df.columns):
        both = df["price"].notna() & df["latest_price"].notna() & (df["price"] > 0)
        mismatch = both & ((df["latest_price"] - df["price"]).abs() / df["price"] > 0.40)
        n_mismatch = int(mismatch.sum())
        if n_mismatch:
            print(
                f"  WARNING: dropped {n_mismatch} rows where the yfinance price "
                "diverges >40% from the Finviz price (ticker identity mismatch)."
            )
            if both.any() and n_mismatch / int(both.sum()) > 0.30:
                print(
                    "  WARNING: >30% of rows failed the identity check — the "
                    "universe fetch is likely corrupted; treat this watchlist "
                    "with suspicion.",
                    file=sys.stderr,
                )
            df = df[~mismatch]

    dropped = before - len(df)
    if dropped > 0:
        print(f"  Validation: dropped {dropped} rows failing sanity checks")
    return df


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

# 110 calendar days ~ 75 sessions. 60 gave ~40, so the 50-day SMA check never
# computed (above_sma50 was absent from every watchlist until 2026-09-14) and the
# distance-from-50-day rule could not be enforced here.
LOOKBACK_DAYS = 110
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
        record = _calculate_signals(tk, hist, iwm_ret_20d, row.get("earnings"))
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

    # Drop any bar for a session that has not closed yet. Running during market
    # hours otherwise feeds a PARTIAL day into every signal: `volume.iloc[-1]`
    # becomes a few minutes of trading against a 20-day average, collapsing
    # volume_ratio (2026-08-31: every candidate scored 0.10-0.38x, versus
    # 1.2-8.3x on the same screen run pre-open), and momentum/SMA/BBW all read
    # off an intraday price rather than a close.
    cutoff = last_completed_session()
    for tk, hist in list(result.items()):
        trimmed = hist[hist.index.tz_localize(None) <= cutoff] if hist.index.tz is not None \
            else hist[hist.index <= cutoff]
        if trimmed.empty:
            result.pop(tk)
        else:
            result[tk] = trimmed

    return result


def _parse_finviz_earnings(value: object, asof: pd.Timestamp) -> tuple[pd.Timestamp, str] | None:
    """'Aug 12/b' -> (Timestamp('YYYY-08-12'), 'b').

    Finviz omits the year: take the reading nearest `asof`, so a date more than
    six months ahead belongs to last year and one more than six months back to next.
    """
    if not isinstance(value, str) or "/" not in value:
        return None
    day_txt, _, timing = value.strip().partition("/")
    try:
        d = pd.Timestamp(datetime.strptime(f"{day_txt.strip()} {asof.year}", "%b %d %Y"))
    except ValueError:
        return None
    if d - asof > pd.Timedelta(days=183):
        d = d.replace(year=asof.year - 1)
    elif asof - d > pd.Timedelta(days=183):
        d = d.replace(year=asof.year + 1)
    return d, timing.strip().lower()[:1]


def _calculate_signals(ticker: str, hist: pd.DataFrame | None, iwm_ret_20d: float,
                       earnings: object = None) -> dict:
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

    # SMA checks + distance from base (entry-discipline.md: <=20% above the 20-day,
    # <=40% above the 50-day)
    last = float(close.iloc[-1])
    if n >= 20:
        sma20_v = float(close.tail(20).mean())
        base["above_sma20"] = bool(last > sma20_v)
        base["pct_vs_sma20"] = round((last / sma20_v - 1) * 100, 2)
    if n >= 50:
        sma50_v = float(close.tail(50).mean())
        base["above_sma50"] = bool(last > sma50_v)
        base["pct_vs_sma50"] = round((last / sma50_v - 1) * 100, 2)

    # ATR(14) as % of price (simple mean of true range) and the last session's
    # range -- inputs to the deal-pinned screen
    if {"High", "Low"}.issubset(hist.columns) and n >= 15 and last > 0:
        high, low = hist["High"].astype(float), hist["Low"].astype(float)
        prev_close = close.astype(float).shift(1)
        true_range = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        base["atr_pct"] = round(float(true_range.iloc[-14:].mean()) / last * 100, 3)
        base["last_range_pct"] = round(float(high.iloc[-1] - low.iloc[-1]) / last * 100, 3)

    # Composite signals adopted from the Phase 2 factor study
    # ("Experiment Details/Screener Factor Study — Phase 2.md"). Each is stored so a HIGHER
    # value ranks better, which is what score_and_rank assumes.
    if n >= 21:
        base["low_vol"] = round(-float(close.pct_change().tail(20).std()) * 100, 4)
    if n >= 60:
        sixty_high = float(hist["High"].tail(60).max())
        if sixty_high > 0:
            base["near_high"] = round((last / sixty_high - 1) * 100, 2)
    if n >= 50 and float(volume.tail(50).mean()) > 0:
        base["vol_5_50"] = round(float(volume.tail(5).mean() / volume.tail(50).mean()), 3)

    # Breakout age (entry-discipline.md: avoid days 1-3 of a new 20-day breakout when
    # the move is >+10%). A breakout session closes above the prior 20 closes; the
    # breakout starts at the first session of the latest unbroken run of them, and
    # the move is measured from the close before it.
    if n >= 21:
        c = close.astype(float).to_numpy()
        is_breakout = np.zeros(n, dtype=bool)
        for i in range(20, n):
            is_breakout[i] = c[i] > c[i - 20:i].max()
        if is_breakout.any():
            start = int(np.flatnonzero(is_breakout)[-1])
            while start > 20 and is_breakout[start - 1]:
                start -= 1
            base["breakout_session"] = n - start          # 1 = the last close was day 1
            base["breakout_move_pct"] = round((c[-1] / c[start - 1] - 1) * 100, 2)

    # Post-earnings jump (entry-discipline.md cooldown: no buy within 3 sessions of a
    # print at >5% above the reference close). The reaction session is the print day
    # for a before-open report and the next session for an after-close one. The
    # reference is the lower of the pre-print and post-print closes: the rule text
    # names the post-print close, but its origin case (ARLO, bought below the
    # post-print close yet +12% above the pre-print one) only fails against the
    # pre-print close -- the lower of the two catches both readings.
    idx = pd.DatetimeIndex(close.index)
    idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    parsed = _parse_finviz_earnings(earnings, idx[-1])
    if parsed is not None:
        e_date, timing = parsed
        reacted = (idx > e_date) if timing == "a" else (idx >= e_date)
        if reacted.any():
            r = int(np.argmax(reacted))
            if r >= 1:
                base["earnings_sessions_ago"] = n - r     # 1 = the last close was the reaction
                ref = min(float(close.iloc[r - 1]), float(close.iloc[r]))
                base["post_earnings_move_pct"] = round((last / ref - 1) * 100, 2)

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

# Composite inputs, adopted 2026-09-15 from the Phase 2 factor study. The six signals that
# passed the pre-registered test, equal-weighted on percentile ranks. A stock missing any of
# them cannot be ranked and fails the "incomplete signals" gate.
COMPOSITE_INPUTS = ("low_vol", "near_high", "bb_width", "vol_5_50", "volume_ratio", "pct_vs_sma50")

# Gate thresholds mirror .claude/rules/entry-discipline.md -- change them there first.
MIN_DOLLAR_VOLUME = 500_000
MAX_PCT_ABOVE_SMA50 = 40.0
MAX_PCT_ABOVE_SMA20 = 20.0
BREAKOUT_MAX_SESSIONS, BREAKOUT_MAX_MOVE_PCT = 3, 10.0
EARNINGS_COOLDOWN_SESSIONS, EARNINGS_MAX_MOVE_PCT = 3, 5.0
# Deal-pinned = ATR alone. On the 2026-09-14 screen every stock under 0.8% ATR had a
# pinned profile (the highest was 0.45%) while the universe's 2nd-percentile ATR was
# 1.45%. The first version also required |20d momentum| <= 1% and price above target
# or a tiny range; it let two confirmed all-cash takeover targets through and ranked
# them #1 and #2 -- DV (Nielsen at $13.60, momentum +1.65%) and PAYO (Nuvei at $7.40,
# target 3.6% above the price).
PINNED_MAX_ATR_PCT = 0.75


def apply_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every hard gate BEFORE ranking; failures are listed in `gate_fail`.

    2026-09-14: the composite ranked the whole universe and the rules were applied
    by hand to the top 15 afterwards -- only 5 of the 15 survived, 8 of them had
    moved less than 1% in 20 sessions, and the strongest momentum names sat at
    #37-#48, below the cutoff. A gate whose input is missing passes the name:
    missing data is not evidence of failure, and the PRV gate re-checks every
    candidate on the quote page.
    """
    df = df.copy()

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    checks = {
        "low data confidence": col("data_confidence").eq("LOW"),
        "incomplete signals": pd.concat([col(c) for c in COMPOSITE_INPUTS], axis=1).isna().any(axis=1),
        "illiquid (<$500K/day)": col("avg_dollar_volume") < MIN_DOLLAR_VOLUME,
        "deal-pinned": col("atr_pct") < PINNED_MAX_ATR_PCT,
        ">40% above 50-day SMA": col("pct_vs_sma50") > MAX_PCT_ABOVE_SMA50,
        ">20% above 20-day SMA": col("pct_vs_sma20") > MAX_PCT_ABOVE_SMA20,
        "fresh >10% breakout": (col("breakout_session") <= BREAKOUT_MAX_SESSIONS)
                               & (col("breakout_move_pct") > BREAKOUT_MAX_MOVE_PCT),
        "post-earnings jump": (col("earnings_sessions_ago") <= EARNINGS_COOLDOWN_SESSIONS)
                              & (col("post_earnings_move_pct") > EARNINGS_MAX_MOVE_PCT),
        "shrinking revenue": col("sales_qq") < 0,
    }
    hits = pd.DataFrame({reason: mask.fillna(False).astype(bool) for reason, mask in checks.items()})
    df["gate_fail"] = hits.apply(lambda row: "; ".join(r for r, hit in row.items() if hit), axis=1)

    # Not a gate: industries mixing prohibited and permitted businesses are flagged
    # for a by-hand read of what the company does (analysis-workflow.md PRV gate).
    industry = col("industry").astype(str).str.upper()
    df["review_flag"] = np.where(
        industry.apply(lambda s: any(k in s for k in _REVIEW_INDUSTRIES)), "REVIEW", "")
    df["target_upside"] = ((col("target_price") / col("latest_price") - 1) * 100).round(1)

    print(f"  {len(df)} stocks evaluated (a stock can fail several gates):")
    for reason in hits.columns:
        print(f"    {reason:<24} {int(hits[reason].sum()):>5}")
    print(f"  Survivors: {int((df['gate_fail'] == '').sum())}")
    return df


def score_and_rank(df: pd.DataFrame, top_n: int = 50,
                   max_per_sector: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank gate survivors: 40% momentum + 30% volume breakout + 30% volatility squeeze.

    Returns (watchlist, scored): the top_n survivors with at most max_per_sector
    from any one sector, and every evaluated row with its composite (NaN where a
    gate failed) for the history file. Weights are unchanged pending the factor
    study.
    """
    scored = df.copy()
    survivors = scored["gate_fail"] == ""
    if not survivors.any():
        print("  WARNING: No stocks passed the hard gates.", file=sys.stderr)
        return scored.iloc[0:0], scored

    # Percentile ranks among survivors only, so gated names cannot shift them
    s = scored.loc[survivors]
    ranks = {
        "low_vol": s["low_vol"].rank(pct=True),              # calmer ranks higher
        "near_high": s["near_high"].rank(pct=True),           # closer to the 60-day high
        # BB squeeze: LOWER width = TIGHTER = better setup (rank ascending, invert)
        "squeeze": 1 - s["bb_width"].rank(pct=True),
        "vol_5_50": s["vol_5_50"].rank(pct=True),
        "vol_ratio": s["volume_ratio"].rank(pct=True),
        "vs_sma50": s["pct_vs_sma50"].rank(pct=True),
    }
    for name, r in ranks.items():
        scored.loc[survivors, f"rank_{name}"] = r
    scored.loc[survivors, "mom_rank"] = s["momentum_20d"].rank(pct=True)

    # Composite: the six signals that passed the Phase 2 pre-registered test, equal-weighted.
    # 20-day momentum is no longer scored -- it showed no ranking skill over 5/10/20 sessions
    # (IC 0.031, t 1.24 at 10 sessions), and it is 0.77 correlated with vs_sma50, which is in.
    scored["composite_score"] = (sum(ranks.values()) / len(ranks)).round(4)

    # Shadow scores -- recorded for the Phase 4 out-of-sample comparison, never used to order
    # the watchlist. "legacy" is the composite this replaced; "dedup" drops the near-duplicates
    # (squeeze ~ low_vol 0.80, near_high ~ vs_sma50 0.75) and scored best in-sample, but its
    # signal set was chosen after seeing the data, so it has to prove itself on fresh screens.
    scored["composite_legacy"] = (0.40 * scored["mom_rank"] + 0.30 * ranks["vol_ratio"]
                                  + 0.30 * ranks["squeeze"]).round(4)
    scored["composite_dedup"] = ((ranks["low_vol"] + ranks["vol_5_50"]
                                  + ranks["vol_ratio"] + ranks["vs_sma50"]) / 4).round(4)

    # Sector cap on the list so one hot sector cannot crowd it out. The book holds at
    # most 2 per sector (3 healthcare), but 11 sectors x 3 = 33 cannot fill 50 slots,
    # so the list cap is looser than the holdings cap.
    ranked = scored.loc[survivors].sort_values("composite_score", ascending=False)
    sector = ranked["sector"].fillna("Unknown") if "sector" in ranked.columns \
        else pd.Series("Unknown", index=ranked.index)
    ranked = ranked[sector.groupby(sector).cumcount() < max_per_sector].head(top_n).copy()
    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked.reset_index(drop=True), scored


def _save_history(scored: pd.DataFrame, data_dir: Path) -> None:
    """Save every evaluated stock -- signals, fundamentals, gate result, score.

    Only the top 15 used to survive a run (watchlist.csv, overwritten weekly), so the
    screener's ranking skill could not be measured. Named for the last completed
    session the signals were computed on. The Finviz fundamentals matter most:
    yfinance has no point-in-time fundamentals, so these files are the only record
    of what the numbers were that day.
    """
    hist_dir = data_dir / "screener_history"
    hist_dir.mkdir(exist_ok=True)
    path = hist_dir / f"screen_{pd.Timestamp(last_completed_session()).date().isoformat()}.csv"
    scored.sort_values("composite_score", ascending=False, na_position="last").to_csv(path, index=False)
    print(f"  Full gated universe saved to {path} ({len(scored)} rows)")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_watchlist(df: pd.DataFrame, data_dir: Path) -> str:
    """Write watchlist CSV and return formatted table for stdout."""
    csv_path = data_dir / "watchlist.csv"

    # Columns for output
    out_cols = [
        "rank", "ticker", "company", "sector", "industry", "latest_price", "market_cap",
        "momentum_20d", "momentum_5d", "volume_ratio", "vol_5_50", "rs_vs_iwm", "bb_width",
        "low_vol", "near_high", "pct_vs_sma20", "pct_vs_sma50", "atr_pct", "sales_qq",
        "eps_qq", "fwd_pe", "recom", "target_upside", "beta", "earnings", "short_float",
        "review_flag", "data_confidence", "composite_score", "composite_legacy", "composite_dedup",
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
    def _cell(v: object, fmt: str, width: int) -> str:
        return "n/a".rjust(width) if v is None or pd.isna(v) else format(v, fmt).rjust(width)

    header = (f"  {'Rk':>3} {'Ticker':<7} {'Sector':<16} {'Price':>8} {'Mkt Cap':>8} {'Mom20d':>7} "
              f"{'VolRat':>6} {'V5/50':>6} {'Near%':>6} {'vs50d':>7} {'ATR%':>5} {'SalesQ':>7} "
              f"{'Upside':>7} {'Score':>6}  Flag")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for _, r in out.iterrows():
        lines.append(
            f"  {int(r.get('rank', 0)):>3} "
            f"{str(r.get('ticker', '')):<7} "
            f"{str(r.get('sector', 'N/A'))[:15]:<16} "
            f"{_cell(r.get('latest_price'), '.2f', 8)} "
            f"{str(r.get('market_cap_display', 'N/A')):>8} "
            f"{_cell(r.get('momentum_20d'), '+.1f', 6)}% "
            f"{_cell(r.get('volume_ratio'), '.1f', 5)}x "
            f"{_cell(r.get('vol_5_50'), '.1f', 5)}x "
            f"{_cell(r.get('near_high'), '+.1f', 5)}% "
            f"{_cell(r.get('pct_vs_sma50'), '+.1f', 6)}% "
            f"{_cell(r.get('atr_pct'), '.1f', 5)} "
            f"{_cell(r.get('sales_qq'), '+.1f', 6)}% "
            f"{_cell(r.get('target_upside'), '+.0f', 6)}% "
            f"{_cell(r.get('composite_score'), '.3f', 6)}  "
            f"{r.get('review_flag', '') or ''}"
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
    parser.add_argument("--top-n", type=int, default=50, help="Number of top candidates to output (default: 50)")
    parser.add_argument("--max-per-sector", type=int, default=6,
                        help="Most candidates from any one sector in the watchlist (default: 6)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("Micro-Cap Screener")
    print("=" * 40)

    # Step 1: Get universe
    print("\n[1/5] Fetching universe...")
    universe = get_universe(data_dir)

    # Step 2: Enrich with signals
    print("\n[2/5] Enriching with technical signals...")
    enriched = enrich_with_signals(universe)
    enriched = _validate_enriched(enriched)

    # Step 3: Hard gates -- rules first, ranking second
    print("\n[3/5] Applying hard gates...")
    gated = apply_gates(enriched)

    # Step 4: Score and rank the survivors; keep the full universe for research
    print("\n[4/5] Scoring and ranking survivors...")
    ranked, scored = score_and_rank(gated, top_n=args.top_n, max_per_sector=args.max_per_sector)
    _save_history(scored, data_dir)

    if len(ranked) == 0:
        print("\n  No candidates passed all filters.", file=sys.stderr)
        sys.exit(1)

    # Step 5: Format and output
    print("\n[5/5] Generating watchlist...")
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
