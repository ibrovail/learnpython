"""Stage 1b — assemble the price panel through the ticker map, then report coverage.

Order matters and is enforced here: the ticker reuse/rename map is applied BEFORE any
price is keyed, and the coverage report is produced BEFORE any return is computed. The
pre-registration makes coverage a gate, not a diagnostic -- below 90% of expected
member-years the study is reported inconclusive rather than published with a number.

Three things are done to the raw per-ticker CSVs, in this order:

  1. RENAME. A membership ticker that is an old name for a current company reads the
     current company's series (FB -> META). An unmapped rename finds no file and is
     counted MISSING, which costs coverage rather than correctness.

  2. REUSE MASK. A ticker that two unrelated companies have held is blanked outside the
     era belonging to the member company, and blanked for the whole year when the two
     eras meet inside it (FOXA, 2019). Without this a return is manufactured out of a
     discontinuity, and it looks like a number rather than an error.

  3. WHAT IS MISSING IS NAMED. A symbol absent because the source has no history for it
     and a symbol absent because the API quota refused it are different facts, and the
     coverage report keeps them apart. Collapsing them would let a quota wall read as a
     property of the market.

    python3 -I research/build_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ticker_map as tm  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data.nosync"
PRICES = CACHE / "prices_tiingo"
OUT = ROOT / "output"

FIRST_FORMATION, LAST_FORMATION = 2011, 2024


def load_membership() -> dict[int, set[str]]:
    raw = json.loads((CACHE / "membership.json").read_text())
    return {int(y): set(v) for y, v in raw.items()}


def _read(series_key: str) -> pd.DataFrame | None:
    path = PRICES / f"{series_key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def build(membership: dict[int, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Wide adjusted-close and intraday-low panels, keyed by MEMBERSHIP ticker."""
    wanted: set[str] = set()
    for y in range(FIRST_FORMATION, LAST_FORMATION + 1):
        wanted |= membership.get(y, set())

    close_cols: dict[str, pd.Series] = {}
    low_cols: dict[str, pd.Series] = {}
    resolved_to: dict[str, list[str]] = {}
    absent: list[str] = []

    for ticker in sorted(wanted):
        key = tm.resolve(ticker)
        df = _read(key)
        if df is None or df.empty:
            absent.append(ticker)
            continue
        resolved_to.setdefault(key, []).append(ticker)
        close_cols[ticker] = df["adjclose"].astype(float)
        low_cols[ticker] = df["low"].astype(float) if "low" in df else df["adjclose"].astype(float)

    close = pd.DataFrame(close_cols).sort_index()
    low = pd.DataFrame(low_cols).reindex(close.index)

    # --- reuse mask -------------------------------------------------------
    masked_years: list[tuple[str, int, str]] = []
    for ticker in [t for t in close.columns if tm.is_reused(t)]:
        for year in range(FIRST_FORMATION - 1, LAST_FORMATION + 2):
            rows = close.index.year == year
            if not rows.any():
                continue
            window = tm.valid_window(ticker, year)
            if window is None:
                close.loc[rows, ticker] = float("nan")
                low.loc[rows, ticker] = float("nan")
                masked_years.append((ticker, year, "no single company held the ticker"))
                continue
            lo, hi = window
            outside = rows & ((close.index < lo) | (close.index > hi))
            if outside.any():
                close.loc[outside, ticker] = float("nan")
                low.loc[outside, ticker] = float("nan")
                masked_years.append((ticker, year, f"trimmed to {lo.date()}..{hi.date()}"))

    # Two membership tickers resolving to one series is legitimate across years (CBS,
    # VIAC and PARA are one company) but never inside one year.
    collisions = []
    for year in range(FIRST_FORMATION, LAST_FORMATION + 1):
        seen: dict[str, str] = {}
        for ticker in membership.get(year, set()):
            key = tm.resolve(ticker)
            if key in seen and ticker in close.columns and seen[key] in close.columns:
                collisions.append((year, seen[key], ticker, key))
            seen.setdefault(key, ticker)

    diag = {
        "wanted": len(wanted),
        "with_prices": len(close.columns),
        "absent": absent,
        "masked_years": masked_years,
        "collisions": collisions,
    }
    return close, low, diag


def coverage(membership: dict[int, set[str]], close: pd.DataFrame,
             quota_blocked: set[str], no_history: set[str]) -> pd.DataFrame:
    """Member-years usable for RANKING, per formation year.

    A ticker counts as priced for formation year Y only if its series actually spans Y:
    a company that listed or was acquired mid-year has no full-year return, and scoring
    it on a stub is the error the thread audit records against Sandisk in 2025.
    """
    rows = []
    years = [y for y in range(FIRST_FORMATION, LAST_FORMATION + 1) if y in membership]
    for year in years:
        members = sorted(membership[year])
        sessions = close.index[close.index.year == year]
        if len(sessions) == 0:
            rows.append({"formation_year": year, "members": len(members), "priced": 0,
                         "full_year": 0, "coverage": 0.0, "missing_quota": len(members),
                         "missing_no_history": 0, "missing_other": 0})
            continue
        open_by, close_by = sessions[min(4, len(sessions) - 1)], sessions[-min(5, len(sessions))]
        priced = full = 0
        miss_q = miss_h = miss_o = 0
        for t in members:
            if t not in close.columns:
                key = tm.resolve(t)
                if key in quota_blocked or t in quota_blocked:
                    miss_q += 1
                elif key in no_history or t in no_history:
                    miss_h += 1
                else:
                    miss_o += 1
                continue
            col = close[t].reindex(sessions).dropna()
            if col.empty:
                miss_o += 1
                continue
            priced += 1
            if col.index[0] <= open_by and col.index[-1] >= close_by:
                full += 1
        rows.append({
            "formation_year": year,
            "members": len(members),
            "priced": priced,
            "full_year": full,
            "coverage": full / len(members) if members else float("nan"),
            "missing_quota": miss_q,
            "missing_no_history": miss_h,
            "missing_other": miss_o,
        })
    return pd.DataFrame(rows)


def main() -> int:
    membership = load_membership()
    quota_blocked = set((CACHE / "quota_blocked_tiingo.txt").read_text().split()) \
        if (CACHE / "quota_blocked_tiingo.txt").exists() else set()
    no_history = set((CACHE / "no_history_tiingo.txt").read_text().split()) \
        if (CACHE / "no_history_tiingo.txt").exists() else set()

    close, low, diag = build(membership)
    OUT.mkdir(exist_ok=True)
    if not close.empty:
        close.to_parquet(CACHE / "prices_close.parquet")
        low.to_parquet(CACHE / "prices_low.parquet")

    print(f"membership tickers wanted (2011-2024): {diag['wanted']}")
    print(f"with a price series:                   {diag['with_prices']}")
    print(f"absent:                                {len(diag['absent'])}")
    print(f"  of which quota-refused:              "
          f"{len([t for t in diag['absent'] if tm.resolve(t) in quota_blocked or t in quota_blocked])}")
    print(f"  of which no history at source:       "
          f"{len([t for t in diag['absent'] if tm.resolve(t) in no_history or t in no_history])}")
    print(f"reuse masks applied:                   {len(diag['masked_years'])}")
    for t, y, why in diag["masked_years"]:
        print(f"    {t} {y}: {why}")
    if diag["collisions"]:
        print(f"TICKER COLLISIONS INSIDE A YEAR: {diag['collisions']}")

    cov = coverage(membership, close, quota_blocked, no_history)
    cov.to_csv(OUT / "coverage.csv", index=False)
    pd.DataFrame(diag["masked_years"],
                 columns=["ticker", "year", "action"]).to_csv(
        OUT / "ticker_map_applied.csv", index=False)

    print("\nCoverage of expected member-years (the pre-registration's gate):")
    print(cov.to_string(index=False))
    overall = cov["full_year"].sum() / cov["members"].sum() if cov["members"].sum() else 0.0
    print(f"\nOVERALL COVERAGE: {overall:.1%} of expected member-years")
    print("GATE: the pre-registration reports the study INCONCLUSIVE below 90%.")
    print("VERDICT:", "PASS -- proceed to Stage 2" if overall >= 0.90
          else "FAIL -- Stage 2 must not publish a number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
