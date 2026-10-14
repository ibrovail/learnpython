"""Data acquisition for Study 01 — point-in-time S&P 500 membership and prices.

STATUS: not yet runnable. This session's environment denies network egress to the
required hosts, so nothing here has been executed against live data. The module
defines the interface the engine expects and fails loudly rather than returning
anything synthetic -- a mocked panel that silently reaches the engine would produce
a publishable-looking number with no data behind it.

What Stage 1 must produce, per the pre-registration:

  membership: dict[int, set[str]]
      Index members as of 31 December of each year, 1996..2024. Reconstructed by
      rolling a current constituent list BACKWARD through the published record of
      additions and removals. Rolling forward from an old list silently drops every
      company that was later acquired or delisted, which is the survivorship bias
      this study exists to avoid.

  prices: pd.DataFrame  (date index, ticker columns, adjusted close)
  lows:   pd.DataFrame  (same shape, intraday low; optional, needed for stops)
      MUST include delisted and acquired names for the years they were members.
      Coverage is reported before any return is computed; below 90% of expected
      member-years the study is reported inconclusive.

Known hazards to handle in Stage 1, each of which has silently broken this kind of
study before:

  * Ticker reuse. Symbols are recycled between unrelated companies over 30 years.
    Key on a permanent identifier where one is available, not on the ticker string.
  * Partial-year members. A company added mid-year has no full prior-year return and
    must be excluded from that formation year, not scored on a stub.
  * Spin-offs. A stock that debuts mid-year cannot have its partial return compared
    with full-year returns. The audited thread did exactly this with Sandisk in 2025.
  * Splits and total return. Use adjusted series; state per period whether dividends
    are included.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "data.nosync"
FIRST_FORMATION_YEAR = 1996   # formation for the 1997 holding year
LAST_FORMATION_YEAR = 2024    # formation for the 2025 holding year; 2026 is incomplete


class DataUnavailable(RuntimeError):
    """Raised when the inputs are not present. Never substituted with synthetic data."""


def load_membership() -> dict[int, set[str]]:
    path = CACHE / "membership.json"
    if not path.exists():
        raise DataUnavailable(
            f"No membership reconstruction at {path}. Stage 1 has not been run.\n"
            "It needs network access to the index-change record, which this "
            "environment currently denies."
        )
    import json

    raw = json.loads(path.read_text())
    return {int(year): set(tickers) for year, tickers in raw.items()}


def load_prices(kind: str = "close") -> pd.DataFrame:
    path = CACHE / f"prices_{kind}.parquet"
    if not path.exists():
        raise DataUnavailable(
            f"No price panel at {path}. Stage 1 has not been run.\n"
            "It must cover delisted and acquired names, not just current members."
        )
    return pd.read_parquet(path)


def coverage_report(membership: dict[int, set[str]], prices: pd.DataFrame) -> pd.DataFrame:
    """A LOOSE UPPER BOUND on coverage: member-years whose ticker has a column at all.

    NOT the gate. `build_panel.coverage` computes the gate number, and it is stricter in
    two ways that both matter:

      * it counts a member-year only if the series actually SPANS the formation year, so
        a name that listed or was acquired mid-year does not count as priced for a year
        it cannot be ranked on;
      * it attributes every absence -- quota refusal, no history at source, or
        unexplained -- because an API limit and a delisting are different facts and
        collapsing them lets a quota wall read as a property of the market.

    This function can therefore report a higher number than the gate does, and only the
    gate number may be quoted against the pre-registration's 90% bar. Kept because a
    quick "is there a column for this name" check is still useful while debugging a
    fetch; never cite it as coverage.
    """
    rows = []
    have = set(prices.columns)
    for year in sorted(membership):
        members = membership[year]
        present = len(members & have)
        rows.append(
            {
                "formation_year": year,
                "members": len(members),
                "priced": present,
                "coverage": present / len(members) if members else float("nan"),
            }
        )
    return pd.DataFrame(rows)
