"""Stage 1 — reconstruct point-in-time S&P 500 membership, and report its quality.

Method: take the current constituent list and roll it BACKWARD through the published
record of index changes, snapshotting at each 31 December. Rolling backward is the only
safe direction -- rolling an old list forward silently drops every company that was later
acquired or delisted, which is the survivorship bias this study exists to avoid.

Rolling backward is only as good as the change record. This script therefore reports,
before anything else, how complete that record is per year, measured against the index's
real turnover of roughly 20-25 names a year. The pre-registration makes that a gate.

    python3 -I research/build_membership.py
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent / "data.nosync"
UA = {"User-Agent": "NoCapExperiment/0.1 (portfolio research; contact via github.com/ibrovail)"}
CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

# The index replaces roughly this many names a year. Used only to measure how much of the
# change record is missing -- never to fabricate the missing entries.
TYPICAL_CHANGES_PER_YEAR = 20


def _get(url: str, name: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.html"
    if path.exists():
        return path.read_text()
    delay = 3
    for _ in range(5):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200:
            path.write_text(r.text)
            return r.text
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"could not fetch {url}")


def load_current() -> set[str]:
    t = pd.read_html(io.StringIO(_get(CURRENT_URL, "sp500_wiki")))[0]
    return {str(s).strip().upper() for s in t["Symbol"]}


def load_changes() -> pd.DataFrame:
    t = pd.read_html(io.StringIO(_get(CHANGES_URL, "sp500_historical_wiki")))[0]
    t.columns = ["date", "add_t", "add_n", "rem_t", "rem_n", "reason", "refs"]
    t = t[t["date"] != "Effective Date"].copy()
    t["dt"] = pd.to_datetime(t["date"], format="mixed", errors="coerce")
    t = t.dropna(subset=["dt"]).sort_values("dt", ascending=False).reset_index(drop=True)
    # Normalise to plain strings. Never None/NaN: a NaN is TRUTHY, so `if row["add_t"]`
    # would pass and put a float into the member set.
    for c in ("add_t", "rem_t"):
        t[c] = (t[c].astype(str).str.strip().str.upper()
                .replace({"NAN": "", "NONE": "", "<NA>": ""}))
        t[c] = t[c].where(t[c].str.fullmatch(r"[A-Z0-9.\-]{1,8}"), "")
    return t


def roll_backward(current: set[str], changes: pd.DataFrame,
                  first_year: int, last_year: int) -> dict[int, set[str]]:
    """Snapshot membership at each 31 December, walking the changes newest-first."""
    members = set(current)
    snapshots: dict[int, set[str]] = {}
    years = list(range(last_year, first_year - 1, -1))
    yi = 0
    for _, row in changes.iterrows():
        # Record every year-end the cursor has now passed.
        while yi < len(years) and row["dt"] <= pd.Timestamp(f"{years[yi]}-12-31"):
            snapshots[years[yi]] = set(members)
            yi += 1
        if yi >= len(years):
            break
        # Reverse the change to step before it.
        if row["add_t"]:
            members.discard(row["add_t"])
        if row["rem_t"]:
            members.add(row["rem_t"])
    while yi < len(years):
        snapshots[years[yi]] = set(members)
        yi += 1
    return snapshots


def quality_report(changes: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    """Recorded changes per year against the index's real turnover.

    This is the gate. A year whose record is 10% complete cannot support a backtest,
    however good the prices are.
    """
    counts = changes.groupby(changes["dt"].dt.year).size()
    rows = []
    for year in range(first_year, last_year + 1):
        n = int(counts.get(year, 0))
        rows.append({
            "year": year,
            "changes_recorded": n,
            "completeness_vs_typical": round(min(n / TYPICAL_CHANGES_PER_YEAR, 1.0), 3),
            "usable": n >= TYPICAL_CHANGES_PER_YEAR * 0.5,
        })
    return pd.DataFrame(rows)


def main() -> None:
    current, changes = load_current(), load_changes()
    print(f"current constituents: {len(current)}")
    print(f"recorded changes:     {len(changes)}  "
          f"({changes['dt'].min().date()} .. {changes['dt'].max().date()})\n")

    first, last = 1996, 2025
    q = quality_report(changes, first, last)
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    q.to_csv(out / "membership_quality.csv", index=False)

    usable = q[q["usable"]]
    print("Change-record completeness (vs ~20 real changes/yr):")
    print(q.to_string(index=False))
    print(f"\nFIRST USABLE FORMATION YEAR: {int(usable['year'].min())}")
    print(f"Years failing the 50% bar:   {sorted(q.loc[~q['usable'], 'year'])}")

    snaps = roll_backward(current, changes, first, last)
    Path(CACHE / "membership.json").write_text(
        json.dumps({str(y): sorted(v) for y, v in sorted(snaps.items())})
    )
    sizes = pd.DataFrame(
        [{"year": y, "members": len(v)} for y, v in sorted(snaps.items())]
    )
    sizes.to_csv(out / "membership_sizes.csv", index=False)
    print("\nReconstructed set size by year (a real S&P 500 is ~500 every year):")
    print(sizes.to_string(index=False))


if __name__ == "__main__":
    main()
