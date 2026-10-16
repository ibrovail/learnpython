"""Stage 1 validation — can the reconstructed membership possibly be right?

Set size is not a validity test: an unrecorded addition and an unrecorded removal cancel
out in the count while leaving the composition wrong. This tests composition directly,
by asking whether a snapshot contains companies that did not yet trade on that date.

Any such name is a company the backward roll failed to remove -- which means the snapshot
is contaminated with modern survivors, the exact bias that makes a momentum backtest look
good for free.

    python3 -I research/validate_membership.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent / "data.nosync"
OUT = Path(__file__).resolve().parent / "output"

# First calendar year each ticker could possibly have been an S&P 500 member, from its
# IPO, direct listing or spin-off. Deliberately generous: the year of listing itself is
# allowed, so a hit is unambiguous rather than an off-by-one.
# NEW: the company did not exist as a listed entity before this year, and the ticker was
# not in use by anyone else. A hit here is pure survivorship contamination -- a modern
# winner the backward roll failed to remove.
ANCHORS_NEW = {
    "GOOGL": 2004, "GOOG": 2004, "META": 2022, "TSLA": 2010, "PYPL": 2015,
    "MRNA": 2018, "PLTR": 2020, "ABNB": 2020, "V": 2008, "MA": 2006,
    "KHC": 2015, "CARR": 2020, "OTIS": 2020, "COIN": 2021, "RBLX": 2021,
    "DASH": 2020, "GEV": 2024, "SOLV": 2024, "VLTO": 2023, "KVUE": 2023,
    "CTVA": 2019, "NFLX": 2002, "CRM": 2004,
}

# REUSED: the ticker was legitimately a member earlier, but belonged to a DIFFERENT
# company. A hit is not survivorship -- it is worse in one way, because price data keyed
# on the ticker silently splices two unrelated companies into one series.
ANCHORS_REUSED = {
    "DELL": 2016,   # Dell Technologies; previously Dell Computer, a member to 2013
    "DOW": 2019,    # Dow Inc; previously Dow Chemical
    "FOXA": 2019,   # Fox Corp; previously Twenty-First Century Fox
    "CEG": 2022,    # Constellation Energy Corp; previously Constellation Energy Group
    "SNDK": 2025,   # Sandisk spin-off; previously SanDisk Corp, listed from 1995
    "LIN": 2018,    # Linde plc after the Praxair merger
    "GM": 2010,     # post-bankruptcy General Motors Company
}
ANCHORS = {**ANCHORS_NEW, **ANCHORS_REUSED}


def main() -> None:
    membership = {int(y): set(v) for y, v in
                  json.loads((CACHE / "membership.json").read_text()).items()}

    rows = []
    for year in sorted(membership):
        new_hits = sorted(t for t, f in ANCHORS_NEW.items()
                          if t in membership[year] and year < f)
        reuse_hits = sorted(t for t, f in ANCHORS_REUSED.items()
                            if t in membership[year] and year < f)
        rows.append({
            "snapshot_year": year,
            "members": len(membership[year]),
            "survivorship": len(new_hits),
            "ticker_reuse": len(reuse_hits),
            "impossible_members": len(new_hits) + len(reuse_hits),
            "examples": ", ".join((new_hits + reuse_hits)[:6]),
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "membership_validation.csv", index=False)

    print("Companies present in a snapshot BEFORE they could have been members")
    print(f"(tested against {len(ANCHORS_NEW)} never-before-listed anchors and "
          f"{len(ANCHORS_REUSED)} reused tickers; 0 is the only passing score)\n")
    print(df.to_string(index=False))

    clean = df[df["impossible_members"] == 0]["snapshot_year"]
    print(f"\nSnapshots with zero impossible members: "
          f"{int(clean.min()) if len(clean) else 'none'} onward")
    worst = df.loc[df["impossible_members"].idxmax()]
    print(f"Worst snapshot: {int(worst['snapshot_year'])} with "
          f"{int(worst['impossible_members'])} of {len(ANCHORS)} anchors impossible")


if __name__ == "__main__":
    main()
