"""Stage 1 gate — measure price coverage in MEMBER-YEARS, before any return is computed.

The pre-registration makes this a gate, not a diagnostic: below 90% of expected
member-years the study is reported inconclusive rather than published with a number.

Member-years, not tickers, is the right unit. A name that was a member for two years
matters less than one that was a member for fourteen, so a ticker-level miss rate
overstates the damage. Saying what a figure is *per* is a standing rule here.

THIS IS THE PANEL-COMPARISON MEASURE, AND IT IS AN UPPER BOUND. It asks only whether a
file exists for the ticker. `build_panel.coverage` computes the number the gate is
actually read against, and it is stricter in two ways: it requires the series to SPAN the
formation year, so a name that listed or was acquired mid-year does not count toward a
year it cannot be ranked on; and it runs after the ticker reuse/rename map, so a ticker
held by two different companies is not credited for years that belong to the other one.

The gate figure is therefore LOWER than the figure here, never higher. Use this one to
compare two sources against each other -- that is what it was written for, and the
comparison is sound because both panels are measured the same way. Do not quote it
against the pre-registration's 90% bar.

Run against either panel:
    python3 -I research/coverage.py yahoo
    python3 -I research/coverage.py tiingo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data.nosync"
OUT = ROOT / "output"
PANELS = {"yahoo": CACHE / "prices", "tiingo": CACHE / "prices_tiingo"}
FIRST_FORMATION, LAST_FORMATION = 2011, 2024
GATE = 0.90


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "yahoo"
    panel = PANELS[which]
    if not panel.exists():
        print(f"no panel at {panel}", file=sys.stderr)
        return 2

    have = {p.stem for p in panel.glob("*.csv")}
    membership = {int(k): set(v) for k, v in
                  json.loads((CACHE / "membership.json").read_text()).items()}

    rows = []
    for year in range(FIRST_FORMATION, LAST_FORMATION + 1):
        members = membership.get(year, set())
        priced = len(members & have)
        rows.append({
            "formation_year": year,
            "members": len(members),
            "priced": priced,
            "missing": len(members) - priced,
            "coverage": round(priced / len(members), 4) if members else float("nan"),
        })
    df = pd.DataFrame(rows)

    total_my = int(df["members"].sum())
    priced_my = int(df["priced"].sum())
    overall = priced_my / total_my

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"coverage_{which}.csv", index=False)

    print(f"PANEL: {which}   files on disk: {len(have)}")
    print(df.to_string(index=False))
    print(f"\nmember-years expected {total_my:,} | priced {priced_my:,} | "
          f"missing {total_my - priced_my:,}")
    print(f"OVERALL COVERAGE: {overall:.1%}   gate {GATE:.0%}   "
          f"-> {'PASS' if overall >= GATE else 'FAIL — study is inconclusive on this panel'}")

    # Which names are missing matters as much as how many. If the misses are the
    # acquired and bankrupt ones, the panel is survivorship-biased in the direction
    # that flatters a momentum strategy.
    ever = set().union(*[membership[y] for y in range(FIRST_FORMATION, LAST_FORMATION + 1)])
    missing = sorted(ever - have)
    (OUT / f"missing_{which}.txt").write_text("\n".join(missing))
    print(f"\nmissing tickers: {len(missing)} of {len(ever)} ever-members "
          f"-> research/output/missing_{which}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
