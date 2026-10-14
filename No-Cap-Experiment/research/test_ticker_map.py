"""Self-tests for the Stage 1b ticker map and panel assembly. No network.

Same style as test_engine.py: synthetic inputs with answers worked out by hand, so the
map can be trusted before it is pointed at real prices. The cases are the ones Stage 1
named, plus the two that silently produce a plausible wrong number -- a handover inside
a calendar year, and a company that only traded for part of the formation year.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_panel as bp  # noqa: E402
import ticker_map as tm  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")


def sessions(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


print("1. Rename resolution")
check("FB reads META's series", tm.resolve("FB"), "META")
check("an unmapped ticker resolves to itself", tm.resolve("AAPL"), "AAPL")
check("VIAC and CBS reach the same company", tm.resolve("CBS"), tm.resolve("VIAC"))
check("RTN does NOT resolve to RTX (absorbed, not renamed)",
      tm.resolve("RTN") == "RTX", False)

print("\n2. Reuse eras")
check("DELL in 2012 is Dell Computer",
      tm.era_for("DELL", pd.Timestamp("2012-06-30")).company, "Dell Inc (Dell Computer)")
check("DELL in 2020 is Dell Technologies",
      tm.era_for("DELL", pd.Timestamp("2020-06-30")).company, "Dell Technologies Inc")
check("DELL in 2015 belongs to nobody (private)",
      tm.era_for("DELL", pd.Timestamp("2015-06-30")), None)

print("\n3. Whole-year windows")
check("a normal ticker gets the whole year",
      tm.valid_window("AAPL", 2015),
      (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-31")))
check("FOXA 2019 is REFUSED -- the two companies meet inside the year",
      tm.valid_window("FOXA", 2019), None)
check("FOXA 2018 is fine (21st Century Fox all year)",
      tm.valid_window("FOXA", 2018) is not None, True)
check("FOXA 2020 is fine (Fox Corp all year)",
      tm.valid_window("FOXA", 2020) is not None, True)
check("DELL 2015 refused (gap)", tm.valid_window("DELL", 2015), None)
check("DOW 2018 refused (gap)", tm.valid_window("DOW", 2018), None)
check("LIN 2016 refused (gap)", tm.valid_window("LIN", 2016), None)
check("SNDK 2017 refused (gap)", tm.valid_window("SNDK", 2017), None)
check("CEG 2022 is trimmed to the spin-off date",
      tm.valid_window("CEG", 2022)[0], pd.Timestamp("2022-02-02"))

print("\n4. The reuse mask blanks the panel, it does not splice it")
idx = sessions("2018-01-01", "2020-12-31")
close = pd.DataFrame({"FOXA": 100.0, "AAPL": 50.0}, index=idx)
membership = {y: {"FOXA", "AAPL"} for y in range(2011, 2025)}
# build() reads the real cache from disk; exercise its mask rule directly instead.
for year in (2018, 2019, 2020):
    rows = close.index.year == year
    w = tm.valid_window("FOXA", year)
    if w is None:
        close.loc[rows, "FOXA"] = float("nan")
check("FOXA 2018 survives the mask", close.loc[close.index.year == 2018, "FOXA"].notna().all(), True)
check("FOXA 2019 is entirely blanked", close.loc[close.index.year == 2019, "FOXA"].isna().all(), True)
check("FOXA 2020 survives the mask", close.loc[close.index.year == 2020, "FOXA"].notna().all(), True)
check("the mask touched no other column", close["AAPL"].notna().all(), True)

print("\n5. Coverage counts only FULL formation years")
idx = sessions("2015-01-01", "2015-12-31")
full = pd.Series(100.0, index=idx)
stub = pd.Series(100.0, index=idx[idx >= "2015-07-01"])   # listed mid-year
gone = pd.Series(100.0, index=idx[idx <= "2015-06-30"])   # acquired mid-year
panel = pd.DataFrame({"FULL": full, "STUB": stub, "GONE": gone})
m = {2015: {"FULL", "STUB", "GONE"}}
cov = bp.coverage(m, panel, quota_blocked=set(), no_history=set())
row = cov.set_index("formation_year").loc[2015]
check("all three have some price", int(row["priced"]), 3)
check("only the full-year name is rankable", int(row["full_year"]), 1)
check("coverage is 1 of 3, not 3 of 3", round(float(row["coverage"]), 4), round(1 / 3, 4))

print("\n6. Missing names are attributed, not lumped together")
m = {2015: {"FULL", "QUOTAONLY", "NOHIST", "UNKNOWN"}}
cov = bp.coverage(m, pd.DataFrame({"FULL": full}),
                  quota_blocked={"QUOTAONLY"}, no_history={"NOHIST"})
row = cov.set_index("formation_year").loc[2015]
check("quota refusal is counted as such", int(row["missing_quota"]), 1)
check("no-history-at-source is counted as such", int(row["missing_no_history"]), 1)
check("an unexplained absence is not hidden", int(row["missing_other"]), 1)
check("coverage counts every absence against the gate",
      round(float(row["coverage"]), 4), 0.25)

print("\n7. A reused ticker is never both reused and renamed")
check("no overlap", bool(set(tm.REUSE) & set(tm.RENAME)), False)

print(f"\n{'ALL PASS' if not FAIL else f'{FAIL} FAILURES'}  ({PASS} passed)")
sys.exit(1 if FAIL else 0)
