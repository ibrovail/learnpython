"""Self-test for the Stage 2 runner's machinery, on synthetic prices. No network.

This proves the Stage 2 pipeline works WITHOUT producing a study result: the panel is
invented, the answers are worked out by hand, and nothing here touches market data. The
point is that when real prices do arrive, the only new thing is the data.

    python3 -I research/test_run_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_study as R  # noqa: E402
import winners as W  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")


def panel(spec: dict[str, dict[int, tuple[float, float]]]) -> pd.DataFrame:
    cols = {}
    for ticker, by_year in spec.items():
        parts = []
        for year, (a, b) in sorted(by_year.items()):
            days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
            parts.append(pd.Series(np.linspace(a, b, len(days)), index=days))
        cols[ticker] = pd.concat(parts)
    return pd.DataFrame(cols)


YEARS = range(2011, 2026)
SPEC = {
    # A steady winner, a steady loser, a middling name, and the benchmark.
    "WIN": {y: (100.0, 130.0) for y in YEARS},
    "MID": {y: (100.0, 110.0) for y in YEARS},
    "LOSE": {y: (100.0, 70.0) for y in YEARS},
    "LMT": {y: (100.0, 160.0) for y in YEARS},   # on the rule 6 defence list
    "SPY": {y: (100.0, 108.0) for y in YEARS},
}
PRICES = panel(SPEC)
MEMBERS = {y: {"WIN", "MID", "LOSE", "LMT"} for y in YEARS}

print("1. The configuration count is the search space, not the backtest count")
check("6 N values", len(R.NS), 6)
check("8 stop levels including no-stop", len(R.STOPS), 8)
check("search space is 6 x 8 x 2 exclusion arms", len(R.NS) * len(R.STOPS) * 2, 96)

print("\n2. Periods are the amended split, and 2026 is excluded")
check("TRAIN", R.TRAIN, (2012, 2018))
check("TEST", R.TEST, (2019, 2025))
check("no holding year past 2025", R.LAST_HOLDING_YEAR, 2025)

print("\n3. Benchmark returns come from the panel, per calendar year")
bench = R.benchmark_returns(PRICES, 2012, 2018)
check("seven benchmark years", len(bench), 7)
check("each is +8%", round(float(bench.iloc[0]), 6), 0.08)
check("benchmark CAGR is +8%", round(R.cagr(bench), 6), 0.08)

print("\n4. A missing benchmark stops the study rather than silently skipping it")
try:
    R.benchmark_returns(PRICES.drop(columns=["SPY"]), 2012, 2018)
    check("raises without SPY", False, True)
except SystemExit:
    check("raises without SPY", True, True)

print("\n5. summarise() reports a proportion and an exact p, never a bare count")
res = W.run(PRICES, MEMBERS, n=1, start_year=2012, end_year=2018)
row = R.summarise(res, bench, "TRAIN", 1, None, False)
check("N=1 picks the best prior year (LMT, +60%)", res.years[0].picks, ["LMT"])
check("seven holding years", row["years"], 7)
check("LMT's +60% beats the +8% benchmark every year", row["years_ahead"], 7)
check("reported as a proportion", row["years_ahead_proportion"], 1.0)
check("with an exact one-sided binomial p = 1/2^7",
      round(row["binomial_p_one_sided"], 6), round(1 / 2 ** 7, 6))
check("and it beats the benchmark on CAGR", row["beats_benchmark"], True)

print("\n6. The exclusion list substitutes the next-ranked name, and counts it")
res_x = W.run(PRICES, MEMBERS, n=1, start_year=2012, end_year=2018,
              excluded=frozenset({"LMT"}))
row_x = R.summarise(res_x, bench, "TRAIN", 1, None, True)
check("LMT excluded, WIN substitutes", res_x.years[0].picks, ["WIN"])
check("one substitution per year, seven years", row_x["substitutions"], 7)
check("the exclusion arm is stamped with the list's completeness",
      row_x["exclusion_list_complete"], R.X.COMPLETE)
check("and the unexcluded arm carries no such stamp",
      row["exclusion_list_complete"], "")

print("\n7. A stop is recorded when it fires, and costs exactly its level")
res_s = W.run(PRICES, MEMBERS, n=4, start_year=2012, end_year=2012, stop_pct=0.20)
row_s = R.summarise(res_s, bench, "TRAIN", 4, 0.20, False)
check("LOSE falls 30%, so the 20% stop fires on it", res_s.years[0].stopped, ["LOSE"])
check("one position stopped", row_s["stop_fired_positions"], 1)
check("in one year", row_s["stop_fired_years"], 1)
check("stop level is reported, not inferred", row_s["stop_pct"], "0.20")

print("\n8. The coverage gate is read from the committed report, not assumed")
overall, how = R.read_gate()
check("the gate returns a fraction", isinstance(overall, float), True)
check("and the bar is the pre-registered 90%", R.COVERAGE_GATE, 0.90)

print(f"\n{'ALL PASS' if not FAIL else f'{FAIL} FAILURES'}  ({PASS} passed)")
sys.exit(1 if FAIL else 0)
