"""Self-test for the Study 01 engine, against synthetic prices with known answers.

Runs with no network and no market data: every expected value below is derived by hand,
so the engine's mechanics are verified even while the real data is unavailable.

    python3 -I research/test_engine.py
"""
from __future__ import annotations

import sys
from math import comb, isclose

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import winners as W  # noqa: E402

# Per (ticker, year): (start price, end price). Calendar return is end/start - 1.
SPEC = {
    "AAA": {2020: (100, 200), 2021: (100, 50), 2022: (100, 110)},   # +100%, -50%, +10%
    "BBB": {2020: (100, 150), 2021: (100, 120), 2022: (100, 110)},  #  +50%, +20%, +10%
    "CCC": {2020: (100, 110), 2021: (100, 130), 2022: (100, 110)},  #  +10%, +30%, +10%
    "DDD": {2020: (100, 80),  2021: (100, 200), 2022: (100, 110)},  #  -20%, +100%, +10%
    "XXX": {2020: (100, 900), 2021: (100, 101), 2022: (100, 110)},  # huge, but not a member
}
MEMBERSHIP = {y: {"AAA", "BBB", "CCC", "DDD"} for y in (2020, 2021, 2022)}

FAILED: list[str] = []


def build_prices() -> pd.DataFrame:
    frames = {}
    for ticker, by_year in SPEC.items():
        parts = []
        for year, (start, end) in by_year.items():
            days = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
            parts.append(pd.Series(np.linspace(start, end, len(days)), index=days))
        frames[ticker] = pd.concat(parts)
    return pd.DataFrame(frames)


def build_lows(prices: pd.DataFrame, dips: dict[tuple[str, int], float],
               bar: str = "mid") -> pd.DataFrame:
    """Copy of the close panel with one bar of the year pushed down to a chosen level."""
    lows = prices.copy()
    for (ticker, year), level in dips.items():
        idx = lows.index[lows.index.year == year]
        pos = 0 if bar == "first" else len(idx) // 2
        lows.loc[idx[pos], ticker] = level
    return lows


def check(name: str, got, want, tol: float = 1e-9) -> None:
    ok = isclose(got, want, abs_tol=tol) if isinstance(want, float) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILED.append(name)


def main() -> int:
    prices = build_prices()
    print("1. Selection and look-ahead")
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021)
    check("top-1 for 2021 is AAA (best of 2020)", r.years[0].picks, ["AAA"])
    check("and earns AAA's 2021 return", r.years[0].ret, -0.50)

    check("non-member XXX is never picked despite +800%",
          all("XXX" not in y.picks for y in
              W.run(prices, MEMBERSHIP, n=3, start_year=2021, end_year=2022).years), True)

    print("\n2. Equal weighting")
    r = W.run(prices, MEMBERSHIP, n=2, start_year=2021, end_year=2021)
    check("top-2 for 2021 is AAA, BBB", r.years[0].picks, ["AAA", "BBB"])
    check("mean(-50%, +20%) = -15%", r.years[0].ret, -0.15)

    r = W.run(prices, MEMBERSHIP, n=4, start_year=2021, end_year=2021)
    check("top-4 = mean(-50,+20,+30,+100)/4 = +25%", r.years[0].ret, 0.25)

    print("\n3. Compounding")
    # 2021 picks AAA (-50%); 2022 ranks on 2021, where DDD won (+100%), earning +10%.
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2022)
    check("2022 picks DDD (best of 2021)", r.years[1].picks, ["DDD"])
    check("terminal = 0.50 * 1.10", r.terminal, 0.55)
    check("CAGR over 2 years", r.cagr, 0.55**0.5 - 1.0)
    # Curve is 1.00 -> 0.50 -> 0.55, so the worst peak-to-trough is the full -50%.
    check("max drawdown is the -50% year", r.max_drawdown, -0.50)

    print("\n4. Stop-loss")
    # AAA's 2021 closes fall 100 -> 50, so a 35% stop (trigger 65) must fire regardless.
    lows = build_lows(prices, {("AAA", 2021): 64.0})
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021,
              stop_pct=0.35, lows=lows)
    check("breached stop fires", r.years[0].stopped, ["AAA"])
    check("stopped position returns exactly -35%", r.years[0].ret, -0.35)

    # BBB rises 100 -> 120 all year, so only the injected low can ever trigger a stop.
    # That makes it the only fixture that can test the NOT-fired branch.
    no_aaa = frozenset({"AAA"})
    lows = build_lows(prices, {("BBB", 2021): 66.0})
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021,
              stop_pct=0.35, lows=lows, excluded=no_aaa)
    check("low of 66 clears the 65 trigger", r.years[0].stopped, [])
    check("unstopped position keeps its +20%", r.years[0].ret, 0.20)

    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021,
              stop_pct=0.30, lows=lows, excluded=no_aaa)
    check("a tighter 30% stop (trigger 70) fires where 35% did not",
          r.years[0].stopped, ["BBB"])
    check("and returns exactly -30%", r.years[0].ret, -0.30)

    # A stop cannot fire before the position exists; the entry bar is skipped by design.
    lows = build_lows(prices, {("BBB", 2021): 1.0}, bar="first")
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021,
              stop_pct=0.35, lows=lows, excluded=no_aaa)
    check("entry-bar low does not trigger the stop", r.years[0].stopped, [])

    print("\n5. Exclusion list")
    r = W.run(prices, MEMBERSHIP, n=1, start_year=2021, end_year=2021,
              excluded=frozenset({"AAA"}))
    check("excluded AAA substitutes BBB", r.years[0].picks, ["BBB"])
    check("substitution is counted", r.years[0].substitutions, 1)
    check("and earns BBB's +20%", r.years[0].ret, 0.20)

    print("\n6. Years-ahead exact binomial")
    strat = pd.Series({y: 0.10 for y in range(2000, 2030)})
    bench = pd.Series({y: (0.05 if y < 2019 else 0.20) for y in range(2000, 2030)})
    wins, total, p = W.years_ahead(strat, bench)
    check("wins", wins, 19)
    check("total", total, 30)
    check("p matches exact binomial",
          p, sum(comb(30, i) for i in range(19, 31)) / 2**30)
    check("19/30 is not significant at 5%", p > 0.05, True)

    print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
