"""Stage 2 — the raw backtest output for Study 01. Computes; does not judge.

Everything here is pre-registered in
`Experiment Details/Study 01 — Winner Rotation (pre-registration).md`, as amended by
`Study 01 — Stage 1 findings.md`:

    TRAIN 2012-2018, TEST 2019-2025, split and reported separately
    N in 1, 3, 5, 10, 20, 50
    stops at no-stop and 20/25/30/35/40/45/50%
    both with and without the rule 6 exclusion list, substitutions counted
    benchmark: S&P 500 total return via SPY
    2026 excluded entirely -- it is an incomplete year

THE COVERAGE GATE IS ENFORCED HERE, IN CODE. The pre-registration says that below 90%
coverage of expected member-years the study is "reported as inconclusive rather than
published with a number". A prose rule is one a later run can forget, so this script
reads the committed coverage report and REFUSES TO WRITE RESULTS below the bar. Passing
--gate-override records an explicit override in the output rather than hiding it, and is
not a way to publish a number: it exists so the machinery can be exercised on partial
data without the result ever being mistaken for a finding.

    python3 -I research/run_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exclusions as X  # noqa: E402
import winners as W  # noqa: E402
from build_panel import load_membership  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data.nosync"
OUT = ROOT / "output"

TRAIN = (2012, 2018)
TEST = (2019, 2025)
PERIODS = {"TRAIN": TRAIN, "TEST": TEST}
NS = [1, 3, 5, 10, 20, 50]
STOPS: list[float | None] = [None, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BENCHMARK = "SPY"
COVERAGE_GATE = 0.90
LAST_HOLDING_YEAR = 2025          # 2026 is incomplete and is excluded entirely


def read_gate() -> tuple[float, str]:
    path = OUT / "coverage.csv"
    if not path.exists():
        return 0.0, "no coverage report: Stage 1b has not been run"
    cov = pd.read_csv(path)
    if not cov["members"].sum():
        return 0.0, "coverage report is empty"
    overall = cov["full_year"].sum() / cov["members"].sum()
    return float(overall), f"{overall:.1%} of expected member-years"


def benchmark_returns(prices: pd.DataFrame, start: int, end: int) -> pd.Series:
    if BENCHMARK not in prices.columns:
        raise SystemExit(
            f"No {BENCHMARK} series in the panel. The benchmark is not an index member, "
            "so it is fetched separately; without it there is nothing to compare against."
        )
    out = {}
    for year in range(start, end + 1):
        r = W._calendar_return(prices, BENCHMARK, year)
        if np.isfinite(r):
            out[year] = r
    return pd.Series(out, dtype=float)


def cagr(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() ** (1.0 / len(returns)) - 1.0)


def summarise(res: W.RunResult, bench: pd.Series, period: str, n: int,
              stop: float | None, excl_on: bool) -> dict:
    r = res.returns
    paired_bench = bench.reindex(r.index).dropna()
    r = r.reindex(paired_bench.index)
    wins, total, p = W.years_ahead(r, paired_bench)
    strat_cagr, bench_cagr = cagr(r), cagr(paired_bench)
    return {
        "period": period,
        "n": n,
        "stop_pct": "none" if stop is None else f"{stop:.2f}",
        "exclusions_applied": excl_on,
        "exclusion_list_complete": X.COMPLETE if excl_on else "",
        "years": len(r),
        "terminal_growth_of_1": round(float((1.0 + r).prod()), 6),
        "cagr": round(strat_cagr, 6),
        "benchmark_cagr": round(bench_cagr, 6),
        "excess_cagr": round(strat_cagr - bench_cagr, 6),
        "beats_benchmark": bool(strat_cagr > bench_cagr),
        "max_drawdown": round(res.max_drawdown, 6),
        "years_ahead": wins,
        "years_total": total,
        "years_ahead_proportion": round(wins / total, 4) if total else float("nan"),
        "binomial_p_one_sided": round(p, 6),
        "substitutions": res.total_substitutions,
        "stop_fired_years": sum(1 for y in res.years if y.stopped),
        "stop_fired_positions": sum(len(y.stopped) for y in res.years),
        "picks_dropped_no_price": sum(len(y.dropped) for y in res.years),
    }


def main() -> int:
    override = "--gate-override" in sys.argv
    overall, how = read_gate()
    print(f"Coverage gate: {how} (bar is {COVERAGE_GATE:.0%})")
    if overall < COVERAGE_GATE:
        print("\nGATE FAILED.")
        print("The pre-registration reports the study INCONCLUSIVE rather than publishing")
        print("a number below this bar. No result files are written.")
        if not override:
            return 3
        print("\n--gate-override given: running anyway, and every output row is stamped")
        print("INCONCLUSIVE. These numbers are machinery checks, not findings.\n")

    membership = load_membership()
    close = pd.read_parquet(CACHE / "prices_close.parquet")
    low_path = CACHE / "prices_low.parquet"
    low = pd.read_parquet(low_path) if low_path.exists() else None

    start_t, end_t = TRAIN[0], min(TEST[1], LAST_HOLDING_YEAR)
    bench_all = benchmark_returns(close, start_t, end_t)

    rows, year_rows = [], []
    evaluations = 0
    for period, (y0, y1) in PERIODS.items():
        y1 = min(y1, LAST_HOLDING_YEAR)
        bench = bench_all.loc[(bench_all.index >= y0) & (bench_all.index <= y1)]
        for excl_on in (False, True):
            excluded = X.EXCLUDED if excl_on else frozenset()
            for n in NS:
                for stop in STOPS:
                    evaluations += 1
                    res = W.run(close, membership, n=n, start_year=y0, end_year=y1,
                                stop_pct=stop, lows=low, excluded=excluded)
                    rows.append(summarise(res, bench, period, n, stop, excl_on))
                    for y in res.years:
                        year_rows.append({
                            "period": period, "n": n,
                            "stop_pct": "none" if stop is None else f"{stop:.2f}",
                            "exclusions_applied": excl_on, "year": y.year,
                            "ret": round(y.ret, 6),
                            "benchmark_ret": round(float(bench.get(y.year, float("nan"))), 6),
                            "picks": "|".join(y.picks),
                            "stopped": "|".join(y.stopped),
                            "substitutions": y.substitutions,
                            "dropped_no_price": "|".join(y.dropped),
                        })

    frame = pd.DataFrame(rows)
    years = pd.DataFrame(year_rows)
    # The multiple-comparisons burden is the size of the SEARCH SPACE -- the distinct
    # strategy configurations a result could have been picked from -- not the number of
    # backtests run. Each configuration is evaluated twice, once per period, and the two
    # evaluations are one look, not two.
    configs = len(NS) * len(STOPS) * 2
    frame.insert(0, "inconclusive_below_coverage_gate", overall < COVERAGE_GATE)
    frame.insert(1, "configurations_tried", configs)
    frame.insert(2, "evaluations_run", evaluations)

    OUT.mkdir(exist_ok=True)
    frame[frame["stop_pct"] == "none"].to_csv(OUT / "stage2_n_curve.csv", index=False)
    frame.to_csv(OUT / "stage2_stop_surface.csv", index=False)
    years.to_csv(OUT / "stage2_year_detail.csv", index=False)

    print(f"\nCONFIGURATIONS TRIED: {configs}  "
          f"({len(NS)} values of N x {len(STOPS)} stop levels x 2 exclusion arms)")
    print(f"  evaluations run: {evaluations} (each configuration on both periods)")
    print(f"  a one-sided p of 0.05 among {configs} looks needs roughly "
          f"p < {0.05 / configs:.4f} to mean the same thing")
    print("  (the best of N tries is biased upward by roughly the spread of N draws)")
    print(f"Exclusion list: {X.summary()}")
    print(f"Benchmark: {BENCHMARK} total return. 2026 excluded as an incomplete year.")
    print("\nN curve, no stop:")
    cols = ["period", "n", "exclusions_applied", "years", "cagr", "benchmark_cagr",
            "beats_benchmark", "max_drawdown", "years_ahead", "years_total",
            "binomial_p_one_sided", "substitutions"]
    print(frame[frame["stop_pct"] == "none"][cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
