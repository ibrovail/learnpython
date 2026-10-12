"""Winner-rotation backtest engine for Study 01.

Each January, buy the top-N performers of the completed prior calendar year from that
year-end's index membership, equal-weighted; hold to year-end; repeat.

The engine is deliberately data-source agnostic: it takes a price panel and a membership
map, so it can be unit-tested against synthetic prices with known answers without any
network access. See research/test_engine.py.

Pre-registration: Experiment Details/Study 01 — Winner Rotation (pre-registration).md
Decision rules live there, not here. This file computes; it does not judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class YearResult:
    year: int
    picks: list[str]
    ret: float
    stopped: list[str] = field(default_factory=list)
    substitutions: int = 0
    dropped: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    years: list[YearResult]
    n: int
    stop_pct: float | None

    @property
    def returns(self) -> pd.Series:
        return pd.Series({y.year: y.ret for y in self.years}, dtype=float)

    @property
    def curve(self) -> pd.Series:
        """Growth of 1.0, indexed by year-end, starting the year before the first."""
        r = self.returns
        c = (1.0 + r).cumprod()
        first = r.index.min() - 1 if len(r) else 0
        return pd.concat([pd.Series({first: 1.0}, dtype=float), c])

    @property
    def terminal(self) -> float:
        return float(self.curve.iloc[-1])

    @property
    def cagr(self) -> float:
        n_years = len(self.returns)
        return self.terminal ** (1.0 / n_years) - 1.0 if n_years else float("nan")

    @property
    def max_drawdown(self) -> float:
        """Worst peak-to-trough on the YEAR-END curve. Intra-year lows are not captured."""
        c = self.curve
        return float((c / c.cummax() - 1.0).min())

    @property
    def total_substitutions(self) -> int:
        return sum(y.substitutions for y in self.years)


def _calendar_return(prices: pd.DataFrame, ticker: str, year: int) -> float:
    """Total return over a calendar year. NaN if either end is missing."""
    col = prices[ticker].dropna()
    window = col[col.index.year == year]
    if len(window) < 2:
        return float("nan")
    first, last = window.iloc[0], window.iloc[-1]
    if not np.isfinite(first) or first <= 0 or not np.isfinite(last):
        return float("nan")
    return last / first - 1.0


def _hold_return(
    prices: pd.DataFrame,
    lows: pd.DataFrame | None,
    ticker: str,
    year: int,
    stop_pct: float | None,
) -> tuple[float, bool]:
    """Return over the holding year, and whether the stop fired.

    The stop sells at exactly the stop price; the proceeds then sit in cash earning
    nothing for the rest of the year. Real exits can be worse when a stock gaps through
    the level -- this is the same optimistic assumption the audited thread made, and it
    is recorded here so the bias is known rather than hidden.
    """
    col = prices[ticker].dropna()
    window = col[col.index.year == year]
    if len(window) < 2:
        return float("nan"), False

    entry = float(window.iloc[0])
    if not np.isfinite(entry) or entry <= 0:
        return float("nan"), False

    if stop_pct is not None:
        trigger = entry * (1.0 - stop_pct)
        # Intraday lows when available, otherwise closes (the year is flagged upstream).
        if lows is not None and ticker in lows.columns:
            path = lows[ticker].dropna()
            path = path[path.index.year == year]
        else:
            path = window
        # Ignore the entry bar itself: a stop cannot fire before the position exists.
        if (path.iloc[1:] <= trigger).any():
            return -stop_pct, True

    return float(window.iloc[-1]) / entry - 1.0, False


def run(
    prices: pd.DataFrame,
    membership: dict[int, set[str]],
    n: int,
    start_year: int,
    end_year: int,
    stop_pct: float | None = None,
    lows: pd.DataFrame | None = None,
    excluded: frozenset[str] = frozenset(),
) -> RunResult:
    """Run the rotation over [start_year, end_year] inclusive.

    membership[y] is the set of index members as of 31 December of year y, so the picks
    for holding year Y are ranked on year Y-1 using membership[Y-1]. Nothing from year Y
    is used to choose them.
    """
    years: list[YearResult] = []

    for year in range(start_year, end_year + 1):
        formation = year - 1
        members = membership.get(formation)
        if not members:
            continue

        scored = []
        for ticker in members:
            if ticker not in prices.columns:
                continue
            r = _calendar_return(prices, ticker, formation)
            if np.isfinite(r):
                scored.append((r, ticker))
        if not scored:
            continue

        # Rank descending by formation-year return; ties broken by ticker for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))

        picks: list[str] = []
        substitutions = 0
        for _, ticker in scored:
            if len(picks) == n:
                break
            if ticker in excluded:
                substitutions += 1
                continue
            picks.append(ticker)
        if not picks:
            continue

        rets, stopped, dropped = [], [], []
        for ticker in picks:
            r, hit = _hold_return(prices, lows, ticker, year, stop_pct)
            if not np.isfinite(r):
                dropped.append(ticker)
                continue
            rets.append(r)
            if hit:
                stopped.append(ticker)
        if not rets:
            continue

        years.append(
            YearResult(
                year=year,
                picks=picks,
                ret=float(np.mean(rets)),  # equal-weighted
                stopped=stopped,
                substitutions=substitutions,
                dropped=dropped,
            )
        )

    return RunResult(years=years, n=n, stop_pct=stop_pct)


def years_ahead(strategy: pd.Series, benchmark: pd.Series) -> tuple[int, int, float]:
    """Count years the strategy beat the benchmark, with an EXACT one-sided binomial p.

    Reported instead of a bare count because, per the pre-registration, 19 wins in 30 years
    is not distinguishable from a coin flip (p = 0.100).
    """
    from math import comb

    pair = pd.concat([strategy, benchmark], axis=1).dropna()
    total = len(pair)
    if total == 0:
        return 0, 0, float("nan")
    wins = int((pair.iloc[:, 0] > pair.iloc[:, 1]).sum())
    p = sum(comb(total, i) for i in range(wins, total + 1)) / 2**total
    return wins, total, p
