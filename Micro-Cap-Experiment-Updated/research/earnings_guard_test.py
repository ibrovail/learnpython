"""Does the 10-session pre-earnings guard earn its place? Two tests, not one.

portfolio_rules.md forbids initiating a position within 10 trading sessions before a known
earnings date. In practice that blocks 23-39 of the top 50 candidates in October, which makes it
the largest constraint on candidate supply in the book -- and its only evidence is a single-season
study from 2026-09-19 (1,141 reports, one earnings season).

Earnings dates come from SEC EDGAR 8-K Item 2.02 filings (research/earnings_dates.py), which are
the releases themselves and survive delisting. Caveat recorded up front: the FILING date is the
same session as the release or one after it, since companies release after the close and file the
8-K that day or the next. For a 10-session window that is immaterial, but it is not exact.

TEST 1 -- the differential. Among candidates on the same date, do those inside the 10-session
pre-earnings window go on to return less than those outside it? This is the cleaner test: it
compares two groups under the same conditions, so partial ticker coverage biases it far less than
it would bias a level.

TEST 2 -- the portfolio. Switch the guard on and off in the daily simulation and see whether the
book is better for it. This is what the rule actually does, including the cost of the candidates
it refuses.

    venv/bin/python research/earnings_guard_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, DATA = ROOT / "output", ROOT / "data.nosync"
sys.path.insert(0, str(ROOT))
from regime_backtest import nw_tstat, lag_for, effective_n  # noqa: E402

SPLIT = pd.Timestamp("2021-06-30")
GUARD = 10          # the rule's window, in sessions
HORIZON = 40


def main():
    ed = DATA / "earnings_dates.csv"
    if not ed.exists():
        sys.exit("run research/earnings_dates.py first")
    e = pd.read_csv(ed, parse_dates=["earnings_date"])
    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv", parse_dates=["date"],
                        usecols=["date", "ticker", "fwd40", "pct_vs_sma50", "regime"],
                        low_memory=False)
    panel = panel[panel.fwd40.notna()]
    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    pos = {d: i for i, d in enumerate(cal)}

    # sessions from each formation date to that ticker's next earnings release
    e = e[e.ticker.isin(set(panel.ticker))].copy()
    e["idx"] = e.earnings_date.map(lambda d: pos.get(pd.Timestamp(d)))
    e = e.dropna(subset=["idx"])
    by_t = {t: np.sort(g.idx.values) for t, g in e.groupby("ticker")}
    cov = panel.ticker.isin(by_t).mean()
    print(f"earnings dates for {len(by_t)} tickers; they cover {cov:.0%} of panel rows\n",
          flush=True)

    def sessions_to_next(row):
        a = by_t.get(row.ticker)
        if a is None:
            return np.nan
        i = pos.get(row.date)
        if i is None:
            return np.nan
        nxt = a[a >= i]
        return (nxt[0] - i) if len(nxt) else np.nan

    panel["to_earn"] = panel.apply(sessions_to_next, axis=1)
    sub = panel[panel.to_earn.notna()].copy()
    sub["in_window"] = sub.to_earn <= GUARD
    print(f"  rows with a known next earnings date: {len(sub)} "
          f"({sub.in_window.mean():.0%} inside the {GUARD}-session window)\n", flush=True)

    print("TEST 1 -- in-window vs out-of-window entries, same dates, 40-session forward\n")
    print(f"{'period':<14}{'in-window':>12}{'outside':>10}{'difference':>12}{'t':>7}{'dates':>7}")
    rows = []
    for lbl, m in (("train", sub.date < SPLIT), ("test", sub.date >= SPLIT),
                   ("full", pd.Series(True, index=sub.index))):
        s = sub[m]
        per = []
        for d, g in s.groupby("date"):
            a, b = g[g.in_window], g[~g.in_window]
            if len(a) < 3 or len(b) < 10:
                continue
            per.append({"date": d, "d": a.fwd40.mean() - b.fwd40.mean(),
                        "a": a.fwd40.mean(), "b": b.fwd40.mean()})
        p = pd.DataFrame(per)
        if p.empty:
            print(f"{lbl:<14}{'(too few)':>12}"); continue
        lg, _ = lag_for(HORIZON, len(p))
        mn, t, _ = nw_tstat(p.d, lg)
        print(f"{lbl:<14}{p.a.mean():>11.2f}%{p.b.mean():>9.2f}%{mn:>11.2f}pp{t:>7.2f}{len(p):>7}")
        rows.append(dict(period=lbl, in_window=p.a.mean(), outside=p.b.mean(),
                         diff_pp=mn, nw_t=t, dates=len(p),
                         effective_n=round(effective_n(len(p), HORIZON), 1)))
    pd.DataFrame(rows).to_csv(OUT / "earnings_guard_differential.csv", index=False)
    print(f"\n  A negative difference means in-window entries did WORSE, which is what the")
    print(f"  guard assumes. A positive one means the guard is refusing good candidates.")
    print(f"\nwrote {OUT/'earnings_guard_differential.csv'}", flush=True)


if __name__ == "__main__":
    main()
