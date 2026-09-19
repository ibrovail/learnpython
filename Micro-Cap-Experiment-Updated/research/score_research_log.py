"""Score the research log: did the names research BOUGHT beat the names it PASSED?

Reads "Start Your Own/research_log.csv" (written by log_research.py since 2026-09-19). Each
research date is one cross-sectional comparison: forward returns of that weekend's BUY names
against its PASS names, from the last close on or before the research date, over 20/40/60
sessions. Follows .claude/rules/research-methods.md:

  - like with like only: BUY mean vs PASS mean, BUY median vs PASS median
  - a verdict needs >= 5 INDEPENDENT research dates at the horizon -- dates whose forward windows
    do not overlap (weekly dates overlap ~h/5 deep); below that the output is descriptive only
  - per-reason breakdown: did the names passed for a given reason go on to lag? That tests each
    gate as research actually applies it, not as the screener codes it

Run: venv/bin/python research/score_research_log.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "Start Your Own" / "research_log.csv"
HORIZONS = (20, 40, 60)
MIN_INDEPENDENT = 5


def independent_dates(dates: list[pd.Timestamp], h: int, sessions: pd.DatetimeIndex) -> int:
    """Greedy count of dates whose h-session forward windows do not overlap."""
    n, last = 0, None
    for d in sorted(dates):
        i = sessions.searchsorted(d)
        if last is None or i - last >= h:
            n, last = n + 1, i
    return n


def main() -> None:
    if not LOG.exists():
        sys.exit(f"{LOG} does not exist yet -- nothing has been logged.")
    log = pd.read_csv(LOG)
    if log.empty:
        sys.exit("research log is empty.")
    log["date"] = pd.to_datetime(log["date"])
    log["ticker"] = log["ticker"].str.upper().str.strip()
    print(f"{len(log)} rows, {log['date'].nunique()} research dates, "
          f"{(log['decision'] == 'BUY').sum()} BUY / {(log['decision'] == 'PASS').sum()} PASS / "
          f"{(log['decision'] == 'WATCH').sum()} WATCH")

    start = (log["date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    px = yf.download(sorted(set(log["ticker"])) + ["SPY"], start=start, auto_adjust=True,
                     progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    sessions = px.index

    def fwd(t: str, d: pd.Timestamp, h: int) -> float:
        i = sessions.searchsorted(d, side="right") - 1          # last close on/before the date
        if i < 0 or i + h >= len(sessions) or t not in px:
            return np.nan
        a, b = px[t].iloc[i], px[t].iloc[i + h]
        return b / a - 1 if pd.notna(a) and pd.notna(b) and a > 0 else np.nan

    for h in HORIZONS:
        log[f"r{h}"] = [fwd(t, d, h) for t, d in zip(log["ticker"], log["date"])]
        log[f"x{h}"] = log[f"r{h}"] - [fwd("SPY", d, h) for d in log["date"]]

    for h in HORIZONS:
        per_date = []
        for d, g in log.dropna(subset=[f"r{h}"]).groupby("date"):
            b, p = g[g["decision"] == "BUY"][f"r{h}"], g[g["decision"] == "PASS"][f"r{h}"]
            if len(b) and len(p):
                per_date.append({"date": d, "mean_diff": b.mean() - p.mean(),
                                 "median_diff": b.median() - p.median()})
        pdd = pd.DataFrame(per_date)
        print(f"\n=== {h} sessions ===")
        if pdd.empty:
            print("  no research date yet has both BUY and PASS names with forward data")
            continue
        ind = independent_dates(list(pdd["date"]), h, sessions)
        tag = "" if ind >= MIN_INDEPENDENT else f"  -> DESCRIPTIVE ONLY ({ind} < {MIN_INDEPENDENT})"
        print(f"  dates with both BUY and PASS: {len(pdd)};  independent at this horizon: {ind}{tag}")
        print(f"  BUY minus PASS, mean vs mean:     {pdd['mean_diff'].mean() * 100:+.2f}pp "
              f"(BUY ahead on {(pdd['mean_diff'] > 0).mean():.0%} of dates)")
        print(f"  BUY minus PASS, median vs median: {pdd['median_diff'].mean() * 100:+.2f}pp "
              f"(BUY ahead on {(pdd['median_diff'] > 0).mean():.0%} of dates)")

        passed = log[(log["decision"] == "PASS")].dropna(subset=[f"x{h}"])
        if not passed.empty:
            by = passed.groupby("reason_code")[f"x{h}"].agg(["count", "mean", "median"])
            by[["mean", "median"]] = (by[["mean", "median"]] * 100).round(2)
            print("  PASSED names vs the S&P, by reason (negative = the pass was right):")
            print("    " + by.rename(columns={"mean": "mean pp", "median": "median pp"})
                  .to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
