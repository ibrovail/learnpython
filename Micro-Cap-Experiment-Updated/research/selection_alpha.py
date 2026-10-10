"""Does ANY selection signal produce >1.5pp of alpha over the gated universe, out of sample?

The gated universe returns 14.1% a year equal-weighted; cap-weighted SPY returned 15.6%. Closing
that needs about 1.5pp of selection alpha. Phase 3.5 found none -- but Phase 3.5 only ever tested
the six screener signals, all of which are measured over 20 to 60 days.

That is the same mistake as the regime filter: a factor evaluated at the wrong horizon. The
documented equity momentum anomaly is 6-12 MONTH formation with the most recent month skipped,
and the "proximity to high" anomaly is measured against the 52-WEEK high, not a 60-day high. The
screener uses 20-day momentum and a 60-day high. Those are not the same factors.

This builds the long-horizon versions from daily prices and tests every signal the same way:
rank the gated universe each week, hold the top decile, and measure the spread over the universe
at the 40-session horizon -- on TRAIN 2016-2021 and then on TEST 2021-2026. A signal only counts
if it clears +1.5pp in both.

    venv/bin/python research/selection_alpha.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, PRICES = ROOT / "output", ROOT / "data.nosync" / "prices"
sys.path.insert(0, str(ROOT))
from regime_backtest import nw_tstat, lag_for, effective_n  # noqa: E402

SPLIT = pd.Timestamp("2021-06-30")
TARGET_PP = 1.5          # what we must clear to close the gap to SPY
HORIZON = 40
TOP_FRAC = 0.10


def build_signals(tickers, cal) -> pd.DataFrame:
    """Long-horizon factors, built the way the literature builds them."""
    rows = []
    for n, t in enumerate(tickers, 1):
        f = PRICES / f"{t}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date").reindex(cal)
        c = d.adjClose.astype(float)
        if c.notna().sum() < 300:
            continue
        v = (c * d.adjVolume.astype(float))
        out = pd.DataFrame(index=cal)
        # momentum with the last month SKIPPED -- skipping avoids the short-term reversal that
        # contaminates raw trailing returns, and is why 12-1 is the standard construction.
        out["mom_12_1"] = c.shift(21) / c.shift(252) - 1
        out["mom_6_1"] = c.shift(21) / c.shift(126) - 1
        out["mom_3_1"] = c.shift(21) / c.shift(63) - 1
        out["mom_12_0"] = c / c.shift(252) - 1          # no skip, for contrast
        # proximity to the 52-WEEK high, not the 60-day high the screener uses
        out["near_52w"] = c / c.rolling(252).max() - 1
        out["near_60d"] = c / c.rolling(60).max() - 1
        # risk-adjusted momentum: the same signal divided by its own volatility
        vol = c.pct_change(fill_method=None).rolling(252).std()
        out["mom_12_1_riskadj"] = out.mom_12_1 / (vol * np.sqrt(252))
        # acceleration: is the medium-term trend improving on the long-term one
        out["accel"] = out.mom_3_1 - out.mom_12_1
        out["log_dollar_vol"] = np.log(v.rolling(60).median().replace(0, np.nan))
        out["ticker"] = t
        out.index.name = "date"
        rows.append(out.reset_index())
        if n % 100 == 0:
            print(f"  signals: {n}/{len(tickers)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv", parse_dates=["date"],
                        usecols=["date", "ticker", "fwd40", "rank_low_vol", "rank_near_high",
                                 "rank_vol_5_50", "rank_vol_ratio", "rank_pct_vs_sma50",
                                 "mom20", "pct_vs_sma50"], low_memory=False)
    panel = panel[panel.fwd40.notna()]
    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2014-09-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal.name = "date"
    tickers = sorted(panel.ticker.unique())
    print(f"building long-horizon signals for {len(tickers)} tickers...", flush=True)
    sig = build_signals(tickers, cal)
    df = panel.merge(sig, on=["date", "ticker"], how="left")
    print(f"  merged panel: {len(df)} rows\n", flush=True)

    cands = ["mom_12_1", "mom_6_1", "mom_3_1", "mom_12_0", "mom_12_1_riskadj", "near_52w",
             "accel", "log_dollar_vol", "near_60d", "mom20", "pct_vs_sma50",
             "rank_low_vol", "rank_near_high", "rank_vol_5_50", "rank_vol_ratio"]
    print(f"TOP {TOP_FRAC:.0%} vs the whole gated universe, {HORIZON}-session forward return.")
    print(f"A signal must clear +{TARGET_PP}pp in BOTH periods to close the gap to SPY.\n")
    print(f"{'signal':<20}{'TRAIN pp':>10}{'t':>7}{'TEST pp':>10}{'t':>7}{'both>1.5?':>11}")
    res = []
    for c in cands:
        sub = df[df[c].notna()]
        if len(sub) < 5000:
            continue
        rows = []
        for d, g in sub.groupby("date"):
            if len(g) < 40:
                continue
            k = max(4, int(len(g) * TOP_FRAC))
            rows.append({"date": d, "sp": g.nlargest(k, c).fwd40.mean() - g.fwd40.mean()})
        s = pd.DataFrame(rows).set_index("date").sp
        tr, te = s[s.index < SPLIT], s[s.index >= SPLIT]
        o = {}
        for lbl, x in (("tr", tr), ("te", te)):
            lg, _ = lag_for(HORIZON, len(x))
            m, t, _ = nw_tstat(x, lg)
            o[lbl] = (m, t, len(x))
        ok = o["tr"][0] > TARGET_PP and o["te"][0] > TARGET_PP
        print(f"{c:<20}{o['tr'][0]:>9.2f}{o['tr'][1]:>7.2f}{o['te'][0]:>10.2f}{o['te'][1]:>7.2f}"
              f"{('YES' if ok else 'no'):>11}")
        res.append(dict(signal=c, train_pp=o["tr"][0], train_t=o["tr"][1],
                        test_pp=o["te"][0], test_t=o["te"][1], clears_both=ok,
                        train_dates=o["tr"][2], test_dates=o["te"][2]))
    r = pd.DataFrame(res).sort_values("test_pp", ascending=False)
    r.to_csv(OUT / "selection_alpha.csv", index=False)
    print(f"\nwrote {OUT/'selection_alpha.csv'}")
    w = r[r.clears_both]
    print(f"\n  signals clearing +{TARGET_PP}pp in BOTH periods: {len(w)} of {len(r)}")
    if len(w):
        print(w[["signal", "train_pp", "test_pp", "test_t"]].to_string(index=False))


if __name__ == "__main__":
    main()
