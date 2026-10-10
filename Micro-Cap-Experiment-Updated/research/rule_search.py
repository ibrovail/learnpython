"""Search the rule space for a configuration that beats SPY buy-and-hold -- honestly.

The goal set for this session was to find rules beating SPY's 15.6% CAGR over 2016-2026. The
danger in that instruction is obvious and has to be designed around: searching a large space on
one sample until something clears a target WILL produce a winner, and it will usually be noise.
This project has already been burnt by that (Phase 3.5 printed t-statistics of 15.3 from a broken
variance estimate), which is why pre-registration exists here at all.

So the search is split:
  * TRAIN 2016-01 .. 2021-06  -- every configuration is ranked here and only here
  * TEST  2021-06 .. 2026-09  -- the top few are then run on data the search never saw

A configuration is only reported as beating SPY if it clears SPY's CAGR in BOTH periods. The
number of configurations tried is printed, because with N tries the best train result is biased
upward by roughly the spread of the N draws and that bias has to be visible.

    venv/bin/python research/rule_search.py
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import live_simulation as L  # noqa: E402

SPLIT = pd.Timestamp("2021-06-30")


def spy_cagr(a, b):
    import yfinance as yf
    h = yf.Ticker("SPY").history(start=a.strftime("%Y-%m-%d"), end=b.strftime("%Y-%m-%d"),
                                 auto_adjust=True)["Close"]
    yrs = (h.index[-1] - h.index[0]).days / 365.25
    return (h.iloc[-1] / h.iloc[0]) ** (1 / yrs) - 1


def main():
    panel = pd.read_csv(ROOT / "output/regime_backtest_panel_survivors.csv",
                        parse_dates=["date"], low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                                "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)
    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
    px = L.load_prices(set(panel.ticker.unique()), cal)
    tr_cal, te_cal = cal[cal < SPLIT], cal[cal >= SPLIT]
    tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]

    spy_tr = spy_cagr(tr_cal[0], tr_cal[-1])
    spy_te = spy_cagr(te_cal[0], te_cal[-1])
    spy_all = spy_cagr(cal[0], cal[-1])
    print(f"SPY buy-and-hold:  train {spy_tr:.1%}   test {spy_te:.1%}   full {spy_all:.1%}\n",
          flush=True)

    grid = dict(
        max_hold=[20, 40, 60],
        ceiling=[5, 10, 20],
        cash_floor=[0.0, 0.15],
        rank_by=["composite", "mom20", "pct_vs_sma50", "low_vol"],
        rules=["live", "none"],
    )
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"searching {len(combos)} configurations on TRAIN only\n", flush=True)

    rows = []
    for n, c in enumerate(combos, 1):
        kw = dict(zip(keys, c))
        rules = kw.pop("rules")
        kw["use_regime"] = rules == "live"
        kw["calm"] = 0.90 if rules == "live" else None
        kw["raise_policy"] = "never"
        try:
            r = L.run(tr_p, px, tr_cal, capital=10_000.0, **kw)
        except Exception as e:
            continue
        rows.append(dict(**dict(zip(keys[:-1], c[:-1])), rules=rules,
                         train_cagr=r["cagr"], train_dd=r["max_drawdown"],
                         train_sharpe=r["sharpe_like"], trades=r["trades"]))
        if n % 24 == 0:
            print(f"  {n}/{len(combos)}...", flush=True)

    res = pd.DataFrame(rows).sort_values("train_cagr", ascending=False)
    res.to_csv(ROOT / "output/rule_search_train.csv", index=False)
    print(f"\nTOP 10 ON TRAIN (SPY train = {spy_tr:.1%}):", flush=True)
    print(res.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    print(f"\n=== now the only test that counts: top 6 on UNSEEN 2021-2026 data ===", flush=True)
    print(f"{'config':<56}{'train':>8}{'TEST':>8}{'beats SPY both?':>18}", flush=True)
    out = []
    for _, x in res.head(6).iterrows():
        kw = dict(max_hold=int(x.max_hold), ceiling=int(x.ceiling), cash_floor=float(x.cash_floor),
                  rank_by=x.rank_by, raise_policy="never",
                  use_regime=x.rules == "live", calm=0.90 if x.rules == "live" else None)
        r = L.run(te_p, px, te_cal, capital=10_000.0, **kw)
        both = x.train_cagr > spy_tr and r["cagr"] > spy_te
        lbl = (f"hold{int(x.max_hold)} n{int(x.ceiling)} floor{x.cash_floor:.0%} "
               f"{x.rank_by} {x.rules}")
        print(f"{lbl:<56}{x.train_cagr:>7.1%}{r['cagr']:>8.1%}"
              f"{('YES' if both else 'no'):>18}", flush=True)
        out.append(dict(config=lbl, train_cagr=x.train_cagr, test_cagr=r["cagr"],
                        test_dd=r["max_drawdown"], test_sharpe=r["sharpe_like"],
                        beats_spy_both=both))
    pd.DataFrame(out).to_csv(ROOT / "output/rule_search_test.csv", index=False)
    winners = [o for o in out if o["beats_spy_both"]]
    print(f"\n  configurations clearing SPY in BOTH periods: {len(winners)} of {len(out)} tested "
          f"(from {len(combos)} searched)", flush=True)
    if not winners:
        print("  -> nothing survived out of sample. The honest answer is that this rule family,"
              "\n     on this universe, did not beat passive SPY exposure.", flush=True)


if __name__ == "__main__":
    main()
