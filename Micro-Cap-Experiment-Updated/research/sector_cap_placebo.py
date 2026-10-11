"""40-seed placebo for the sector cap. 12 seeds resolve to 1/12 = 8%, too coarse to read a
percentile off. Same construction, more draws, and the real-label result is read from the
committed results file rather than recomputed."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "research")
import live_simulation as L, yfinance as yf

SPLIT = pd.Timestamp("2021-06-30"); CAPITAL = 710.0
BASE = dict(rank_by="pct_vs_sma50", max_hold=60, cash_floor=0.05, raise_policy="never",
            slippage_bps=25, use_regime=True, calm=None, fractional=True,
            risk_per_trade=0.02, ceiling=10, day1_rule=True)
lab = pd.read_csv("research/output/sector_labels.csv")
sectors = lab.set_index("ticker").sector.to_dict()
panel = pd.read_csv("research/output/regime_backtest_panel_survivors.csv",
                    parse_dates=["date"], low_memory=False)
panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                            "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)
iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
px = L.load_prices(set(panel.ticker.unique()), cal)
tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]

res = pd.read_csv("research/output/sector_cap_results.csv")
real = res.query("variant.str.startswith('full') and cap == 3").iloc[0]
tk, sc = list(sectors), list(sectors.values())
rows = []
for seed in range(40):
    rng = np.random.default_rng(1000 + seed)
    shuf = dict(zip(tk, rng.permutation(sc)))
    kw = {**BASE, "sectors": shuf, "sector_cap": 3}
    a = L.run(tr_p, px, tr_c, capital=CAPITAL, **kw)
    b = L.run(te_p, px, te_c, capital=CAPITAL, **kw)
    rows.append(dict(seed=seed, train=a["cagr"], test=b["cagr"],
                     test_dd=b["max_drawdown"], train_dd=a["max_drawdown"]))
    if (seed + 1) % 10 == 0:
        print(f"  {seed+1}/40 seeds", flush=True)
p = pd.DataFrame(rows)
p.to_csv("research/output/sector_cap_placebo40.csv", index=False)
print(f"\nreal labels : train {real.train:.1%}  test {real.test:.1%}  test DD {real.test_dd:.1%}")
print(f"placebo mean: train {p.train.mean():.1%}  test {p.test.mean():.1%}  "
      f"test DD {p.test_dd.mean():.1%}")
print(f"placebo sd  : {p.test.std():.1%} on test return, {p.test_dd.std():.1%} on test DD")
print(f"placebo 90th pct test DD {p.test_dd.quantile(0.90):.1%} | "
      f"best of 40 {p.test_dd.max():.1%}")
print(f"\nreal beats {(p.test_dd < real.test_dd).mean():.0%} of 40 placebos on test drawdown")
print(f"real beats {(p.test < real.test).mean():.0%} of 40 placebos on test return")
print(f"empirical one-sided p on drawdown: {(p.test_dd >= real.test_dd).mean():.3f}")
print(f"empirical one-sided p on return  : {(p.test >= real.test).mean():.3f}")
