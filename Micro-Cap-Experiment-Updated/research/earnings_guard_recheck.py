"""The pre-earnings guard, re-measured. The first portfolio test of it was wrong.

WHAT WENT WRONG. live_simulation.run() counts sessions with `i`, an index into the calendar it
is handed. The first guard test pre-indexed earnings dates against the FULL calendar and then
ran the test period as a separate call, so every index was shifted by the length of the train
period. The guard still blocked candidates -- it just blocked them off train-period earnings
dates while trading the test period. It looked like it was working, and the "3.5pp cost" it
appeared to carry was the cost of refusing candidates at random. run() now takes DATES and
indexes them itself, so the drift cannot recur.

Re-measured on the final rule set, the guard is worth +1.5pp of test return AND 3.1pp less
drawdown. That reverses the verdict I gave, and it takes the adopted set above SPY out of
sample, so it gets checked harder rather than accepted.

TWO CHECKS, both pre-registered here
  A. LENGTH SWEEP -- 0/5/10/15/20 sessions. If 10 is a lucky point and its neighbours are bad,
     the result is noise. A plateau or a monotone run is credible.
  B. PLACEBO -- refuse the same SHARE of candidates at random, 20 seeds. If random refusal buys
     the same improvement, the guard is just "trade less" and the gain is not about earnings.
     This is the control that settled the sector cap, and it is the one that matters.

    venv/bin/python research/earnings_guard_recheck.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, DATA = ROOT / "output", ROOT / "data.nosync"
sys.path.insert(0, str(ROOT))
import live_simulation as L  # noqa: E402

SPLIT = pd.Timestamp("2021-06-30"); CAPITAL = 710.0
ADOPTED = dict(use_regime=True, calm=None, raise_policy="never", fractional=True, ceiling=10,
               cash_floor=0.05, risk_per_trade=0.02, day1_rule=True, breaker=False,
               reentry_ban=0, sector_cap=3, max_hold=60, rank_by="pct_vs_sma50",
               slippage_bps=25)


def main() -> None:
    lab = pd.read_csv(OUT / "sector_labels.csv")
    sectors = lab.set_index("ticker").sector.to_dict()
    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv", parse_dates=["date"],
                        low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                                "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)
    e = pd.read_csv(DATA / "earnings_dates.csv", parse_dates=["earnings_date"])
    earn = {tk: g.earnings_date.values for tk, g in e.groupby("ticker")}

    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
    px = L.load_prices(set(panel.ticker.unique()), cal)
    tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
    tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]

    def sim(**over):
        kw = {**ADOPTED, **over}
        a = L.run(tr_p, px, tr_c, capital=CAPITAL, sectors=sectors, earn_idx=earn, **kw)
        b = L.run(te_p, px, te_c, capital=CAPITAL, sectors=sectors, earn_idx=earn, **kw)
        return a, b

    print("=== A. LENGTH SWEEP on the final rule set ===")
    print(f"{'guard':<22}{'train':>8}{'TEST':>8}{'test DD':>10}{'trades':>8}"
          f"{'% cands blocked':>17}")
    rows, rate10 = [], 0.0
    for g in (0, 5, 10, 15, 20):
        a, b = sim(earnings_guard=g)
        br = b["guard_block_rate"]
        if g == 10:
            rate10 = br
        print(f"{(str(g) + ' sessions') if g else 'off':<22}{a['cagr']:>7.1%}{b['cagr']:>8.1%}"
              f"{b['max_drawdown']:>10.1%}{b['trades']:>8}{br:>17.0%}")
        rows.append(dict(test="length", guard=g, train=a["cagr"], test_cagr=b["cagr"],
                         test_dd=b["max_drawdown"], trades=b["trades"], block_rate=br))

    print(f"\n=== B. PLACEBO -- refuse {rate10:.0%} of candidates AT RANDOM, 20 seeds ===")
    real = [r for r in rows if r["guard"] == 10][0]
    pl = []
    for seed in range(20):
        a, b = sim(block_rate=rate10, block_seed=seed)
        pl.append(dict(test="placebo", seed=seed, train=a["cagr"], test_cagr=b["cagr"],
                       test_dd=b["max_drawdown"], trades=b["trades"]))
    pl = pd.DataFrame(pl)
    off = [r for r in rows if r["guard"] == 0][0]
    print(f"{'':<22}{'train':>8}{'TEST':>8}{'test DD':>10}")
    print(f"{'guard off':<22}{off['train']:>7.1%}{off['test_cagr']:>8.1%}"
          f"{off['test_dd']:>10.1%}")
    print(f"{'real guard (10)':<22}{real['train']:>7.1%}{real['test_cagr']:>8.1%}"
          f"{real['test_dd']:>10.1%}")
    print(f"{'placebo mean':<22}{pl.train.mean():>7.1%}{pl.test_cagr.mean():>8.1%}"
          f"{pl.test_dd.mean():>10.1%}")
    print(f"{'placebo best':<22}{pl.train.max():>7.1%}{pl.test_cagr.max():>8.1%}"
          f"{pl.test_dd.max():>10.1%}")
    p_ret = float((pl.test_cagr >= real["test_cagr"]).mean())
    p_dd = float((pl.test_dd >= real["test_dd"]).mean())
    print(f"\n  real guard beats {1 - p_ret:.0%} of placebos on test return, "
          f"{1 - p_dd:.0%} on test drawdown")
    print(f"  empirical one-sided p: return {p_ret:.3f}, drawdown {p_dd:.3f}")
    print("\n  VERDICT: the guard survives only if it beats the placebo on BOTH. Random refusal")
    print("  at the same rate is the whole alternative explanation -- a book that trades less")
    print("  in a falling market looks better for reasons that have nothing to do with earnings.")
    pd.concat([pd.DataFrame(rows), pl]).to_csv(OUT / "earnings_guard_recheck.csv", index=False)
    print(f"\nwrote {OUT/'earnings_guard_recheck.csv'}", flush=True)


if __name__ == "__main__":
    main()
