"""Is "2% risk per trade" a real rule, or does the 30% ceiling make it nominal?

THE FINDING, handed over by the live session on 2026-10-12 and corrected here. It observed that
TREX could not be sized at the full 2% risk because its widest stop candidate was 6.1% and the
book needed 6.67%. Two of its claims do not survive arithmetic:

  - It called this a MINIMUM STOP WIDTH that excludes names. It is not an exclusion.
    portfolio_rules.md already says actual size is the MINIMUM of the risk figure, the 30%
    ceiling and available cash. A tight-stop name is buildable at every width; below 6.67% the
    ceiling simply binds first and the book deploys LESS than 2% risk.
  - It said the constraint relaxes as equity grows. It does not. The threshold is
    risk_per_trade / name_cap = 0.02 / 0.30 = 6.67%. Both terms scale with equity, so the ratio
    is constant from $710 to $100,000. Book size cancels out entirely.

What survives is larger than what was claimed. At a 1.75xATR stop the full 2% risk needs
ATR >= 3.81% of price, and on the live watchlist 45 of 50 names (90%) sit below that; the median
ATR is 2.71%, implying ~1.0% of equity at risk rather than 2%. The screener is built to favour
low volatility -- it is one of the six composite inputs and the one with the highest rank IC --
so THE BOOK SYSTEMATICALLY UNDER-RISKS EXACTLY THE NAMES IT IS DESIGNED TO FIND.

That matters because the 10/11 ablation already priced this: cutting risk per trade to 1% cost
5.4pp of test return. If the ceiling silently holds most entries near 1%, the book may already be
paying that penalty without anyone having chosen it.

DECISION RULES, fixed before the first result
N1  Report the share of entries where the CEILING bound rather than the risk budget. If it binds
    on most entries at the current 30%, then "2% risk per trade" is nominal, and that is a
    finding whatever the return numbers say.
N2  A higher ceiling is adopted only if it improves test CAGR AND does not worsen test max
    drawdown by more than 2.0pp -- the same bar the parking test failed. The drawdown advantage is
    the only verified edge this book has.
N3  Test a LOWER risk budget at the current ceiling too (1.5%/30%). If the problem is the ratio
    rather than the ceiling, cutting risk to make the budget bind should be strictly worse, which
    distinguishes "the ceiling is too tight" from "the two constants disagree".
N4  Concentration is the risk being taken here: a 50% ceiling means two names can be the whole
    book. Report max drawdown and say so plainly rather than reading CAGR alone.

    venv/bin/python research/name_cap_test.py
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
               cash_floor=0.05, risk_per_trade=0.02, day1_rule=True, breaker=True,
               reentry_ban=10, sector_cap=3, max_hold=60, rank_by="pct_vs_sma50",
               slippage_bps=25, earnings_guard=10)


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

    print("=== DOES THE 30% CEILING MAKE 2% RISK NOMINAL? ===")
    print(f"{'risk / ceiling':<22}{'min stop':>10}{'train':>8}{'TEST':>8}{'d TEST':>9}"
          f"{'test DD':>10}{'d DD':>8}{'ceiling bound':>15}")
    rows, base = [], None
    grid = [(0.02, 0.30), (0.02, 0.40), (0.02, 0.50), (0.02, 1.00), (0.015, 0.30), (0.03, 0.30)]
    for risk, cap in grid:
        a, b = sim(risk_per_trade=risk, name_cap=cap)
        lbl = f"{risk:.1%} / {cap:.0%}"
        r = dict(risk=risk, name_cap=cap, min_stop=risk/cap, train=a["cagr"], test=b["cagr"],
                 test_dd=b["max_drawdown"], cap_bound=b["cap_bound_share"], trades=b["trades"])
        if base is None:
            base = r; dt = dd = ""
        else:
            r["d_test"] = b["cagr"]-base["test"]; r["d_dd"] = b["max_drawdown"]-base["test_dd"]
            dt, dd = f"{r['d_test']*100:+.1f}pp", f"{r['d_dd']*100:+.1f}pp"
        print(f"{lbl:<22}{risk/cap:>9.2%}{a['cagr']:>8.1%}{b['cagr']:>8.1%}{dt:>9}"
              f"{b['max_drawdown']:>10.1%}{dd:>8}{b['cap_bound_share']:>14.0%}")
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "name_cap_results.csv", index=False)
    print(f"\nwrote {OUT/'name_cap_results.csv'} -- commit before interpreting (R15)")
    b0 = rows[0]
    print(f"\nN1 -> at the live 2%/30%, the CEILING bound {b0['cap_bound']:.0%} of entries. "
          f"{'2% risk per trade is NOMINAL.' if b0['cap_bound'] > 0.5 else 'The risk budget mostly binds.'}")


if __name__ == "__main__":
    main()
