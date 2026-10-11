"""One clean ablation: every adopted change measured against the SAME final rule set.

Why this exists. The audit tested rules as it went, so each effect (+2.2pp regime, -5.8pp
defensive profile, +2.8pp never-raise, +2.7pp fractional/10, 3.5pp earnings guard) was measured
against whatever baseline was in force at the time. Rule effects interact -- the earnings guard
test proved it, since the guard only looked useful while stops were being raised mechanically --
so those figures are directionally sound but NOT additive, and an amendment that cites them side
by side would be quoting numbers from different worlds.

So: fix one baseline, flip one rule at a time back to what portfolio_rules.md says today, and
report the difference. Every number in the amendment comes from this file.

TWO REFERENCE CONFIGURATIONS
  STATUS QUO -- the rules as written today: RISK-ON buys freely and RISK-OFF buys only names
    passing the defensive profile, stops raised whenever eligible, whole shares, 5 positions,
    15% cash floor, 10-session pre-earnings guard, circuit breaker and re-entry ban live.
  ADOPTED -- the audit's result: regime gate closes the screener lane in RISK-OFF, no defensive
    profile, stops never raised, fractional shares, 10 positions, 5% cash floor, no earnings
    guard, sector cap 3.

WHAT THIS CANNOT MODEL, stated before the output
  - The catalyst lane. Under the adopted set RISK-OFF closes the SCREENER lane; the catalyst
    lane stays open at full risk. History carries no catalyst labels, so the adopted column is
    the pessimistic case -- it shows RISK-OFF as fully closed.
  - The $5Bn market-cap ceiling (tested separately; lifting it was worse).
  - The driver cap and the binary-thesis prohibition, which are human labels.
  - rank_by is NOT a rule. The book picks by research, not by score; it appears here as a
    diagnostic row only, because the simulation has to pick somehow.

SCALE MATTERS. Fractional shares and a 10-position ceiling are worth nothing to a $710 book
that cannot buy 10 positions. Every row is reported at the live book's $710 AND at $10,000.

    venv/bin/python research/rule_ablation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, DATA = ROOT / "output", ROOT / "data.nosync"
sys.path.insert(0, str(ROOT))
import live_simulation as L  # noqa: E402

SPLIT = pd.Timestamp("2021-06-30")
CAPITALS = (710.0, 10_000.0)

ADOPTED = dict(use_regime=True, calm=None, calm_when="all", raise_policy="never",
               fractional=True, ceiling=10, cash_floor=0.05, risk_per_trade=0.02,
               earnings_guard=0, day1_rule=True, breaker=False, reentry_ban=0,
               sector_cap=3, max_hold=60, rank_by="pct_vs_sma50", slippage_bps=25)

STATUS_QUO = dict(ADOPTED, use_regime=False, calm=0.90, calm_when="risk_off",
                  raise_policy="mechanical", fractional=False, ceiling=5, cash_floor=0.15,
                  earnings_guard=10, breaker=True, reentry_ban=10, rank_by="composite")

# Each flip returns ONE adopted choice to what the rules say today. The label names the rule.
FLIPS = [
    ("regime: RISK-OFF opens on the defensive profile",
     dict(use_regime=False, calm=0.90, calm_when="risk_off")),
    ("regime: no gate at all, no profile", dict(use_regime=False, calm=None)),
    ("stops: raise whenever eligible", dict(raise_policy="mechanical")),
    ("stops: raise once only", dict(raise_policy="once")),
    ("shares: whole only", dict(fractional=False)),
    ("positions: ceiling 5", dict(ceiling=5)),
    ("shares+positions: whole AND 5", dict(fractional=False, ceiling=5)),
    ("cash floor: 15%", dict(cash_floor=0.15)),
    ("earnings guard: 10 sessions", dict(earnings_guard=10)),
    ("sector cap: off", dict(sector_cap=0)),
    ("day-1 rule: off", dict(day1_rule=False)),
    ("risk per trade: 1%", dict(risk_per_trade=0.01)),
    ("circuit breaker: on", dict(breaker=True)),
    ("re-entry ban: 10 sessions", dict(reentry_ban=10)),
    ("[diagnostic, not a rule] rank by composite", dict(rank_by="composite")),
]


def main() -> None:
    lab = pd.read_csv(OUT / "sector_labels.csv")
    sectors = lab.set_index("ticker").sector.to_dict()
    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv", parse_dates=["date"],
                        low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                                "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)

    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
    px = L.load_prices(set(panel.ticker.unique()), cal)
    tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
    tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]

    spy = yf.Ticker("SPY").history(start="2015-08-01", end="2026-09-19",
                                   auto_adjust=True)["Close"]
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    bar = {}
    for lbl, c in (("train", tr_c), ("test", te_c)):
        s = spy[(spy.index >= c[0]) & (spy.index <= c[-1])]
        y = (s.index[-1] - s.index[0]).days / 365.25
        bar[lbl] = (s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1
    print(f"SPY buy-and-hold: train {bar['train']:.1%}  test {bar['test']:.1%}  "
          f"(the scoreboard)\n", flush=True)

    def sim(kw, cap):
        a = L.run(tr_p, px, tr_c, capital=cap, sectors=sectors, **kw)
        b = L.run(te_p, px, te_c, capital=cap, sectors=sectors, **kw)
        return a, b

    rows = []
    base = {}
    for cap in CAPITALS:
        a, b = sim(ADOPTED, cap)
        base[cap] = (a, b)
        rows.append(dict(capital=cap, config="ADOPTED SET (baseline)", train=a["cagr"],
                         test=b["cagr"], test_dd=b["max_drawdown"], trades=b["trades"],
                         stop_rate=b["stop_exits"] / max(b["trades"], 1),
                         avg_hold=b["avg_hold"], d_train=0.0, d_test=0.0, d_dd=0.0))
        a, b = sim(STATUS_QUO, cap)
        rows.append(dict(capital=cap, config="STATUS QUO (rules as written)", train=a["cagr"],
                         test=b["cagr"], test_dd=b["max_drawdown"], trades=b["trades"],
                         stop_rate=b["stop_exits"] / max(b["trades"], 1),
                         avg_hold=b["avg_hold"],
                         d_train=a["cagr"] - base[cap][0]["cagr"],
                         d_test=b["cagr"] - base[cap][1]["cagr"],
                         d_dd=b["max_drawdown"] - base[cap][1]["max_drawdown"]))
        for lbl, over in FLIPS:
            a, b = sim({**ADOPTED, **over}, cap)
            rows.append(dict(capital=cap, config=lbl, train=a["cagr"], test=b["cagr"],
                             test_dd=b["max_drawdown"], trades=b["trades"],
                             stop_rate=b["stop_exits"] / max(b["trades"], 1),
                             avg_hold=b["avg_hold"],
                             d_train=a["cagr"] - base[cap][0]["cagr"],
                             d_test=b["cagr"] - base[cap][1]["cagr"],
                             d_dd=b["max_drawdown"] - base[cap][1]["max_drawdown"]))
            print(f"  [{cap:,.0f}] {lbl}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rule_ablation.csv", index=False)

    for cap in CAPITALS:
        v = res[res.capital == cap]
        print(f"\n{'='*94}")
        print(f"=== FLIP EACH ADOPTED RULE BACK, AT ${cap:,.0f} "
              f"(negative delta = the adopted choice is better) ===")
        print(f"{'='*94}")
        print(f"{'configuration':<48}{'train':>7}{'TEST':>7}{'d TEST':>8}{'test DD':>9}"
              f"{'d DD':>7}{'trds':>6}{'stop%':>7}{'hold':>6}")
        for _, r in v.iterrows():
            dt = "" if r.config.startswith("ADOPTED") else f"{r.d_test:+.1%}"
            dd = "" if r.config.startswith("ADOPTED") else f"{r.d_dd:+.1%}"
            print(f"{r.config:<48}{r.train:>6.1%}{r.test:>7.1%}{dt:>8}{r.test_dd:>9.1%}"
                  f"{dd:>7}{r.trades:>6.0f}{r.stop_rate:>7.0%}{r.avg_hold:>6.0f}")
        b = v[v.config == "ADOPTED SET (baseline)"].iloc[0]
        print(f"\n  vs SPY on test: adopted {b.test:.1%} against {bar['test']:.1%} "
              f"= {b.test - bar['test']:+.1f}pp, at {b.test_dd:.1%} drawdown")
    print(f"\nwrote {OUT/'rule_ablation.csv'} -- commit before interpreting (R15)", flush=True)


if __name__ == "__main__":
    main()
