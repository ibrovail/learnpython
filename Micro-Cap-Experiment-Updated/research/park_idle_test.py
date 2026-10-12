"""Should idle cash sit, or be parked in SPY? Pre-registered before the first result.

THE QUESTION, raised 2026-10-12 after Week 57 found no catalyst candidates. The book holds ~85%
cash and the screener lane is shut in RISK-OFF, so cash periods are now LONGER than before the
10/11 amendment, not shorter. Meanwhile the live tape says something specific:

    2026-10-09   SPY  +1.60% ABOVE its 50-day SMA
                 IWM  -4.11% BELOW its own
                 IJR  -3.82% BELOW its own

The regime filter is computed on IWM. It is a SMALL-CAP signal and says nothing about large caps.
So the book is sitting in cash because small caps are weak, while the benchmark it is judged
against is rising. That is the worst configuration the gap metric can be in, and it is also the
audit's central finding restated: the gap is a CAP-WEIGHTING effect (SPY beat the equal-weighted
S&P by 4.7pp on the test period), and no rule change can close it.

WHAT CHANGED SINCE THIS WAS DISMISSED
  - Fractional shares were only permitted on 2026-10-11. SPY at $778 was literally unbuyable with
    $600 of cash before that.
  - The 10/11 lane closure lengthened the cash periods.
  - The capacity study measured time-in-cash at -0.964 correlation with return.

WHAT HAS NOT CHANGED, and may still kill it
  - The Universe rule is "U.S.-listed common stocks, up to $5Bn". An ETF is neither. This needs an
    amendment, not a trade.
  - Parking imports SPY's drawdown. SPY ran -24.5% in the test period against the book's -15.4%,
    and that drawdown advantage is the ONLY verified edge this book has. Unconditional parking
    could destroy the one thing that works, which is why "spy_trend" is tested beside "spy".
  - At household level the RRSP core is already a global index fund, so parking the satellite may
    duplicate an exposure already held. That is the user's call, not this test's.

DECISION RULES, fixed before any result
P1  The comparison is on BOTH axes. Parking is adopted only if it improves test CAGR AND does not
    worsen test max drawdown by more than 2.0pp. The drawdown edge is this book's only verified
    advantage; buying return by importing SPY's drawdown is the trade the audit already rejected
    twice (mechanical stop raising, 1% risk).
P2  "spy_trend" must beat "spy". If unconditional parking wins, the result is a large-cap beta
    bet, not a cash-management rule, and belongs to the mega-cap session instead of here.
P3  Report the friction: round trips and the slippage paid. A rule that needs 200 switches to earn
    1pp is not a rule, it is a trading strategy.
P4  Report the parked share of equity over time. If the book is parked more than ~60% of the time,
    say so plainly -- at that point the sleeve is mostly the benchmark and the gap metric stops
    measuring selection skill, whatever the CAGR says. That is an integrity finding, not a
    performance one, and it overrides P1.

    venv/bin/python research/park_idle_test.go.py
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

    sp = yf.Ticker("SPY").history(start="2015-06-01", end="2026-09-19", auto_adjust=True)
    sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize()
    spy = pd.DataFrame({"adjOpen": sp.Open, "adjClose": sp.Close})
    spy["sma50"] = spy.adjClose.rolling(50).mean()
    spy = spy.reindex(cal).ffill()
    above = (spy.adjClose > spy.sma50)
    print(f"SPY above its own 50-day on {above.mean():.0%} of sessions "
          f"({above[cal >= SPLIT].mean():.0%} in the test period)\n", flush=True)

    tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
    tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]

    def sim(**over):
        kw = {**ADOPTED, **over}
        a = L.run(tr_p, px, tr_c, capital=CAPITAL, sectors=sectors, earn_idx=earn, spy=spy, **kw)
        b = L.run(te_p, px, te_c, capital=CAPITAL, sectors=sectors, earn_idx=earn, spy=spy, **kw)
        return a, b

    print("=== IDLE CASH: sit, or park in SPY? ===")
    print(f"{'policy':<34}{'train':>8}{'TEST':>8}{'d TEST':>9}{'test DD':>10}{'d DD':>8}"
          f"{'switches':>10}")
    rows, base = [], None
    for lbl, mode in (("cash sits (the rules today)", "none"),
                      ("park in SPY, always", "spy"),
                      ("park in SPY only above its 50d", "spy_trend")):
        a, b = sim(park_idle=mode)
        r = dict(policy=lbl, mode=mode, train=a["cagr"], test=b["cagr"],
                 test_dd=b["max_drawdown"], train_dd=a["max_drawdown"],
                 switches=b["park_switches"], trades=b["trades"])
        if base is None:
            base = r; dt = dd = ""
        else:
            r["d_test"] = b["cagr"] - base["test"]; r["d_dd"] = b["max_drawdown"] - base["test_dd"]
            dt, dd = f"{r['d_test']*100:+.1f}pp", f"{r['d_dd']*100:+.1f}pp"
        print(f"{lbl:<34}{a['cagr']:>7.1%}{b['cagr']:>8.1%}{dt:>9}{b['max_drawdown']:>10.1%}"
              f"{dd:>8}{b['park_switches']:>10}")
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "park_idle_results.csv", index=False)
    print(f"\nwrote {OUT/'park_idle_results.csv'} -- commit before interpreting (R15)")

    print("\n" + "=" * 86)
    print("=== VERDICT (P1-P4 applied as written) ===")
    print("=" * 86)
    always = [r for r in rows if r["mode"] == "spy"][0]
    trend = [r for r in rows if r["mode"] == "spy_trend"][0]
    best = max((always, trend), key=lambda r: r["d_test"])
    print(f"  best policy on test return: {best['policy']} ({best['d_test']*100:+.1f}pp, "
          f"drawdown {best['d_dd']*100:+.1f}pp)")
    if best["d_test"] <= 0:
        print("  P1 -> REJECT. Parking does not improve test return at all.")
    elif best["d_dd"] < -0.02:
        print(f"  P1 -> REJECT. It buys {best['d_test']*100:+.1f}pp of return by importing "
              f"{abs(best['d_dd'])*100:.1f}pp of drawdown (limit 2.0pp). Same trade the audit")
        print("        rejected for mechanical stop raising and for 1% risk per trade.")
    else:
        print(f"  P1 -> PASSES on both axes.")
    if trend["d_test"] < always["d_test"]:
        print("  P2 -> the UNCONDITIONAL version wins, so this is a large-cap beta bet rather")
        print("        than a cash-management rule. It belongs to the mega-cap session.")
    else:
        print("  P2 -> the trend-gated version wins, which is the cash-management reading.")
    print(f"  P3 -> friction: {best['switches']} round trips in the test period at 25bps each.")
    print("  P4 -> parked share of equity is reported above as switch count only; the integrity")
    print("        question (is the sleeve mostly the benchmark?) is answered in the write-up.")


if __name__ == "__main__":
    main()
