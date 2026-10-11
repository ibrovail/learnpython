"""The post-earnings cooldown: the last testable rule in the book. Pre-registered here.

THE RULE (`.claude/rules/entry-discipline.md`): never buy within 3 trading days of an earnings
print at a price more than +5% above the post-print close. Both conditions -- a name that has
come back down to its post-print close is not blocked.

ITS ONLY EVIDENCE is one trade. ARLO, Week 34: entered $15.25 on the Monday three sessions after
a Wednesday print, post-print close $15.32, gave back -10.9% on entry day. One case, and the
entry was not even above the post-print close, so the rule as written would not have blocked it.
That is reason enough to measure it.

DECISION RULES, fixed before the first result
C1  INERT if the cooldown blocks under 2% of candidate rows. Record it as a stated discipline
    that cannot be shown to do anything, as the re-entry ban was, and stop.
C2  DROP if it costs test CAGR and does not improve test drawdown. A rule that costs return and
    gives no risk reduction has no remaining justification. (Same rule as S2 for the sector cap.)
C3  KEEP if it improves test drawdown by >= 1.0pp at a test CAGR cost <= 1.0pp, OR improves test
    CAGR at no cost in drawdown.
C4  The result must BEAT A PLACEBO that refuses the same share of candidates at random, 20
    seeds, on whichever metric C3 was satisfied on -- at the 75th percentile or better. This is
    what the sector cap and the earnings guard were both held to, and the earnings guard's
    return claim died here while its drawdown claim survived.
C5  DOSE-RESPONSE: the parameter sweep must not contradict the verdict. A rule that works at
    3 sessions / +5% but is harmful at 5 / +5% and at 3 / +10% is a lucky point, not a rule.

CAVEAT, stated up front: the EDGAR filing date is the release session or the one after it, so a
3-session window carries roughly one session of slack. That matters more for a 3-session rule
than it did for the 10-session guard, and it is the main reason the sweep below is wider than
the rule.

    venv/bin/python research/cooldown_test.py
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
# The rule set adopted 2026-10-11, earnings guard included.
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

    print("=== A. THE RULE, AND THE SWEEP AROUND IT (C5) ===")
    print(f"{'cooldown':<26}{'train':>8}{'TEST':>8}{'test DD':>10}{'trades':>8}{'blocked':>9}")
    rows, base, rule = [], None, None
    grid = [(0, 0.05), (3, 0.05), (3, 0.00), (3, 0.10), (5, 0.05), (10, 0.05), (3, 0.20)]
    for ses, pct in grid:
        a, b = sim(cooldown_sessions=ses, cooldown_pct=pct)
        lbl = "off" if ses == 0 else f"{ses} sessions / +{pct:.0%}"
        br = b["cooldown_block_rate"]
        print(f"{lbl:<26}{a['cagr']:>7.1%}{b['cagr']:>8.1%}{b['max_drawdown']:>10.1%}"
              f"{b['trades']:>8}{br:>9.1%}")
        r = dict(test="sweep", sessions=ses, pct=pct, train=a["cagr"], test_cagr=b["cagr"],
                 test_dd=b["max_drawdown"], trades=b["trades"], block_rate=br)
        rows.append(r)
        if ses == 0:
            base = r
        if (ses, pct) == (3, 0.05):
            rule = r
    d_ret = rule["test_cagr"] - base["test_cagr"]
    d_dd = rule["test_dd"] - base["test_dd"]
    print(f"\n  the rule as written vs off: test CAGR {d_ret*100:+.1f}pp, "
          f"drawdown {d_dd*100:+.1f}pp (positive = shallower)")
    print(f"  it blocks {rule['block_rate']:.1%} of candidate rows")

    print(f"\n=== B. PLACEBO (C4) -- refuse {rule['block_rate']:.1%} at random, 20 seeds ===")
    pl = []
    for seed in range(20):
        a, b = sim(block_rate=rule["block_rate"], block_seed=seed)
        pl.append(dict(test="placebo", seed=seed, train=a["cagr"], test_cagr=b["cagr"],
                       test_dd=b["max_drawdown"], trades=b["trades"]))
    pl = pd.DataFrame(pl)
    print(f"{'':<26}{'train':>8}{'TEST':>8}{'test DD':>10}")
    print(f"{'cooldown off':<26}{base['train']:>7.1%}{base['test_cagr']:>8.1%}"
          f"{base['test_dd']:>10.1%}")
    print(f"{'rule as written':<26}{rule['train']:>7.1%}{rule['test_cagr']:>8.1%}"
          f"{rule['test_dd']:>10.1%}")
    print(f"{'placebo mean':<26}{pl.train.mean():>7.1%}{pl.test_cagr.mean():>8.1%}"
          f"{pl.test_dd.mean():>10.1%}")
    print(f"{'placebo best':<26}{pl.train.max():>7.1%}{pl.test_cagr.max():>8.1%}"
          f"{pl.test_dd.max():>10.1%}")
    beat_ret = float((pl.test_cagr < rule["test_cagr"]).mean())
    beat_dd = float((pl.test_dd < rule["test_dd"]).mean())
    print(f"\n  rule beats {beat_ret:.0%} of placebos on test return, {beat_dd:.0%} on drawdown")
    pd.concat([pd.DataFrame(rows), pl]).to_csv(OUT / "cooldown_results.csv", index=False)
    print(f"\nwrote {OUT/'cooldown_results.csv'} -- commit before interpreting (R15)")

    print("\n" + "=" * 78)
    print("=== VERDICT (C1-C5 applied as written) ===")
    print("=" * 78)
    if rule["block_rate"] < 0.02:
        print(f"C1 -> INERT. It blocks only {rule['block_rate']:.1%} of candidate rows.")
    elif d_ret < 0 and d_dd <= 0:
        print("C2 -> DROP. It costs test return and does not reduce drawdown.")
    elif (d_dd >= 0.01 and d_ret >= -0.01) or (d_ret > 0 and d_dd >= 0):
        print("C3 -> KEEP, subject to C4 and C5 below.")
        metric, beat = (("drawdown", beat_dd) if d_dd >= 0.01 else ("return", beat_ret))
        print(f"C4 -> it beats {beat:.0%} of placebos on {metric}; "
              f"{'PASSES' if beat >= 0.75 else 'FAILS'} the 75% bar.")
    else:
        print("neither C2 nor C3 fires -> unclear on the pre-registered thresholds.")
    sw = pd.DataFrame(rows).query("sessions > 0")
    agree = int(((sw.test_cagr - base["test_cagr"]) > 0).sum())
    print(f"C5 -> {agree} of {len(sw)} sweep points improve test return; "
          f"{int(((sw.test_dd - base['test_dd']) > 0).sum())} improve drawdown.")


if __name__ == "__main__":
    main()
