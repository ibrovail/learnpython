"""Does the 3-per-sector cap earn its place? Pre-registered before the labels were read.

portfolio_rules.md caps the book at 3 positions in any one GICS sector. It has never been tested,
and until now could not be: sector labels for a survivorship-free frame do not exist in any free
current-listing source. research/sector_labels.py builds them from SEC EDGAR SIC codes, which
survive delisting.

=== CAVEAT, stated before the result (R15) =====================================================
A SIC-derived sector is a PROXY for GICS. Section 1 measures the disagreement against the
screener's own Finviz sector labels on the overlapping names and prints the agreement rate, so
the proxy error is a number in the output rather than a disclaimer. Names EDGAR could not resolve
carry no label; under the primary variant an unlabelled name counts as its own sector and is
therefore never refused by the cap, which understates the cap's bite rather than inventing one.
Variant (b) re-runs on labelled names only, where no name can escape it.

=== DECISION RULES, fixed in advance ===========================================================
S1  INERT -- if the UNCAPPED run's largest same-sector holding count reaches 3 on under 5% of
    sessions, the cap cannot be doing material work. Record it as kept-but-inert, as the
    re-entry ban was, and stop: the remaining rules cannot separate it from no rule at all.
S2  The cap's stated purpose is correlated-risk control, not return, so it is judged on risk
    like the defensive profile was (Phase 3.8 R7): DROP if it costs test CAGR and does NOT
    improve test max drawdown. A rule that costs return and gives no risk reduction has no
    remaining justification.
S3  KEEP at 3 if it improves test max drawdown by >= 1.0pp at a test CAGR cost <= 1.0pp.
    (Sign note, corrected after the first run: drawdowns are negative, so "improves" means the
    difference is POSITIVE -- shallower. The first implementation tested <= -0.01 and so could
    only ever fire on a WORSE drawdown. The thresholds are unchanged; the comparison was wrong.)
S4  CHANGE the number only if another cap value beats 3 on BOTH train and test, on both CAGR
    and drawdown. One-sided improvement is not enough -- 160 configurations of this book have
    already shown how easily a single split rewards an arbitrary threshold.
S5  Both variants (full panel, labelled-only) must agree for any verdict other than "unclear".
S6  PLACEBO, added after the first run on an observation the pre-registration had not
    anticipated: a cap of FIVE, which never changes the largest same-sector count the book
    actually holds, improved test drawdown almost as much as a cap of three. If merely
    perturbing which names get picked improves drawdown, then the cap's apparent benefit is not
    correlated-risk control. So the cap is also run at 3 against SHUFFLED sector labels, 12
    seeds, preserving the sector size distribution. If the real labels sit inside the placebo
    distribution, the verdict is that the cap does nothing identifiable and the earlier
    S3 "KEEP" is an artifact. This is post hoc and labelled as such.

    venv/bin/python research/sector_cap_test.py
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
CAPITAL = 710.0        # the live book's actual equity
# The ADOPTED rule set from the audit: regime filter kept, defensive profile removed, stops never
# raised, fractional shares, 10 positions, 2% risk, 5% cash floor, no pre-earnings guard.
BASE = dict(rank_by="pct_vs_sma50", max_hold=60, cash_floor=0.05, raise_policy="never",
            slippage_bps=25, use_regime=True, calm=None, fractional=True,
            risk_per_trade=0.02, ceiling=10, day1_rule=True)

FINVIZ = {"Basic Materials": "Materials", "Consumer Cyclical": "Consumer Discretionary",
          "Consumer Defensive": "Consumer Staples", "Financial": "Financials",
          "Healthcare": "Health Care", "Technology": "Information Technology"}


def label_quality(lab: pd.DataFrame) -> None:
    """Section 1 -- how far is a SIC sector from the GICS-style sector the screener uses?"""
    hist = sorted((ROOT.parent / "Start Your Own" / "screener_history").glob("screen_*.csv"))
    if not hist:
        print("  no screener history to check the labels against\n"); return
    fv = pd.concat([pd.read_csv(f, usecols=["ticker", "sector"]) for f in hist])
    fv = fv.dropna().groupby("ticker").sector.first()
    fv = fv.map(lambda s: FINVIZ.get(s, s))
    m = lab.set_index("ticker").sector.to_frame("sic").join(fv.rename("finviz"), how="inner")
    m = m[m.finviz.notna()]
    if m.empty:
        print("  no overlap with the screener's labels\n"); return
    agree = (m.sic == m.finviz).mean()
    print(f"  SIC sector vs the screener's Finviz sector, {len(m)} overlapping names: "
          f"{agree:.0%} agree")
    bad = m[m.sic != m.finviz]
    if len(bad):
        pairs = bad.groupby(["finviz", "sic"]).size().sort_values(ascending=False).head(6)
        print("  the disagreements, most common first (finviz -> sic):")
        for (f, s), n in pairs.items():
            print(f"    {f:<24} -> {s:<24} {n}")
    print()


def main() -> None:
    sl = OUT / "sector_labels.csv"
    if not sl.exists():
        sys.exit("run research/sector_labels.py first")
    lab = pd.read_csv(sl)
    lab = lab[lab.sector != "Unclassified"]
    sectors = lab.set_index("ticker").sector.to_dict()

    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv", parse_dates=["date"],
                        low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                                "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)
    names = set(panel.ticker.unique())
    print(f"=== 1. THE LABELS ===")
    print(f"  {len(sectors)} of {len(names)} panel names labelled "
          f"({len(sectors)/len(names):.0%}); "
          f"delisted {int(lab.delisted.sum())} of {len(lab)} labelled names")
    print(f"  sector mix: " + ", ".join(f"{k} {v}" for k, v in
                                        lab.sector.value_counts().head(6).items()))
    label_quality(lab)

    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
    px = L.load_prices(names, cal)

    rows = []
    for variant, pan in (("full panel (unknown = own sector)", panel),
                         ("labelled names only", panel[panel.ticker.isin(sectors)])):
        tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
        tr_p, te_p = pan[pan.date < SPLIT], pan[pan.date >= SPLIT]
        print(f"=== 2. THE CAP -- {variant} ===")
        print(f"    {pan.ticker.nunique()} names, {len(pan)} rows")
        print(f"{'cap':<12}{'train':>8}{'TEST':>8}{'test DD':>9}{'trades':>8}"
              f"{'max/sector':>12}{'>=3 sectr':>11}{'>=4':>7}")
        base = None
        for cap in (0, 2, 3, 4, 5):
            kw = {**BASE, "sectors": sectors, "sector_cap": cap}
            a = L.run(tr_p, px, tr_c, capital=CAPITAL, **kw)
            b = L.run(te_p, px, te_c, capital=CAPITAL, **kw)
            lbl = "none" if cap == 0 else f"{cap} per sector"
            print(f"{lbl:<12}{a['cagr']:>7.1%}{b['cagr']:>8.1%}{b['max_drawdown']:>9.1%}"
                  f"{b['trades']:>8}{b['sector_max']:>12.0f}{b['sector_ge3']:>11.0%}"
                  f"{b['sector_ge4']:>7.0%}")
            r = dict(variant=variant, cap=cap, train=a["cagr"], test=b["cagr"],
                     test_dd=b["max_drawdown"], trades=b["trades"],
                     sector_max=b["sector_max"], sector_ge3=b["sector_ge3"],
                     sector_ge4=b["sector_ge4"], train_dd=a["max_drawdown"])
            if cap == 0:
                base = r
            else:
                r["d_train"] = a["cagr"] - base["train"]
                r["d_test"] = b["cagr"] - base["test"]
                r["d_dd"] = b["max_drawdown"] - base["test_dd"]
            rows.append(r)
        print()
    # --- 3. placebo: the same cap on shuffled labels (post hoc, see S6) --------------------
    print("=== 3. PLACEBO -- cap 3 on SHUFFLED sector labels (post hoc) ===")
    print("    if random labels diversify just as well, the cap is not doing what it claims")
    tr_c, te_c = cal[cal < SPLIT], cal[cal >= SPLIT]
    tr_p, te_p = panel[panel.date < SPLIT], panel[panel.date >= SPLIT]
    real = pd.DataFrame(rows).query("variant.str.startswith('full') and cap == 3").iloc[0]
    tk, sc = list(sectors), list(sectors.values())
    pl = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        shuf = dict(zip(tk, rng.permutation(sc)))
        kw = {**BASE, "sectors": shuf, "sector_cap": 3}
        a = L.run(tr_p, px, tr_c, capital=CAPITAL, **kw)
        b = L.run(te_p, px, te_c, capital=CAPITAL, **kw)
        pl.append(dict(variant="placebo (shuffled labels)", cap=3, seed=seed,
                       train=a["cagr"], test=b["cagr"], test_dd=b["max_drawdown"],
                       trades=b["trades"], sector_max=b["sector_max"],
                       sector_ge3=b["sector_ge3"], sector_ge4=b["sector_ge4"],
                       train_dd=a["max_drawdown"]))
    pl = pd.DataFrame(pl)
    print(f"{'':<12}{'train':>8}{'TEST':>8}{'test DD':>9}")
    print(f"{'real labels':<12}{real.train:>7.1%}{real.test:>8.1%}{real.test_dd:>9.1%}")
    print(f"{'placebo mean':<12}{pl.train.mean():>7.1%}{pl.test.mean():>8.1%}"
          f"{pl.test_dd.mean():>9.1%}")
    print(f"{'placebo min':<12}{pl.train.min():>7.1%}{pl.test.min():>8.1%}"
          f"{pl.test_dd.min():>9.1%}")
    print(f"{'placebo max':<12}{pl.train.max():>7.1%}{pl.test.max():>8.1%}"
          f"{pl.test_dd.max():>9.1%}")
    pct_dd = float((pl.test_dd < real.test_dd).mean())
    pct_ret = float((pl.test < real.test).mean())
    print(f"\n  real labels beat {pct_dd:.0%} of placebos on test drawdown and "
          f"{pct_ret:.0%} on test return")
    rows += pl.to_dict("records")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sector_cap_results.csv", index=False)
    print(f"wrote {OUT/'sector_cap_results.csv'} -- commit before interpreting (R15)\n")

    print("=" * 78)
    print("=== VERDICT (decision rules applied as written) ===")
    print("=" * 78)
    for variant in [x for x in res.variant.unique() if not x.startswith("placebo")]:
        v = res[res.variant == variant]
        none = v[v.cap == 0].iloc[0]
        three = v[v.cap == 3].iloc[0]
        print(f"\n{variant}:")
        print(f"  uncapped book reaches 3 in a sector on {none.sector_ge3:.0%} of sessions "
              f"(max {none.sector_max:.0f})")
        if none.sector_ge3 < 0.05:
            print("  S1 -> INERT. The cap cannot bind often enough to matter. Keep it as a "
                  "stated discipline, record that it is untested in effect.")
            continue
        dte, ddd = three.d_test, three.d_dd
        print(f"  cap 3 vs none: test CAGR {dte:+.1%}, test drawdown {ddd:+.1%} "
              f"(negative = deeper)")
        if dte < 0 and ddd <= 0:
            print("  S2 -> DROP. It costs test return and does not reduce drawdown.")
        elif ddd >= 0.01 and dte >= -0.01:
            print("  S3 -> KEEP at 3. Drawdown improves by >=1pp at a cost of <=1pp.")
        else:
            print("  neither S2 nor S3 fires -> unclear on the pre-registered thresholds.")
        better = v[(v.cap > 0) & (v.d_train > three.d_train) & (v.d_test > dte)
                   & (v.d_dd > three.d_dd)]
        if len(better):
            print(f"  S4 -> a different cap dominates 3 on both splits: "
                  f"{', '.join(str(int(c)) for c in better.cap)}")
        else:
            print("  S4 -> no other cap value dominates 3 on both splits; the number stands.")
    print(f"\n  S6 placebo ({len(pl)} seeds here; research/sector_cap_placebo.py runs 40, which")
    print(f"  is the figure of record -- 12 seeds resolve only to 1/12 = 8%):")
    print(f"    real labels beat {pct_dd:.0%} of placebos on test drawdown, {pct_ret:.0%} on return")
    if pct_dd >= 0.75:
        print("    -> the drawdown gain is NOT selection perturbation; the S3 KEEP survives.")
    else:
        print("    -> random labels diversify as well as real ones. The cap's apparent benefit")
        print("       is selection perturbation and the S3 KEEP above does not survive.")


if __name__ == "__main__":
    main()
