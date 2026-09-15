"""Phase 2 diagnostics: post-hoc checks on the pre-registered factor study.

NOT part of the pre-registration. Added after the first results showed positive rank ICs
alongside near-zero or negative quintile *mean* spreads. The questions:

  1. Skew     -- how far is the survivors' mean forward return from their median?
  2. Quintiles -- do signals separate medians, means, or both?
  3. Return-magnitude IC -- correlation of signal rank with winsorized returns, not return ranks.
  4. Partial IC -- does a signal carry information once low volatility is controlled for?
  5. Direction -- do ICs depend on whether the survivors' median return was falling?
  6. Top-N, unbiased -- top 15/50 mean vs survivors' MEAN, median vs survivors' MEDIAN.
  7. Watchlists, unbiased -- the same comparison for the lists actually delivered.
  8. Deal-pinned -- absolute returns of the gated group, not just returns vs the median.
  9. Redundancy -- mean cross-sectional rank correlation among the passing signals.
 10. Composites side by side -- current, rule 5, rule 5 + mom20 (rule 4 reading), and rule 5
     with near-duplicates removed: of any passing pair with rank correlation >= 0.70, the one
     with the lower 10-session return-magnitude t is dropped. Rule 5 did not anticipate
     collinear signals; this dedup rule is post hoc and is labelled as such wherever used.

The pre-registered verdicts stand whatever this shows; it informs how to act on them.
"rule5" (equal-weighted passing signals) is evaluated IN-SAMPLE -- its signal set was chosen
on this data, so its numbers are optimistic by construction.

Usage: venv/bin/python research/factor_diagnostics.py   (after factor_study.py cached prices)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factor_study as fs  # noqa: E402

TOP_SETS = ("composite", "rule5", "rule5_mom20", "near_high", "low_vol")


def spearman(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < fs.MIN_NAMES or x[ok].nunique() < 2:
        return np.nan
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def agg_t(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    """Mean, Newey-West t and hit rate of a per-date statistic."""
    out = []
    for key, grp in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        s = grp.set_index("date")[value].sort_index()
        h = dict(zip(keys, key)).get("horizon", 10)
        out.append({**dict(zip(keys, key)), "dates": int(s.notna().sum()), "mean": s.mean(),
                    "nw_t": fs.nw_t(s, fs.NW_LAGS.get(h, 1)), "share_pos": (s.dropna() > 0).mean()})
    return pd.DataFrame(out)


def main() -> None:
    universes, watchlists = fs.load_snapshots("universe"), fs.load_snapshots("watchlist")
    tickers = sorted(set().union(*(set(df["ticker"]) for _, df in universes + watchlists)) | {"IWM"})
    panel = fs.build_panel(fs.load_prices(tickers, refresh=False))
    sessions = panel["sessions"]
    summary = pd.read_csv(fs.OUT / "ic_summary.csv")
    passing = sorted(summary.loc[(summary["horizon"] == fs.PRIMARY) & (summary["signal"] != "composite")
                                 & summary["verdict_10d"].str.startswith("PASS"), "signal"])
    print("Rule-2 passing signals (10 sessions):", ", ".join(passing))
    signals = [s for s in fs.SIGNALS] + ["rule5", "rule5_mom20"]

    weekly_last = pd.Series(sessions, index=sessions).groupby(sessions.to_period("W-FRI")).max()
    formation = [d for d in weekly_last
                 if d >= fs.FIRST_FORMATION and sessions.get_loc(d) + min(fs.HORIZONS) < len(sessions)]

    skew, quint, mag, partial, top, pinned, ics = [], [], [], [], [], [], []
    frames: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in formation:
        pop = fs.population(d, universes, panel)
        if pop is None:
            continue
        sv = pop[pop["survivor"]].copy()
        ranks = {s: sv[s].rank(pct=True) for s in passing + ["mom20"]}
        sv["rule5"] = sum(ranks[s] for s in passing) / len(passing)
        sv["rule5_mom20"] = (sum(ranks[s] for s in passing) + ranks["mom20"]) / (len(passing) + 1)
        low_vol_rank = sv["low_vol"].rank(pct=True)
        frames[d] = sv
        for h in fs.HORIZONS:
            y = sv[f"fwd{h}"]
            if y.notna().sum() < fs.MIN_NAMES:
                continue
            skew.append({"date": d, "horizon": h, "median": y.median(), "mean": y.mean()})
            lo, hi = y.quantile(0.01), y.quantile(0.99)
            for sig in signals:
                x = sv[sig]
                ok = x.notna() & y.notna()
                if ok.sum() < fs.MIN_NAMES or x[ok].nunique() < 5:
                    continue
                ics.append({"date": d, "horizon": h, "signal": sig, "ic": spearman(x, y)})
                q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
                for k in range(5):
                    yk = y[ok][q == k]
                    quint.append({"date": d, "horizon": h, "signal": sig, "q": k + 1,
                                  "median": yk.median(), "mean": yk.mean()})
                mag.append({"date": d, "horizon": h, "signal": sig,
                            "ic_magnitude": float(np.corrcoef(x[ok].rank(), y[ok].clip(lo, hi))[0, 1])})
                if sig != "low_vol":
                    ok2 = ok & low_vol_rank.notna()
                    xr, lr = x[ok2].rank(pct=True), low_vol_rank[ok2]
                    resid = xr - np.polyval(np.polyfit(lr, xr, 1), lr)
                    partial.append({"date": d, "horizon": h, "signal": sig,
                                    "partial_ic": float(np.corrcoef(resid.rank(), y[ok2].rank())[0, 1])})
            for sig in TOP_SETS:
                for n in (15, 50):
                    t = sv.dropna(subset=[sig]).nlargest(n, sig)[f"fwd{h}"].dropna()
                    top.append({"date": d, "horizon": h, "signal": sig, "top_n": n,
                                "mean_vs_mean": t.mean() - y.mean(), "median_vs_median": t.median() - y.median()})
            if h == fs.PRIMARY:
                g = pop.loc[pop["g_pinned"], f"fwd{h}"].dropna()
                if len(g):
                    pinned.append({"date": d, "names": len(g), "pinned_mean": g.mean(), "pinned_median": g.median(),
                                   "survivor_mean": y.mean(), "survivor_median": y.median()})

    skew, quint, mag, partial, top, ics = map(pd.DataFrame, (skew, quint, mag, partial, top, ics))
    pinned = pd.DataFrame(pinned)

    wl = []
    for when, df in watchlists:
        closed = sessions[(sessions + pd.Timedelta(hours=16)) <= when]
        if len(closed) == 0 or closed[-1] < fs.FIRST_FORMATION - pd.Timedelta(days=7):
            continue
        d = closed[-1]
        pop = fs.population(d, universes, panel)
        if pop is None:
            continue
        for h in fs.HORIZONS:
            if sessions.get_loc(d) + h >= len(sessions):
                continue
            fwd = panel["fwd"][h].loc[d]
            names = fwd.reindex([t for t in df["ticker"] if t in fwd.index]).dropna()
            base = pop[f"fwd{h}"].dropna()
            if len(names):
                wl.append({"formation": d, "horizon": h, "names": len(names),
                           "mean_vs_mean": names.mean() - base.mean(),
                           "median_vs_median": names.median() - base.median()})
    wl = pd.DataFrame(wl)

    # Direction: per-date IC against the survivors' median forward return that date
    direction = []
    for (sig, h), grp in ics.groupby(["signal", "horizon"]):
        m = grp.merge(skew[["date", "horizon", "median"]], on=["date", "horizon"])
        if len(m) >= 5:
            falling, rising = m[m["median"] < 0], m[m["median"] >= 0]
            direction.append({"signal": sig, "horizon": h, "corr_ic_vs_median": m["ic"].corr(m["median"]),
                              "ic_when_median_falling": falling["ic"].mean(), "dates_falling": len(falling),
                              "ic_when_median_rising": rising["ic"].mean(), "dates_rising": len(rising)})
    direction = pd.DataFrame(direction)

    out = {"diag_skew": skew, "diag_quintiles": quint, "diag_magnitude_ic": mag, "diag_partial_ic": partial,
           "diag_top_n": top, "diag_watchlists": wl, "diag_direction": direction, "diag_pinned": pinned,
           "diag_ic_incl_rule5": ics}
    for name, frame in out.items():
        frame.to_csv(fs.OUT / f"{name}.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n=== 1. Skew: survivors' forward return, mean across dates ===")
    sk = skew.groupby("horizon").agg(median=("median", "mean"), mean=("mean", "mean"),
                                     dates_median_negative=("median", lambda s: int((s < 0).sum())),
                                     dates=("median", "size"))
    sk["mean_minus_median"] = sk["mean"] - sk["median"]
    print(sk.round(4).to_string())

    for h in (10, 20):
        qp = quint[quint["horizon"] == h].groupby(["signal", "q"])[["median", "mean"]].mean().unstack("q")
        tbl = pd.DataFrame({f"Q{k}_med": qp[("median", k)] for k in range(1, 6)})
        tbl["Q5-Q1_median"] = qp[("median", 5)] - qp[("median", 1)]
        tbl["Q5-Q1_mean"] = qp[("mean", 5)] - qp[("mean", 1)]
        print(f"\n=== 2. Quintile profile, {h} sessions (Q5 = best-ranked; mean across dates) ===")
        print(tbl.sort_values("Q5-Q1_median", ascending=False).round(4).to_string())

    for title, frame, value in (("3. Return-magnitude IC (signal rank vs winsorized return)", mag, "ic_magnitude"),
                                ("4. Partial IC after controlling for low volatility", partial, "partial_ic")):
        print(f"\n=== {title} ===")
        a = agg_t(frame, value, ["signal", "horizon"])
        print(a.pivot(index="signal", columns="horizon", values=["mean", "nw_t"]).round(3).to_string())

    print("\n=== 5. Direction: IC vs the survivors' median forward return ===")
    print(direction[direction["horizon"].isin([10, 20])].round(3).to_string(index=False))

    print("\n=== 6. Top-N vs survivors, unbiased (mean vs mean, median vs median) ===")
    tt = top.groupby(["signal", "top_n", "horizon"]).agg(
        mean_vs_mean=("mean_vs_mean", "mean"), share_pos_mean=("mean_vs_mean", lambda s: (s > 0).mean()),
        median_vs_median=("median_vs_median", "mean"),
        share_pos_median=("median_vs_median", lambda s: (s > 0).mean()), dates=("mean_vs_mean", "size"))
    print(tt.round(4).to_string())

    print("\n=== 7. Delivered watchlists vs eligible universe, unbiased ===")
    if not wl.empty:
        print(wl.groupby("horizon").agg(lists=("names", "size"), mean_vs_mean=("mean_vs_mean", "mean"),
                                        share_pos_mean=("mean_vs_mean", lambda s: (s > 0).mean()),
                                        median_vs_median=("median_vs_median", "mean"),
                                        share_pos_median=("median_vs_median", lambda s: (s > 0).mean()))
              .round(4).to_string())

    print("\n=== 8. Deal-pinned group, absolute 10-session returns ===")
    if not pinned.empty:
        print(pinned.round(4).to_string(index=False))
        print(pinned[["pinned_mean", "pinned_median", "survivor_mean", "survivor_median"]].mean().round(4).to_string())

    # 9. Redundancy among passing signals (rank() then Pearson = Spearman, pairwise NaNs dropped)
    cols = passing + ["mom20", "composite"]
    corr = sum(f[cols].rank().corr() for f in frames.values()) / len(frames)
    corr.to_csv(fs.OUT / "diag_signal_corr.csv")
    print("\n=== 9. Mean cross-sectional rank correlation, passing signals ===")
    print(corr.round(2).to_string())

    # 10. Post-hoc dedup of rule 5, then the composites side by side (all in-sample)
    mag_t = agg_t(mag[mag["horizon"] == fs.PRIMARY], "ic_magnitude", ["signal", "horizon"]).set_index("signal")["nw_t"]
    keep = list(passing)
    pairs = sorted(((corr.loc[a, b], a, b) for i, a in enumerate(passing) for b in passing[i + 1:]), reverse=True)
    for rho, a, b in pairs:
        if rho >= 0.70 and a in keep and b in keep:
            drop = a if mag_t[a] < mag_t[b] else b
            keep.remove(drop)
            print(f"  dedup: {a} ~ {b} (rho {rho:.2f}) -> drop {drop} (magnitude t {mag_t[drop]:.2f})")
    print("  rule5_dedup keeps:", ", ".join(keep))
    for f in frames.values():
        f["rule5_dedup"] = sum(f[s].rank(pct=True) for s in keep) / len(keep)

    rows = []
    for name in ("composite", "rule5", "rule5_mom20", "rule5_dedup"):
        for h in (10, 20):
            per = []
            for d, f in frames.items():
                x, y = f[name], f[f"fwd{h}"]
                ok = x.notna() & y.notna()
                if ok.sum() < fs.MIN_NAMES:
                    continue
                lo, hi = y[ok].quantile(0.01), y[ok].quantile(0.99)
                q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
                t15 = f.loc[ok].nlargest(15, name)[f"fwd{h}"]
                per.append({"date": d, "ic": spearman(x, y),
                            "ic_mag": float(np.corrcoef(x[ok].rank(), y[ok].clip(lo, hi))[0, 1]),
                            "q_med": y[ok][q == 4].median() - y[ok][q == 0].median(),
                            "q_mean": y[ok][q == 4].mean() - y[ok][q == 0].mean(),
                            "t15_mean": t15.mean() - y[ok].mean(), "t15_median": t15.median() - y[ok].median(),
                            "median": y[ok].median()})
            p = pd.DataFrame(per).set_index("date").sort_index()
            lag = fs.NW_LAGS[h]
            rows.append({"composite": name, "horizon": h, "dates": len(p),
                         "ic": p["ic"].mean(), "ic_t": fs.nw_t(p["ic"], lag),
                         "ic_mag": p["ic_mag"].mean(), "ic_mag_t": fs.nw_t(p["ic_mag"], lag),
                         "q5q1_med": p["q_med"].mean(), "q5q1_mean": p["q_mean"].mean(),
                         "top15_mean_vs_mean": p["t15_mean"].mean(), "top15_mean_pos": (p["t15_mean"] > 0).mean(),
                         "top15_med_vs_med": p["t15_median"].mean(), "top15_med_pos": (p["t15_median"] > 0).mean(),
                         "ic_median_falling": p.loc[p["median"] < 0, "ic"].mean(),
                         "ic_median_rising": p.loc[p["median"] >= 0, "ic"].mean()})
    comp = pd.DataFrame(rows)
    comp.to_csv(fs.OUT / "diag_composites.csv", index=False)
    print("\n=== 10. Composites side by side (IN-SAMPLE; rule5_dedup's rule is post hoc) ===")
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
