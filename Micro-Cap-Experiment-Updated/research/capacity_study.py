"""Capacity rules study — are the rules that gate deployment the right rules?

Pre-registered in "Experiment Details/Capacity Rules Study — Phase 3.8.md" (Part 1), committed
BEFORE this script was written. The decision rules in §1.6 are applied mechanically; nothing here
chooses an outcome.

Why it exists: a post-hoc check in Phase 3.75 found that the weeks the rules label RISK-OFF went on
to return +1.89% against RISK-ON's +0.53% over 2016-2026 -- the regime signal has the wrong sign at
this horizon. Every rule that references regime is therefore in question, not just the calm
threshold the earlier study was built to test.

Method (§1.3): portfolio simulation, not cross-sectional spreads. For each rule set and formation
date, buy every qualifying name equal-weighted and hold for the horizon. A date with no qualifying
name HOLDS CASH and records 0.00% -- that is the rule's actual consequence, and dropping those
dates would hide the capacity cost entirely. Primary horizon 40 sessions (§1.4), the midpoint of
the book's stated hold, because capacity rules bind over a whole holding period.

Reuses Phase 3.75's survivorship-free panel. No API allowance is spent: the regime variants come
from IWM, SPY and ^VIX, all still listed.

    venv/bin/python research/capacity_study.py

Reads research/output/regime_backtest_panel_survivors.csv (built by regime_backtest.py signals).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT))
from regime_backtest import nw_tstat, lag_for, effective_n  # noqa: E402

PRIMARY = 40                 # §1.4
HORIZONS = (20, 40, 60)
MIN_EFFECTIVE_N = 10         # §1.6 R13
T_CRIT = 2.0
T_ABORT = 5.0
MIN_UNIVERSE = 50            # a week needs this many eligible names to count
BOOK_N = 10                  # names held in the baseline book for Q1/Q2/Q3
MATERIALITY_PP = 0.5
MIN_GROUP_FOR_MEDIAN = 10    # §1.6 R9


# ---------------------------------------------------------------------------
# Regime variants (§1.5)
# ---------------------------------------------------------------------------

def _series(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if h.empty:
        raise SystemExit(f"could not fetch {ticker}")
    s = h["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()


def _banded(pct: pd.Series, band: float = 1.0) -> pd.Series:
    """Walk the +-band rule forward. Path-dependent, so it cannot be vectorised."""
    out, cur = [], "ON"
    for p in pct:
        if p < -band:
            cur = "OFF"
        elif p > band:
            cur = "ON"
        out.append(cur)
    return pd.Series(out, index=pct.index)


def build_regimes(start: str, end: str) -> pd.DataFrame:
    """One column per regime variant; "ON" means deploy, "OFF" means hold cash."""
    iwm, spy = _series("IWM", start, end), _series("SPY", start, end)
    vix = _series("^VIX", start, end)
    r = pd.DataFrame(index=iwm.index)

    r["none"] = "ON"                                      # no regime concept at all
    pct50 = (iwm / iwm.rolling(50).mean() - 1) * 100
    r["current"] = _banded(pct50)                         # the rule in force
    r["inverted"] = r["current"].map({"ON": "OFF", "OFF": "ON"})
    pct200 = (iwm / iwm.rolling(200).mean() - 1) * 100
    r["current_200d"] = _banded(pct200)
    spct = (spy / spy.rolling(50).mean() - 1) * 100
    r["spy_50d"] = _banded(spct)

    # Q2 alternatives -- different concepts, not different lookbacks
    v = vix.reindex(r.index).ffill()
    vmed = v.rolling(252).median()
    r["vix_low"] = np.where(v <= vmed, "ON", "OFF")       # deploy when fear is below normal
    r["vix_high"] = np.where(v > vmed, "ON", "OFF")       # deploy when fear is above normal
    dd = iwm / iwm.rolling(252).max() - 1
    r["drawdown"] = np.where(dd > -0.10, "ON", "OFF")     # deploy unless >10% off the 1-yr high
    r["drawdown_inv"] = np.where(dd <= -0.10, "ON", "OFF")
    return r


# ---------------------------------------------------------------------------
# Simulation (§1.3)
# ---------------------------------------------------------------------------

def simulate(panel: pd.DataFrame, regime_col: pd.Series, h: int,
             calm: float | None = None, top_n: int | None = BOOK_N,
             cash_floor: float = 0.15, breadth: pd.Series | None = None) -> pd.DataFrame:
    """Per-week portfolio return for one rule set.

    `regime_col` maps date -> "ON"/"OFF"; "OFF" means the rule forbids deployment, so the week
    records 0.00% and counts as forced cash. `calm` applies the defensive profile. `top_n` holds
    the best N by composite. `cash_floor` scales the invested fraction, so a 15% floor can only
    ever put 85% to work.
    """
    col = f"fwd{h}"
    rows = []
    for d, g in panel.groupby("date"):
        g = g[g[col].notna()]
        if len(g) < MIN_UNIVERSE:
            continue
        on = regime_col.get(d, "ON") == "ON"
        if breadth is not None and on:
            on = breadth.get(d, True)
        sel = g
        if calm is not None:
            sel = sel[(sel.rank_low_vol >= calm) & (sel.vol_5_50 > 1.0)
                      & (sel.above_sma50.astype(bool))]
        if top_n is not None and len(sel) > top_n:
            sel = sel.nlargest(top_n, "composite")
        invested = (1.0 - cash_floor)
        if not on or len(sel) == 0:
            rows.append({"date": d, "r": 0.0, "n": 0, "cash": 1})
        else:
            rows.append({"date": d, "r": float(sel[col].mean()) * invested,
                         "n": len(sel), "cash": 0})
    return pd.DataFrame(rows).set_index("date")


def describe(sim: pd.DataFrame, h: int, label: str, base: pd.DataFrame | None = None) -> dict:
    s = sim.r
    n = len(s)
    lags, bound = lag_for(h, n)
    mean, t, _ = nw_tstat(s, lags)
    sd = float(s.std())
    row = {"rule": label, "horizon": h, "weeks": n,
           "effective_n": round(effective_n(n, h), 1),
           "mean_pct": round(float(s.mean()), 3), "median_pct": round(float(s.median()), 3),
           "pct_weeks_positive": round(float((s > 0).mean()), 3),
           "pct_weeks_cash": round(float(sim.cash.mean()), 3),
           "avg_names": round(float(sim.n.mean()), 2),
           "median_names_when_held": round(float(sim.n[sim.n > 0].median()) if (sim.n > 0).any() else 0, 1),
           "worst_week": round(float(s.min()), 2), "pct5": round(float(s.quantile(.05)), 2),
           "sd": round(sd, 3), "mean_over_sd": round(float(s.mean()) / sd, 3) if sd else np.nan,
           "nw_t": round(t, 3) if np.isfinite(t) else None, "lag_cap_bound": bound}
    if base is not None:
        d = (s - base.r).dropna()
        lg, _ = lag_for(h, len(d))
        dm, dt, _ = nw_tstat(d, lg)
        row["vs_base_pp"] = round(dm, 3)
        row["vs_base_t"] = round(dt, 3) if np.isfinite(dt) else None
        # non-inferiority interval (§1.6 R8): does it exclude a loss worse than the margin?
        se = abs(dm / dt) if (np.isfinite(dt) and dt != 0) else np.inf
        row["vs_base_ci_low"] = round(dm - 1.96 * se, 3) if np.isfinite(se) else None
    return row


def halves(panel, regime_col, h, **kw) -> tuple[float, float]:
    dates = sorted(panel.date.unique())
    mid = dates[len(dates) // 2]
    a = simulate(panel[panel.date < mid], regime_col, h, **kw).r.mean()
    b = simulate(panel[panel.date >= mid], regime_col, h, **kw).r.mean()
    return float(a), float(b)


def main() -> None:
    path = OUT / "regime_backtest_panel_survivors.csv"
    if not path.exists():
        sys.exit("Run regime_backtest.py signals first.")
    panel = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    # composite = equal-weighted percentile ranks, as screener.py builds it. bb_width is ranked
    # here because the panel carries the raw value rather than its rank.
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    rc = ["rank_low_vol", "rank_near_high", "rank_vol_5_50", "rank_vol_ratio",
          "rank_pct_vs_sma50", "rank_bb"]
    panel["composite"] = panel[rc].mean(axis=1)

    reg = build_regimes("2015-08-01", "2026-09-19")
    # breadth: share of the eligible universe above its own 50-day SMA, vs its median
    br = panel.groupby("date").above_sma50.mean()
    breadth_on = br > br.expanding(min_periods=52).median()

    print(f"panel {len(panel)} rows, {panel.date.nunique()} weeks, "
          f"{panel.ticker.nunique()} companies", flush=True)
    print(f"primary horizon {PRIMARY} sessions, book of {BOOK_N} names, 15% cash floor\n",
          flush=True)

    results = []
    base_none = simulate(panel, reg["none"], PRIMARY)

    # ---- Q1: does any index-trend regime filter earn its place? -------------------------
    print("=== Q1 -- regime filters on trial (book of 10, no defensive profile) ===", flush=True)
    print(f"{'rule':<16}{'mean':>8}{'median':>8}{'mean/sd':>9}{'t':>7}{'vs none':>9}"
          f"{'t':>7}{'%cash':>7}{'worst':>8}", flush=True)
    q1 = {}
    for name in ("none", "current", "inverted", "current_200d", "spy_50d"):
        sim = simulate(panel, reg[name], PRIMARY)
        row = describe(sim, PRIMARY, f"Q1:{name}", base=base_none)
        results.append(row); q1[name] = row
        print(f"{name:<16}{row['mean_pct']:>7.2f}%{row['median_pct']:>7.2f}%"
              f"{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
              f"{row.get('vs_base_pp', 0):>8.2f}%{row.get('vs_base_t') or 0:>7.2f}"
              f"{row['pct_weeks_cash']:>6.0%}{row['worst_week']:>7.1f}%", flush=True)

    # ---- Q2: alternative regime concepts ------------------------------------------------
    print("\n=== Q2 -- alternative regime concepts ===", flush=True)
    q2 = {}
    for name in ("vix_low", "vix_high", "drawdown", "drawdown_inv"):
        sim = simulate(panel, reg[name], PRIMARY)
        row = describe(sim, PRIMARY, f"Q2:{name}", base=base_none)
        results.append(row); q2[name] = row
        h1, h2 = halves(panel, reg[name], PRIMARY)
        row["half1"], row["half2"] = round(h1, 3), round(h2, 3)
        print(f"{name:<16}{row['mean_pct']:>7.2f}%{row['median_pct']:>7.2f}%"
              f"{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
              f"{row['vs_base_pp']:>8.2f}%{row['vs_base_t'] or 0:>7.2f}"
              f"{row['pct_weeks_cash']:>6.0%}{row['worst_week']:>7.1f}%"
              f"   halves {h1:+.2f}/{h2:+.2f}", flush=True)
    sim = simulate(panel, reg["none"], PRIMARY, breadth=breadth_on)
    row = describe(sim, PRIMARY, "Q2:breadth", base=base_none); results.append(row); q2["breadth"] = row
    print(f"{'breadth':<16}{row['mean_pct']:>7.2f}%{row['median_pct']:>7.2f}%"
          f"{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
          f"{row['vs_base_pp']:>8.2f}%{row['vs_base_t'] or 0:>7.2f}"
          f"{row['pct_weeks_cash']:>6.0%}{row['worst_week']:>7.1f}%", flush=True)

    # ---- Q3: does the defensive profile add anything? -----------------------------------
    print("\n=== Q3 -- the defensive profile, regime-free (R7: judged on mean/sd) ===", flush=True)
    base_q3 = simulate(panel, reg["none"], PRIMARY)
    q3 = {}
    for lbl, calm in (("no profile", None), ("calm 0.50", 0.50), ("calm 0.60", 0.60),
                      ("calm 0.75", 0.75), ("calm 0.80", 0.80), ("calm 0.90", 0.90)):
        sim = simulate(panel, reg["none"], PRIMARY, calm=calm)
        row = describe(sim, PRIMARY, f"Q3:{lbl}", base=base_q3)
        results.append(row); q3[lbl] = row
        print(f"{lbl:<16}{row['mean_pct']:>7.2f}%{row['median_pct']:>7.2f}%"
              f"{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
              f"{row['vs_base_pp']:>8.2f}%{row['vs_base_t'] or 0:>7.2f}"
              f"{row['pct_weeks_cash']:>6.0%}"
              f"   held {row['median_names_when_held']:>4.0f} names", flush=True)

    # ---- Q4: book shape -----------------------------------------------------------------
    print("\n=== Q4 -- how many names, and the cash floor ===", flush=True)
    base_q4 = simulate(panel, reg["none"], PRIMARY, top_n=5)
    q4 = {}
    for n_names in (3, 5, 8, 12, 20):
        sim = simulate(panel, reg["none"], PRIMARY, top_n=n_names)
        row = describe(sim, PRIMARY, f"Q4:top{n_names}", base=base_q4)
        results.append(row); q4[f"top{n_names}"] = row
        print(f"{'top '+str(n_names):<16}{row['mean_pct']:>7.2f}%{row['median_pct']:>7.2f}%"
              f"{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
              f"{row['vs_base_pp']:>8.2f}%{row['vs_base_t'] or 0:>7.2f}"
              f"{row['worst_week']:>8.1f}%", flush=True)
    for floor in (0.15, 0.05, 0.0):
        sim = simulate(panel, reg["none"], PRIMARY, cash_floor=floor)
        row = describe(sim, PRIMARY, f"Q4:floor{floor:.2f}", base=base_q4)
        results.append(row); q4[f"floor{floor:.2f}"] = row
        print(f"{'cash floor '+format(floor,'.0%'):<16}{row['mean_pct']:>7.2f}%"
              f"{row['median_pct']:>7.2f}%{row['mean_over_sd']:>9.3f}{row['nw_t'] or 0:>7.2f}"
              f"{row['vs_base_pp']:>8.2f}%{row['vs_base_t'] or 0:>7.2f}"
              f"{row['worst_week']:>8.1f}%", flush=True)

    # ---- secondary horizons -------------------------------------------------------------
    for h in (20, 60):
        for name in ("none", "current"):
            results.append(describe(simulate(panel, reg[name], h), h, f"H{h}:{name}"))

    res = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "capacity_study_results.csv", index=False)
    print(f"\nwrote {OUT/'capacity_study_results.csv'} -- commit before interpreting (R15).",
          flush=True)

    big = res[res.nw_t.abs() > T_ABORT] if "nw_t" in res else pd.DataFrame()
    if not big.empty:
        print(f"\n!!! ABORT (R14): |t| > {T_ABORT} on {len(big)} statistics. Broken variance "
              f"estimate, not a signal. Raw output written; find the cause first.", flush=True)
        print(big[["rule", "weeks", "effective_n", "mean_pct", "nw_t"]].to_string(index=False))
        sys.exit(2)

    # ---- verdicts (§1.6), applied mechanically ------------------------------------------
    print("\n" + "=" * 72 + "\n=== VERDICTS (§1.6, applied mechanically) ===\n" + "=" * 72,
          flush=True)
    if q1["none"]["effective_n"] < MIN_EFFECTIVE_N:
        print(f"R13: effective n {q1['none']['effective_n']} < {MIN_EFFECTIVE_N}. NO DECISIONS.",
              flush=True)
        return

    cur, non, inv = q1["current"], q1["none"], q1["inverted"]
    cur_beats = ((cur["mean_pct"] > non["mean_pct"] or cur["mean_over_sd"] > non["mean_over_sd"])
                 and (cur.get("vs_base_t") or 0) >= T_CRIT)
    none_beats = (non["mean_pct"] > cur["mean_pct"] and non["mean_over_sd"] > cur["mean_over_sd"]
                  and abs(cur.get("vs_base_t") or 0) >= T_CRIT)
    print("\nQ1 -- the regime filter:", flush=True)
    if cur_beats:
        print("  R1 -- RETAIN the current filter.", flush=True)
    elif none_beats:
        print("  R2 -- REMOVE the regime filter. Capacity becomes regime-independent.", flush=True)
    else:
        print("  R4 -- REMOVE the regime filter: it cannot be shown to help in either "
              "direction, so it is unjustified complexity gating real capital.", flush=True)
    inv_t = inv.get("vs_base_t") or 0
    if inv["mean_pct"] > cur["mean_pct"] and inv["mean_over_sd"] > cur["mean_over_sd"]:
        print(f"  R3 -- the inverted signal beats the live one (mean {inv['mean_pct']:+.2f}% vs "
              f"{cur['mean_pct']:+.2f}%). Recorded as CONFIRMATION that the sign is wrong. "
              f"Inversion is NOT adopted: the honest response to a signal with the wrong sign is "
              f"to stop using it, not to trade it backwards.", flush=True)

    print("\nQ2 -- a replacement signal:", flush=True)
    adopted = None
    for name, row in q2.items():
        if row.get("half1") is None:
            continue
        ok = (row["mean_pct"] > non["mean_pct"] and row["mean_over_sd"] > non["mean_over_sd"]
              and (row.get("vs_base_t") or 0) >= T_CRIT
              and row["half1"] > 0 and row["half2"] > 0)
        if ok:
            if adopted is None or row["worst_week"] > q2[adopted]["worst_week"]:
                adopted = name           # R6: better worst week wins, not higher mean
    print(f"  {'R5 -- adopt ' + adopted if adopted else 'R5 -- NO replacement qualifies; capacity becomes regime-independent'}",
          flush=True)

    print("\nQ3 -- the defensive profile:", flush=True)
    npf, p90 = q3["no profile"], q3["calm 0.90"]
    keep = (p90["mean_over_sd"] > npf["mean_over_sd"] and abs(p90.get("vs_base_t") or 0) >= T_CRIT)
    if not keep:
        print(f"  R7 -- REMOVE the defensive profile. mean/sd {p90['mean_over_sd']:.3f} with it "
              f"vs {npf['mean_over_sd']:.3f} without, t={p90.get('vs_base_t')}. Its stated purpose "
              f"is defensive, so return alone cannot justify it.", flush=True)
    else:
        print("  R7 -- RETAIN the defensive profile; now set the threshold under R8.", flush=True)
        small = [k for k, v in q3.items() if 0 < v["median_names_when_held"] < MIN_GROUP_FOR_MEDIAN]
        if small:
            print(f"  R9 -- these hold fewer than {MIN_GROUP_FOR_MEDIAN} names and cannot decide "
                  f"anything: {small}", flush=True)

    print("\nQ4 -- book shape:", flush=True)
    b5 = q4["top5"]
    best = max((k for k in q4 if k.startswith("top")),
               key=lambda k: (q4[k]["mean_over_sd"], q4[k]["mean_pct"]))
    if best != "top5" and (q4[best].get("vs_base_t") or 0) >= T_CRIT:
        print(f"  R10 -- adopt {best} (mean/sd {q4[best]['mean_over_sd']:.3f} vs "
              f"{b5['mean_over_sd']:.3f} at 5 names)", flush=True)
    else:
        print(f"  R10 -- keep ~5 positions; no count beats it with t >= {T_CRIT}", flush=True)
    f15, f00 = q4["floor0.15"], q4["floor0.00"]
    if f00["mean_over_sd"] > f15["mean_over_sd"] and abs(f00.get("vs_base_t") or 0) >= T_CRIT:
        print("  R11 -- LOWER the cash floor: it costs return and buys no stability.", flush=True)
    else:
        print(f"  R11 -- KEEP the 15% cash floor (mean/sd {f15['mean_over_sd']:.3f} vs "
              f"{f00['mean_over_sd']:.3f} at 0%).", flush=True)
    print("\n  R12 -- risk-per-trade stays at 2%: this simulation is equal-weighted and carries "
          "no stops, so it cannot speak to sizing.", flush=True)
    print("\n  R16 -- no rule is amended until Part 2 of the pre-registration is written and "
          "names the decision rule that produced each change.", flush=True)


if __name__ == "__main__":
    main()
