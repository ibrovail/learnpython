"""Screener factor study: rank IC of screener signals on point-in-time universes.

Phase 2 pre-registered in "Experiment Details/Screener Factor Study — Phase 2.md" (Part 1).
Phase 3.5 extends it to 40- and 60-session horizons, pre-registered in
"Experiment Details/Horizon Factor Study — Phase 3.5.md" (Part 1), committed before this
script was run at those horizons. The verdicts it prints apply those rules mechanically.

Inputs (research/data/, gitignored -- rebuilt from git history and yfinance):
  universe_commits.txt, watchlist_commits.txt   one "<sha>|<committer ISO time>" per line
  universes/<date>_<sha7>.csv, watchlists/<date>_<sha7>.csv
  prices.pkl                                    yfinance cache (--refresh-prices rebuilds it)
Outputs: research/output/*.csv, plus a printed summary.

Usage: venv/bin/python research/factor_study.py [--refresh-prices]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import screener as scr  # noqa: E402  -- exclusion lists, ticker repair, gate thresholds

DATA, OUT = ROOT / "data", ROOT / "output"
PRICE_START, PRICE_END = "2025-11-03", "2026-09-18"   # end is exclusive
FIRST_FORMATION = pd.Timestamp("2026-04-17")
HORIZONS = (5, 10, 20, 40, 60)
# Weekly formation dates overlap every horizon beyond 5 sessions. Phase 2 used 0/1/3 for
# 5/10/20 and those are kept unchanged for continuity. Phase 3.5 pre-registers ceil(h/5) for
# the new horizons: at 40 and 60 sessions consecutive observations share 7 of 8 and 11 of 12
# of their forward windows, so Phase 2's lag structure would badly overstate t-statistics.
NW_LAGS = {5: 0, 10: 1, 20: 3, 40: 8, 60: 12}
PRIMARY = 40                     # Phase 3.5 rule 1: the midpoint of the adopted 40-60 hold
CONTINUITY = (5, 10, 20)         # reported, but cannot trigger a decision rule
MIN_DATES = 8                    # Phase 3.5 rule 10: below this a horizon is descriptive only
MIN_NAMES = 30                   # fewer names with data on a date -> no IC for that date
BATCH = 100
SIGNALS = ["mom20", "vol_ratio", "squeeze", "composite",
           "mom5", "mom60_skip5", "mom60_riskadj", "vol_5_50", "squeeze_own",
           "low_vol", "near_high", "vs_sma50", "vs_sma20"]
GATES = {"g_pinned": "deal-pinned (ATR<0.75%)", "g_sma50": ">40% above 50-day SMA",
         "g_sma20": ">20% above 20-day SMA", "g_breakout": "fresh >10% breakout"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_snapshots(kind: str) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """(commit time in New York, frame) for every committed snapshot, oldest first."""
    commits = DATA / f"{kind}_commits.txt"
    if not commits.exists():
        sys.exit(f"{commits} is missing: extract the snapshots from git history first.")
    snaps = []
    for line in commits.read_text().splitlines():
        if "|" not in line:
            continue
        sha, iso = line.strip().split("|")
        path = DATA / f"{kind}s" / f"{iso[:10]}_{sha[:7]}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if "ticker" not in df.columns or df.empty:
            continue
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = scr._repair_ticker_corruption(df)
        when = pd.Timestamp(iso).tz_convert("America/New_York").tz_localize(None)
        snaps.append((when, df))
    return sorted(snaps, key=lambda s: s[0])


def load_prices(tickers: list[str], refresh: bool) -> dict:
    """Adjusted daily Close/High/Low/Volume as date x ticker frames (cached)."""
    cache = DATA / "prices.pkl"
    if cache.exists() and not refresh:
        panel = pd.read_pickle(cache)
        if set(tickers) <= set(panel["attempted"]):
            return panel
    frames: dict[str, list[pd.DataFrame]] = {f: [] for f in ("Close", "High", "Low", "Volume")}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            raw = yf.download(batch, start=PRICE_START, end=PRICE_END, auto_adjust=True,
                              progress=False, threads=True, group_by="column")
        if raw is not None and not raw.empty:
            for field in frames:
                if field not in raw.columns.get_level_values(0):
                    continue
                block = raw[field]
                if isinstance(block, pd.Series):
                    block = block.to_frame(batch[0])
                frames[field].append(block)
        print(f"  prices: {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)
    panel: dict = {}
    for field, parts in frames.items():
        wide = pd.concat(parts, axis=1)
        wide = wide.loc[:, ~wide.columns.duplicated()]
        idx = pd.DatetimeIndex(wide.index)
        wide.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        panel[field] = wide.sort_index()
    panel["attempted"] = list(tickers)
    pd.to_pickle(panel, cache)
    return panel


# ---------------------------------------------------------------------------
# Signals, gates, populations
# ---------------------------------------------------------------------------

def build_panel(p: dict) -> dict:
    """Every signal, gate input and forward return as a date x ticker frame."""
    sessions = p["Close"]["IWM"].dropna().index   # the trading calendar
    C, H, L, V = (p[f].reindex(sessions) for f in ("Close", "High", "Low", "Volume"))
    ret = C.pct_change(fill_method=None)
    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    bbw = 4 * C.rolling(20).std() / sma20
    mom60_skip5 = C.shift(5) / C.shift(60) - 1
    prev = C.shift(1)
    true_range = np.maximum(H - L, np.maximum((H - prev).abs(), (L - prev).abs()))
    return {
        "sessions": sessions,
        "close": C,
        "signals": {
            # current composite components, defined exactly as screener.py does
            "mom20": C / C.shift(19) - 1,
            "vol_ratio": V / V.rolling(20).mean(),
            "squeeze": -bbw,                                  # tighter ranks higher
            # candidates
            "mom5": C / C.shift(4) - 1,
            "mom60_skip5": mom60_skip5,
            "mom60_riskadj": mom60_skip5 / ret.rolling(60).std(),
            "vol_5_50": V.rolling(5).mean() / V.rolling(50).mean(),
            "squeeze_own": -(bbw / bbw.rolling(60).median()),
            "low_vol": -ret.rolling(20).std(),                # calmer ranks higher
            "near_high": C / H.rolling(60).max() - 1,
            "vs_sma50": C / sma50 - 1,
            "vs_sma20": C / sma20 - 1,
        },
        "atr_pct": true_range.rolling(14).mean() / C * 100,
        "adv20": (C * V).rolling(20).mean(),
        "is_breakout": C > C.shift(1).rolling(20).max(),
        "fwd": {h: C.shift(-h) / C - 1 for h in HORIZONS},
        "iwm_riskon": C["IWM"] > C["IWM"].rolling(50).mean(),
    }


def fresh_breakout(panel: dict, i: int) -> pd.Series:
    """screener.py's breakout gate at session i: days 1-3 of a >10% 20-day breakout."""
    bo = panel["is_breakout"].iloc[: i + 1].to_numpy()
    close = panel["close"].iloc[: i + 1].to_numpy()
    rows, n = bo.shape
    cols = np.arange(n)
    any_bo = bo.any(axis=0)
    start = rows - 1 - np.argmax(bo[::-1], axis=0)        # latest breakout session
    active = any_bo.copy()
    while active.any():                                    # walk back to the run's first day
        prev = start - 1
        cont = active & (prev >= 0) & bo[np.clip(prev, 0, rows - 1), cols]
        start = np.where(cont, prev, start)
        active = cont
    base = close[np.clip(start - 1, 0, rows - 1), cols]
    move_pct = (close[-1] / base - 1) * 100
    flag = (any_bo & (start >= 1) & (rows - start <= scr.BREAKOUT_MAX_SESSIONS)
            & (move_pct > scr.BREAKOUT_MAX_MOVE_PCT))
    return pd.Series(flag, index=panel["close"].columns)


def population(d: pd.Timestamp, snaps: list, panel: dict) -> pd.DataFrame | None:
    """Eligible stocks at session d, from the latest snapshot committed before the next open."""
    sessions = panel["sessions"]
    i = sessions.get_loc(d)
    nxt = sessions[i + 1] if i + 1 < len(sessions) else d + pd.Timedelta(days=3)
    usable = [df for when, df in snaps if when < nxt + pd.Timedelta(hours=9, minutes=30)]
    if not usable:
        return None
    snap = usable[-1].drop_duplicates("ticker").set_index("ticker")
    snap = snap.loc[snap.index.intersection(panel["close"].columns)]

    def text(col: str) -> pd.Series:
        return snap[col].astype(str).str.upper() if col in snap.columns else pd.Series("", index=snap.index)

    tickers = pd.Series(snap.index, index=snap.index)
    excluded = (text("sector").str.contains("|".join(scr._EXCLUDED_TYPES), na=False)
                | text("industry").str.contains("|".join(scr._EXCLUDED_TYPES), na=False)
                | tickers.str.contains(scr._EXCLUDED_TICKER_SUFFIX, regex=True, na=False)
                | tickers.isin(scr._PROHIBITED_TICKERS))
    for name in scr._PROHIBITED_INDUSTRIES:
        excluded |= text("industry").str.contains(name, regex=False, na=False)
    cap = (pd.to_numeric(snap["market_cap"], errors="coerce") if "market_cap" in snap.columns
           else pd.Series(np.nan, index=snap.index))
    close = panel["close"].loc[d, snap.index]
    adv = panel["adv20"].loc[d, snap.index]
    eligible = ~excluded & (cap.isna() | (cap <= scr.MAX_MARKET_CAP)) & (close >= 1) \
        & (adv >= scr.MIN_DOLLAR_VOLUME)
    tk = snap.index[eligible.to_numpy()]

    df = pd.DataFrame(index=tk)
    for name, frame in panel["signals"].items():
        df[name] = frame.loc[d, tk]
    df["g_pinned"] = panel["atr_pct"].loc[d, tk] < scr.PINNED_MAX_ATR_PCT
    df["g_sma50"] = df["vs_sma50"] * 100 > scr.MAX_PCT_ABOVE_SMA50
    df["g_sma20"] = df["vs_sma20"] * 100 > scr.MAX_PCT_ABOVE_SMA20
    df["g_breakout"] = fresh_breakout(panel, i).reindex(tk).fillna(False).astype(bool)
    incomplete = df[["mom20", "vol_ratio", "squeeze"]].isna().any(axis=1)
    df["survivor"] = ~(df[list(GATES)].any(axis=1) | incomplete)
    sv = df["survivor"]
    df.loc[sv, "composite"] = (0.40 * df.loc[sv, "mom20"].rank(pct=True)
                               + 0.30 * df.loc[sv, "vol_ratio"].rank(pct=True)
                               + 0.30 * df.loc[sv, "squeeze"].rank(pct=True))
    for h in HORIZONS:
        df[f"fwd{h}"] = panel["fwd"][h].loc[d, tk] if i + h < len(sessions) else np.nan
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def rank_ic(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < MIN_NAMES or x[ok].nunique() < 2:
        return np.nan, n
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1]), n


def nw_t(values: pd.Series, lags: int) -> float:
    """Newey-West t-statistic for the mean of a time series."""
    x = values.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / (n - 1)
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def quintile_spread(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < MIN_NAMES or x[ok].nunique() < 5:
        return np.nan
    q = pd.qcut(x[ok].rank(method="first"), 5, labels=False)
    return float(y[ok][q == 4].mean() - y[ok][q == 0].mean())


def nonoverlapping_phases(dates: list, h: int, sessions: pd.DatetimeIndex) -> list[list]:
    """Partition formation dates into maximal subsets whose forward windows never overlap.

    A date at session index i owns the window [i, i+h). Two dates are independent only if
    their indices differ by at least h. Starting from each of the first `step` dates and
    greedily taking every date at least h sessions later yields several non-overlapping
    "phases" that between them use all the data. Each phase is internally independent, so a
    t-statistic on it needs no Newey-West correction -- which is the whole point. The phases
    are NOT independent of each other, so they are reported side by side rather than pooled:
    the spread across phases shows how much the answer depends on which slice you took.

    Adopted for Phase 4 (decided 2026-09-17) after Phase 3.5 found that weekly formation
    dates at a 40-session horizon give an effective n of 1.75 while appearing to give 14.
    """
    if not dates:
        return []
    idx = {d: sessions.get_loc(d) for d in dates}
    step = max(1, min(len(dates), int(round(h / 5)) or 1))
    phases = []
    for start in range(step):
        picked, last = [], None
        for d in dates[start:]:
            if last is None or idx[d] - idx[last] >= h:
                picked.append(d)
                last = d
        if len(picked) >= 2 and picked not in phases:
            phases.append(picked)
    return phases


def verdict(mean_ic: float, t: float, hit: float) -> str:
    """Pre-registered rules 2-4, applied at PRIMARY."""
    if np.isnan(mean_ic) or np.isnan(t):
        return "NO DATA"
    if mean_ic > 0 and t >= 2.0 and hit >= 0.60:
        return "PASS (rule 2)"
    if mean_ic <= 0 or t < 1.0:
        return "DROP (rule 3)"
    return "UNPROVEN (rule 4)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh-prices", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    universes, watchlists = load_snapshots("universe"), load_snapshots("watchlist")
    print(f"{len(universes)} universe snapshots ({universes[0][0]:%Y-%m-%d} -> {universes[-1][0]:%Y-%m-%d}), "
          f"{len(watchlists)} watchlists")
    tickers = sorted(set().union(*(set(df["ticker"]) for _, df in universes + watchlists)) | {"IWM"})
    panel = build_panel(load_prices(tickers, args.refresh_prices))
    sessions = panel["sessions"]
    print(f"prices for {panel['close'].notna().any().sum()} of {len(tickers)} tickers, "
          f"{sessions[0]:%Y-%m-%d} -> {sessions[-1]:%Y-%m-%d}")

    weekly_last = pd.Series(sessions, index=sessions).groupby(sessions.to_period("W-FRI")).max()
    formation = [d for d in weekly_last if d >= FIRST_FORMATION and sessions.get_loc(d) + min(HORIZONS) < len(sessions)]

    ic_rows, cover_rows, gate_rows, spread_rows = [], [], [], []
    for d in formation:
        pop = population(d, universes, panel)
        if pop is None:
            continue
        sv = pop[pop["survivor"]]
        regime = "RISK-ON" if bool(panel["iwm_riskon"].loc[d]) else "RISK-OFF"
        cover_rows.append({"date": d.date(), "regime": regime, "eligible": len(pop), "survivors": len(sv),
                           **{f"fwd{h}_coverage": round(sv[f"fwd{h}"].notna().mean(), 3) for h in HORIZONS},
                           "squeeze_vs_lowvol_rho": rank_ic(sv["squeeze"], sv["low_vol"])[0]})
        for h in HORIZONS:
            y_sv, y_el = sv[f"fwd{h}"], pop[f"fwd{h}"]
            if y_sv.notna().sum() < MIN_NAMES:
                continue
            for sig in SIGNALS:
                ic_sv, n_sv = rank_ic(sv[sig], y_sv)
                ic_el, _ = rank_ic(pop[sig], y_el) if sig != "composite" else (np.nan, 0)
                ic_rows.append({"date": d, "regime": regime, "horizon": h, "signal": sig, "ic": ic_sv,
                                "ic_eligible": ic_el, "names": n_sv, "q5_q1": quintile_spread(sv[sig], y_sv)})
            top = sv.nlargest(50, "composite")
            spread_rows.append({"date": d.date(), "horizon": h,
                                "top50_minus_median": top[f"fwd{h}"].mean() - y_sv.median()})
            if h == PRIMARY:
                for g, label in GATES.items():
                    hit = pop[pop[g]]
                    for tk, r in hit.iterrows():
                        if pd.notna(r[f"fwd{h}"]):
                            gate_rows.append({"date": d.date(), "gate": label, "ticker": tk,
                                              "excess_vs_survivor_median": r[f"fwd{h}"] - y_sv.median()})

    ic = pd.DataFrame(ic_rows)

    # ---- Non-overlapping formation dates (Phase 4 basis, decided 2026-09-17) ----
    # Phase 3.5 established that weekly dates at h sessions overlap ~h/5 deep, so the
    # pooled t-statistic counts near-duplicate observations as independent evidence.
    # Within a phase the windows never overlap, so lags = 0 is correct.
    nonov_rows = []
    for h in HORIZONS:
        hdates = sorted(ic[ic["horizon"] == h]["date"].unique())
        hdates = [pd.Timestamp(d) for d in hdates]
        for pi, phase in enumerate(nonoverlapping_phases(hdates, h, sessions)):
            sub_ic = ic[(ic["horizon"] == h) & (ic["date"].isin(phase))]
            for sig, grp in sub_ic.groupby("signal"):
                v = grp.set_index("date")["ic"].sort_index().dropna()
                if len(v) < 2:
                    continue
                nonov_rows.append({
                    "horizon": h, "phase": pi, "signal": sig, "n": len(v),
                    "mean_ic": v.mean(), "sd": v.std(ddof=1),
                    "t_indep": (v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))) if v.std(ddof=1) > 0 else np.nan,
                    "all_positive": bool((v > 0).all()),
                    "first": str(phase[0].date()), "last": str(phase[-1].date()),
                })
    nonov = pd.DataFrame(nonov_rows)
    if not nonov.empty:
        nonov.to_csv(OUT / "ic_nonoverlapping.csv", index=False)

    summary = []
    for (sig, h), grp in ic.groupby(["signal", "horizon"], sort=False):
        s = grp.set_index("date")["ic"].sort_index()
        summary.append({"signal": sig, "horizon": h, "dates": int(s.notna().sum()),
                        "mean_ic": s.mean(), "nw_t": nw_t(s, NW_LAGS[h]), "hit_rate": (s.dropna() > 0).mean(),
                        "mean_ic_eligible": grp["ic_eligible"].mean(), "q5_q1": grp["q5_q1"].mean(),
                        "mean_names": grp["names"].mean()})
    summary = pd.DataFrame(summary)
    prim = summary[summary["horizon"] == PRIMARY].set_index("signal")
    summary["verdict_10d"] = summary["signal"].map(
        {sig: verdict(r["mean_ic"], r["nw_t"], r["hit_rate"]) for sig, r in prim.iterrows()})

    # Rule 7: timing modes only if a signal passes at 5 or 20 sessions and is <= 0 at the other
    by = summary.set_index(["signal", "horizon"])
    modes = []
    for sig in SIGNALS:
        for a, b in ((5, 20), (20, 5)):
            if (sig, a) in by.index and (sig, b) in by.index:
                ra, rb = by.loc[(sig, a)], by.loc[(sig, b)]
                if verdict(ra["mean_ic"], ra["nw_t"], ra["hit_rate"]).startswith("PASS") and rb["mean_ic"] <= 0:
                    modes.append(f"{sig}: passes at {a} sessions, mean IC {rb['mean_ic']:+.3f} at {b}")

    regime = (ic[ic["horizon"] == PRIMARY].groupby(["signal", "regime"])["ic"]
              .agg(["mean", "count"]).unstack("regime"))
    gates = pd.DataFrame(gate_rows)
    gate_summary = (gates.groupby("gate")["excess_vs_survivor_median"]
                    .agg(stock_dates="count", mean="mean", median="median",
                         share_positive=lambda s: (s > 0).mean()) if not gates.empty else pd.DataFrame())
    spreads = pd.DataFrame(spread_rows)

    # Delivered watchlists vs the eligible universe median
    wl_rows = []
    for when, wl in watchlists:
        closed = sessions[(sessions + pd.Timedelta(hours=16)) <= when]
        if len(closed) == 0 or closed[-1] < sessions[60]:
            continue
        d = closed[-1]
        pop = population(d, universes, panel)
        if pop is None:
            continue
        row = {"committed": when.strftime("%Y-%m-%d %H:%M"), "formation": d.date(), "listed": len(wl)}
        for h in HORIZONS:
            if sessions.get_loc(d) + h >= len(sessions):
                continue
            fwd = panel["fwd"][h].loc[d]
            names = [t for t in wl["ticker"] if t in fwd.index and pd.notna(fwd[t])]
            row[f"found{h}"] = len(names)
            row[f"excess{h}"] = fwd[names].mean() - pop[f"fwd{h}"].median() if names else np.nan
        wl_rows.append(row)
    wl_perf = pd.DataFrame(wl_rows)

    for name, frame in {"ic_by_date": ic, "ic_summary": summary, "ic_by_regime": regime,
                        "coverage": pd.DataFrame(cover_rows), "gate_outcomes": gates,
                        "gate_summary": gate_summary, "top50_spread": spreads,
                        "watchlist_performance": wl_perf}.items():
        frame.to_csv(OUT / f"{name}.csv", index=name in ("ic_by_regime", "gate_summary"))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\nFormation dates: {len(formation)} ({formation[0]:%Y-%m-%d} -> {formation[-1]:%Y-%m-%d})")
    print(pd.DataFrame(cover_rows).to_string(index=False))
    for h in HORIZONS:
        t = summary[summary["horizon"] == h].sort_values("nw_t", ascending=False)
        _n = int(t["dates"].max()) if not t.empty else 0
        _tag = ("  [PRIMARY]" if h == PRIMARY else ("  [continuity only]" if h in CONTINUITY else ""))
        if _n < MIN_DATES:
            _tag += f"  [DESCRIPTIVE ONLY -- {_n} dates < {MIN_DATES} (rule 10)]"
        print(f"(horizon {h}: {_n} dates{_tag})")
        print(f"\n=== Rank IC among gate survivors, {h}-session forward return ===")
        print(t[["signal", "dates", "mean_ic", "nw_t", "hit_rate", "mean_ic_eligible", "q5_q1", "mean_names"]
                + (["verdict_10d"] if h == PRIMARY else [])].round(3).to_string(index=False))
    print("\n=== 10-session IC by regime (descriptive only, rule 8) ===")
    print(regime.round(3).to_string())
    print("\n=== Top-50 composite minus survivor median (mean across dates) ===")
    print(spreads.groupby("horizon")["top50_minus_median"].agg(["mean", "count",
          lambda s: (s > 0).mean()]).round(4).to_string())
    print("\n=== Gated groups, 10-session return minus survivor median ===")
    print(gate_summary.round(4).to_string() if not gate_summary.empty else "none")
    print("\n=== Delivered watchlists vs eligible-universe median ===")
    print(wl_perf.round(4).to_string(index=False))
    if not wl_perf.empty:
        print(wl_perf[[c for c in wl_perf.columns if c.startswith("excess")]].agg(["mean", "count"]).round(4).to_string())
    print("\n=== NON-OVERLAPPING formation dates (Phase 4 basis) ===")
    print("Within a phase, forward windows never overlap, so t needs no NW correction.")
    print("Phases are not independent OF EACH OTHER -- the spread across them is the point.\n")
    if nonov.empty:
        print("  no horizon has >=2 non-overlapping formation dates yet")
    else:
        six = ["low_vol", "near_high", "squeeze", "vol_5_50", "vol_ratio", "vs_sma50"]
        for h in sorted(nonov["horizon"].unique()):
            hh = nonov[nonov["horizon"] == h]
            nph, nobs = hh["phase"].nunique(), int(hh["n"].max())
            print(f"  horizon {h}: {nph} phase(s), up to {nobs} independent observations each")
            agg = (hh[hh["signal"].isin(six + ["composite"])]
                   .groupby("signal")
                   .agg(phases=("phase", "nunique"), n=("n", "max"),
                        mean_ic=("mean_ic", "mean"), ic_min=("mean_ic", "min"),
                        ic_max=("mean_ic", "max"), t_min=("t_indep", "min"),
                        t_max=("t_indep", "max"), always_pos=("all_positive", "all"))
                   .sort_values("mean_ic", ascending=False))
            print(agg.round(3).to_string())
            if nobs < 5:
                print(f"  -> {nobs} independent observations: DESCRIPTIVE ONLY, no verdict.\n")
            else:
                print()

    print("\n=== Rule 7 (timing modes) ===")
    print("\n".join(modes) if modes else "No signal passes at one horizon while <= 0 at the other: one mode.")


if __name__ == "__main__":
    main()
