"""Regime profile backtest — survivorship-free, point-in-time.

Pre-registered in "Experiment Details/Regime Profile Backtest — Phase 3.75.md" (Part 1),
committed `1cd9fa6` BEFORE this script was written. The decision rules in §1.8 are applied
mechanically; nothing here chooses a threshold.

Why this exists: the RISK-OFF defensive profile's calm threshold (0.90) is recorded in
portfolio_rules.md as "chosen by judgment, not by the study", and the whole RISK-OFF screener
allowance sunsets at Phase 4. The signals are price-and-volume only, so their history already
exists -- including for companies that no longer trade.

Survivorship-free by construction (§1.2):
  * Membership comes from Tiingo's `supported_tickers.csv`, which carries a startDate AND an
    endDate for every ticker. A name is in the universe on date d iff startDate <= d <= endDate.
    That removes survivorship bias (delisted names are present on the dates they traded) and
    listing bias (IPOs are absent before they listed) in one rule.
  * Prices come from Tiingo's EOD API, which serves delisted tickers. Verified 2026-10-07:
    TWTR, ATVI and VMW all return full daily history over dates they traded, where yfinance and
    Yahoo's own chart endpoint return zero rows.

Stages are separate so each is checkable before the next, and so an interrupted download does
not waste the free tier's 500-symbol monthly allowance:

    venv/bin/python research/regime_backtest.py sample    # free, no API calls
    venv/bin/python research/regime_backtest.py prices    # spends the allowance, resumable
    venv/bin/python research/regime_backtest.py signals
    venv/bin/python research/regime_backtest.py metrics

Requires TIINGO_API_KEY in .env (gitignored). `requests` is used rather than urllib: this venv's
urllib fails SSL verification against api.tiingo.com for want of a CA bundle, while requests
bundles certifi.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "output"
PROJECT = ROOT.parent

# ---------------------------------------------------------------------------
# Pre-registered constants. Changing any of these invalidates the pre-registration.
# ---------------------------------------------------------------------------
SEED = 20261007                      # §1.3, recorded before the draw
SAMPLE_N = 500                       # Tiingo free tier: 500 unique symbols/month
SPAN0 = pd.Timestamp("2016-01-01")   # §1.4 first formation date
SPAN1 = pd.Timestamp("2026-09-18")   # §1.4 last date with a full forward window
PRICE_START = pd.Timestamp("2015-10-01")   # pre-loads the 60-day and 50-day windows
LIVE_CUTOFF = pd.Timestamp("2026-09-01")   # endDate at/after this = still trading
MIN_NAMES = 100                      # §1.3 floor: fewer usable names -> no observation
FRAME_DELISTED_SHARE = 0.379         # §1.8 R12 reference, re-measured 2026-10-07 on the clean frame
EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "NYSE MKT")

TICKER_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"


def _key() -> str:
    """Read TIINGO_API_KEY from .env. Never logged, never written to output."""
    env = PROJECT / ".env"
    if not env.exists():
        sys.exit("No .env found. Copy .env.example to .env and add TIINGO_API_KEY.")
    for line in env.read_text().splitlines():
        if line.startswith("TIINGO_API_KEY="):
            k = line.split("=", 1)[1].strip()
            if k and k != "your-key-here":
                return k
    sys.exit("TIINGO_API_KEY is missing or still the placeholder in .env.")


# ---------------------------------------------------------------------------
# Stage: sample
# ---------------------------------------------------------------------------

def fetch_ticker_snapshot(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """The point-in-time membership file, cached with its hash for reproducibility (§1.2).

    Re-downloading later yields a different file, so the snapshot's date, size and SHA-256 are
    recorded alongside the results. No API key is needed for this endpoint.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    cached = DATA / "supported_tickers.csv"
    meta = DATA / "supported_tickers.meta.txt"
    if refresh or not cached.exists():
        r = requests.get(TICKER_URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            cached.write_bytes(z.read(name))
        meta.write_text(
            f"downloaded={pd.Timestamp.now(tz='UTC').isoformat()}\n"
            f"url={TICKER_URL}\nbytes={cached.stat().st_size}\n"
            f"sha256={hashlib.sha256(cached.read_bytes()).hexdigest()}\n")
    info = dict(l.split("=", 1) for l in meta.read_text().strip().splitlines())
    return pd.read_csv(cached), info


#  Tiingo's assetType=="Stock" includes preferred shares, warrants, units and rights -- 28.4% of
#  the raw span-overlap frame. Leaving them in would have invalidated this study specifically:
#  preferreds are bond-like and SPAC units sit at $10 until their deal, so both are calm by
#  construction and would have flooded `rank_low_vol >= 0.90` -- the exact leg under test. The
#  result would have read "the calm decile outperforms" when it actually said "the calm decile is
#  full of preferred shares". Found 2026-10-07 because the download's first four tickers included
#  ABLLW (a warrant) and ABR-P-E (Arbor Realty preferred series E).
#
#  Nasdaq's fifth-letter convention marks a 5-CHARACTER ticker ending W/U/R as a warrant, unit or
#  right; a 3-4 character ticker ending in those letters is usually ordinary common, so the
#  pattern is anchored at five characters to keep LOW, WOW, FLOW, PLOW, TWOU and U. Dual-class
#  common (-A/-B/-C: BRK-B, HEI-A, LEN-B) is common equity and is kept.
#  Validated on 24 known commons (none excluded) and 10 known non-commons (all excluded).
NON_COMMON = re.compile(r"-P(?:-|$)|-W(?:S|$)|-U$|-R$|^[A-Z]{4}[WUR]$")


def build_frame(tickers: pd.DataFrame) -> pd.DataFrame:
    """US common stocks whose listed interval overlaps the span, with strata assigned."""
    f = tickers[tickers.exchange.isin(EXCHANGES)
                & (tickers.assetType == "Stock")
                & (tickers.priceCurrency == "USD")].copy()
    f["start"] = pd.to_datetime(f.startDate, errors="coerce")
    f["end"] = pd.to_datetime(f.endDate, errors="coerce")
    f = f.dropna(subset=["start", "end"])
    f = f[(f.end >= SPAN0) & (f.start <= SPAN1)].copy()

    f = f[~f.ticker.astype(str).str.contains(NON_COMMON, regex=True)].copy()
    #  Symbol recycling: 695 symbols (4.5%) name different companies in different eras -- AAC is
    #  three of them. Tiingo's price endpoint is keyed by symbol alone, so a recycled symbol's
    #  history cannot be attributed to the right company, and joining it to one listing interval
    #  would splice two businesses into one series. They are dropped rather than guessed at.
    recycled = set(f.loc[f.duplicated("ticker", keep=False), "ticker"])
    f = f[~f.ticker.isin(recycled)].drop_duplicates("ticker").copy()
    # Strata (§1.3): delisting status x listing era. Sampling proportionally within these is what
    # keeps the sample from quietly becoming a survivor sample again.
    f["delisted"] = f.end < LIVE_CUTOFF
    f["era"] = np.where(f.start <= SPAN0, "at-start", "mid-span")
    f["stratum"] = f.era + "/" + np.where(f.delisted, "delisted", "survived")
    return f.reset_index(drop=True)


def draw_sample(frame: pd.DataFrame, n: int = SAMPLE_N) -> pd.DataFrame:
    """Proportional stratified draw with the pre-registered seed.

    Proportional allocation, largest-remainder rounded so the sizes sum to exactly n. Each
    stratum is then sampled with its own derived seed so that adding a later tranche (§1.3's
    fallback) extends the draw rather than redrawing it.
    """
    sizes = frame.stratum.value_counts().sort_index()
    exact = sizes / sizes.sum() * n
    take = np.floor(exact).astype(int)
    while take.sum() < n:                       # largest remainder
        take[(exact - take).idxmax()] += 1
    parts = []
    for i, (stratum, k) in enumerate(take.items()):
        pool = frame[frame.stratum == stratum]
        parts.append(pool.sample(n=min(k, len(pool)), random_state=SEED + i))
    out = pd.concat(parts).sort_values("ticker").reset_index(drop=True)
    out["tranche"] = 1
    return out


def stage_sample(args) -> None:
    tickers, info = fetch_ticker_snapshot(refresh=args.refresh)
    frame = build_frame(tickers)
    samp = draw_sample(frame)

    OUT.mkdir(parents=True, exist_ok=True)
    samp.to_csv(OUT / "regime_backtest_sample.csv", index=False)

    dates = pd.date_range(SPAN0, SPAN1, freq="W-FRI")
    listed = pd.Series([((samp.start <= d) & (samp.end >= d)).sum() for d in dates], index=dates)
    dl_share = samp.delisted.mean()

    print("=== snapshot (recorded for reproducibility, §1.2) ===")
    for k in ("downloaded", "bytes", "sha256"):
        print(f"  {k:<11} {info[k]}")

    print(f"\n=== frame === {len(frame)} names overlapping {SPAN0.date()}..{SPAN1.date()}")
    print(f"  delisted share of frame: {frame.delisted.mean():.1%}")
    print(frame.stratum.value_counts().sort_index().to_string())

    print(f"\n=== sample === n={len(samp)}, seed={SEED}")
    print(samp.stratum.value_counts().sort_index().to_string())
    print(f"  delisted share of sample: {dl_share:.1%}  (frame {frame.delisted.mean():.1%})")

    print(f"\n=== per-date coverage over {len(dates)} weekly formation dates ===")
    print(f"  listed names: min {listed.min()}  q10 {listed.quantile(.10):.0f}  "
          f"median {listed.median():.0f}  max {listed.max()}")
    print(f"  dates below MIN_NAMES={MIN_NAMES} before the liquidity gate: "
          f"{(listed < MIN_NAMES).sum()} of {len(dates)}")

    # R12 (§1.8) applied here at sample level: the join must not have dropped delisted names.
    print("\n=== R12 pre-check: is the point-in-time join intact? ===")
    lo, hi = FRAME_DELISTED_SHARE - 0.08, FRAME_DELISTED_SHARE + 0.08
    if lo <= dl_share <= hi:
        print(f"  PASS  sample delisted share {dl_share:.1%} within {lo:.0%}-{hi:.0%} "
              f"of the frame's {FRAME_DELISTED_SHARE:.1%}")
    else:
        print(f"  FAIL  sample delisted share {dl_share:.1%} outside {lo:.0%}-{hi:.0%} "
              f"-- the stratified draw is not reproducing the frame. Stop.")
        sys.exit(1)
    print(f"\nwrote {OUT / 'regime_backtest_sample.csv'}")
    print("Nothing was downloaded from the price API: this stage spends no allowance.")


# ---------------------------------------------------------------------------
# Stage: prices
# ---------------------------------------------------------------------------
# Tiingo's free tier allows 500 UNIQUE SYMBOLS per month and 50 requests per hour. The symbol
# allowance is the scarce resource, so every outcome is recorded per ticker and never re-requested:
# a ticker that legitimately has no data must not be retried for the rest of the month. The pace is
# one request every PACE_SECONDS, which is why the full run takes ~10 hours; it is resumable at any
# point, so stopping it costs nothing.
PACE_SECONDS = 74           # 50/hour with headroom
STATUS_OK, STATUS_NODATA, STATUS_ERROR = "ok", "nodata", "error"
LIQUIDITY_FLOOR = 1_000_000  # $1M median 20-day dollar volume -- the live screener's own gate


def _price_paths() -> tuple[Path, Path]:
    d = DATA / "prices"
    d.mkdir(parents=True, exist_ok=True)
    return d, DATA / "prices_status.csv"


def _load_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df.ticker.astype(str), df.status.astype(str)))


def fetch_prices(session: requests.Session, ticker: str, key: str) -> tuple[str, pd.DataFrame]:
    """One ticker's full daily history. Returns (status, frame)."""
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": PRICE_START.date().isoformat(),
              "endDate": SPAN1.date().isoformat(), "token": key}
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            if attempt == 3:
                return f"{STATUS_ERROR}: {type(e).__name__}", pd.DataFrame()
            time.sleep(30 * (attempt + 1))
            continue
        if r.status_code == 429:                      # throttled -- back off, do not give up
            time.sleep(300 * (attempt + 1))
            continue
        if r.status_code == 404:
            return STATUS_NODATA, pd.DataFrame()
        if r.status_code != 200:
            if attempt == 3:
                return f"{STATUS_ERROR}: HTTP {r.status_code}", pd.DataFrame()
            time.sleep(30 * (attempt + 1))
            continue
        try:
            rows = r.json()
        except ValueError:
            return f"{STATUS_ERROR}: bad json", pd.DataFrame()
        if not isinstance(rows, list) or not rows:
            return STATUS_NODATA, pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df.date, utc=True).dt.tz_localize(None).dt.normalize()
        # Adjusted series for signals AND returns (see the pre-registration §1.5 correction):
        # in an unadjusted series a 2:1 split is a -50% daily return, which corrupts low_vol and
        # near_high for every name that ever split. The raw close is kept for reference only.
        keep = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close",
                "splitFactor", "divCash"]
        return STATUS_OK, df[[c for c in keep if c in df.columns]].sort_values("date")
    return f"{STATUS_ERROR}: retries exhausted", pd.DataFrame()


def stage_prices(args) -> None:
    key = _key()
    sample_csv = OUT / "regime_backtest_sample.csv"
    if not sample_csv.exists():
        sys.exit("Run the 'sample' stage first.")
    samp = pd.read_csv(sample_csv)
    pdir, spath = _price_paths()
    status = _load_status(spath)

    todo = [t for t in samp.ticker.astype(str)
            if status.get(t) not in (STATUS_OK, STATUS_NODATA)]
    done_ok = sum(1 for v in status.values() if v == STATUS_OK)
    print(f"sample {len(samp)} tickers | already done {len(status)} "
          f"(ok {done_ok}) | to fetch {len(todo)}", flush=True)
    if not todo:
        print("nothing to fetch.", flush=True)
    eta_h = len(todo) * PACE_SECONDS / 3600
    print(f"pacing {PACE_SECONDS}s/request -> ETA ~{eta_h:.1f}h. Resumable: safe to stop.\n",
          flush=True)

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    rows = []
    liq_pass = liq_total = 0
    for i, t in enumerate(todo, 1):
        st, df = fetch_prices(session, t, key)
        rec = {"ticker": t, "status": st.split(":")[0], "detail": st, "bars": len(df),
               "first": "", "last": "", "median_dollar_vol": np.nan}
        if st == STATUS_OK and not df.empty:
            df.to_csv(pdir / f"{t}.csv", index=False)
            rec["first"], rec["last"] = str(df.date.iloc[0].date()), str(df.date.iloc[-1].date())
            # liquidity proxy: median daily dollar volume over the whole history
            dv = (df.adjClose * df.adjVolume).median()
            rec["median_dollar_vol"] = float(dv)
            liq_total += 1
            liq_pass += int(dv >= LIQUIDITY_FLOOR)
        rows.append(rec)
        # append after every ticker so an interruption loses at most one request
        pd.DataFrame([rec]).to_csv(spath, mode="a", header=not spath.exists(), index=False)
        if i % 10 == 0 or i == len(todo):
            rate = f"{liq_pass}/{liq_total} ({liq_pass/liq_total:.0%})" if liq_total else "n/a"
            print(f"  [{i}/{len(todo)}] {t:<6} {st:<8} bars={len(df):<5} "
                  f"| liquidity>=1M so far: {rate} "
                  f"| ~{(len(todo)-i)*PACE_SECONDS/3600:.1f}h left", flush=True)
        if i < len(todo):
            time.sleep(PACE_SECONDS)

    print("\n=== done ===", flush=True)
    if liq_total:
        print(f"  liquidity pass rate: {liq_pass}/{liq_total} ({liq_pass/liq_total:.0%}) "
              f"-- this decides whether 500 names clears MIN_NAMES={MIN_NAMES} per date",
              flush=True)


# ---------------------------------------------------------------------------
# Stage: signals
# ---------------------------------------------------------------------------
HORIZONS = (5, 10, 20, 40, 60)
REGIME_BAND_PCT = 1.0        # the +-1% band in force since 2026-09-19 (D3)


def master_calendar() -> pd.Series:
    """IWM's adjusted closes -- the master session calendar and the regime input.

    IWM is still listed, so yfinance serves it and no Tiingo symbol allowance is spent. Its date
    index defines what "a session" means for every signal and forward-return count, so each
    ticker is reindexed onto it rather than onto its own ragged index.
    """
    import yfinance as yf
    h = yf.Ticker("IWM").history(start=(PRICE_START - pd.Timedelta(days=120)).date().isoformat(),
                                 end=(SPAN1 + pd.Timedelta(days=1)).date().isoformat(),
                                 auto_adjust=True)
    if h.empty:
        sys.exit("Could not fetch IWM for the master calendar.")
    s = h["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna()


def compute_regime(iwm: pd.Series) -> pd.DataFrame:
    """RISK-ON / RISK-OFF with the +-1% band, recomputed over the whole span.

    Path-dependent by design: the regime only flips on a close more than 1% beyond the SMA and
    holds inside the band, so it must be walked forward rather than vectorised. This is the one
    input that is exactly reconstructable, which is why it is reconciled against the live
    regime_history.csv below.
    """
    sma = iwm.rolling(50).mean()
    df = pd.DataFrame({"iwm": iwm, "sma50": sma}).dropna()
    df["pct_vs_sma50"] = (df.iwm / df.sma50 - 1) * 100
    reg, cur = [], "RISK-ON"
    for p in df.pct_vs_sma50:
        if p < -REGIME_BAND_PCT:
            cur = "RISK-OFF"
        elif p > REGIME_BAND_PCT:
            cur = "RISK-ON"
        reg.append(cur)
    df["regime"] = reg
    return df


def reconcile_regime(df: pd.DataFrame) -> None:
    """Check the reconstructed regime against the live file where they overlap."""
    live_path = PROJECT / "Start Your Own" / "regime_history.csv"
    if not live_path.exists():
        print("  (no live regime_history.csv to reconcile against)", flush=True)
        return
    live = pd.read_csv(live_path)
    live["date"] = pd.to_datetime(live.date)
    j = live.set_index("date")[["regime"]].join(df[["regime"]], how="inner", rsuffix="_calc")
    if j.empty:
        print("  (no overlap with regime_history.csv)", flush=True)
        return
    agree = (j.regime == j.regime_calc).mean()
    print(f"  reconciliation vs regime_history.csv: {len(j)} overlapping sessions, "
          f"{agree:.1%} agree", flush=True)
    if agree < 0.95:
        print("  WARNING: reconstructed regime disagrees with the live file on more than 5% "
              "of sessions -- the band walk or the IWM series differs. Investigate before "
              "trusting any regime-conditional result.", flush=True)


def ticker_signals(px: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Rolling signals for one ticker on the master calendar.

    Computed once over the full series and sampled at formation dates later -- O(sessions), not
    O(dates x window). All of it uses ADJUSTED prices (see the §1.5 correction): an unadjusted
    split would read as a -50% return and poison low_vol and near_high.
    """
    d = px.set_index("date").reindex(cal)
    c, v = d.adjClose.astype(float), d.adjVolume.astype(float)
    hi, lo = d.adjHigh.astype(float), d.adjLow.astype(float)
    out = pd.DataFrame(index=cal)
    out["close"] = c
    ret = c.pct_change()
    out["low_vol"] = -(ret.rolling(20).std() * 100)
    out["near_high"] = (c / c.rolling(60).max() - 1) * 100
    v50 = v.rolling(50).mean()
    out["vol_5_50"] = v.rolling(5).mean() / v50
    out["vol_ratio"] = v / v50
    sma50, sma20 = c.rolling(50).mean(), c.rolling(20).mean()
    out["pct_vs_sma50"] = (c / sma50 - 1) * 100
    out["pct_vs_sma20"] = (c / sma20 - 1) * 100
    out["above_sma50"] = c > sma50
    bb = c.rolling(20).std()
    out["bb_width"] = (bb * 4) / sma20 * 100
    # ATR(14) on the true range, for the deal-pinned gate
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_pct"] = tr.rolling(14).mean() / c * 100
    out["dollar_vol_20"] = (c * v).rolling(20).median()
    # days since a >10% 20-day breakout, for the fresh-breakout gate
    hi20 = c.rolling(20).max()
    at_high = (c >= hi20 * 0.999)
    out["mom20"] = (c / c.shift(20) - 1) * 100
    out["days_since_breakout"] = at_high[::-1].groupby((~at_high[::-1]).cumsum()).cumcount()[::-1]
    return out


def stage_signals(args) -> None:
    pdir, spath = _price_paths()
    if not spath.exists():
        sys.exit("Run the 'prices' stage first.")
    status = pd.read_csv(spath).drop_duplicates("ticker", keep="last")
    ok = status[status.status == STATUS_OK].ticker.astype(str).tolist()
    samp = pd.read_csv(OUT / "regime_backtest_sample.csv")
    meta = samp.set_index("ticker")[["start", "end", "delisted", "stratum"]].to_dict("index")
    print(f"tickers with price data: {len(ok)} of {len(samp)}", flush=True)
    if len(ok) < 50:
        sys.exit("Too few tickers downloaded to compute anything yet -- let the prices stage run.")

    print("\n=== regime (IWM, +-1% band) ===", flush=True)
    iwm = master_calendar()
    reg = compute_regime(iwm)
    reconcile_regime(reg)
    cal = reg.index
    print(f"  calendar: {len(cal)} sessions {cal.min().date()}..{cal.max().date()}", flush=True)
    print(f"  RISK-OFF share: {(reg.regime=='RISK-OFF').mean():.1%}", flush=True)

    # formation dates: weekly Fridays with a full 60-session forward window
    fridays = pd.date_range(SPAN0, SPAN1, freq="W-FRI")
    pos = {d: i for i, d in enumerate(cal)}
    dates = [d for d in fridays if d in pos and pos[d] + max(HORIZONS) < len(cal)]
    print(f"  formation dates: {len(dates)} "
          f"({sum(reg.regime.get(d)=='RISK-OFF' for d in dates)} RISK-OFF)", flush=True)

    rows, truncated = [], {h: 0 for h in HORIZONS}
    for n, t in enumerate(ok, 1):
        try:
            px = pd.read_csv(pdir / f"{t}.csv", parse_dates=["date"])
        except Exception:
            continue
        sig = ticker_signals(px, cal)
        m = meta.get(t, {})
        t_start, t_end = pd.Timestamp(m.get("start")), pd.Timestamp(m.get("end"))
        last_px_date = px.date.max()
        c = sig.close
        for d in dates:
            # point-in-time membership (§1.2) -- this is what removes survivorship bias
            if not (t_start <= d <= t_end) or pd.isna(sig.at[d, "low_vol"]):
                continue
            i = pos[d]
            r = {"date": d, "ticker": t, "delisted": bool(m.get("delisted")),
                 "stratum": m.get("stratum"), "regime": reg.regime.get(d)}
            r.update({k: sig.at[d, k] for k in
                      ("close", "low_vol", "near_high", "vol_5_50", "vol_ratio", "pct_vs_sma50",
                       "pct_vs_sma20", "above_sma50", "bb_width", "atr_pct", "dollar_vol_20",
                       "mom20", "days_since_breakout")})
            p0 = c.iloc[i]
            for h in HORIZONS:
                j = i + h
                seg = c.iloc[i:j + 1].dropna()
                if len(seg) < 2 or pd.isna(p0) or p0 <= 0:
                    r[f"fwd{h}"] = np.nan
                    continue
                # Delisting truncation (§1.7 metric 10): a name that stops trading inside the
                # window contributes its realised partial return to its last traded price. This
                # is the mechanism by which survivorship bias is actually removed, so the count
                # of truncated windows is reported rather than assumed.
                end_date = cal[j]
                trunc = end_date > last_px_date
                r[f"fwd{h}"] = (seg.iloc[-1] / p0 - 1) * 100
                r[f"trunc{h}"] = trunc
                if trunc:
                    truncated[h] += 1
            rows.append(r)
        if n % 50 == 0:
            print(f"  signals: {n}/{len(ok)} tickers, {len(rows)} panel rows", flush=True)

    panel = pd.DataFrame(rows)
    if panel.empty:
        sys.exit("No panel rows produced -- check the membership join.")

    # gates (§1.5), applied before ranking, exactly as screener.py does
    g_pinned = panel.atr_pct < 0.75
    g_sma50 = panel.pct_vs_sma50 > 40
    g_sma20 = panel.pct_vs_sma20 > 20
    g_breakout = (panel.days_since_breakout <= 2) & (panel.mom20 > 10)
    g_liquidity = panel.dollar_vol_20 < LIQUIDITY_FLOOR
    panel["gate_fail"] = np.select(
        [g_pinned, g_liquidity, g_sma50, g_sma20, g_breakout],
        ["pinned", "liquidity", "sma50_ext", "sma20_ext", "fresh_breakout"], default="")
    surv = panel[panel.gate_fail == ""].copy()
    # percentile ranks among that date's gate survivors -- matches screener.py exactly
    for col in ("low_vol", "near_high", "vol_5_50", "vol_ratio", "pct_vs_sma50"):
        surv["rank_" + col] = surv.groupby("date")[col].rank(pct=True)

    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "regime_backtest_panel_all.csv", index=False)
    surv.to_csv(OUT / "regime_backtest_panel_survivors.csv", index=False)

    per_date = surv.groupby("date").size()
    print("\n=== panel ===", flush=True)
    print(f"  rows before gates: {len(panel)}   after gates: {len(surv)}", flush=True)
    print(f"  gate failures: \n{panel.gate_fail.value_counts().to_string()}", flush=True)
    print(f"\n  usable names per date: min {per_date.min()}  median {per_date.median():.0f}  "
          f"max {per_date.max()}", flush=True)
    print(f"  dates below MIN_NAMES={MIN_NAMES}: {(per_date < MIN_NAMES).sum()} of "
          f"{len(per_date)}", flush=True)
    print(f"\n  R12 check -- delisted share of usable rows: {surv.delisted.mean():.1%} "
          f"(frame {FRAME_DELISTED_SHARE:.1%})", flush=True)
    print(f"  truncated forward windows (delisted mid-window): "
          f"{ {h: truncated[h] for h in HORIZONS} }", flush=True)
    print(f"\nwrote {OUT/'regime_backtest_panel_survivors.csv'}", flush=True)


def stage_todo(args) -> None:
    sys.exit(f"stage '{args.stage}' is not written yet -- see the module docstring for the order.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["sample", "prices", "signals", "metrics"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the ticker snapshot (changes the recorded hash)")
    args = ap.parse_args()
    {"sample": stage_sample, "prices": stage_prices,
     "signals": stage_signals}.get(args.stage, stage_todo)(args)


if __name__ == "__main__":
    main()
