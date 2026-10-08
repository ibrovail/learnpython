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
import atexit
import hashlib
import io
import os
import re
import subprocess
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
FRAME_DELISTED_SHARE = 0.375         # §1.8 R12b reference, re-measured after the length filter
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
    #  A US common-stock ticker is at most 5 characters. Longer ones are non-standard instruments
    #  that Tiingo files under assetType "Stock": bonds with the coupon in the symbol
    #  ("CFX 5.75", "SO 6.75 08-01-22"), hyphenless preferreds ("ALLPDCL", "ALLYPRA"), warrants
    #  ("AACTWS", "CAPTW(EXP20260807)") and even exchange TEST symbols ("ATEST-A", "NTEST-WD").
    #  106 names, 1.1%. Found 2026-10-07 by chasing ALLPDCL, which the hyphen patterns missed.
    f = f[f.ticker.astype(str).str.replace("-", "", regex=False).str.len() <= 5].copy()
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


def already_downloaded() -> set[str]:
    """Symbols already fetched this month. Tiingo's free tier caps UNIQUE SYMBOLS per month, so a
    symbol already spent is free to reuse and a new one is not."""
    _, spath = _price_paths()
    if not spath.exists():
        return set()
    try:
        d = pd.read_csv(spath).drop_duplicates("ticker", keep="last")
        return set(d.ticker.astype(str))
    except Exception:
        return set()


def draw_sample(frame: pd.DataFrame, n: int = SAMPLE_N,
                include: set[str] | None = None) -> pd.DataFrame:
    """Proportional stratified draw with the pre-registered seed.

    Proportional allocation, largest-remainder rounded so the sizes sum to exactly n. Each
    stratum is sampled with its own derived seed, so adding a later tranche extends the draw
    rather than redrawing it.

    `include` locks in symbols already paid for. Refining the frame (three times on 2026-10-07:
    share classes, recycled symbols, ticker length) redraws the sample, and a redraw that ignored
    the symbols already fetched would spend the monthly allowance twice for the same coverage.
    Locked-in names were themselves drawn proportionally from a near-identical frame, so the
    union remains a random sample -- which stage_sample verifies by comparing the sample's
    stratum shares and delisted share against the frame's rather than assuming it.
    """
    include = {t for t in (include or set()) if t in set(frame.ticker)}
    sizes = frame.stratum.value_counts().sort_index()
    exact = sizes / sizes.sum() * n
    take = np.floor(exact).astype(int)
    while take.sum() < n:
        take[(exact - take).idxmax()] += 1
    locked = frame[frame.ticker.isin(include)]
    parts = [locked] if len(locked) else []
    for i, (stratum, k) in enumerate(take.items()):
        have = int((locked.stratum == stratum).sum()) if len(locked) else 0
        need = max(0, int(k) - have)
        pool = frame[(frame.stratum == stratum) & (~frame.ticker.isin(include))]
        if need and len(pool):
            parts.append(pool.sample(n=min(need, len(pool)), random_state=SEED + i))
    out = pd.concat(parts).drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    out["reused"] = out.ticker.isin(include)
    return out


def stage_sample(args) -> None:
    tickers, info = fetch_ticker_snapshot(refresh=args.refresh)
    frame = build_frame(tickers)
    spent = already_downloaded()
    budget = max(0, SAMPLE_N - len(spent - set(frame.ticker)))
    samp = draw_sample(frame, n=budget, include=spent)

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

    print(f"\n=== sample === n={len(samp)}, seed={SEED}, reused {int(samp.reused.sum())} already-paid symbols")
    print(f"  monthly unique-symbol budget: {len(spent)} spent, "
          f"{len(set(samp.ticker) - spent)} new -> {len(spent | set(samp.ticker))} of 500")
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


def _other_instances() -> list[tuple[int, str]]:
    """Other live `prices` runs of this script, found in the process table.

    The process table rather than a lock file alone: a lock can be stale after a SIGKILL, and a
    run started before this guard existed has no lock at all. `ps` is read directly instead of
    shelling out to pgrep, whose -f matching also picks up the pgrep invocation itself.
    """
    me = os.getpid()
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == me or "regime_backtest.py" not in cmd:
            continue
        # A shell whose command text merely MENTIONS the script is not a running copy of it.
        # Without this, any `bash -c "... regime_backtest.py prices ..."` -- including the shell
        # that is about to launch the real thing -- counts as a conflict and blocks a legitimate
        # start. Found 2026-10-07 while testing the guard, which flagged its own test harness.
        if re.search(r"(?:^|/)(?:z?sh|bash|dash)\b", cmd) or " -c " in cmd:
            continue
        # Only a `prices` run conflicts; sample/signals/metrics are read-only and safe alongside.
        if re.search(r"(?:^|/)python[\d.]*\s+\S*regime_backtest\.py\s+prices(?:\s|$)",
                     cmd, re.I):
            found.append((pid, cmd.strip()))
    return found


def _guard_single_instance() -> None:
    """Refuse to start a second concurrent download.

    Two copies share one to-do list, so they request the same tickers in parallel at twice the
    paced rate -- 100/hour against Tiingo's 50/hour limit. That means 429 throttling, interleaved
    writes to prices_status.csv, and symbol allowance spent twice over. Origin: 2026-10-07, a
    second copy was started by hand while the first was still running and had to be killed before
    it issued a request.
    """
    others = _other_instances()
    if others:
        print("REFUSING TO START: another download is already running.\n", file=sys.stderr)
        for pid, cmd in others:
            print(f"  pid {pid}: {cmd}", file=sys.stderr)
        print("\nIt is resumable, so there is nothing to recover -- just let it run.\n"
              "To take over deliberately:\n"
              '  pkill -f "regime_backtest.py prices"; sleep 2; '
              "then start this again.", file=sys.stderr)
        sys.exit(3)
    lock = DATA / "prices.lock"
    DATA.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()}\n{pd.Timestamp.now().isoformat()}\n")
    atexit.register(lambda: lock.unlink(missing_ok=True))


def stage_prices(args) -> None:
    _guard_single_instance()
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
    # fill_method=None: a gap in a thinly traded name must not be padded into a 0% return,
    # which would understate low_vol for exactly the illiquid names most likely to delist.
    ret = c.pct_change(fill_method=None)
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
    # R12 (§1.8), stated correctly: the question is not what share of ROWS are delisted -- a
    # delisted name is listed for less of the span, so it legitimately appears on fewer formation
    # dates -- but whether any GATE strips delisted names disproportionately. A gate that does
    # would reintroduce the survivorship bias this whole design exists to remove.
    print("\n  R12 -- do the gates select on survival?", flush=True)
    print(f"    delisted share: frame {FRAME_DELISTED_SHARE:.1%} (names) | "
          f"usable names {surv.groupby('ticker').delisted.first().mean():.1%} | "
          f"usable rows {surv.delisted.mean():.1%} (rows are not comparable to names)", flush=True)
    #  The `pinned` gate is exempt, on evidence rather than convenience. It rejects ATR < 0.75%,
    #  which is the signature of a pre-deal SPAC trading within pennies of its ~$10 trust value --
    #  confirmed on the three largest contributors: ALAC median $10.75 (ATR 0.10%), APCA $10.55
    #  (0.00%), ARIZ $10.17 (0.00%). SPAC common shares carry ordinary 4-letter tickers and
    #  assetType "Stock", so no share-class filter catches them, and they ALL delist, on merger or
    #  liquidation. So a large differential here is the gate removing a class that is both calm by
    #  construction and delisting-prone -- it is protecting the ranking, not biasing it. This gate
    #  is therefore load-bearing for universe hygiene, which is worth stating rather than assuming.
    PINNED_EXEMPT = "pinned"
    worst = 0.0
    for g in [x for x in panel.gate_fail.unique() if x]:
        by = panel.assign(f=panel.gate_fail.eq(g)).groupby("delisted").f.mean()
        if len(by) == 2:
            gap = abs(by.get(True, 0) - by.get(False, 0))
            exempt = g == PINNED_EXEMPT
            if not exempt:
                worst = max(worst, gap)
            print(f"    gate {g:<15} fails on {by.get(True,0):6.1%} of delisted rows vs "
                  f"{by.get(False,0):6.1%} of survived  (gap {gap:+.1%})"
                  f"{'   [exempt: removes pinned SPACs by design]' if exempt else ''}", flush=True)
    print(f"    -> {'PASS' if worst <= 0.10 else 'FAIL'}: largest differential {worst:.1%} "
          f"(threshold 10pp)", flush=True)
    print(f"  truncated forward windows (delisted mid-window): "
          f"{ {h: truncated[h] for h in HORIZONS} }", flush=True)
    print(f"\nwrote {OUT/'regime_backtest_panel_survivors.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Stage: metrics
# ---------------------------------------------------------------------------
PRIMARY = 20                 # §1.6 primary horizon
THRESHOLDS = (0.50, 0.60, 0.75, 0.80, 0.90)   # §1.8 Q2
MIN_GROUP = 3                # a date needs this many names in the group to yield a spread
MATERIALITY_PP = 0.5         # §1.8 R3/R4/R6
T_CRIT = 2.0
T_ABORT = 5.0                # §1.6: a larger |t| means a broken variance estimate, not a signal
SIGNALS = ("low_vol", "near_high", "vol_5_50", "vol_ratio", "pct_vs_sma50")


def nw_tstat(x: pd.Series, lags: int) -> tuple[float, float, int]:
    """Newey-West t-statistic on a per-date series. Returns (mean, t, lags_used).

    Bartlett kernel. The lag length is capped by the caller (§1.6): Phase 3.5 put 12 lags on 10
    observations and printed t = 15.3, which is an undefined estimator rather than a conservative
    one.
    """
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n < 3:
        return (x.mean() if n else np.nan), np.nan, lags
    xc = x - x.mean()
    g0 = float((xc ** 2).sum() / n)
    s = g0
    for j in range(1, min(lags, n - 1) + 1):
        gj = float((xc.iloc[j:].values * xc.iloc[:-j].values).sum() / n)
        s += 2.0 * (1.0 - j / (lags + 1.0)) * gj
    if s <= 0:
        return float(x.mean()), np.nan, lags
    return float(x.mean()), float(x.mean() / np.sqrt(s / n)), lags


def lag_for(h: int, n_dates: int) -> tuple[int, bool]:
    """§1.6: min(ceil(h/5), floor(n/5)). Second value says whether the cap bound."""
    want = int(np.ceil(h / 5))
    cap = max(0, int(n_dates // 5))
    return min(want, cap), want > cap


def effective_n(n_dates: int, h: int) -> float:
    return n_dates / (h / 5)


def profile_mask(df: pd.DataFrame, calm: float, use_near_high: bool = False) -> pd.Series:
    """The defensive profile as portfolio_rules.md defines it, plus the hard trend rule.

    Two legs since 2026-10-05 (`rank_low_vol`, `vol_5_50`); `use_near_high` restores the removed
    third leg for Q3. `above_sma50` is a separate hard entry rule that applies in every lane.
    """
    m = (df.rank_low_vol >= calm) & (df.vol_5_50 > 1.0) & (df.above_sma50.astype(bool))
    if use_near_high:
        m &= df.near_high >= -5
    return m


def per_date_spread(panel: pd.DataFrame, h: int, calm: float,
                    use_near_high: bool = False) -> pd.DataFrame:
    """Group-vs-universe spread per formation date, mean-vs-mean AND median-vs-median (§1.7 #4).

    Like for like: the group's mean against the universe's mean, the group's median against the
    universe's median. Never a group mean against a universe median -- that error cost about half
    the claimed edge in the first Phase 3.5 write-up.
    """
    col = f"fwd{h}"
    rows = []
    for d, g in panel.groupby("date"):
        g = g[g[col].notna()]
        if len(g) < MIN_NAMES:
            continue
        sel = g[profile_mask(g, calm, use_near_high)]
        if len(sel) < MIN_GROUP:
            rows.append({"date": d, "n_univ": len(g), "n_group": len(sel),
                         "mm": np.nan, "dd": np.nan, "univ_med": g[col].median()})
            continue
        rows.append({"date": d, "n_univ": len(g), "n_group": len(sel),
                     "mm": sel[col].mean() - g[col].mean(),
                     "dd": sel[col].median() - g[col].median(),
                     "univ_med": g[col].median()})
    return pd.DataFrame(rows)


def summarise(series: pd.Series, h: int, label: str) -> dict:
    n = int(series.dropna().shape[0])
    lags, bound = lag_for(h, n)
    mean, t, _ = nw_tstat(series, lags)
    return {"metric": label, "horizon": h, "n_dates": n, "effective_n": round(effective_n(n, h), 1),
            "mean_pp": mean, "nw_t": t, "nw_lags": lags, "lag_cap_bound": bound}


def stage_metrics(args) -> None:
    path = OUT / "regime_backtest_panel_survivors.csv"
    if not path.exists():
        sys.exit("Run the 'signals' stage first.")
    panel = pd.read_csv(path, parse_dates=["date"])
    off = panel[panel.regime == "RISK-OFF"]
    print(f"panel {len(panel)} rows | RISK-OFF {len(off)} rows "
          f"({off.date.nunique()} dates of {panel.date.nunique()})", flush=True)

    results, raw = [], {}

    # ---- R12: the join must not have dropped delisted names -------------------------------
    dl = panel.delisted.mean()
    print(f"\n=== R12 -- point-in-time join intact? ===", flush=True)
    ok12 = abs(dl - FRAME_DELISTED_SHARE) <= 0.12
    print(f"  delisted share of usable rows {dl:.1%} vs frame {FRAME_DELISTED_SHARE:.1%} "
          f"-> {'PASS' if ok12 else 'FAIL'}", flush=True)

    # ---- Q1: does the profile beat the universe in RISK-OFF? ------------------------------
    print("\n=== Q1 -- profile vs gate survivors, by regime ===", flush=True)
    for regime, sub in (("RISK-OFF", off), ("RISK-ON", panel[panel.regime == "RISK-ON"])):
        for h in HORIZONS:
            sp = per_date_spread(sub, h, 0.90)
            raw[f"q1_{regime}_{h}"] = sp
            for key, lbl in (("mm", "mean-vs-mean"), ("dd", "median-vs-median")):
                r = summarise(sp[key], h, f"Q1 {regime} {lbl}")
                r["regime"] = regime
                results.append(r)
                if h == PRIMARY:
                    print(f"  {regime:<9} {lbl:<18} {r['mean_pp']:+7.2f}pp  "
                          f"t={r['nw_t']:+6.2f}  dates={r['n_dates']:<4} "
                          f"eff_n={r['effective_n']:<5}"
                          f"{'  [lag cap bound]' if r['lag_cap_bound'] else ''}", flush=True)

    # ---- Q2: the threshold curve, paired ---------------------------------------------------
    print(f"\n=== Q2 -- calm threshold curve, RISK-OFF, {PRIMARY} sessions ===", flush=True)
    base = per_date_spread(off, PRIMARY, 0.90).set_index("date")
    curve = []
    for c in THRESHOLDS:
        sp = per_date_spread(off, PRIMARY, c).set_index("date")
        raw[f"q2_{c}"] = sp.reset_index()
        row = {"calm": c, "median_group_n": sp.n_group.median()}
        for key, lbl in (("mm", "mean_vs_mean"), ("dd", "median_vs_median")):
            r = summarise(sp[key], PRIMARY, f"Q2 calm={c} {lbl}")
            row[f"{lbl}_pp"], row[f"{lbl}_t"] = r["mean_pp"], r["nw_t"]
            # paired difference against 0.90 -- same dates, same names
            diff = (sp[key] - base[key]).dropna()
            dr = summarise(diff, PRIMARY, f"Q2 paired {c}-0.90 {lbl}")
            row[f"{lbl}_vs090_pp"], row[f"{lbl}_vs090_t"] = dr["mean_pp"], dr["nw_t"]
            results.extend([r, dr])
        curve.append(row)
    curve = pd.DataFrame(curve)
    print(curve[["calm", "median_group_n", "median_vs_median_pp", "median_vs_median_t",
                 "median_vs_median_vs090_pp", "median_vs_median_vs090_t"]]
          .to_string(index=False, float_format=lambda v: f"{v:7.2f}"), flush=True)

    # ---- Q3: three legs vs two ------------------------------------------------------------
    print(f"\n=== Q3 -- does near_high add anything? RISK-OFF, {PRIMARY} sessions ===", flush=True)
    two = per_date_spread(off, PRIMARY, 0.90).set_index("date")
    three = per_date_spread(off, PRIMARY, 0.90, use_near_high=True).set_index("date")
    raw["q3_three_leg"] = three.reset_index()
    q3 = {}
    for key, lbl in (("mm", "mean-vs-mean"), ("dd", "median-vs-median")):
        d = (three[key] - two[key]).dropna()
        r = summarise(d, PRIMARY, f"Q3 three-minus-two {lbl}")
        results.append(r); q3[key] = r
        print(f"  {lbl:<18} three-leg minus two-leg {r['mean_pp']:+7.2f}pp  "
              f"t={r['nw_t']:+6.2f}  dates={r['n_dates']}", flush=True)

    # ---- raw output committed BEFORE interpretation (§1.8 R10) ----------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT / "regime_backtest_results.csv", index=False)
    curve.to_csv(OUT / "regime_backtest_threshold_curve.csv", index=False)
    for k, v in raw.items():
        v.to_csv(OUT / f"regime_backtest_raw_{k}.csv", index=False)
    print(f"\nwrote {OUT/'regime_backtest_results.csv'} and "
          f"{len(raw)+1} raw files -- commit these before interpreting (R10).", flush=True)

    # ---- abort check (§1.6) ----------------------------------------------------------------
    big = res[res.nw_t.abs() > T_ABORT]
    if not big.empty:
        print(f"\n!!! ABORT (§1.6): {len(big)} statistics with |t| > {T_ABORT}. A t this large on "
              f"overlapping windows is evidence of a broken variance estimate, not a strong "
              f"signal. Raw output is written; find the cause before reading any verdict.",
              flush=True)
        print(big[["metric", "horizon", "n_dates", "effective_n", "mean_pp", "nw_t"]]
              .to_string(index=False), flush=True)
        sys.exit(2)

    # ---- the pre-registered verdicts, applied mechanically (§1.8) --------------------------
    print("\n" + "=" * 72 + "\n=== VERDICTS (§1.8, applied mechanically) ===\n" + "=" * 72,
          flush=True)
    if not ok12:
        print("R12 FAIL -- the point-in-time join looks broken. Every result below is void.",
              flush=True)

    q1 = {k: next(r for r in results
                  if r["metric"] == f"Q1 RISK-OFF {v}" and r["horizon"] == PRIMARY)
          for k, v in (("mm", "mean-vs-mean"), ("dd", "median-vs-median"))}
    both_pos = all(q1[k]["mean_pp"] > 0 and q1[k]["nw_t"] >= T_CRIT for k in ("mm", "dd"))
    print(f"\nQ1: {'R1 -- allowance CONFIRMED' if both_pos else 'R2 -- allowance REVERTS to a freeze at Phase 4'}",
          flush=True)
    for k in ("mm", "dd"):
        print(f"     {q1[k]['metric']}: {q1[k]['mean_pp']:+.2f}pp t={q1[k]['nw_t']:+.2f} "
              f"(eff_n {q1[k]['effective_n']})", flush=True)

    # R3: loosest threshold that is not significantly worse and not worse by > 0.5pp
    adopted = 0.90
    for c in sorted(THRESHOLDS):
        row = curve[curve.calm == c].iloc[0]
        t_ok = not (row.median_vs_median_vs090_t <= -T_CRIT)
        mat_ok = row.median_vs_median_vs090_pp >= -MATERIALITY_PP
        if t_ok and mat_ok:
            adopted = c
            break
    print(f"\nQ2: R3 adopts calm >= {adopted:.2f}", flush=True)
    if adopted != 0.90:
        row = curve[curve.calm == adopted].iloc[0]
        # R5: sign disagreement between the two measures -> keep 0.90
        if np.sign(row.median_vs_median_vs090_pp) != np.sign(row.mean_vs_mean_vs090_pp):
            adopted = 0.90
            print("     R5 overrides: the two measures disagree in sign -> keep 0.90", flush=True)
        else:
            # R5b: must win or tie in both halves
            dates = sorted(off.date.unique())
            mid = dates[len(dates) // 2]
            halves = []
            for lo, hi in ((dates[0], mid), (mid, dates[-1])):
                sub = off[(off.date >= lo) & (off.date <= hi)]
                a = per_date_spread(sub, PRIMARY, adopted).set_index("date")["dd"]
                b = per_date_spread(sub, PRIMARY, 0.90).set_index("date")["dd"]
                halves.append(float((a - b).dropna().mean()))
            print(f"     R5b stability: half-1 {halves[0]:+.2f}pp, half-2 {halves[1]:+.2f}pp "
                  f"(split {pd.Timestamp(mid).date()})", flush=True)
            if min(halves) < -MATERIALITY_PP:
                adopted = 0.90
                print("     R5b VETO: loses in one half -> keep 0.90, report the instability",
                      flush=True)
    print(f"     => calm threshold: {adopted:.2f}", flush=True)

    reinstate = (q3["dd"]["mean_pp"] > MATERIALITY_PP and q3["mm"]["mean_pp"] > MATERIALITY_PP)
    print(f"\nQ3: {'R6 -- reinstate near_high as rank_near_high >= 0.90' if reinstate else 'R7 -- the removal STANDS'}",
          flush=True)
    print("\n(Record these in Part 2 of the pre-registration, then amend the rules.)", flush=True)


def stage_todo(args) -> None:
    sys.exit(f"stage '{args.stage}' is not written yet -- see the module docstring for the order.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["sample", "prices", "signals", "metrics"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the ticker snapshot (changes the recorded hash)")
    args = ap.parse_args()
    {"sample": stage_sample, "prices": stage_prices, "signals": stage_signals,
     "metrics": stage_metrics}.get(args.stage, stage_todo)(args)


if __name__ == "__main__":
    main()
