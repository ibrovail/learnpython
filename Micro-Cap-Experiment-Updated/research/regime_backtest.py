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
import sys
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
FRAME_DELISTED_SHARE = 0.477         # §1.8 R12 reference, measured 2026-10-07
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


def build_frame(tickers: pd.DataFrame) -> pd.DataFrame:
    """US common stocks whose listed interval overlaps the span, with strata assigned."""
    f = tickers[tickers.exchange.isin(EXCHANGES)
                & (tickers.assetType == "Stock")
                & (tickers.priceCurrency == "USD")].copy()
    f["start"] = pd.to_datetime(f.startDate, errors="coerce")
    f["end"] = pd.to_datetime(f.endDate, errors="coerce")
    f = f.dropna(subset=["start", "end"])
    f = f[(f.end >= SPAN0) & (f.start <= SPAN1)].copy()
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
# Later stages are written after this one is reviewed (see the module docstring).
# ---------------------------------------------------------------------------

def stage_todo(args) -> None:
    sys.exit(f"stage '{args.stage}' is not written yet -- see the module docstring for the order.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["sample", "prices", "signals", "metrics"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-download the ticker snapshot (changes the recorded hash)")
    args = ap.parse_args()
    {"sample": stage_sample}.get(args.stage, stage_todo)(args)


if __name__ == "__main__":
    main()
