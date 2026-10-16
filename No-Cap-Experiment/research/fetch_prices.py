"""Stage 1b — daily prices for every ticker that was ever an S&P 500 member in the window.

Delisted and acquired names MUST be present. A panel holding only today's survivors would
make any momentum backtest look good for free, which is the failure this study exists to
avoid. Coverage is measured and reported before a single return is computed.

Resume-safe: already-downloaded tickers are skipped, so it can be re-run after an
interruption or a rate limit.

    python3 -I research/fetch_prices.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data.nosync"
PRICES = CACHE / "prices"
UA = {"User-Agent": "NoCapExperiment/0.1 (portfolio research; contact via github.com/ibrovail)"}

FIRST_FORMATION, LAST_FORMATION = 2011, 2024
PERIOD1 = int(pd.Timestamp("2010-01-01").timestamp())
PERIOD2 = int(pd.Timestamp("2026-01-01").timestamp())
BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{}"

PAUSE = 0.7          # polite spacing between tickers
MAX_TRIES = 4


def universe() -> list[str]:
    m = {int(k): set(v) for k, v in
         json.loads((CACHE / "membership.json").read_text()).items()}
    u: set[str] = set()
    for y in range(FIRST_FORMATION, LAST_FORMATION + 1):
        u |= m.get(y, set())
    return sorted(u)


def yahoo_symbol(ticker: str) -> str:
    """Index tickers use dots for share classes; Yahoo uses hyphens.

    BRK.B 404s while BRK-B returns 4,024 rows. Without this, two S&P 500 members are
    silently dropped from the panel as if they had never existed.
    """
    return ticker.replace(".", "-")


def fetch_one(ticker: str) -> pd.DataFrame | None:
    """Daily OHLC + adjusted close. Returns None only when Yahoo has no history at all.

    A 400 means "no data in the requested window", NOT "unknown symbol": AVB answers 400
    for a 2010 start yet has 427 rows under range=max. Discarding a ticker on 400 threw
    away every name whose Yahoo history begins later than the window, so this re-asks for
    the full range and keeps what exists. Partial history is usable -- the engine scores a
    year only when that year has prices.
    """
    sym = yahoo_symbol(ticker)
    windowed = {"period1": PERIOD1, "period2": PERIOD2, "interval": "1d",
                "events": "div|split", "includeAdjustedClose": "true"}
    full = {"range": "max", "interval": "1d",
            "events": "div|split", "includeAdjustedClose": "true"}
    delay = 2.0
    for attempt in range(1, MAX_TRIES + 1):
        params = windowed if attempt == 1 else full
        try:
            r = requests.get(BASE.format(sym), params=params, headers=UA, timeout=45)
            if r.status_code == 404:
                return None            # symbol genuinely unknown to Yahoo
            if r.status_code == 400 and attempt < MAX_TRIES:
                time.sleep(1.0)
                continue               # no data in that window -- retry with range=max
            if r.status_code == 200:
                res = r.json().get("chart", {}).get("result")
                if not res:
                    return None
                res = res[0]
                q = res["indicators"]["quote"][0]
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
                if not res.get("timestamp"):
                    return None
                df = pd.DataFrame({
                    "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
                    "close": q.get("close"),
                    "low": q.get("low"),
                    "adjclose": adj if adj is not None else q.get("close"),
                })
                df = df.dropna(subset=["date"]).set_index("date")
                lo = pd.Timestamp(PERIOD1, unit="s")
                hi = pd.Timestamp(PERIOD2, unit="s")
                df = df[(df.index >= lo) & (df.index <= hi)]
                return df if not df.empty else None
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 45)
    raise RuntimeError(f"{ticker}: exhausted retries")


def main() -> None:
    PRICES.mkdir(parents=True, exist_ok=True)
    tickers = universe()
    missing_log = CACHE / "no_history.txt"
    known_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()

    todo = [t for t in tickers
            if not (PRICES / f"{t}.csv").exists() and t not in known_missing]
    print(f"universe {len(tickers)} | already have "
          f"{len(tickers) - len(todo) - len(known_missing & set(tickers))} | "
          f"no history {len(known_missing)} | to fetch {len(todo)}", flush=True)

    got = 0
    for i, t in enumerate(todo, 1):
        try:
            df = fetch_one(t)
        except RuntimeError as e:
            print(f"  [{i}/{len(todo)}] {e}", flush=True)
            continue
        if df is None or df.empty:
            known_missing.add(t)
            missing_log.write_text("\n".join(sorted(known_missing)))
        else:
            df.to_csv(PRICES / f"{t}.csv")
            got += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(todo)}] fetched={got} no-history={len(known_missing)}",
                  flush=True)
        time.sleep(PAUSE)

    print(f"\nDONE. files={len(list(PRICES.glob('*.csv')))} "
          f"no_history={len(known_missing)} of {len(tickers)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
