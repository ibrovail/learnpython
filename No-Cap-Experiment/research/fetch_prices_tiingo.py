"""Stage 1b (preferred source) — daily prices from Tiingo, including delisted tickers.

Yahoo was measured and rejected for this study. Of the 755 tickers that were ever S&P 500
members in the study window, roughly a third return no history at all -- overwhelmingly the
acquired and bankrupt names (ABMD, ACE, AET, AGN, ALTR, ALXN, ANDV, APC, ATVI, BBBY ...).
That is the survivorship bias the study exists to avoid, and it biases in the flattering
direction: a prior-year winner that later collapsed and delisted vanishes from the panel
instead of recording its loss. Yahoo also serves broken history for some LIVE members --
AVB, a current constituent, returns only 2026-07-17 onward.

Tiingo retains delisted tickers, which is why the predecessor experiment used it.

Credentials: read from the environment variable TIINGO_API_KEY. It belongs in the cloud
environment's secrets, never in a file in this repository and never in chat.

    python3 -I research/fetch_prices_tiingo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data.nosync"
PRICES = CACHE / "prices_tiingo"

FIRST_FORMATION, LAST_FORMATION = 2011, 2024
START, END = "2010-01-01", "2025-12-31"
BASE = "https://api.tiingo.com/tiingo/daily/{}/prices"

PAUSE = 0.6
MAX_TRIES = 4


class MissingKey(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        raise MissingKey(
            "TIINGO_API_KEY is not set.\n"
            "Add it as a secret in the cloud environment's settings, then start a new\n"
            "session so the container picks it up. Do not put it in a file in this repo."
        )
    return key


def universe() -> list[str]:
    m = {int(k): set(v) for k, v in
         json.loads((CACHE / "membership.json").read_text()).items()}
    u: set[str] = set()
    for y in range(FIRST_FORMATION, LAST_FORMATION + 1):
        u |= m.get(y, set())
    return sorted(u)


def tiingo_symbol(ticker: str) -> str:
    """Tiingo uses hyphens for share classes, as Yahoo does: BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def fetch_one(ticker: str, key: str) -> pd.DataFrame | None:
    """Adjusted daily close and low. None only when Tiingo has no history for the symbol."""
    params = {"startDate": START, "endDate": END, "format": "json"}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {key}"}
    delay = 2.0
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(BASE.format(tiingo_symbol(ticker)),
                             params=params, headers=headers, timeout=45)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                # Free tier is rate limited. Back off hard rather than burning the quota.
                time.sleep(min(delay * 4, 120))
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                rows = r.json()
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
                out = df[["date", "adjClose", "adjLow", "close"]].rename(
                    columns={"adjClose": "adjclose", "adjLow": "low"}
                )
                return out.set_index("date")
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"{ticker}: exhausted retries")


def main() -> int:
    try:
        key = api_key()
    except MissingKey as e:
        print(e, file=sys.stderr)
        return 2

    PRICES.mkdir(parents=True, exist_ok=True)
    tickers = universe()
    missing_log = CACHE / "no_history_tiingo.txt"
    known_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()
    todo = [t for t in tickers
            if not (PRICES / f"{t}.csv").exists() and t not in known_missing]
    print(f"universe {len(tickers)} | to fetch {len(todo)}", flush=True)

    got = 0
    for i, t in enumerate(todo, 1):
        try:
            df = fetch_one(t, key)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
