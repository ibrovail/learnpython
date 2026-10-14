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

THE FREE TIER'S THREE LIMITS, all measured against this key on 2026-10-14:

  * 50 requests per HOUR.   Over it: HTTP 429, "over your hourly request allocation".
  * 1000 requests per day.
  * 500 UNIQUE SYMBOLS per MONTH, and the universe is 755. Over it Tiingo answers
    HTTP 200 with a JSON OBJECT -- {"detail": "You have run over your 500 symbol look up
    for this month."} -- not the usual array.

The 200-with-an-object is the dangerous one. A quota refusal recorded as "no history"
would silently shrink the universe, and the names it dropped would be whatever came last
in the sweep rather than a random sample. Quota refusals are therefore logged separately
and counted against coverage as MISSING, which is the honest accounting. Symbols already
looked up this month do not consume new budget, so a re-run still recovers them.

SWEEP ORDER IS A FIXED-SEED RANDOM PERMUTATION, not alphabetical. The hourly cap means
any run inside a session covers only part of the universe, so the part it covers has to be
an unbiased sample -- otherwise the measured coverage describes the start of the alphabet
rather than the universe, and coverage is the study's gate.

    python3 -I research/fetch_prices_tiingo.py
"""
from __future__ import annotations

import collections
import json
import os
import random
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

REQUESTS_PER_HOUR = 50          # the measured free-tier ceiling
SWEEP_SEED = 20261014           # fixed, so the sample is reproducible
MAX_TRIES = 3


class MissingKey(RuntimeError):
    pass


class QuotaExceeded(RuntimeError):
    """The month's unique-symbol allowance is spent. Not the same as "no such ticker"."""


class HourlyLimit(RuntimeError):
    """The hourly request allowance is spent. Recoverable by waiting."""


class RateGate:
    """Hold the caller to REQUESTS_PER_HOUR by waiting out the oldest request."""

    def __init__(self, per_hour: int = REQUESTS_PER_HOUR) -> None:
        self.per_hour = per_hour
        self.stamps: collections.deque[float] = collections.deque()

    def take(self) -> None:
        now = time.time()
        while self.stamps and now - self.stamps[0] > 3600:
            self.stamps.popleft()
        if len(self.stamps) >= self.per_hour:
            wait = 3600 - (now - self.stamps[0]) + 2
            print(f"    rate gate: sleeping {wait/60:.1f} min for the hourly window",
                  flush=True)
            time.sleep(max(wait, 0))
            self.take()
            return
        self.stamps.append(time.time())


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


def sweep_order(tickers: list[str]) -> list[str]:
    order = list(tickers)
    random.Random(SWEEP_SEED).shuffle(order)
    return order


def tiingo_symbol(ticker: str) -> str:
    """Tiingo uses hyphens for share classes, as Yahoo does: BRK.B -> BRK-B."""
    return ticker.replace(".", "-")


def fetch_one(ticker: str, key: str, gate: RateGate) -> pd.DataFrame | None:
    """Adjusted daily close and low. None only when Tiingo has no history for the symbol."""
    params = {"startDate": START, "endDate": END, "format": "json"}
    headers = {"Content-Type": "application/json", "Authorization": f"Token {key}"}
    delay = 2.0
    for _ in range(MAX_TRIES):
        gate.take()
        try:
            r = requests.get(BASE.format(tiingo_symbol(ticker)),
                             params=params, headers=headers, timeout=45)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                raise HourlyLimit(r.text[:120])
            if r.status_code == 200:
                rows = r.json()
                if isinstance(rows, dict):
                    # Tiingo reports quota refusals and other errors as a 200 + object.
                    detail = str(rows.get("detail", rows))
                    if "symbol look up" in detail or "over your" in detail:
                        raise QuotaExceeded(detail)
                    return None
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                df["date"] = (pd.to_datetime(df["date"], utc=True)
                              .dt.tz_localize(None).dt.normalize())
                out = df[["date", "adjClose", "adjLow", "close"]].rename(
                    columns={"adjClose": "adjclose", "adjLow": "low"}
                )
                return out.set_index("date")
        except (QuotaExceeded, HourlyLimit):
            raise
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
    quota_log = CACHE / "quota_blocked_tiingo.txt"
    known_missing = set(missing_log.read_text().split()) if missing_log.exists() else set()
    # Quota blocks are NOT sticky: the monthly allowance resets, and a symbol blocked in
    # one run is retried in the next. Only a real absence is remembered across runs.
    blocked: set[str] = set()
    todo = [t for t in sweep_order(tickers)
            if not (PRICES / f"{t}.csv").exists() and t not in known_missing]
    print(f"universe {len(tickers)} | already on disk "
          f"{len(tickers) - len(todo) - len(known_missing)} | to fetch {len(todo)}",
          flush=True)

    gate, got, attempted = RateGate(), 0, 0
    for i, t in enumerate(todo, 1):
        try:
            df = fetch_one(t, key, gate)
        except HourlyLimit:
            # The gate underestimated the window (earlier requests in this hour came from
            # outside this process). Wait a full window out rather than burning 429s.
            print(f"  [{i}/{len(todo)}] hourly limit hit at {t}; sleeping 61 min",
                  flush=True)
            time.sleep(61 * 60)
            gate = RateGate()
            continue
        except QuotaExceeded:
            blocked.add(t)
            quota_log.write_text("\n".join(sorted(blocked)))
            attempted += 1
            continue
        except RuntimeError as e:
            print(f"  [{i}/{len(todo)}] {e}", flush=True)
            continue
        attempted += 1
        if df is None or df.empty:
            known_missing.add(t)
            missing_log.write_text("\n".join(sorted(known_missing)))
        else:
            df.to_csv(PRICES / f"{t}.csv")
            got += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(todo)}] fetched={got} no-history={len(known_missing)} "
                  f"quota-blocked={len(blocked)}", flush=True)

    print(f"\nDONE. files={len(list(PRICES.glob('*.csv')))} "
          f"no_history={len(known_missing)} quota_blocked={len(blocked)} "
          f"of {len(tickers)}", flush=True)
    if blocked:
        print(f"Quota wall: {len(blocked)} of {attempted} symbols attempted this run were\n"
              "refused for the month's 500-unique-symbol cap. They are MISSING for coverage\n"
              "purposes -- not absent data. Re-run after the allowance resets.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
