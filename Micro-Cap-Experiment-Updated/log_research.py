"""
log_research.py

Append one row per researched candidate to "Start Your Own/research_log.csv" -- bought,
passed or put on watch. Adopted 2026-09-19 (review recommendation R4).

Why: the 2026-09-19 review scored every closed trade and could not say whether research adds
value beyond the screener list it chose from, because only BUYS were ever recorded. A pass is
half the evidence. Scoring passes against buys over the same windows is the only way to learn
it (`research/score_research_log.py`).

Append-only by design. The file is never rewritten, so a malformed call cannot truncate it --
the failure mode that once emptied the portfolio ledger (2026-06-23).

Usage:
  venv/bin/python log_research.py --week 54 --ticker XYZ --source "screener #12" \
      --decision PASS --reason-code extended --reason "44% above the 50-day" --ref-price 12.34

  # several rows at once (a JSON list of objects with the same keys, dashes -> underscores)
  venv/bin/python log_research.py --batch rows.json
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

LOG = Path("Start Your Own") / "research_log.csv"

COLUMNS = ["date", "week", "ticker", "source", "rank", "sector", "market_cap_bn", "decision",
           "reason_code", "reason", "ref_price", "conviction", "driver"]

DECISIONS = {"BUY", "PASS", "WATCH"}

REASON_CODES = {
    # why a name was bought
    "thesis": "thesis clears every gate; the buy case",
    # why a name was passed or watched
    "shrinking-revenue": "TTM revenue or Sales Q/Q negative",
    "negative-earnings": "TTM EPS / net income negative without a credible path",
    "extended": "too far above the 20- or 50-day SMA, or days 1-3 of a >10% breakout",
    "earnings-window": "earnings inside the next 10 sessions (no-initiation guard)",
    "post-earnings": "inside the post-earnings cooldown",
    "binary-thesis": "the thesis is a pass/fail event",
    "prohibited": "prohibited business",
    "driver-stale": "thesis driver falling or unverifiable (thesis-input freshness)",
    "weak-catalyst": "no credible catalyst or thesis",
    "valuation": "price at or above analyst targets; upside unconvincing",
    "liquidity": "fails ADV, spread or float limits",
    "correlated": "would breach the driver cap or the sector cap",
    "deal-pinned": "trading under a pending acquisition",
    "capacity": "no slot or no cash; a good name that could not be bought",
    "other": "anything else -- say what in --reason",
}


def _row(d: dict) -> dict:
    row = {c: d.get(c, "") for c in COLUMNS}
    row["date"] = row["date"] or date.today().isoformat()
    row["ticker"] = str(row["ticker"]).strip().upper()
    row["decision"] = str(row["decision"]).strip().upper()
    row["reason_code"] = str(row["reason_code"]).strip().lower()
    errors = []
    if not row["ticker"]:
        errors.append("ticker is required")
    if row["decision"] not in DECISIONS:
        errors.append(f"decision must be one of {sorted(DECISIONS)}")
    if row["reason_code"] not in REASON_CODES:
        errors.append(f"reason_code must be one of {sorted(REASON_CODES)}")
    if not str(row["source"]).strip():
        errors.append('source is required, e.g. "screener #12" or "off-list"')
    if row["reason_code"] == "other" and not str(row["reason"]).strip():
        errors.append('reason_code "other" needs --reason')
    for num in ("ref_price", "market_cap_bn", "rank", "week", "conviction"):
        if str(row[num]).strip():
            try:
                float(row[num])
            except ValueError:
                errors.append(f"{num} must be a number")
    if str(row["conviction"]).strip() and not (1 <= float(row["conviction"]) <= 5):
        errors.append("conviction must be 1-5")
    if errors:
        raise ValueError(f"{row['ticker'] or '?'}: " + "; ".join(errors))
    return row


def append(rows: list[dict]) -> None:
    clean = [_row(r) for r in rows]  # validate everything before writing anything
    new = not LOG.exists() or LOG.stat().st_size == 0
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(clean)
    for r in clean:
        print(f"logged {r['date']} {r['ticker']:<6} {r['decision']:<5} {r['reason_code']:<18} "
              f"({r['source']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--batch", help="JSON file: a list of row objects")
    ap.add_argument("--codes", action="store_true", help="List the reason codes and exit")
    for c in COLUMNS:
        ap.add_argument("--" + c.replace("_", "-"), dest=c, default="")
    a = ap.parse_args()

    if a.codes:
        for k, v in REASON_CODES.items():
            print(f"  {k:<18} {v}")
        return
    try:
        if a.batch:
            rows = json.loads(Path(a.batch).read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("--batch file must contain a JSON list")
        else:
            rows = [{c: getattr(a, c) for c in COLUMNS}]
        append(rows)
    except (ValueError, json.JSONDecodeError) as e:
        sys.exit(f"log_research: nothing written -- {e}")


if __name__ == "__main__":
    main()
