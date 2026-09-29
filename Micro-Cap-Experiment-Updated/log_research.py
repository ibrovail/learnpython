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
  venv/bin/python log_research.py --week 54 --ticker XYZ --source "screener #12" --stage 1 \
      --decision PASS --reason-code shrinking-revenue --reason "TTM revenue -6%" --ref-price 12.34

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

COLUMNS = ["date", "week", "ticker", "source", "rank", "sector", "market_cap_bn", "stage",
           "decision", "reason_code", "reason", "ref_price", "conviction", "driver",
           "pct_vs_sma50"]

# pct_vs_sma50: the candidate's distance above its 50-day SMA at decision time, in percent.
# Added 2026-09-29. The trend gate is a hard rule at entry but says nothing about MARGIN, and
# the log could not be asked whether a thin pass predicts a worse outcome -- the question the
# HOPE post-mortem raised, HOPE having been bought at +0.15% and stopped out six sessions later
# while CON at +6.16% held. Auto-filled from that week's watchlist by ticker, so it cannot be
# forgotten; pass --pct-vs-sma50 explicitly for an off-list name the screen never ranked.
WATCHLISTS = (Path("Start Your Own") / "watchlist.csv",
              Path("Start Your Own") / "watchlist_extended.csv")

# Research funnel (2026-09-19): stage 1 = quick quote-page check, stage 2 = full research.
# A name killed at stage 1 is logged as PASS with stage 1 -- cheap, and the most useful passes
# for testing which filters work.
STAGES = {"1", "2"}

DECISIONS = {"BUY", "PASS", "WATCH"}

REASON_CODES = {
    # why a name was bought
    "thesis": "thesis clears every gate; the buy case",
    # why a name was passed or watched
    "shrinking-revenue": "TTM revenue or Sales Q/Q negative",
    "negative-earnings": "TTM EPS / net income negative without a credible path",
    "extended": "too far above the 20- or 50-day SMA, or days 1-3 of a >10% breakout",
    "below-50d": "trading below its 50-day SMA (trend rule, applied at stage 1)",
    "earnings-window": "earnings inside the next 10 sessions (no-initiation guard)",
    "post-earnings": "inside the post-earnings cooldown",
    "re-entry-ban": "inside the 10-session blackout after a stop-out",
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


def _vs_sma50(ticker: str) -> str:
    """This week's pct_vs_sma50 for a ticker, from the watchlists; "" when it is not on them."""
    for path in WATCHLISTS:
        if not path.exists():
            continue
        try:
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if str(row.get("ticker", "")).strip().upper() == ticker:
                        val = str(row.get("pct_vs_sma50", "")).strip()
                        if val:
                            return f"{float(val):.2f}"
        except (OSError, ValueError, csv.Error):
            continue  # a missing or malformed watchlist must never block logging
    return ""


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
    row["stage"] = str(row["stage"]).strip()
    if row["stage"] not in STAGES:
        errors.append("stage must be 1 (quick check) or 2 (full research)")
    elif row["decision"] == "BUY" and row["stage"] != "2":
        errors.append("a BUY must come from stage 2 (full research)")
    if row["reason_code"] == "other" and not str(row["reason"]).strip():
        errors.append('reason_code "other" needs --reason')
    if not str(row["pct_vs_sma50"]).strip() and row["ticker"]:
        row["pct_vs_sma50"] = _vs_sma50(row["ticker"])
    for num in ("ref_price", "market_cap_bn", "rank", "week", "conviction", "pct_vs_sma50"):
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
    if not new:
        # Appending rows whose fields do not match the file's own header writes silently
        # misaligned data -- worse than failing, because nothing looks wrong afterwards.
        with LOG.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise ValueError(
                f"{LOG} header does not match COLUMNS. Found {len(header)} columns, expected "
                f"{len(COLUMNS)}. Missing: {[c for c in COLUMNS if c not in header]}. Migrate the "
                f"file (add the column with empty values for existing rows) before logging again."
            )
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
