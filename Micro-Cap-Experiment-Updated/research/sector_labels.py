"""Sector labels for the backtest frame, from SEC EDGAR SIC codes. Free, survivorship-free.

The sector cap (at most 3 positions in one GICS sector) is one of the last untested rules in
portfolio_rules.md. Testing it needs a sector label for every name in the frame INCLUDING the
delisted ones -- and the obvious sources (yfinance info, Finviz) serve current listings only, so
they would reintroduce exactly the survivor bias the panel was built to remove.

EDGAR assigns every filer a SIC code and keeps it after delisting. 261 of the 373 panel names
already have a validated CIK from the earnings harvest; the rest are recovered the same way
(current map, else full-text search validated against the ticker's Tiingo listing window).

A SIC-derived sector is a PROXY for GICS, not GICS. The two disagree at the edges -- SIC 2834
is Health Care in both, but SIC 7372 (prepackaged software) spans GICS Information Technology
and Communication Services. What the cap actually needs is a correlation cluster, and a 2-digit
SIC group is a defensible one; the disagreements are reported, not hidden.

    venv/bin/python research/sector_labels.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT, DATA = ROOT / "output", ROOT / "data.nosync"
S = requests.Session()
S.headers.update({"User-Agent": "micro-cap research iayenajeh@gmail.com"})
PAUSE = 0.15

# 4-digit SIC codes that sit in a different sector from their 2-digit group. Checked first.
SIC4 = {
    "2833": "Health Care", "2834": "Health Care", "2835": "Health Care", "2836": "Health Care",
    "3826": "Health Care", "3841": "Health Care", "3842": "Health Care", "3843": "Health Care",
    "3844": "Health Care", "3845": "Health Care", "3851": "Health Care", "8731": "Health Care",
    "6798": "Real Estate", "6500": "Real Estate", "6510": "Real Estate", "6512": "Real Estate",
    "6519": "Real Estate", "6531": "Real Estate", "6552": "Real Estate", "6798": "Real Estate",
    "3711": "Consumer Discretionary", "3713": "Consumer Discretionary",
    "3714": "Consumer Discretionary", "3716": "Consumer Discretionary",
    "3751": "Consumer Discretionary", "5912": "Consumer Staples", "5411": "Consumer Staples",
    "7812": "Communication Services", "7819": "Communication Services",
    "7822": "Communication Services", "7829": "Communication Services",
    "7841": "Communication Services", "7900": "Communication Services",
    "2711": "Communication Services", "2721": "Communication Services",
    "2731": "Communication Services", "2741": "Communication Services",
}
# 2-digit SIC major group -> sector. The fallback, and what most names resolve on.
SIC2 = {
    "01": "Consumer Staples", "02": "Consumer Staples", "07": "Consumer Staples",
    "08": "Materials", "09": "Consumer Staples",
    "10": "Materials", "12": "Energy", "13": "Energy", "14": "Materials",
    "15": "Industrials", "16": "Industrials", "17": "Industrials",
    "20": "Consumer Staples", "21": "Consumer Staples", "22": "Consumer Discretionary",
    "23": "Consumer Discretionary", "24": "Industrials", "25": "Consumer Discretionary",
    "26": "Materials", "27": "Communication Services", "28": "Materials", "29": "Energy",
    "30": "Materials", "31": "Consumer Discretionary", "32": "Materials", "33": "Materials",
    "34": "Industrials", "35": "Information Technology", "36": "Information Technology",
    "37": "Industrials", "38": "Health Care", "39": "Consumer Discretionary",
    "40": "Industrials", "41": "Industrials", "42": "Industrials", "44": "Industrials",
    "45": "Industrials", "46": "Energy", "47": "Industrials",
    "48": "Communication Services", "49": "Utilities",
    "50": "Industrials", "51": "Industrials",
    "52": "Consumer Discretionary", "53": "Consumer Discretionary",
    "54": "Consumer Staples", "55": "Consumer Discretionary", "56": "Consumer Discretionary",
    "57": "Consumer Discretionary", "58": "Consumer Discretionary", "59": "Consumer Discretionary",
    "60": "Financials", "61": "Financials", "62": "Financials", "63": "Financials",
    "64": "Financials", "65": "Real Estate", "67": "Financials",
    "70": "Consumer Discretionary", "72": "Consumer Discretionary",
    "73": "Information Technology", "75": "Consumer Discretionary",
    "76": "Industrials", "78": "Communication Services", "79": "Consumer Discretionary",
    "80": "Health Care", "81": "Industrials", "82": "Consumer Discretionary",
    "83": "Industrials", "87": "Industrials", "89": "Industrials", "99": "Industrials",
}


def get(url):
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def sector_of(sic: str) -> str:
    sic = str(sic).strip().zfill(4)
    if sic in SIC4:
        return SIC4[sic]
    return SIC2.get(sic[:2], "Unclassified")


def submission(cik: str) -> dict | None:
    r = get(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")
    return r.json() if r is not None else None


def search_ciks(ticker: str) -> list[str]:
    r = get(f'https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22')
    if r is None:
        return []
    counts: dict[str, int] = {}
    for h in r.json().get("hits", {}).get("hits", []):
        for c in h.get("_source", {}).get("ciks", []):
            counts[c] = counts.get(c, 0) + 1
    return [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:3]


def filing_window_ok(js: dict, start, end) -> bool:
    """The right company's filings sit inside the window the ticker actually traded."""
    rec = js.get("filings", {}).get("recent", {})
    d = pd.to_datetime(pd.Series(rec.get("filingDate", [])), errors="coerce").dropna()
    if d.empty:
        return False
    inside = ((d >= start - pd.Timedelta(days=365))
              & (d <= end + pd.Timedelta(days=365))).mean()
    return bool(inside >= 0.5)


def main() -> None:
    panel_t = set(pd.read_csv(OUT / "regime_backtest_panel_survivors.csv",
                              usecols=["ticker"]).ticker.unique())
    samp = pd.read_csv(OUT / "regime_backtest_sample.csv", parse_dates=["start", "end"])
    samp = samp[samp.ticker.isin(panel_t)].reset_index(drop=True)
    known = (pd.read_csv(DATA / "earnings_dates.csv", dtype={"cik": str})
             .groupby("ticker").cik.first().to_dict())
    print(f"{len(samp)} panel names | {sum(t in known for t in samp.ticker)} with a validated CIK",
          flush=True)

    rows, failed = [], []
    for i, r in samp.iterrows():
        t = str(r.ticker).upper()
        js, cik = None, known.get(t)
        if cik:
            js = submission(cik); time.sleep(PAUSE)
        else:
            for c in search_ciks(t):
                time.sleep(PAUSE)
                j = submission(c); time.sleep(PAUSE)
                if j and filing_window_ok(j, r.start, r.end):
                    js, cik = j, c
                    break
        if js is None or not js.get("sic"):
            failed.append(t); continue
        rows.append({"ticker": t, "cik": str(cik).zfill(10), "sic": str(js["sic"]).zfill(4),
                     "sic_desc": js.get("sicDescription", ""),
                     "company": js.get("name", ""), "delisted": bool(r.delisted),
                     "sector": sector_of(js["sic"])})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samp)} | labelled {len(rows)} | unresolved {len(failed)}",
                  flush=True)

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sector_labels.csv", index=False)
    print(f"\n=== coverage ===", flush=True)
    print(f"  labelled {len(out)} of {len(samp)} names ({len(out)/len(samp):.0%})", flush=True)
    if len(out):
        print(out.groupby("delisted").size().rename("names").to_string(), flush=True)
        print("\n=== sector mix (names) ===", flush=True)
        print(out.sector.value_counts().to_string(), flush=True)
        print(f"\n  unclassified SIC: {(out.sector=='Unclassified').sum()}", flush=True)
    print(f"\n  unresolved: {len(failed)} e.g. {failed[:10]}", flush=True)
    print(f"\nwrote {OUT/'sector_labels.csv'}", flush=True)


if __name__ == "__main__":
    main()
