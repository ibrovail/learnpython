"""Build a survivorship-free earnings-date table from SEC EDGAR. Free, no API key.

The 10-session pre-earnings guard blocks roughly 23-39 of the top 50 candidates in October and is
the largest constraint in portfolio_rules.md that has never been tested. Its only evidence is a
one-season study from 2026-09-19. Testing it needs historical earnings dates for the backtest
sample, INCLUDING the companies that later delisted -- otherwise the test is survivor-only and
reintroduces the bias the whole panel was built to remove.

An 8-K filing carrying Item 2.02, "Results of Operations and Financial Condition", IS the earnings
release, precisely dated. EDGAR keeps delisted companies' filings permanently, which is what makes
this survivorship-free: TWTR has 27 such filings through 2022-12, ATVI 47 through 2024-06.

The obstacle is mapping ticker -> CIK. EDGAR's company_tickers.json is CURRENT-only, so 182 of the
187 delisted names in the sample are missing from it. Full-text search recovers them, but returns
false positives -- one company's filing mentioning another's ticker. So each candidate CIK is
VALIDATED against the ticker's listed window from the Tiingo snapshot: the right company's
earnings filings fall inside the dates it actually traded.

    venv/bin/python research/earnings_dates.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT, DATA = ROOT / "output", ROOT / "data.nosync"
UA = {"User-Agent": "micro-cap research iayenajeh@gmail.com"}
S = requests.Session(); S.headers.update(UA)
PAUSE = 0.15          # SEC asks for <=10 req/s; this is well inside it


def get(url, **kw):
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                time.sleep(1.5 * (attempt + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def current_map() -> dict[str, str]:
    r = get("https://www.sec.gov/files/company_tickers.json")
    if r is None:
        sys.exit("could not fetch company_tickers.json")
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def search_ciks(ticker: str) -> list[str]:
    """Candidate CIKs from full-text search, most frequently cited first."""
    r = get(f'https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=8-K')
    if r is None:
        return []
    counts: dict[str, int] = {}
    for h in r.json().get("hits", {}).get("hits", []):
        for c in h.get("_source", {}).get("ciks", []):
            counts[c] = counts.get(c, 0) + 1
    return [c for c, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:4]


def earnings_8ks(cik: str) -> list[str]:
    r = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if r is None:
        return []
    rec = r.json().get("filings", {}).get("recent", {})
    f = pd.DataFrame({k: rec.get(k, []) for k in ("form", "filingDate", "items")})
    if f.empty:
        return []
    e = f[(f.form == "8-K") & f["items"].fillna("").str.contains("2.02")]
    return sorted(e.filingDate.tolist())


def main():
    samp = pd.read_csv(OUT / "regime_backtest_sample.csv", parse_dates=["start", "end"])
    cur = current_map()
    print(f"sample {len(samp)} tickers; current EDGAR map has {len(cur)} entries\n", flush=True)

    rows, resolved, failed = [], 0, []
    for i, r in samp.iterrows():
        t = str(r.ticker).upper()
        cands = [cur[t]] if t in cur else search_ciks(t)
        time.sleep(PAUSE)
        chosen, dates = None, []
        for c in cands:
            d = earnings_8ks(c)
            time.sleep(PAUSE)
            if not d:
                continue
            # VALIDATION: the right company's earnings filings sit inside the window the ticker
            # actually traded. This is what rejects a false-positive CIK from full-text search.
            dd = pd.to_datetime(d)
            inside = ((dd >= r.start - pd.Timedelta(days=120))
                      & (dd <= r.end + pd.Timedelta(days=120))).mean()
            if inside >= 0.6:
                chosen, dates = c, d
                break
        if chosen:
            resolved += 1
            for d in dates:
                rows.append({"ticker": t, "cik": chosen, "earnings_date": d})
        else:
            failed.append(t)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samp)} tickers | resolved {resolved} | "
                  f"{len(rows)} earnings dates", flush=True)

    out = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA / "earnings_dates.csv", index=False)
    cov = samp.assign(ok=samp.ticker.str.upper().isin(set(out.ticker)))
    print(f"\n=== coverage ===", flush=True)
    print(f"  tickers resolved: {resolved} of {len(samp)} ({resolved/len(samp):.0%})", flush=True)
    print(f"  earnings dates collected: {len(out)}", flush=True)
    print(cov.groupby("delisted").ok.agg(["size", "sum", "mean"]).rename(
        columns={"size": "tickers", "sum": "resolved", "mean": "rate"}).to_string(), flush=True)
    print(f"\n  unresolved: {len(failed)}  e.g. {failed[:10]}", flush=True)
    print(f"\nwrote {DATA/'earnings_dates.csv'}", flush=True)


if __name__ == "__main__":
    main()
