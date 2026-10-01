"""
perplexity_brief.py

Build the discovery brief for one holding or candidate: a verified peer/sector comparison
computed here, plus the Perplexity Finance URL to open for candidate EXPLANATIONS.

The division of labour is the one in .claude/rules/price-data-integrity.md: Perplexity and
WebSearch are discovery -- they tell you what you did not know to look for -- and price data,
filings and quote pages supply the facts. So this script computes the numbers (peer moves,
sector proxy, relative move) and prints the page's URL with a checklist, rather than scraping
a narrative and believing it.

Why it exists: on 2026-09-30 ATRC fell 3.19% and the five-category unexplained-move check cost
six browser fetches and two web searches, of which the decisive one -- that the sector proxy
should have been XLV, not biotech XBI -- was found by hand. Two headlines surfaced in that
search would have explained the move and both were stale (2026-07-14 and 2025-09-25); dating
them took two more fetches. The peer set here is picked from the screener's own cached universe
by industry and market cap, so it is reproducible rather than ad hoc.

Usage:
  venv/bin/python perplexity_brief.py ATRC              # one name
  venv/bin/python perplexity_brief.py                   # every current holding
  venv/bin/python perplexity_brief.py ATRC --peers 7
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trading_script as ts  # noqa: E402  (price fetching + the sector proxy map)


def _universe(data_dir: Path) -> pd.DataFrame:
    cache = data_dir / "universe_cache.csv"
    if not cache.exists():
        raise SystemExit(f"perplexity_brief: {cache} not found -- run `make screen` first.")
    df = pd.read_csv(cache, usecols=["ticker", "sector", "industry", "market_cap"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.dropna(subset=["market_cap"])


def pick_peers(ticker: str, uni: pd.DataFrame, n: int) -> tuple[list[str], str]:
    """Nearest peers by market cap within the same industry, falling back to the sector."""
    row = uni[uni.ticker == ticker]
    if row.empty:
        return [], f"{ticker} is not in the cached universe -- no peers chosen"
    industry, sector, cap = row.iloc[0].industry, row.iloc[0].sector, float(row.iloc[0].market_cap)
    pool = uni[(uni.industry == industry) & (uni.ticker != ticker)]
    basis = f"same industry ({industry})"
    if len(pool) < n:
        pool = uni[(uni.sector == sector) & (uni.ticker != ticker)]
        basis = f"same sector ({sector}) -- too few in {industry}"
    # Closeness in LOG market cap: a $1Bn gap means something very different at $0.5Bn than
    # at $5Bn, and this universe spans both.
    pool = pool.assign(gap=(pool.market_cap.apply(
        lambda m: abs(math.log(max(m, 1.0)) - math.log(max(cap, 1.0))))))
    return pool.nsmallest(n, "gap").ticker.tolist(), basis


def moves(tickers: list[str], end: pd.Timestamp) -> dict[str, tuple[float, float]]:
    """{ticker: (1-session %, 1-week %)} from price history, not from any page."""
    out = {}
    for t in tickers:
        try:
            df = ts.download_price_data(t, start=end - pd.Timedelta(days=21),
                                        end=end + pd.Timedelta(days=1), progress=False).df
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            c = df[df.index <= end]["Close"].astype(float).dropna()
            out[t] = (100 * (c.iloc[-1] / c.iloc[-2] - 1),
                      100 * (c.iloc[-1] / c.iloc[-6] - 1) if len(c) >= 6 else float("nan"))
        except Exception:
            out[t] = (float("nan"), float("nan"))
    return out


def brief(ticker: str, uni: pd.DataFrame, n: int, end: pd.Timestamp) -> None:
    peers, basis = pick_peers(ticker, uni, n)
    row = uni[uni.ticker == ticker]
    sector = row.iloc[0].sector if not row.empty else "UNKNOWN"
    proxy = ts.SECTOR_PROXY.get(sector, ts.SECTOR_PROXY_FALLBACK)
    m = moves([ticker] + peers + [proxy], end)
    own = m[ticker][0]
    peer_days = sorted(v[0] for v in (m[p] for p in peers) if v[0] == v[0])
    peer_med = peer_days[len(peer_days) // 2] if peer_days else float("nan")

    print(f"\n=== {ticker} · {sector} · session {end.date()} ===")
    print(f"peers: {', '.join(peers) or '(none)'}  [{basis}]")
    print(f"\nOPEN (discovery -- candidate explanations; date every claim before weighing it):")
    print(f"  https://www.perplexity.ai/finance/{ticker}?comparing={','.join([ticker] + peers)}")
    print("\nVERIFIED HERE (price history):")
    print(f"  {'ticker':8}{'1-session':>11}{'1-week':>10}")
    for t in [ticker] + peers + [proxy]:
        tag = "  <- subject" if t == ticker else ("  <- sector proxy" if t == proxy else "")
        print(f"  {t:8}{m[t][0]:>+10.2f}%{m[t][1]:>+9.2f}%{tag}")
    rel_sector = own - m[proxy][0]
    print(f"\n  vs sector proxy : " +
          (f"{rel_sector:+.2f}pp" if rel_sector == rel_sector else "n/a (no proxy data)"))
    print("  vs peer median  : " +
          (f"{own - peer_med:+.2f}pp" if peer_med == peer_med else "n/a (no peers)"))
    print("""
CHECKLIST (price-data-integrity.md -- the page is discovery, not evidence):
  [ ] date every claim on the page; a headline that reads current is often months old
  [ ] any claim that would move a decision -> verify on the quote page, filing or price history
  [ ] insider sale -> shares as a % of the session's volume, and $ vs market cap, before weighing
  [ ] analyst action -> confirm the consensus target actually moved; the wire never carries these
  [ ] index event -> S&P DJI / FTSE Russell release, which no single-name source carries""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("tickers", nargs="*", help="default: every current holding")
    ap.add_argument("--peers", type=int, default=5)
    ap.add_argument("--data-dir", default="Start Your Own")
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    ts.set_data_dir(data_dir)
    uni = _universe(data_dir)
    end = ts.last_completed_session()

    tickers = [t.upper() for t in a.tickers]
    if not tickers:
        pf, _ = ts.load_latest_portfolio_state()
        pf = pd.DataFrame(pf) if isinstance(pf, list) else pf
        tickers = [] if pf.empty else [str(t).upper() for t in pf["ticker"]]
    if not tickers:
        raise SystemExit("perplexity_brief: no holdings and no tickers given.")
    for t in tickers:
        brief(t, uni, a.peers, end)


if __name__ == "__main__":
    main()
