"""Research-arm review (2026-09-19): every closed trade scored against SPY and IWM over its own
holding window, exits by type, what each stock did in the 20 sessions after it was sold, and
whether research picks beat the screener top-15 they were chosen from. Reconciles to the Week 52
readout (post-pivot to 9/09: 25 sell events, +$83.59). Run: venv/bin/python research/arm_review.py
"""
import pandas as pd, numpy as np, yfinance as yf, glob, os, warnings
warnings.filterwarnings("ignore")
PIVOT = pd.Timestamp("2026-04-13")

tl = pd.read_csv("Start Your Own/chatgpt_trade_log.csv")
tl["Date"] = pd.to_datetime(tl["Date"])
tl["is_buy"] = tl["Reason"].str.contains("BUY", case=False, na=False)
tl["is_sell"] = tl["Reason"].str.contains("SELL", case=False, na=False)

# ---- pair each sell with the entry of its open position ----
trades, open_since, held = [], {}, {}
for _, r in tl.sort_values("Date", kind="stable").iterrows():
    t = r["Ticker"]
    if r["is_buy"]:
        open_since.setdefault(t, r["Date"])
        held[t] = held.get(t, 0) + float(r["Shares Bought"] or 0)
    elif r["is_sell"] and pd.notna(r["Shares Sold"]) and r["Shares Sold"] > 0:
        sh, px, pnl = float(r["Shares Sold"]), float(r["Sell Price"]), float(r["PnL"])
        cost = px - pnl / sh
        trades.append({"ticker": t, "entry": open_since.get(t, pd.NaT), "exit": r["Date"],
                       "cost": cost, "exit_px": px, "shares": sh, "pnl": pnl,
                       "ret": px / cost - 1 if cost > 0 else np.nan,
                       "kind": "stop" if "STOP" in str(r["Reason"]).upper() else "discretionary"})
        # close the position only when the share count reaches zero; a partial sale
        # leaves the entry date in place for the remaining lot
        held[t] = held.get(t, 0) - sh
        if held[t] <= 1e-9:
            open_since.pop(t, None); held[t] = 0
tr = pd.DataFrame(trades)
print(f"sell events: {len(tr)}; with a matched entry: {tr['entry'].notna().sum()}")
tr = tr.dropna(subset=["entry"])
tr["era"] = np.where(tr["entry"] >= PIVOT, "post-pivot", "pre-pivot")

# ---- prices ----
tick = sorted(set(tr["ticker"])) + ["SPY", "IWM"]
px = yf.download(tick, start="2025-09-01", end="2026-09-19", auto_adjust=True, progress=False)["Close"]
sess = px.index
def on_or_after(d):
    i = sess.searchsorted(pd.Timestamp(d)); return sess[min(i, len(sess)-1)]
def bench(sym, a, b):
    a, b = on_or_after(a), on_or_after(b); return px[sym].loc[b] / px[sym].loc[a] - 1
tr["spy"] = [bench("SPY", a, b) for a, b in zip(tr["entry"], tr["exit"])]
tr["iwm"] = [bench("IWM", a, b) for a, b in zip(tr["entry"], tr["exit"])]
tr["vs_spy"] = tr["ret"] - tr["spy"]
tr["vs_iwm"] = tr["ret"] - tr["iwm"]
tr["days"] = [np.searchsorted(sess, on_or_after(b)) - np.searchsorted(sess, on_or_after(a)) for a, b in zip(tr["entry"], tr["exit"])]

# after-exit: what did the stock do in the next 20 sessions (did exiting help?)
def after(t, d, n=20):
    i = np.searchsorted(sess, on_or_after(d))
    if i + n >= len(sess) or t not in px or pd.isna(px[t].iloc[i]) or pd.isna(px[t].iloc[i+n]): return np.nan
    return px[t].iloc[i+n] / px[t].iloc[i] - 1 - (px["SPY"].iloc[i+n] / px["SPY"].iloc[i] - 1)
tr["after20_vs_spy"] = [after(t, d) for t, d in zip(tr["ticker"], tr["exit"])]

def summ(g):
    return pd.Series({"trades": len(g), "win%": (g["ret"] > 0).mean()*100,
                      "avg ret%": g["ret"].mean()*100, "med ret%": g["ret"].median()*100,
                      "avg vs S&P pp": g["vs_spy"].mean()*100, "med vs S&P pp": g["vs_spy"].median()*100,
                      "beat S&P %": (g["vs_spy"] > 0).mean()*100,
                      "avg vs IWM pp": g["vs_iwm"].mean()*100,
                      "avg days held": g["days"].mean(), "net $": g["pnl"].sum()})

print("=== 1. EVERY CLOSED TRADE vs the S&P over its own holding window ===")
print(tr.groupby("era").apply(summ).round(1).T.to_string())

print("\n=== 2. BY EXIT TYPE (post-pivot) ===")
pp = tr[tr["era"] == "post-pivot"]
print(pp.groupby("kind").apply(summ).round(1).T.to_string())

print("\n=== 3. AFTER THE EXIT: stock vs S&P over the next 20 sessions (positive = selling cost you) ===")
for era in ("pre-pivot", "post-pivot"):
    for k in ("stop", "discretionary"):
        g = tr[(tr["era"] == era) & (tr["kind"] == k)]["after20_vs_spy"].dropna()
        if len(g):
            print(f"  {era:<10} {k:<13} n={len(g):>2}  avg {g.mean()*100:+6.2f}pp  median {g.median()*100:+6.2f}pp  "
                  f"kept rising after exit: {(g > 0).mean():.0%}")

print("\n=== 4. WINNERS vs LOSERS (post-pivot): where did the money come from? ===")
w, l = pp[pp["pnl"] > 0], pp[pp["pnl"] <= 0]
print(f"  winners n={len(w)} total ${w['pnl'].sum():+.2f}  avg {w['ret'].mean()*100:+.1f}%  held {w['days'].mean():.0f}d")
print(f"  losers  n={len(l)} total ${l['pnl'].sum():+.2f}  avg {l['ret'].mean()*100:+.1f}%  held {l['days'].mean():.0f}d")
print(f"  largest 3 winners: " + ", ".join(f"{r.ticker} ${r.pnl:+.2f}" for r in pp.nlargest(3, 'pnl').itertuples()))
print(f"  largest 3 losers : " + ", ".join(f"{r.ticker} ${r.pnl:+.2f}" for r in pp.nsmallest(3, 'pnl').itertuples()))

# ---- 5. research selection vs the screener list it chose from ----
wls = []
for f in sorted(glob.glob("research/data/watchlists/*.csv")):
    d = pd.Timestamp(os.path.basename(f)[:10])
    try:
        w_ = pd.read_csv(f); w_["ticker"] = w_["ticker"].astype(str).str.upper().str.strip()
        wls.append((d, w_))
    except Exception: pass
print(f"\n=== 5. DID RESEARCH BEAT THE LIST IT PICKED FROM? ({len(wls)} watchlist snapshots) ===")
rows = []
for r in pp.itertuples():
    prior = [(d, w_) for d, w_ in wls if d <= r.entry]
    if not prior: continue
    d, w_ = prior[-1]
    listed = r.ticker in set(w_["ticker"].head(15))
    names = [t for t in w_["ticker"].head(15) if t in px.columns]
    a, b = on_or_after(r.entry), on_or_after(r.exit)
    lst = [px[t].loc[b] / px[t].loc[a] - 1 for t in names if pd.notna(px[t].loc[a]) and pd.notna(px[t].loc[b])]
    rows.append({"ticker": r.ticker, "on_list": listed, "pick": r.ret,
                 "list_mean": np.mean(lst) if lst else np.nan,
                 "list_median": np.median(lst) if lst else np.nan})
sel = pd.DataFrame(rows)
if not sel.empty:
    sel["pick_vs_list_mean"] = sel["pick"] - sel["list_mean"]
    print(f"  post-pivot entries with a prior watchlist: {len(sel)};  picked from the top-15: {sel['on_list'].sum()}  "
          f"({sel['on_list'].mean():.0%}); from elsewhere: {(~sel['on_list']).sum()}")
    for lab, g in (("picked FROM the list", sel[sel["on_list"]]), ("picked OFF the list", sel[~sel["on_list"]])):
        if len(g):
            print(f"  {lab:<21} n={len(g):>2}  pick avg {g['pick'].mean()*100:+6.1f}%  | same-window list avg "
                  f"{g['list_mean'].mean()*100:+6.1f}%  | pick minus list: avg {g['pick_vs_list_mean'].mean()*100:+6.1f}pp, "
                  f"beat the list on {(g['pick_vs_list_mean'] > 0).mean():.0%}")

tr.to_csv("research/output/trades_scored.csv", index=False)
print(f"\n(trades scored: {len(tr)}; per-trade detail saved)")

print("\n=== RECONCILIATION vs Week 52 readout (post-pivot: 25 closed trades, +$83.59 realised, as of 9/09) ===")
pe = tr[(tr["exit"] >= PIVOT) & (tr["exit"] <= pd.Timestamp("2026-09-09"))]
print(f"  by exit date, pivot -> 9/09: {len(pe)} sell events, net ${pe['pnl'].sum():+.2f}")
print(f"  whole log: {len(tr)} sell events, net ${tr['pnl'].sum():+.2f}   (readout: 82 trades, -$3.84)")
led = pd.read_csv("Start Your Own/chatgpt_portfolio_update.csv")
a = led[(led["Ticker"] == "ATRC")].iloc[-1]
print(f"\n=== THE OPEN POSITION === ATRC on {a['Date']}: {a['Shares']:.0f} sh, cost ${a['Cost Basis']:.2f}, "
      f"value ${a['Total Value']:.2f}, unrealised ${a['PnL']:+.2f}")
first = tl[(tl["Ticker"] == "ATRC") & tl["is_buy"]]["Date"].min()
prior = [(d, w_) for d, w_ in wls if d <= first]
if prior:
    d, w_ = prior[-1]; lst = list(w_["ticker"].head(15))
    print(f"  ATRC first bought {first.date()}; latest watchlist before that ({d.date()}): "
          f"{'ON the top-15 at #' + str(lst.index('ATRC')+1) if 'ATRC' in lst else 'NOT on the top-15'}")
