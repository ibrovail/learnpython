"""Earnings guard study (2026-09-19): is "no initiation within 10 sessions before earnings" earning its keep?

The rule was written 2026-09-17 during the indefinite-rules critique, as a guard against buying a
coin flip by accident. It had never been tested. This asks, for a small-cap bought k sessions
before its report with a stop at 1.75xATR (the book's entry stop):
  1. how often the report takes the stop out, versus same-length windows with no report;
  2. how badly -- the loss in multiples of the planned risk (a gap can blow through a stop, and
     the book's GTC stop-LIMIT orders may not fill at all when price gaps below the limit);
  3. what the rule costs in return -- mean vs mean and median vs median of the window return.
Events come from the Finviz "earnings" column, which the universe snapshots carry only since
2026-09-14 -- so each stock's most recent report: ONE earnings season (Jul-Sep 2026), one market
environment. Prices from the Phase 3.5 cache. Descriptive, not a verdict.
Run: venv/bin/python research/earnings_guard_study.py
"""
import glob, os, re
import numpy as np, pandas as pd

P = pd.read_pickle("research/data/prices.pkl")
C, H, L = P["Close"], P["High"], P["Low"]
S = C.index
prev_c = C.shift(1)
tr = pd.concat({"a": H - L, "b": (H - prev_c).abs(), "c": (L - prev_c).abs()}).groupby(level=1).max()
tr = tr.reindex(S)
ATR = tr.rolling(14).mean()

# ---- events from the snapshots ----
ev = set()
for f in sorted(glob.glob("research/data/universes/*.csv")):
    snap = pd.Timestamp(os.path.basename(f)[:10])
    u = pd.read_csv(f, usecols=lambda c: c in ("ticker", "earnings", "market_cap", "price"))
    if "earnings" not in u.columns: continue          # column exists only from 2026-09-14 on
    u = u[pd.to_numeric(u.get("market_cap"), errors="coerce").fillna(0) <= 5e9]
    for t, e in zip(u["ticker"].astype(str).str.upper(), u["earnings"].astype(str)):
        m = re.match(r"([A-Z][a-z]{2}) (\d{1,2})/([ab])", e)
        if not m or t not in C.columns: continue
        d = pd.to_datetime(f"{m.group(1)} {m.group(2)} {snap.year}", format="%b %d %Y", errors="coerce")
        if pd.isna(d): continue
        if d > snap + pd.Timedelta(days=120): d -= pd.DateOffset(years=1)
        ev.add((t, d.normalize(), m.group(3)))
rows = []
for t, d, when in ev:
    i = S.searchsorted(d)                      # first session on/after the date
    if i >= len(S): continue
    r = i + 1 if (when == "a" and i < len(S) and S[i] == d) else i   # after close -> next session reacts
    if r < 25 or r >= len(S): continue
    rows.append((t, r))
E = pd.DataFrame(rows, columns=["t", "r"]).drop_duplicates()
print(f"earnings events usable: {len(E)} across {E['t'].nunique()} stocks")

# ---- reaction day, in ATR units ----
def at(frame, t, i): return frame[t].iat[i]
mv = []
for t, r in E.itertuples(index=False):
    a, cp = at(ATR, t, r - 1), at(C, t, r - 1)
    if not (a > 0 and cp > 1): continue
    mv.append(((at(C, t, r) - cp) / a, (at(L, t, r) - cp) / a))
mv = pd.DataFrame(mv, columns=["close_atr", "low_atr"]).dropna()
# baseline: ordinary days for the same stocks, excluding +-3 sessions around any event
tick = sorted(set(E["t"])); mask = pd.DataFrame(False, index=S, columns=tick)
for t, r in E.itertuples(index=False): mask.iloc[max(0, r - 3): r + 4, mask.columns.get_loc(t)] = True
cm = ((C[tick] - C[tick].shift(1)) / ATR[tick].shift(1)).where(~mask)
lm = ((L[tick] - C[tick].shift(1)) / ATR[tick].shift(1)).where(~mask)
cm, lm = cm.stack().dropna(), lm.stack().dropna()
print("\n1. THE REPORT DAY vs AN ORDINARY DAY (moves in ATR units; ATR = typical daily range)")
print(f"   median absolute close move:     report {mv['close_atr'].abs().median():.2f}   ordinary {cm.abs().median():.2f}")
print(f"   close moves beyond +-2 ATR:     report {(mv['close_atr'].abs() > 2).mean():.0%}    ordinary {(cm.abs() > 2).mean():.1%}")
print(f"   intraday low below -1.75 ATR:   report {(mv['low_atr'] <= -1.75).mean():.0%}    ordinary {(lm <= -1.75).mean():.1%}")
hit = mv[mv["low_atr"] <= -1.75]
print(f"   when the report breaches -1.75 ATR, the close sits at median {hit['close_atr'].median():.2f} ATR, "
      f"worst decile {hit['close_atr'].quantile(0.1):.2f} ATR")

# ---- the rule's actual scenario: enter k sessions before, stop 1.75 ATR, hold through the report ----
def window(t, e, k):
    """enter at close e, look through e+k; stopped if any low <= entry - 1.75 ATR"""
    a, ce = at(ATR, t, e), at(C, t, e)
    if not (a > 0 and ce > 1) or e + k >= len(S): return None
    lows = L[t].iloc[e + 1: e + k + 1]
    stop = ce - 1.75 * a
    stopped = bool((lows <= stop).any())
    ret = at(C, t, e + k) / ce - 1
    # loss vs plan: a stop fill at the stop costs 1.0x the planned risk; worse = the gap
    first = int(np.argmax((lows <= stop).values)) if stopped else None
    lvl = min(stop, at(C, t, e + 1 + first)) if stopped else None
    worst = (ce - lvl) / (1.75 * a) if stopped else 0.0
    realised = (lvl / ce - 1) if stopped else ret        # what the book actually gets
    return stopped, ret, worst, realised
rng = np.random.default_rng(7)
print("\n2. BUY k SESSIONS BEFORE THE REPORT, STOP AT 1.75 ATR, HOLD THROUGH IT")
print("   Baseline is DATE-MATCHED: for each report window, up to 5 other stocks with no report in")
print("   the same window, same entry date -- so market drift over those dates cancels out.")
for k in (3, 5, 10):
    evw, base = [], []
    for t, r in E.itertuples(index=False):
        e = r - k
        if e < 20: continue
        x = window(t, e, k)
        if not x: continue
        evw.append(x)
        picks = 0
        for j in rng.permutation(len(tick))[:40]:
            u = tick[j]
            if u == t or mask[u].iloc[e: e + k + 4].any(): continue
            y = window(u, e, k)
            if y: base.append(y); picks += 1
            if picks == 5: break
    ev_df, b_df = pd.DataFrame(evw, columns=["stop", "ret", "cost", "real"]), pd.DataFrame(base, columns=["stop", "ret", "cost", "real"])
    print(f"   k={k:>2}: stopped out  {ev_df['stop'].mean():5.1%} vs {b_df['stop'].mean():5.1%}   | "
          f"costly stop-outs (>1.5x planned risk): {(ev_df['cost'] > 1.5).mean():5.1%} vs {(b_df['cost'] > 1.5).mean():4.1%}   | "
          f"return mean {ev_df['ret'].mean()*100:+5.2f}% vs {b_df['ret'].mean()*100:+5.2f}%, "
          f"median {ev_df['ret'].median()*100:+5.2f}% vs {b_df['ret'].median()*100:+5.2f}%   (n={len(ev_df)} / {len(b_df)})")
    print(f"         WITH THE STOP APPLIED (what the book gets): mean {ev_df['real'].mean()*100:+5.2f}% vs "
          f"{b_df['real'].mean()*100:+5.2f}%, median {ev_df['real'].median()*100:+5.2f}% vs {b_df['real'].median()*100:+5.2f}%")

print("\n3. WHAT THE REPORT DOES TO A FRESH POSITION SIZED AT 2% RISK (k=5, loss in % of equity)")
evw = [window(t, r - 5, 5) for t, r in E.itertuples(index=False) if r - 5 >= 20]
d = pd.DataFrame([x for x in evw if x], columns=["stop", "ret", "cost", "real"])
loss = d.loc[d["stop"], "cost"] * 2.0
print(f"   stopped out {d['stop'].mean():.0%} of the time; the loss when stopped: median {loss.median():.1f}% of equity, "
      f"worst decile {loss.quantile(0.9):.1f}%, worst {loss.max():.1f}%   (plan: 2.0%)")
print("   NB: the book's GTC stop-LIMIT orders may not fill at all when price gaps below the limit;")
print("   these figures assume a fill at the lower of the stop and that day's close.")
