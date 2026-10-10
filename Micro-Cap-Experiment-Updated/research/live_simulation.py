"""Daily event-driven backtest of the actual book, not a weekly rebalanced index.

Phase 3.8's capacity study simulated an unconstrained, equal-weighted, weekly-rebalanced portfolio
holding every qualifying name. The live book is nothing like that: it holds about five positions,
sizes them by risk at the stop, is constrained by cash, and holds for 40-60 sessions or until a
trailing stop takes it out. Those differences are not details -- the stop is described in
portfolio_rules.md as "the single most load-bearing line in this file", and the capacity study
could not model it at all.

So this is a daily loop that implements the rules as written:

  Entry (on the session AFTER the formation date, as a weekend decision executes Monday)
    * stop = below the LOWEST of 1.75xATR(14), the 10-session low, and the 50-day SMA
    * shares = (equity x 2%) / (entry - stop), floored to whole shares
    * capped at 30% of equity, and at cash above the 15% floor
    * position ceiling, GICS sector cap not modelled (sector data is not point-in-time here)
  Each session
    * stop raise: candidate = close - 2.0xATR, taken only if it is >= 0.5xATR above the
      current stop and still leaves >= 1.5xATR of room. Never lowered.
    * stop exit: if the low breaches the stop, exit. If the OPEN is already below the stop the
      fill is the open, not the stop -- that is the gap risk the rules warn about, and it is the
      main thing a no-stop simulation gets wrong in the optimistic direction.
    * hard exit at MAX_HOLD sessions (the 60-session re-underwrite, treated as an exit)
  Costs
    * slippage in basis points on both sides, as an explicit parameter. Commission-free: the
      ledger's realised PnL reconciles to shares x (sell - buy) with no fee line.

Reads the Phase 3.75 panel for weekly candidate signals and research/data.nosync/prices/*.csv for
daily OHLC. Spends no API allowance.

    venv/bin/python research/live_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, PRICES = ROOT / "output", ROOT / "data.nosync" / "prices"
sys.path.insert(0, str(ROOT))
from regime_backtest import nw_tstat, lag_for  # noqa: E402

# ---- the live rules, as portfolio_rules.md states them --------------------------------------
RISK_PER_TRADE = 0.02        # 2% of equity at the stop
STOP_ATR_MULT = 1.75        # entry target
RAISE_ATR_MULT = 2.00       # raise target
RAISE_MIN_ATR = 0.50        # anti-ratchet: a raise must move at least this far
ROOM_MIN_ATR = 1.50         # and must still leave this much room
SINGLE_NAME_CAP = 0.30      # 30% of equity
CASH_FLOOR = 0.15           # 15% of equity is not deployable
POSITION_CEILING = 5
MAX_HOLD = 60               # sessions; the re-underwrite treated as an exit
SLIPPAGE_BPS = 25           # each side; 0.25% is realistic for $1M-ADV small caps


class Book:
    """Minimal portfolio: cash, positions, and an equity curve."""

    def __init__(self, capital: float):
        self.cash = capital
        self.pos: dict[str, dict] = {}
        self.curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[dict] = []

    def equity(self, px: dict[str, float]) -> float:
        return self.cash + sum(p["shares"] * px.get(t, p["last"]) for t, p in self.pos.items())


def load_prices(tickers: set[str], cal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Per-ticker daily OHLC plus the levels the stop rules need, on the master calendar."""
    out = {}
    for t in tickers:
        f = PRICES / f"{t}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date").reindex(cal)
        c, h, l = d.adjClose.astype(float), d.adjHigh.astype(float), d.adjLow.astype(float)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        d["atr"] = tr.rolling(14).mean()
        d["low10"] = l.rolling(10).min()
        d["sma50"] = c.rolling(50).mean()
        out[t] = d[["adjOpen", "adjHigh", "adjLow", "adjClose", "atr", "low10", "sma50"]]
    return out


def initial_stop(row) -> float:
    """Below the LOWEST of the three candidates -- entry-discipline.md requires all three."""
    cands = [row.adjClose - STOP_ATR_MULT * row.atr, row.low10, row.sma50]
    cands = [c for c in cands if np.isfinite(c)]
    return min(cands) * 0.999 if cands else np.nan


def run(panel: pd.DataFrame, px: dict, cal: pd.DatetimeIndex, *,
        capital: float = 10_000.0, use_regime: bool = False, calm: float | None = None,
        use_stops: bool = True, ceiling: int = POSITION_CEILING,
        slippage_bps: float = SLIPPAGE_BPS, max_hold: int = MAX_HOLD,
        raise_policy: str = "mechanical", rank_by: str = "composite",
        cash_floor: float = CASH_FLOOR, liq_tercile: str | None = None,
        fractional: bool = False, risk_per_trade: float = RISK_PER_TRADE,
        name_cap: float = SINGLE_NAME_CAP) -> dict:
    """raise_policy decides how the trailing stop is managed, and it dominates everything else.

    portfolio_rules.md makes raising a stop *eligible* mechanically but leaves the decision to
    judgment: "Eligibility is mechanical; using it is not. A qualifying stop MAY be reset, not
    MUST be." The live book exercises that discretion sparingly -- ATRC's stop still sat at
    1.35xATR after 58 sessions, so it had not been tracking the price, and a mechanical ratchet
    to 2xATR below every new high would have taken the position out long before +61%.

    So the policy is a scenario dimension, not a constant:
      "never"      -- the initial stop never moves. Closest to how ATRC was actually handled.
      "mechanical" -- raise whenever the anti-ratchet test passes. The literal rules, maximally
                      applied, and what the first version of this script assumed.
      "profit"     -- raise only once the position is up more than 1xATR, so the stop protects
                      gains rather than chasing them.
      "once"       -- at most one raise per position, which is how a stop-restoration-style
                      allowance would behave if used conservatively.
    """
    """One scenario, start to finish. Returns the equity curve and trade stats."""
    bk = Book(capital)
    slip = slippage_bps / 10_000.0
    cand_by_date = {d: g for d, g in panel.groupby("date")}
    pending: list[str] = []          # decided on the formation date, executed next session

    for i, day in enumerate(cal):
        live = {t: px[t].loc[day] for t in list(bk.pos) if t in px}
        marks = {t: (r.adjClose if np.isfinite(r.adjClose) else bk.pos[t]["last"])
                 for t, r in live.items()}

        # ---- 1. exits, checked on the open first (gap risk) ---------------------------------
        for t in list(bk.pos):
            p, r = bk.pos[t], live.get(t)
            if r is None or not np.isfinite(r.adjClose):
                continue
            p["held"] += 1
            fill = None
            if use_stops and np.isfinite(p["stop"]):
                if r.adjOpen <= p["stop"]:
                    fill = r.adjOpen          # gapped through: the stop does NOT protect here
                elif r.adjLow <= p["stop"]:
                    fill = p["stop"]
            if fill is None and p["held"] >= max_hold:
                fill = r.adjClose
            if fill is not None:
                proceeds = p["shares"] * fill * (1 - slip)
                bk.cash += proceeds
                bk.trades.append({"ticker": t, "entry": p["entry"], "exit": fill,
                                  "held": p["held"], "shares": p["shares"],
                                  "pnl": proceeds - p["cost"],
                                  "reason": "stop" if p["held"] < max_hold else "hold_limit"})
                del bk.pos[t]
                marks.pop(t, None)

        # ---- 2. stop raises (anti-ratchet, subject to raise_policy) -------------------------
        if use_stops and raise_policy != "never":
            for t, p in bk.pos.items():
                r = live.get(t)
                if r is None or not np.isfinite(r.atr) or r.atr <= 0:
                    continue
                if raise_policy == "once" and p["raises"] >= 1:
                    continue
                if raise_policy == "profit" and (r.adjClose - p["entry"]) < r.atr:
                    continue
                cand = r.adjClose - RAISE_ATR_MULT * r.atr
                if (cand - p["stop"]) >= RAISE_MIN_ATR * r.atr and \
                   (r.adjClose - cand) >= ROOM_MIN_ATR * r.atr:
                    p["stop"] = cand
                    p["raises"] += 1

        # ---- 3. execute what was decided on the previous formation date --------------------
        eq = bk.equity(marks)
        for t in pending:
            if len(bk.pos) >= ceiling or t in bk.pos or t not in px:
                continue
            r = px[t].loc[day]
            prev = px[t].iloc[max(0, i - 1)]
            if not np.isfinite(r.adjOpen) or not np.isfinite(prev.atr):
                continue
            entry = r.adjOpen * (1 + slip)
            stop = initial_stop(prev) if use_stops else entry * 0.5
            if not np.isfinite(stop) or stop >= entry:
                continue
            risk_per_share = entry - stop
            rnd = (lambda x: x) if fractional else int
            shares = rnd((eq * risk_per_trade) / risk_per_share)
            shares = min(shares, rnd(eq * name_cap / entry))
            spendable = bk.cash - eq * cash_floor
            shares = min(shares, rnd(spendable / entry) if spendable > 0 else 0)
            if shares < (0.0001 if fractional else 1):
                continue
            cost = shares * entry
            bk.cash -= cost
            bk.pos[t] = {"shares": shares, "entry": entry, "cost": cost, "stop": stop,
                         "held": 0, "last": entry, "raises": 0}
        pending = []

        # ---- 4. on a formation date, pick next session's buys ------------------------------
        if day in cand_by_date and len(bk.pos) < ceiling:
            g = cand_by_date[day]
            if use_regime:
                g = g[g.regime == "RISK-ON"]
            if calm is not None:
                g = g[(g.rank_low_vol >= calm) & (g.vol_5_50 > 1.0) & (g.above_sma50.astype(bool))]
            if liq_tercile is not None:
                # size/liquidity tercile within that week's eligible names. The most liquid third
                # returned 16.9% a year equal-weighted against the universe's 14.1%, which is the
                # one lever found that clears SPY -- so it is testable rather than assumed.
                q = g.dollar_vol_20.quantile([1/3, 2/3])
                if liq_tercile == "large":
                    g = g[g.dollar_vol_20 >= q.iloc[1]]
                elif liq_tercile == "small":
                    g = g[g.dollar_vol_20 <= q.iloc[0]]
            g = g[~g.ticker.isin(bk.pos)]
            pending = g.nlargest(ceiling - len(bk.pos), rank_by).ticker.tolist()

        for t, p in bk.pos.items():
            if t in marks:
                p["last"] = marks[t]
        bk.curve.append((day, bk.equity(marks)))

    cur = pd.Series(dict(bk.curve))
    tr = pd.DataFrame(bk.trades)
    yrs = (cal[-1] - cal[0]).days / 365.25
    total = cur.iloc[-1] / capital - 1
    dd = (cur / cur.cummax() - 1).min()
    rets = cur.pct_change().dropna()
    return {"final": cur.iloc[-1], "total_return": total,
            "cagr": (1 + total) ** (1 / yrs) - 1,
            "max_drawdown": dd,
            "sharpe_like": (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() else np.nan,
            "trades": len(tr),
            "stop_exits": int((tr.reason == "stop").sum()) if len(tr) else 0,
            "win_rate": float((tr.pnl > 0).mean()) if len(tr) else np.nan,
            "avg_hold": float(tr.held.mean()) if len(tr) else np.nan,
            "pct_time_invested": float((cur.index.map(lambda d: 1) * 0).mean()) if False else None,
            "curve": cur}


def main() -> None:
    panel = pd.read_csv(OUT / "regime_backtest_panel_survivors.csv",
                        parse_dates=["date"], low_memory=False)
    panel["above_sma50"] = panel.above_sma50.astype(str).str.lower().isin(["true", "1"])
    panel["rank_bb"] = panel.groupby("date").bb_width.rank(pct=True)
    panel["composite"] = panel[["rank_low_vol", "rank_near_high", "rank_vol_5_50",
                                "rank_vol_ratio", "rank_pct_vs_sma50", "rank_bb"]].mean(axis=1)
    tickers = set(panel.ticker.unique())

    import yfinance as yf
    iwm = yf.Ticker("IWM").history(start="2015-08-01", end="2026-09-19", auto_adjust=True)
    cal = pd.to_datetime(iwm.index).tz_localize(None).normalize()
    cal = cal[(cal >= panel.date.min()) & (cal <= pd.Timestamp("2026-09-18"))]
    print(f"loading daily prices for {len(tickers)} tickers...", flush=True)
    px = load_prices(tickers, cal)
    print(f"  loaded {len(px)}; calendar {len(cal)} sessions "
          f"{cal[0].date()}..{cal[-1].date()}\n", flush=True)

    # The stop-raise policy turned out to dominate every other choice, so it is the first
    # dimension rather than a footnote. Scenarios A-C hold the rules fixed and vary only the
    # policy, to isolate its effect before anything else is compared.
    scenarios = [
        ("A1 live rules, stop NEVER raised", dict(use_regime=True, calm=0.90,
                                                  raise_policy="never")),
        ("A2 live rules, raise once only", dict(use_regime=True, calm=0.90,
                                                raise_policy="once")),
        ("A3 live rules, raise once in profit", dict(use_regime=True, calm=0.90,
                                                     raise_policy="profit")),
        ("A4 live rules, raise mechanically", dict(use_regime=True, calm=0.90,
                                                   raise_policy="mechanical")),
        ("B1 no regime, stop never raised", dict(use_regime=False, calm=0.90,
                                                 raise_policy="never")),
        ("C1 no regime/profile, never raised", dict(use_regime=False, calm=None,
                                                    raise_policy="never")),
        ("C2 no regime/profile, raise in profit", dict(use_regime=False, calm=None,
                                                       raise_policy="profit")),
        ("C3 no regime/profile, mechanical", dict(use_regime=False, calm=None,
                                                  raise_policy="mechanical")),
        ("A  live rules (regime + profile + stops)", dict(use_regime=True, calm=0.90)),
        ("B  drop regime filter", dict(use_regime=False, calm=0.90)),
        ("C  drop profile too", dict(use_regime=False, calm=None)),
        ("D  C, but NO stops (60-session blind hold)", dict(use_regime=False, calm=None,
                                                            use_stops=False)),
        ("E  live rules, NO stops", dict(use_regime=True, calm=0.90, use_stops=False)),
        ("F  C1 with 8 positions", dict(use_regime=False, calm=None, ceiling=8,
                                      raise_policy="never")),
        ("G  C1 with 3 positions", dict(use_regime=False, calm=None, ceiling=3,
                                      raise_policy="never")),
        ("H  C1, 40-session hold", dict(use_regime=False, calm=None, max_hold=40,
                                      raise_policy="never")),
        ("I  C1, zero slippage", dict(use_regime=False, calm=None, slippage_bps=0,
                                     raise_policy="never")),
        ("J  C1, 50bps slippage", dict(use_regime=False, calm=None, slippage_bps=50,
                                      raise_policy="never")),
    ]
    rows = []
    print(f"{'scenario':<44}{'CAGR':>8}{'max DD':>9}{'Sharpe':>8}{'trades':>8}"
          f"{'stops':>7}{'win':>6}{'hold':>6}", flush=True)
    for lbl, kw in scenarios:
        r = run(panel, px, cal, capital=10_000.0, **kw)
        rows.append(dict(scenario=lbl, **{k: v for k, v in r.items() if k != "curve"}))
        print(f"{lbl:<44}{r['cagr']:>7.1%}{r['max_drawdown']:>9.1%}"
              f"{r['sharpe_like']:>8.2f}{r['trades']:>8}{r['stop_exits']:>7}"
              f"{r['win_rate']:>6.0%}{r['avg_hold']:>6.0f}", flush=True)

    # scale check: the live book runs ~$700, where whole-share rounding bites hard
    print(f"\n{'scale check (scenario C)':<44}{'CAGR':>8}{'max DD':>9}{'trades':>8}", flush=True)
    for cap in (700.0, 10_000.0, 100_000.0):
        r = run(panel, px, cal, capital=cap, use_regime=False, calm=None,
                raise_policy="never")
        rows.append(dict(scenario=f"C at ${cap:,.0f}", **{k: v for k, v in r.items() if k != "curve"}))
        print(f"{'  starting capital $'+format(cap,',.0f'):<44}{r['cagr']:>7.1%}"
              f"{r['max_drawdown']:>9.1%}{r['trades']:>8}", flush=True)

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "live_simulation_results.csv", index=False)
    print(f"\nwrote {OUT/'live_simulation_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
