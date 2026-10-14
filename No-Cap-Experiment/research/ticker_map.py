"""Stage 1b — the ticker reuse and rename map, applied before any price is keyed.

Stage 1's composition test found three distinct failure modes in the reconstruction
(`Study 01 — Stage 1 findings.md`). One of them, ticker reuse, is not a membership problem
at all: the ticker really was a member, but of a *different company*. A price series keyed
on the bare ticker string therefore splices two unrelated businesses into one line, and the
splice is invisible -- it produces a plausible number, not an error.

Two directions of failure, and they are not symmetric:

  * REUSE (one ticker, two companies) is the dangerous direction. Unmapped, it silently
    manufactures a return out of a discontinuity. Every reuse case Stage 1 identified is
    listed below, and the panel builder refuses to use a ticker outside the era that
    actually belongs to the member company.

  * RENAME (one company, two tickers) fails safe. Unmapped, the old ticker simply has no
    price series, the name is counted MISSING in the coverage report, and the study's
    coverage gate sees it. That is conservative: it loses data rather than inventing it.
    The renames below are the ones carried by members inside the 2011-2024 formation
    window; the list is not claimed to be exhaustive, and it does not need to be, because
    an omission costs coverage rather than correctness.

Dates are the effective date of the event -- the acquisition close, the spin-off's first
regular-way session, or the ticker change. They are deliberately recorded as the boundary
between eras rather than as "the year of", because FOXA's two companies meet INSIDE a
calendar year and a year-granularity map would not catch it.

Nothing here is derived from a return. This file was written before any backtest ran.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Era:
    """A window in which one ticker belonged to one company."""

    company: str
    start: str | None   # first session under this ticker; None = before the study window
    end: str | None     # last session under this ticker; None = still current
    note: str = ""

    def covers(self, ts: pd.Timestamp) -> bool:
        if self.start is not None and ts < pd.Timestamp(self.start):
            return False
        if self.end is not None and ts > pd.Timestamp(self.end):
            return False
        return True


# ---------------------------------------------------------------------------
# REUSE. One ticker, two unrelated companies. Every case Stage 1 flagged.
# ---------------------------------------------------------------------------
REUSE: dict[str, tuple[Era, ...]] = {
    "DELL": (
        Era("Dell Inc (Dell Computer)", None, "2013-10-29",
            "taken private 2013-10-29; left the index then"),
        Era("Dell Technologies Inc", "2018-12-28", None,
            "Class C relisted 2018-12-28; joined the index 2024-09-23"),
    ),
    "DOW": (
        Era("The Dow Chemical Company", None, "2017-08-31",
            "merged into DowDuPont 2017-08-31"),
        Era("Dow Inc", "2019-04-02", None,
            "spun out of DowDuPont, regular way 2019-04-02"),
    ),
    "FOXA": (
        Era("Twenty-First Century Fox Inc", None, "2019-03-19",
            "Disney acquisition closed 2019-03-19"),
        Era("Fox Corporation", "2019-03-19", None,
            "THE TWO ERAS MEET INSIDE 2019 -- a year-granularity map misses this one"),
    ),
    "CEG": (
        Era("Constellation Energy Group Inc", None, "2012-03-12",
            "acquired by Exelon 2012-03-12"),
        Era("Constellation Energy Corporation", "2022-02-02", None,
            "spun out of Exelon, regular way 2022-02-02"),
    ),
    "SNDK": (
        Era("SanDisk Corporation", None, "2016-05-12",
            "acquired by Western Digital 2016-05-12"),
        Era("Sandisk Corporation", "2025-02-21", None,
            "spun out of Western Digital 2025-02-21; after the last formation year"),
    ),
    "LIN": (
        Era("LIN TV Corp / LIN Media LLC", None, "2014-12-19",
            "merged into Media General 2014-12-19"),
        Era("Linde plc", "2018-10-31", None,
            "Praxair/Linde merger completed 2018-10-31; Praxair had traded as PX"),
    ),
    "GM": (
        Era("General Motors Corporation", None, "2009-06-01",
            "Chapter 11 2009-06-01; became Motors Liquidation (MTLQQ)"),
        Era("General Motors Company", "2010-11-18", None,
            "IPO 2010-11-18; joined the index 2013-06-06. The old GM's era ends before "
            "the 2011 window, so this entry is inert here and is recorded for the audit "
            "trail rather than for its effect."),
    ),
}

# ---------------------------------------------------------------------------
# RENAME. One company, two tickers. Maps the OLD ticker onto the series that
# carries the continuous history, which is normally the current ticker.
# ---------------------------------------------------------------------------
RENAME: dict[str, str] = {
    "FB": "META",        # Facebook -> Meta Platforms, 2022-06-09
    "PCLN": "BKNG",      # Priceline -> Booking Holdings, 2018-02-27
    "TSO": "ANDV",       # Tesoro -> Andeavor, 2017-08-01 (acquired by MPC 2018-10-01)
    "UTX": "RTX",        # United Technologies -> Raytheon Technologies, 2020-04-03
    "CBS": "PARA",       # CBS -> ViacomCBS (VIAC) 2019-12-04 -> Paramount (PARA) 2022-02-16
    "VIAC": "PARA",
    "SYMC": "GEN",       # Symantec -> NortonLifeLock (NLOK) 2019 -> Gen Digital 2022-11-07
    "NLOK": "GEN",
    "CTL": "LUMN",       # CenturyLink -> Lumen Technologies, 2020-09-14
    "ANTM": "ELV",       # Anthem -> Elevance Health, 2022-06-28
    "FISV": "FI",        # Fiserv, 2023-07-10
    "ABC": "COR",        # AmerisourceBergen -> Cencora, 2023-08-30
    "PKI": "RVTY",       # PerkinElmer -> Revvity, 2023-05-16
    "HFC": "DINO",       # HollyFrontier -> HF Sinclair, 2022-03-14
    "WLTW": "WTW",       # Willis Towers Watson, 2022-01-07
    "BBT": "TFC",        # BB&T -> Truist, 2019-12-09
    "COG": "CTRA",       # Cabot Oil & Gas -> Coterra Energy, 2021-10-01
    "RTN": "RTXRTN",     # Raytheon Company was ABSORBED, not renamed; see note below
}
# RTN is listed so the absorption is explicit rather than an omission. Raytheon Company
# ceased to exist in the 2020 merger and its shares converted to RTX at 2.3348; RTX's own
# series is United Technologies' history, NOT Raytheon's. Mapping RTN onto RTX would
# splice two companies exactly as the reuse cases do. The sentinel target has no price
# file, so RTN resolves to nothing and is counted MISSING -- the conservative outcome.

# A ticker that is BOTH reused and renamed would need both rules; none of the cases above
# is, and the panel builder asserts that rather than assuming it.
assert not (set(REUSE) & set(RENAME)), "a ticker is both reused and renamed; handle it"


def resolve(ticker: str) -> str:
    """The price-series key for a membership ticker. Renames only; reuse needs a date."""
    return RENAME.get(ticker, ticker)


def era_for(ticker: str, ts: pd.Timestamp) -> Era | None:
    """Which company held `ticker` on `ts`. None for a ticker no era covers."""
    eras = REUSE.get(ticker)
    if eras is None:
        return None
    for era in eras:
        if era.covers(ts):
            return era
    return None


def is_reused(ticker: str) -> bool:
    return ticker in REUSE


def valid_window(ticker: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The part of calendar `year` in which ONE company held `ticker`.

    Returns None when the ticker changed hands inside the year, because no single
    company's return can be computed from that year -- FOXA in 2019 is the case this
    exists for. The caller must drop the ticker-year rather than compute across it.
    """
    jan1, dec31 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
    eras = REUSE.get(ticker)
    if eras is None:
        return jan1, dec31
    covering = [e for e in eras if e.covers(jan1) or e.covers(dec31)
                or (e.start is not None and jan1 <= pd.Timestamp(e.start) <= dec31)]
    if len(covering) != 1:
        return None          # zero eras (a gap year) or two (a handover year)
    era = covering[0]
    lo = jan1 if era.start is None else max(jan1, pd.Timestamp(era.start))
    hi = dec31 if era.end is None else min(dec31, pd.Timestamp(era.end))
    return (lo, hi) if lo < hi else None
