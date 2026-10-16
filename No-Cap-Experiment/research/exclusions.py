"""portfolio_rules.md section 6, as a list of tickers, defined BEFORE Stage 2 runs.

Rule 6 requires that "any mechanical rule must define its substitute before it is run",
and its origin note is this very study: the 2025 pick was Palantir, which fails the list
on defence and intelligence contracting, in one of the three years carrying the entire
headline result. So the list has to exist before the backtest, not after it.

WHAT THIS FILE IS, AND WHAT IT IS NOT
-------------------------------------
Rule 6 says plainly that defence "can be partly screened by industry" while Israeli
affiliation "cannot and must be checked by hand", and that Security & Protection Services
and Credit Services are "never auto-cleared". Only the industry-screenable part is
encoded below. The hand-checked part is NOT encoded, is NOT guessed at, and is listed as
outstanding -- a list invented here would read in the output exactly like one the owner
had actually checked, and the whole point of rule 6 is that it is checked per name.

COMPLETE is therefore False, and the Stage 2 runner stamps every exclusion-applied row
with that flag. The "without exclusions" arm of the study is unaffected and is the one
the pre-registration treats as the primary comparison.

Scope: direct holdings only. Broad funds are exempt (rule 6), so the SPY benchmark is
not screened.
"""
from __future__ import annotations

# Defence, weapons and firearms. Screenable by industry, per rule 6. S&P 500 members at
# some point in the 2011-2024 formation window; a ticker that was never picked costs
# nothing by being listed.
DEFENCE = {
    "LMT": "Lockheed Martin — prime defence contractor",
    "NOC": "Northrop Grumman — prime defence contractor",
    "GD": "General Dynamics — prime defence contractor, firearms-adjacent munitions",
    "RTN": "Raytheon Company — missiles and defence electronics",
    "RTX": "Raytheon Technologies / RTX — prime defence contractor",
    "UTX": "United Technologies — Pratt & Whitney military propulsion",
    "LHX": "L3Harris — defence communications and electronics",
    "HRS": "Harris Corporation — predecessor of L3Harris",
    "LLL": "L-3 Communications — defence electronics",
    "HII": "Huntington Ingalls — naval warship construction",
    "TXT": "Textron — Bell military rotorcraft, weapons and munitions",
    "TDG": "TransDigm — defence aerospace components",
    "LDOS": "Leidos — defence and intelligence services",
    "COL": "Rockwell Collins — military avionics",
    "BA": "Boeing — Defense, Space & Security segment",
    "PLTR": "Palantir — defence and intelligence contracting (rule 6's origin case)",
    "AXON": "Axon Enterprise — conducted-energy weapons and law-enforcement systems",
    "MSI": "Motorola Solutions — law-enforcement and defence communications",
    "HWM": "Howmet Aerospace — defence aerostructures and engine components",
    "ARNC": "Arconic — defence aerostructures (Howmet predecessor)",
}

# Private corrections and immigration detention. Checked: no operator (CoreCivic, GEO
# Group) was an S&P 500 member inside the window -- both sat in the mid- and small-cap
# indices -- so this category is empty here rather than unconsidered.
PRISONS: dict[str, str] = {}

# Payday, pawn and high-cost subprime instalment credit. No pure operator was an S&P 500
# member in the window. Rule 6 additionally forbids auto-clearing Credit Services, so the
# card and consumer-credit members (COF, DFS, SYF, AXP) are NOT cleared here -- they are
# part of the outstanding hand check below.
PREDATORY_LENDING: dict[str, str] = {}

# Israeli affiliation. Rule 6: "a company's listed country of domicile is not the test;
# affiliation is", and it "must be checked by hand". Not encoded. Not approximated.
ISRAELI_AFFILIATION_OUTSTANDING = True

# Security & Protection Services and Credit Services, which rule 6 never auto-clears.
NEVER_AUTO_CLEARED_OUTSTANDING = True

COMPLETE = not (ISRAELI_AFFILIATION_OUTSTANDING or NEVER_AUTO_CLEARED_OUTSTANDING)

EXCLUDED: frozenset[str] = frozenset(DEFENCE) | frozenset(PRISONS) | frozenset(PREDATORY_LENDING)

REASONS: dict[str, str] = {**DEFENCE, **PRISONS, **PREDATORY_LENDING}


def summary() -> str:
    parts = [f"{len(EXCLUDED)} tickers excluded by industry screen "
             f"(defence {len(DEFENCE)}, prisons {len(PRISONS)}, "
             f"lending {len(PREDATORY_LENDING)})"]
    if not COMPLETE:
        parts.append("LIST INCOMPLETE: Israeli affiliation and the never-auto-cleared "
                     "industries require a per-name hand check that has not been done")
    return "; ".join(parts)
