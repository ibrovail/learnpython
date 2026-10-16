# No-Cap Experiment — working guide

A real-money CAD portfolio in a TFSA, run as a research project. Goal: beat the S&P 500 after
costs, in Canadian dollars. Started 2026-10-12 at CAD $200.

## Before anything else

1. Read `portfolio_rules.md`. It is short and it is binding.
2. For any study, read `.claude/rules/research-methods.md`. Pre-registration is not optional.

## Non-negotiables

- **No trade recommendation without checking `portfolio_rules.md` §6 (exclusions) first**,
  per name, by hand. Israeli affiliation cannot be screened by industry.
- **Never source a live price from web search.** Use a browser on a quote page, with a session
  label and timestamp. Search results are undated cached snippets.
- **Turnover budget: 6 trades a year, hard ceiling 10.** See §5 for why — it is a tax
  constraint, not a preference.
- **The user executes every trade at their broker.** Nothing here places orders.
- **Never commit a result before its pre-registration.** Stage order is enforced by commits.

## Measurement

The balance rises because of deposits, not skill. Only the **injection-neutral time-weighted
series** against the CAD S&P 500 benchmark says whether a decision added value. Lives in
`ledger/`.

## Tech

Python 3.13, pandas, numpy. No virtualenv committed. `python3 -I` for anything reading
downloaded data.

## Current state

- **Rules:** written, v1, 2026-10-12.
- **Study 01** (winner rotation): pre-registered; Stage 1a (membership) and Stage 1b
  (ticker map, exclusion list, panel builder, coverage) done; 83 synthetic self-tests
  passing with no network. **Reported INCONCLUSIVE on the coverage gate** — Tiingo's free
  tier allows 500 unique symbols a month against a universe of 754, so the price panel
  cannot be built. Network egress is open; the limit is the vendor's, not the container's.
  Unblocking it needs an upgraded `TIINGO_API_KEY` in the environment's secrets, or a new
  calendar month. See `Experiment Details/Study 01 — Stage 1b findings.md`.
  **Outstanding for the owner: the §6 hand check** (Israeli affiliation, and the
  never-auto-cleared industries) — `research/exclusions.py` encodes only the
  industry-screenable part and flags itself incomplete.
- **Ledger:** not yet started. Needed before the first deposit.
- **Not yet decided:** broker, and whether the satellite sleeve's picks come from US or
  Canadian listings (FX cost versus universe quality).
