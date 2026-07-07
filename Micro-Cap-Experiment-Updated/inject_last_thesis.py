"""
inject_last_thesis.py

Populates the <last_analyst_thesis> block in weekend_summary.md with the
content of the most recent Weekly Deep Research summary file.

Run as part of `make weekend`, after trading_script.py --weekend-summary.

Logic:
  - Reads the week number N from <week_number> in weekend_summary.md
  - Loads the highest-numbered "Weekly Deep Research (MD)/Week {k} Summary.md"
    with k < N (gap-tolerant: a missing week number, e.g. the skipped Week 39,
    no longer blocks injection or silently leaves a stale thesis in place)
  - Replaces the content inside <last_analyst_thesis>...</last_analyst_thesis>
  - Writes the updated file back
"""

import re
import sys
from pathlib import Path

DATA_DIR = Path("Start Your Own")
WEEKLY_MD_DIR = Path("Weekly Deep Research (MD)")
SUMMARY_FILE = DATA_DIR / "weekend_summary.md"


def main() -> None:
    if not SUMMARY_FILE.exists():
        print(f"inject_last_thesis: {SUMMARY_FILE} not found — skipping.")
        return

    content = SUMMARY_FILE.read_text(encoding="utf-8")

    # Extract current week number from <week_number>27 of 52...</week_number>
    match = re.search(r"<week_number>(\d+)", content)
    if not match:
        print("inject_last_thesis: could not find <week_number> tag — skipping.")
        return

    current_week = int(match.group(1))

    # Find the highest-numbered summary below the current week (gap-tolerant).
    prev_week = 0
    for p in WEEKLY_MD_DIR.glob("Week * Summary.md"):
        m = re.match(r"Week (\d+) Summary\.md$", p.name)
        if m and prev_week < int(m.group(1)) < current_week:
            prev_week = int(m.group(1))

    if prev_week < 1:
        print(f"inject_last_thesis: no summary earlier than week {current_week} found — skipping.")
        return

    thesis_path = WEEKLY_MD_DIR / f"Week {prev_week} Summary.md"

    thesis_content = thesis_path.read_text(encoding="utf-8").strip()

    # Replace content between <last_analyst_thesis> and </last_analyst_thesis>
    updated = re.sub(
        r"(<last_analyst_thesis>).*?(</last_analyst_thesis>)",
        lambda m: f"{m.group(1)}\n{thesis_content}\n{m.group(2)}",
        content,
        count=1,
        flags=re.DOTALL,
    )

    if updated == content:
        print("inject_last_thesis: <last_analyst_thesis> block not found — skipping.")
        return

    SUMMARY_FILE.write_text(updated, encoding="utf-8")
    print(f"inject_last_thesis: injected Week {prev_week} Summary into <last_analyst_thesis>.")


if __name__ == "__main__":
    main()
