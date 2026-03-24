"""
inject_last_thesis.py

Populates the <last_analyst_thesis> block in weekend_summary.md with the
content of the most recent Weekly Deep Research summary file.

Run as part of `make weekend`, after trading_script.py --weekend-summary.

Logic:
  - Reads the week number from <week_number> in weekend_summary.md
  - Loads "Weekly Deep Research (MD)/Week {N-1} Summary.md"
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
    prev_week = current_week - 1

    if prev_week < 1:
        print(f"inject_last_thesis: week {current_week} has no prior week — skipping.")
        return

    thesis_path = WEEKLY_MD_DIR / f"Week {prev_week} Summary.md"
    if not thesis_path.exists():
        print(f"inject_last_thesis: {thesis_path} not found — leaving <last_analyst_thesis> unchanged.")
        return

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
