---
name: update-docs
description: Updates CLAUDE.md Current State and appends an entry to implementation-notes.md after any meaningful code change. Use after modifying trading_script.py, Makefile, any .claude/ file, or any analysis workflow file.
---

## Overview

This skill enforces documentation hygiene after code changes. It updates two places: the 3-line "Current State" section in `CLAUDE.md` (so the next session immediately knows what's happening) and the change log in `.claude/docs/implementation-notes.md` (permanent record). Run this skill at the end of any session where code was changed.

---

## Workflow

1. Read the current `CLAUDE.md` "Current State" section (if it doesn't exist, it will be created).
2. Determine the 3-line summary: what changed, what is currently in progress, what is next.
3. Edit `CLAUDE.md` — replace or write the "Current State" section with exactly 3 lines.
4. Verify `CLAUDE.md` line count is ≤200. If over, identify the longest non-essential section and move it to `.claude/docs/`.
5. Read `.claude/docs/implementation-notes.md`.
6. Append a new entry at the top of the "Change History" section with:
   - Date in `YYYY-MM-DD` format
   - Heading: `### YYYY-MM-DD — [Change Name]`
   - One-paragraph summary
   - Table of key files modified (File | Change columns)
7. Write the updated `implementation-notes.md`.
8. Confirm both files are saved.

---

## Output Format

**CLAUDE.md Current State section** (exactly 3 lines):
```markdown
## Current State
- **Complete**: [what is done and stable]
- **In progress**: [what is actively being worked on, or "nothing"]
- **Next**: [the most important upcoming task]
```

**implementation-notes.md entry** (appended to top of Change History):
```markdown
### YYYY-MM-DD — [Change Name]

[One paragraph: what was changed, why, and the outcome.]

| File | Change |
|------|--------|
| `path/to/file.py` | Description of what changed |
| `path/to/other.md` | Description of what changed |
```

---

## Edge Cases

- **No Current State section in CLAUDE.md**: Add it after the "Daily Workflow" section.
- **CLAUDE.md is over 200 lines after update**: Identify the longest REFERENCE or HISTORICAL section, move it to `.claude/docs/` with a descriptive filename, and add it to the Documentation table.
- **Change is trivial (typo fix, comment)**: Still log it, but the summary can be one sentence and the files table can have one row.
- **Multiple changes in one session**: Write one combined entry covering all changes, not one entry per file.

---

## Examples

### Example 1 — Normal code change

**Input**: User just added a `--dry-run` flag to `trading_script.py`.

**CLAUDE.md Current State output**:
```markdown
## Current State
- **Complete**: Claude Code analysis layer, stop-limit support, capital injection declaration, --dry-run flag
- **In progress**: Nothing
- **Next**: Test --dry-run against historical data
```

**implementation-notes.md entry**:
```markdown
### 2026-03-25 — Add --dry-run Flag to Trading Script

Added a `--dry-run` flag to `trading_script.py` that processes the portfolio and prints the daily XML output without writing to any CSV files. Useful for testing analysis workflows without affecting the live ledger.

| File | Change |
|------|--------|
| `trading_script.py` | Added `--dry-run` argument; skip CSV writes when flag is set |
| `CLAUDE.md` | Updated Current State |
```

---

### Example 2 — CLAUDE.md is over 200 lines

**Input**: After updating CLAUDE.md, `wc -l CLAUDE.md` returns 215.

**Action**:
1. Identify the longest non-essential section (e.g., a detailed troubleshooting entry added inline).
2. Move that section to `.claude/docs/troubleshooting.md` (create if needed).
3. Replace the moved section in CLAUDE.md with a one-line reference: `See .claude/docs/troubleshooting.md`.
4. Add `troubleshooting.md` to the Documentation table in CLAUDE.md.
5. Verify line count drops to ≤200.
