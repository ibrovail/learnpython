# Documentation Rules

After EVERY feature, bug fix, or meaningful change to `trading_script.py`, Makefile, or any `.claude/` config file:

1. Update `CLAUDE.md` "Current State" section (3 lines max: what changed, what's in progress, what's next)
2. Append an entry to `.claude/docs/implementation-notes.md`:
   - Date (YYYY-MM-DD)
   - One-paragraph summary of the change
   - Table of key files modified
3. Keep `CLAUDE.md` under 200 lines — if it grows beyond that, move detail to `.claude/docs/`

---

## Self-Improvement Loop

After every mistake or correction caught during a session:

1. Identify the root cause (wrong assumption, missing context, incorrect pattern)
2. Add a specific, testable rule to the appropriate `.claude/rules/` file
3. Never add vague rules — every rule must describe a specific action

Examples of good rules:
- "When reading chatgpt_portfolio_update.csv, always check for the Stop Limit column and default missing values to Stop Loss"
- "Never suggest fractional shares — the experiment is full shares only"

Examples of bad rules:
- "Be more careful with CSV files"
- "Handle edge cases appropriately"
