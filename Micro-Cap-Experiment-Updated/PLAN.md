# Plan: Automate LLM Analysis in Trading Script

## Problem
After `trading_script.py` runs, the user manually copies terminal output and pastes it into ChatGPT/Claude web for analysis. This needs to happen programmatically — daily analysis on weekdays, deep research on weekends.

## Architecture

```
python trading_script.py --data-dir "Start Your Own" --analyze
                    │
      ┌─────────────┴─────────────┐
      │  process_portfolio()      │   ← interactive (unchanged)
      │  daily_results()          │   ← MODIFIED: also returns output as string
      └─────────────┬─────────────┘
                    │ output_text
                    ▼
         ┌── is --analyze set? ──┐
         │ NO: exit (as before)  │
         │ YES:                  │
         │   detect mode         │
         │   (weekday → daily)   │
         │   (weekend → deep)    │
         │        │              │
         │   build prompt        │
         │   call OpenAI API     │
         │   (with web search)   │
         │        │              │
         │   print response      │
         │   save .md            │
         │   if deep: save .pdf  │
         └───────────────────────┘
```

## Files to Change

### 1. `trading_script.py` — 3 small modifications

**a. `daily_results()` (line 987) — capture output via TeeStream**
- Add a `TeeStream` helper class that writes to both `sys.stdout` and an `io.StringIO` buffer
- Wrap the function body in `sys.stdout = tee` / `sys.stdout = original` try/finally
- Change return type from `None` to `str`
- Zero changes to the ~50 individual `print()` calls — they keep working as-is
- No `input()` calls exist inside this function, so the tee is safe

**b. `main()` (line 1488) — add `analyze` parameter**
```python
def main(data_dir=None, update_stops=False, analyze=False):
    ...
    daily_output = daily_results(chatgpt_portfolio, cash)
    if analyze:
        from llm_analysis import run_analysis
        run_analysis(daily_output, _effective_now(), DATA_DIR)
```

**c. CLI block (line 1502) — add `--analyze` argument**
```python
parser.add_argument("--analyze", action="store_true",
                    help="Send daily results to LLM for automated analysis")
```
Pass through to `main(... analyze=args.analyze)`.

### 2. `simple_automation.py` — replace entirely with `llm_analysis.py`

Delete the current 1,430-line copy of `trading_script.py` in `simple_automation.py`. Replace `simple_automation.py` with a deprecation shim (3 lines). Create `llm_analysis.py` as the real automation module.

**`llm_analysis.py` structure:**

| Function | Purpose |
|----------|---------|
| `run_analysis(output, date, data_dir)` | Main entry — orchestrates everything below |
| `detect_analysis_mode(date)` | Returns `"daily"` if weekday, `"deep_research"` if Sat/Sun |
| `calculate_week_number(date)` | Weeks since experiment start (2025-09-19) |
| `load_config(path)` | Load prompts + settings from `prompts_config.yaml` |
| `get_api_key()` | Read `OPENAI_API_KEY` from env, raise if missing |
| `build_daily_prompt(output, config)` | Assemble system + user messages for daily mode |
| `build_deep_research_prompt(output, config, date, week, last_thesis)` | Assemble deep research messages with template variable substitution |
| `load_last_thesis(week_number)` | Read previous `Week N Summary.md` for context |
| `LLMClient.call(system, user, tools, ...)` | OpenAI Responses API call with `web_search_preview` tool |
| `save_daily_md(response, date, dir)` | Save as `YYYY-MM-DD_analysis.md` |
| `save_deep_research_md(response, week)` | Save as `Week N Summary.md` |
| `save_deep_research_pdf(response, week)` | Convert markdown → PDF via fpdf2 |

**Error handling**: Every failure (missing API key, API error, PDF generation error) prints a warning and returns gracefully — the user can always fall back to manual copy-paste. The script never crashes.

### 3. `prompts_config.yaml` — new file

Externalizes all prompt text so it can be edited without touching code:
- `daily_system_prompt`: The full initialization/system message (from the user's message above)
- `deep_research_system_prompt`: The deep research system message
- `deep_research_user_template`: User message template with `{{WEEK}}`, `{{HOLDINGS_BLOCK}}`, `{{LAST_THESIS}}`, etc.
- `model`: `gpt-4o`
- `temperature`: `0.3`
- `max_output_tokens`: `16000`
- `api_key_env_var`: `OPENAI_API_KEY`
- `experiment_start_date`: `2025-09-19`

### 4. `requirements.txt` — new file (currently missing)

```
pandas==2.2.2
numpy==2.3.2
yfinance>=1.1.0
matplotlib==3.8.4
pandas-datareader>=0.10.0
beautifulsoup4>=4.14.0
requests>=2.32.0
openai>=1.40.0
pyyaml>=6.0
fpdf2>=2.8.0
```

### 5. Supporting updates

- `AUTOMATION_README.md` — rewrite to reference `llm_analysis.py` and `--analyze` flag
- `CLAUDE.md` — add `--analyze` to the argument table and document `llm_analysis.py`
- `Makefile` — add `analyze` target: `python trading_script.py --data-dir "Start Your Own" --analyze $(ARGS)`
- `.gitignore` — add `llm_responses.jsonl`, `.env`, `Daily Analysis/`
- `Start Your Own/README.md` — mention `--analyze` flag in examples

## New Directories

- `Daily Analysis/` — stores daily `.md` recommendation files (`YYYY-MM-DD_analysis.md`)
- Deep research outputs go into existing `Weekly Deep Research (MD)/` and `Weekly Deep Research (PDF)/`

## Dependencies to Install

```bash
pip install openai pyyaml fpdf2
```

## Usage After Implementation

```bash
# Daily (weekday) — interactive trades then automated analysis
export OPENAI_API_KEY="sk-..."
python trading_script.py --data-dir "Start Your Own" --analyze

# Weekend — auto-detects deep research mode
python trading_script.py --data-dir "Start Your Own" --analyze

# Without --analyze — works exactly as before (manual copy-paste)
python trading_script.py --data-dir "Start Your Own"
```

## Verification

1. **Output capture**: Run without `--analyze`, confirm terminal output is identical to before
2. **Dry-run prompt assembly**: `python llm_analysis.py --test --dry-run --mode daily` — verify prompt text is correct with no leftover `{{VARIABLE}}` placeholders
3. **Daily API call**: Run with `--analyze` on a weekday date, confirm response displays + saves `.md`
4. **Deep research API call**: Run with `--analyze --asof 2026-02-01` (Saturday), confirm deep research mode activates, `.md` + `.pdf` saved with correct week number
5. **Missing API key**: Run with `--analyze` but no `OPENAI_API_KEY` set — confirm graceful warning, no crash
6. **Backwards compatibility**: Run without `--analyze` — confirm zero behavior change
