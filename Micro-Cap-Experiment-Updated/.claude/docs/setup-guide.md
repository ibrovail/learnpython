# Setup Guide

## Standard Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or use Make:
```bash
make setup    # Create venv + install deps
make clean    # Remove venv
```

---

## macOS iCloud Setup (Required on this machine)

This project lives on an iCloud-synced Desktop. The virtual environment **must** use `.nosync` to prevent iCloud sync overhead from freezing Python imports:

```bash
python -m venv venv.nosync
ln -s venv.nosync venv        # symlink so "venv" still works everywhere
source venv/bin/activate
pip install -r requirements.txt
```

The `venv` symlink points to `venv.nosync/`, so all commands (`source venv/bin/activate`, `venv/bin/python3`, etc.) work unchanged. Both `venv` and `venv.nosync` are in `.gitignore`.

`make setup` handles this automatically.

---

## Troubleshooting

### Script hangs on startup with no output

**Cause**: iCloud creates conflict copies like `venv 2` that are actively synced, causing filesystem contention that freezes Python's import machinery.

**Fix**: Delete any `venv *` conflict directories and ensure only the `venv -> venv.nosync` symlink exists:
```bash
rm -rf "venv 2"
ls -la venv    # should show: venv -> venv.nosync
```

The Makefile and `.gitignore` prevent this, but iCloud may recreate conflict copies after Finder operations or sync conflicts. Never create a plain `venv` directory — always use `make setup` or the manual `.nosync` + symlink pattern.

---

### Generate_Graph.py IndexError with --end-date

**Symptom**:
```
IndexError: index 0 is out of bounds for axis 0 with size 0
```

**Cause**: Running with `--end-date` before the last capital injection date causes a date mismatch when calculating returns since last injection. The script was loading all capital injections without filtering by date range.

**Fix (applied 2026-02-06)**: Capital injections are now filtered to the portfolio's actual date range before determining `last_injection_date`. Fix is in `Generate_Graph.py` `main()` at lines 350–358.

---

## Trading Script Arguments

```bash
python trading_script.py --data-dir "Start Your Own" --asof 2025-10-01 --log-level INFO -s 5000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | REQUIRED | Data directory for CSV files |
| `--asof` | Today | Treat as 'today' (YYYY-MM-DD) |
| `--log-level` | INFO | DEBUG / INFO / WARNING / ERROR / CRITICAL |
| `-s`, `--starting-equity` | None | Initial cash amount |
| `--update-stops` | False | Update stop-losses only |
| `--weekend-summary` | False | Generate weekend research prompt |
