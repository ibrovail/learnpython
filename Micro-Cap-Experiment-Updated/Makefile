# Virtual environment setup (uses .nosync to prevent iCloud sync issues)
venv:
	@if [ -d "venv.nosync" ]; then \
		echo "venv.nosync already exists — skipping creation."; \
	else \
		python3 -m venv venv.nosync; \
		ln -sf venv.nosync venv; \
		echo "Virtual environment created. Activate with: source venv/bin/activate"; \
	fi

# Install dependencies
install:
	pip install -r requirements.txt

# Setup project (create venv if needed and install dependencies)
setup: venv
	@echo "Installing dependencies..."
	venv/bin/pip install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Activate virtual environment (shows activation command)
activate: 
	@echo "To activate the virtual environment, run:"
	@echo "source venv/bin/activate"
	@echo ""
	@echo "To deactivate later, simply run: deactivate"

# Run the trading script
trade: activate
	venv/bin/python trading_script.py $(ARGS)

graph: activate
	venv/bin/python "Start Your Own/Generate_Graph.py" $(ARGS)

# Daily workflow: run trading script after 4 PM EST, then Claude auto-analyzes the XML output
daily:
	venv/bin/python trading_script.py --data-dir "Start Your Own"

# Weekend workflow: check portfolio is current, update weekend_summary.md, inject last thesis, print for Claude
weekend:
	@venv/bin/python -c "\
import pandas as pd, sys; \
from datetime import date, timedelta; \
df = pd.read_csv('Start Your Own/chatgpt_portfolio_update.csv'); \
last = pd.to_datetime(df['Date']).max().date(); \
today = date.today(); \
d = (today - timedelta(days=max(0, today.weekday() - 4))) if today.weekday() >= 5 else today; \
try: \
    import exchange_calendars as xcals; \
    xnys = xcals.get_calendar('XNYS'); \
    sessions = xnys.sessions_in_range(d - timedelta(days=6), d + timedelta(days=1)); \
    d = max(s.date() for s in sessions if s.date() <= d) if len(sessions) > 0 else d; \
except Exception: pass; \
sys.exit(0 if last >= d else 1)" || \
	(echo "⚠️  Portfolio data is not current for the last trading day. Run '! make daily' first, then re-run '! make weekend'." && exit 1)
	printf '\n' | venv/bin/python trading_script.py --data-dir "Start Your Own" --weekend-summary
	@venv/bin/python inject_last_thesis.py
	@cat "Start Your Own/weekend_summary.md"

# Clean up virtual environment
clean:
	rm -rf venv venv.nosync
	@echo "Virtual environment removed"

.PHONY: venv install setup activate trade graph daily weekend clean

