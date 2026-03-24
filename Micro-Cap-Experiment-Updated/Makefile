# Virtual environment setup (uses .nosync to prevent iCloud sync issues)
venv:
	python3 -m venv venv.nosync
	ln -sf venv.nosync venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Install dependencies
install:
	pip install -r requirements.txt

# Setup project (create venv and install dependencies)
setup: venv activate
	@echo "Activating virtual environment and installing dependencies..."
	pip install -r requirements.txt
	@echo "Setup complete! Activate the environment with: source venv/bin/activate"

# Activate virtual environment (shows activation command)
activate: 
	@echo "To activate the virtual environment, run:"
	@echo "source venv/bin/activate"
	@echo ""
	@echo "To deactivate later, simply run: deactivate"

# Run the trading script
trade: activate
	python trading_script.py $(ARGS)

graph: activate
	python "Start Your Own/Generate_Graph.py" $(ARGS)

# Daily workflow: run trading script after 4 PM EST, then Claude auto-analyzes the XML output
daily:
	python trading_script.py --data-dir "Start Your Own"

# Weekend workflow: check portfolio is current, update weekend_summary.md, inject last thesis, print for Claude
weekend:
	@python3 -c "\
import pandas as pd, sys; \
from datetime import date, timedelta; \
df = pd.read_csv('Start Your Own/chatgpt_portfolio_update.csv'); \
last = pd.to_datetime(df['Date']).max().date(); \
d = date.today(); \
while d.weekday() >= 5: d -= timedelta(days=1); \
sys.exit(0 if last >= d else 1)" || \
	(echo "⚠️  Portfolio data is not current for the last trading day. Run '! make daily' first, then re-run '! make weekend'." && exit 1)
	python trading_script.py --data-dir "Start Your Own" --weekend-summary
	@python3 inject_last_thesis.py
	@cat "Start Your Own/weekend_summary.md"

# Clean up virtual environment
clean:
	rm -rf venv venv.nosync
	@echo "Virtual environment removed"

.PHONY: venv install setup activate trade graph daily weekend clean

