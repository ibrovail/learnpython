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
# Session directives are passed via CLI args (Claude asks questions, then runs with answers):
#   make weekend SECTOR="Biotech" TIMING="Within 10 trading days" RISK="Aggressive" POSITIONS="5"
SECTOR ?=
TIMING ?=
RISK ?=
POSITIONS ?=

screen:
	venv/bin/python screener.py --data-dir "Start Your Own" --top-n 15

weekend:
	@echo "Running screener..."
	@venv/bin/python screener.py --data-dir "Start Your Own" --top-n 15 || \
	echo "⚠️  Screener failed — proceeding with cached watchlist (if available)."
	@venv/bin/python trading_script.py --data-dir "Start Your Own" --check-data-current --log-level ERROR || \
	(echo "⚠️  Portfolio data is not current for the last trading day. Ask Claude to 'run daily' first, then 'run weekend'." && exit 1)
	@SD_ARGS=""; \
	if [ -n "$(SECTOR)" ]; then SD_ARGS="$$SD_ARGS --sector-focus \"$(SECTOR)\""; fi; \
	if [ -n "$(TIMING)" ]; then SD_ARGS="$$SD_ARGS --catalyst-timing \"$(TIMING)\""; fi; \
	if [ -n "$(RISK)" ]; then SD_ARGS="$$SD_ARGS --risk-posture \"$(RISK)\""; fi; \
	if [ -n "$(POSITIONS)" ]; then SD_ARGS="$$SD_ARGS --max-positions \"$(POSITIONS)\""; fi; \
	printf '\n' | eval venv/bin/python trading_script.py --data-dir \"Start Your Own\" --weekend-summary $$SD_ARGS
	@venv/bin/python inject_last_thesis.py
	@cat "Start Your Own/weekend_summary.md"

# Clean up virtual environment
clean:
	rm -rf venv venv.nosync
	@echo "Virtual environment removed"

.PHONY: venv install setup activate trade graph daily screen weekend clean
