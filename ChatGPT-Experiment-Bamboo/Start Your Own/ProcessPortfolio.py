"""Wrapper for the shared trading script using local data directory.

This wrapper is designed for general investment portfolios with:
- Fractional share support
- No market cap restrictions
- Initial portfolio snapshot capability
- Buy-and-hold comparison tracking
"""

from pathlib import Path
import sys

# Allow importing the shared module from the repository root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_script import main, set_data_dir


if __name__ == "__main__":
    # Set the data directory for this portfolio
    data_dir = Path(__file__).resolve().parent
    
    # Configure the global data directory
    set_data_dir(data_dir)
    
    # Run the main trading script
    # This will:
    # 1. Check for initial_portfolio_snapshot.csv (create on first run)
    # 2. Load latest portfolio state
    # 3. Process trades and stop-losses
    # 4. Generate daily results with benchmarks
    main(data_dir=data_dir)