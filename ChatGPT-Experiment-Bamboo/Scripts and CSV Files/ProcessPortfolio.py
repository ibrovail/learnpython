"""Wrapper for the shared trading script using flexible data directory configuration."""

from pathlib import Path
import sys
import argparse

# Allow importing the shared module from the repository root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trading_script import main, set_data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the portfolio trading script with optional data directory override."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional data directory (default: script's local directory)"
    )
    parser.add_argument(
        "--asof",
        type=str,
        default=None,
        help="Treat this YYYY-MM-DD as 'today' (e.g., 2025-08-27)"
    )
    
    args = parser.parse_args()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default to the script's local directory
        data_dir = Path(__file__).resolve().parent
    
    # Call main with the data directory and optional asof date
    main(data_dir=data_dir)