"""
Migrate initial_portfolio_snapshot.csv to chatgpt_portfolio_update.csv

This script reads your existing initial snapshot and creates the proper
portfolio tracking entries so your holdings show up in daily results.

Usage:
    python migrate_initial_snapshot.py
"""

from pathlib import Path
import pandas as pd

# File paths
SCRIPT_DIR = Path(__file__).resolve().parent
INITIAL_SNAPSHOT_CSV = SCRIPT_DIR / "initial_portfolio_snapshot.csv"
PORTFOLIO_CSV = SCRIPT_DIR / "chatgpt_portfolio_update.csv"

def migrate_snapshot():
    """Migrate initial snapshot to portfolio CSV."""
    
    # Check if snapshot exists
    if not INITIAL_SNAPSHOT_CSV.exists():
        print(f"❌ Error: {INITIAL_SNAPSHOT_CSV} not found.")
        print("   Run trading_script.py first to create initial snapshot.")
        return
    
    # Load snapshot
    print(f"Reading {INITIAL_SNAPSHOT_CSV}...")
    snapshot = pd.read_csv(INITIAL_SNAPSHOT_CSV)
    
    if snapshot.empty:
        print("❌ Snapshot is empty. Nothing to migrate.")
        return
    
    print(f"Found {len(snapshot)} holdings in snapshot.")
    
    # Extract info
    date = snapshot["Date"].iloc[0]
    total_value = float(snapshot["Total_Portfolio_Value"].iloc[0])
    holdings_value = float(snapshot["Value"].sum())
    cash = total_value - holdings_value
    
    print(f"\nSnapshot details:")
    print(f"  Date: {date}")
    print(f"  Total Portfolio Value: ${total_value:,.2f}")
    print(f"  Holdings Value: ${holdings_value:,.2f}")
    print(f"  Cash: ${cash:,.2f}")
    
    # Build portfolio entries
    portfolio_results = []
    
    for _, row in snapshot.iterrows():
        portfolio_results.append({
            "Date": date,
            "Ticker": row["Ticker"],
            "Shares": float(row["Shares"]),
            "Buy Price": float(row["Price"]),
            "Cost Basis": float(row["Value"]),
            "Stop Loss": 0.0,
            "Current Price": float(row["Price"]),
            "Total Value": float(row["Value"]),
            "PnL": 0.0,
            "Action": "INITIAL HOLDINGS",
            "Cash Balance": "",
            "Total Equity": ""
        })
    
    # Add TOTAL row
    portfolio_results.append({
        "Date": date,
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": holdings_value,
        "PnL": 0.0,
        "Action": "",
        "Cash Balance": cash,
        "Total Equity": total_value
    })
    
    # Check if portfolio CSV already exists
    if PORTFOLIO_CSV.exists():
        print(f"\n⚠️  Warning: {PORTFOLIO_CSV} already exists.")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            print("Migration cancelled.")
            return
    
    # Write portfolio CSV
    portfolio_df = pd.DataFrame(portfolio_results)
    portfolio_df.to_csv(PORTFOLIO_CSV, index=False)
    
    print(f"\n✓ Migration complete!")
    print(f"  Created {len(snapshot)} holding entries in {PORTFOLIO_CSV}")
    print(f"  Run 'python trading_script.py' to see your holdings.")
    
    # Show summary
    print("\n" + "="*60)
    print("HOLDINGS MIGRATED:")
    print("="*60)
    for _, row in snapshot.iterrows():
        shares = float(row["Shares"])
        ticker = row["Ticker"]
        price = float(row["Price"])
        value = float(row["Value"])
        print(f"  {ticker:8} {shares:12.8f} shares @ ${price:8.2f} = ${value:12,.2f}")
    print("-"*60)
    print(f"  {'CASH':8} {'':<12} {'':>10} = ${cash:12,.2f}")
    print(f"  {'TOTAL':8} {'':<12} {'':>10} = ${total_value:12,.2f}")
    print("="*60)


if __name__ == "__main__":
    migrate_snapshot()