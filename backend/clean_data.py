"""
Standalone script to execute Phase 2 data cleaning pipeline.
Usage:
    python backend/clean_data.py
    python backend/clean_data.py --ticker RELIANCE
"""
import sys
import argparse
from services.data_service import process_and_save_stock_data

def main():
    parser = argparse.ArgumentParser(description="Clean raw stock market data.")
    parser.add_argument("--ticker", type=str, default="TCS", help="Ticker symbol to process (default: TCS)")
    args = parser.parse_args()

    print(f"Cleaning stock data for: {args.ticker}...")
    df_clean, audit = process_and_save_stock_data(ticker=args.ticker)

    print("\n--- Cleaning Summary Audit ---")
    print(f"Initial raw rows:              {audit['initial_rows']}")
    print(f"Exact duplicates removed:      {audit['exact_duplicates_removed']}")
    print(f"Duplicate dates removed:       {audit['duplicate_dates_removed']}")
    print(f"Invalid dates removed:         {audit['invalid_dates_removed']}")
    print(f"Missing core prices dropped:   {audit['missing_core_ohlc_removed']}")
    print(f"Invalid price records dropped: {audit['invalid_price_records_removed']}")
    print(f"Auxiliary NaNs filled:         {audit.get('trades_nans_filled_with_zero', 0)}")
    print(f"Final clean rows:              {audit['final_rows']}")
    print(f"Date range:                    {audit['date_range']}")
    print(f"Clean CSV saved at:            {audit['saved_to']}")
    print("\n[SUCCESS] Phase 2 Data Cleaning completed successfully.")

if __name__ == "__main__":
    main()
