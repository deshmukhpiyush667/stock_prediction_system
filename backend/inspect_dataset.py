import os
import glob
import pandas as pd

def inspect_dataset():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    xlsx_files = glob.glob(os.path.join(raw_dir, "*.xlsx"))
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    all_files = xlsx_files + csv_files

    if not all_files:
        print("No files found in data/raw/")
        return

    print(f"Total files in data/raw/: {len(all_files)}")
    stock_files = [f for f in all_files if "Stock_" in os.path.basename(f)]
    index_files = [f for f in all_files if "Index_" in os.path.basename(f)]
    other_files = [f for f in all_files if f not in stock_files and f not in index_files]
    
    print(f"Stock files count: {len(stock_files)}")
    print(f"Index files count: {len(index_files)}")
    if other_files:
        print(f"Other files count: {len(other_files)} ({[os.path.basename(f) for f in other_files]})")

    # Let's inspect a representative stock file (e.g., Stock_TCS.xlsx or the first stock file)
    sample_file = next((f for f in stock_files if "TCS" in os.path.basename(f)), stock_files[0] if stock_files else all_files[0])
    filename = os.path.basename(sample_file)
    print(f"\n================ Inspecting Representative File: {filename} ================")
    
    if sample_file.endswith(".xlsx"):
        df = pd.read_excel(sample_file)
    else:
        df = pd.read_csv(sample_file)

    num_rows, num_cols = df.shape
    print(f"1. Dataset filename: {filename} (among {len(all_files)} stock/index files in data/raw/)")
    print(f"2. Number of rows: {num_rows}")
    print(f"3. Number of columns: {num_cols}")
    print(f"4. Column names: {list(df.columns)}")
    
    # 5. Date column
    date_candidates = [col for col in df.columns if any(k in col.lower() for k in ['date', 'time', 'timestamp', 'day'])]
    date_col = date_candidates[0] if date_candidates else "Not detected"
    print(f"5. Date column: {date_col}")
    
    # 6. Stock/ticker column
    ticker_candidates = [col for col in df.columns if any(k in col.lower() for k in ['ticker', 'symbol', 'stock', 'name', 'company'])]
    ticker_col = ticker_candidates[0] if ticker_candidates else "Implicit from filename (Stock_<TICKER>.xlsx)"
    print(f"6. Stock/ticker column: {ticker_col}")
    
    # 7. Price columns
    price_cols = [col for col in df.columns if any(k in col.lower() for k in ['open', 'high', 'low', 'close', 'adj close', 'price', 'last', 'vwap', 'prev'])]
    print(f"7. Price columns: {price_cols}")
    
    # 8. Volume column
    vol_cols = [col for col in df.columns if any(k in col.lower() for k in ['vol', 'shares', 'turnover', 'trades'])]
    print(f"8. Volume column(s): {vol_cols}")
    
    # 9. Missing values
    missing = df.isnull().sum()
    print(f"9. Missing values:\n   Total missing values: {missing.sum()}")
    for col, count in missing.items():
        if count > 0:
            print(f"   - {col}: {count} missing")
    if missing.sum() == 0:
        print("   - 0 missing values across all columns.")
        
    # 10. Duplicate rows
    print(f"10. Duplicate rows: {df.duplicated().sum()}")
    
    # 11. Date range
    if date_col != "Not detected":
        date_s = pd.to_datetime(df[date_col], errors='coerce')
        min_date = date_s.min()
        max_date = date_s.max()
        is_sorted = date_s.is_monotonic_increasing
        print(f"11. Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"    Chronologically sorted: {is_sorted}")
    
    # 12. Single vs Multiple
    all_tickers = [os.path.basename(f).replace("Stock_", "").replace(".xlsx", "").replace(".csv", "") for f in stock_files]
    print(f"12. Single stock or multiple stocks: Multiple stocks ({len(stock_files)} stocks, e.g., {all_tickers[:10]}...) plus {len(index_files)} index files")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Last 5 rows ---")
    print(df.tail())
    print("==========================================================================")

if __name__ == "__main__":
    inspect_dataset()
