import os
import glob
import pandas as pd
from typing import Optional, Tuple

def get_raw_filepath(ticker: str = "TCS", raw_dir: Optional[str] = None) -> str:
    """
    Locates the raw file for a given ticker symbol in the raw data directory.
    Supports both .xlsx and .csv formats.
    """
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
        raw_dir = os.path.abspath(raw_dir)

    # Possible naming patterns
    patterns = [
        f"Stock_{ticker}.xlsx",
        f"Stock_{ticker}.csv",
        f"{ticker}.xlsx",
        f"{ticker}.csv",
    ]
    for pattern in patterns:
        path = os.path.join(raw_dir, pattern)
        if os.path.exists(path):
            return path

    # Case-insensitive search fallback
    for file in os.listdir(raw_dir):
        if ticker.lower() in file.lower():
            return os.path.join(raw_dir, file)

    raise FileNotFoundError(f"Could not find raw data file for ticker '{ticker}' in '{raw_dir}'")


def load_raw_stock_data(filepath: str) -> pd.DataFrame:
    """Loads raw Excel or CSV stock data without modifying the original source."""
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    return df


def clean_stock_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Cleans stock market time-series data using robust, non-destructive rules:
    
    1. Removes duplicate rows (exact duplicates).
    2. Parses and validates the Date column.
    3. Sorts chronologically (strictly ascending).
    4. Handles multiple records on the same Date (preserves latest).
    5. Converts numeric columns and enforces correct dtypes.
    6. Performs domain sanity checks (positive prices, High >= Low, Volume >= 0).
    7. Handles missing values rationally (fills auxiliary columns like Trades with 0,
       ensures no NaN in core OHLCV).
    8. Standardizes column names (no spaces or special symbols).
    
    Returns cleaned DataFrame and an audit dictionary of operations performed.
    """
    audit = {}
    audit["initial_rows"] = len(df)
    audit["initial_cols"] = len(df.columns)

    # Work on a copy to never mutate raw data
    data = df.copy()

    # Step 1: Remove exact duplicate rows
    exact_dups = int(data.duplicated().sum())
    data = data.drop_duplicates()
    audit["exact_duplicates_removed"] = exact_dups

    # Step 2: Date identification, conversion and sorting
    date_col = next((c for c in data.columns if any(k in c.lower() for k in ['date', 'time', 'day'])), None)
    if not date_col:
        raise ValueError("Could not find a Date column in the dataset.")

    data['Date'] = pd.to_datetime(data[date_col], errors='coerce')
    invalid_dates = int(data['Date'].isna().sum())
    if invalid_dates > 0:
        data = data.dropna(subset=['Date'])
    audit["invalid_dates_removed"] = invalid_dates

    # Step 3: Sort chronologically ascending
    data = data.sort_values(by='Date').reset_index(drop=True)

    # Step 4: Handle duplicate dates if any remain
    date_dups = int(data.duplicated(subset=['Date']).sum())
    if date_dups > 0:
        data = data.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
    audit["duplicate_dates_removed"] = date_dups

    # Step 5: Standardize column names
    rename_map = {
        'Prev Close': 'Prev_Close',
        'Deliverable Volume': 'Deliverable_Volume',
        '%Deliverble': 'Deliverble_Pct'
    }
    data = data.rename(columns=rename_map)

    # Step 6: Validate numeric types on core OHLCV columns
    core_price_cols = ['Open', 'High', 'Low', 'Close']
    for col in core_price_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    if 'Volume' in data.columns:
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Drop any row missing core prices (can't predict if price is unknown)
    missing_core = int(data[core_price_cols].isna().any(axis=1).sum())
    if missing_core > 0:
        data = data.dropna(subset=core_price_cols).reset_index(drop=True)
    audit["missing_core_ohlc_removed"] = missing_core

    # Step 7: Domain sanity checks
    # Stock prices must be > 0, High must be >= Low
    valid_mask = (
        (data['Open'] > 0) &
        (data['Close'] > 0) &
        (data['High'] >= data['Low']) &
        (data['Volume'] >= 0)
    )
    invalid_records = int((~valid_mask).sum())
    if invalid_records > 0:
        data = data[valid_mask].reset_index(drop=True)
    audit["invalid_price_records_removed"] = invalid_records

    # Step 8: Handle secondary missing values
    # In Indian NSE data, 'Trades' was introduced later in 2011; fill earlier NaNs with 0
    if 'Trades' in data.columns:
        trades_nans = int(data['Trades'].isna().sum())
        data['Trades'] = data['Trades'].fillna(0).astype('int64')
        audit["trades_nans_filled_with_zero"] = trades_nans

    if 'Deliverable_Volume' in data.columns:
        data['Deliverable_Volume'] = data['Deliverable_Volume'].fillna(0).astype('int64')
    if 'Deliverble_Pct' in data.columns:
        data['Deliverble_Pct'] = data['Deliverble_Pct'].fillna(0.0).astype('float64')

    audit["final_rows"] = len(data)
    audit["final_cols"] = len(data.columns)
    audit["date_range"] = f"{data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}"

    return data, audit


def process_and_save_stock_data(
    ticker: str = "TCS",
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    End-to-end execution of Phase 2:
    Loads raw data -> Cleans it -> Saves to data/processed/processed_stock_data.csv
    """
    raw_filepath = get_raw_filepath(ticker)
    df_raw = load_raw_stock_data(raw_filepath)

    df_cleaned, audit = clean_stock_data(df_raw)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "processed", "processed_stock_data.csv"
        )
        output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    audit["saved_to"] = output_path

    return df_cleaned, audit


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING PHASE 2: DATA CLEANING PIPELINE")
    print("==================================================")
    ticker = "TCS"
    df_clean, audit = process_and_save_stock_data(ticker=ticker)
    
    print(f"\nProcessing stock: {ticker}")
    print(f"- Initial rows: {audit['initial_rows']}")
    print(f"- Exact duplicate rows removed: {audit['exact_duplicates_removed']}")
    print(f"- Duplicate dates removed: {audit['duplicate_dates_removed']}")
    print(f"- Missing core price rows: {audit['missing_core_ohlc_removed']}")
    print(f"- Invalid price records: {audit['invalid_price_records_removed']}")
    print(f"- Trades NaNs handled: {audit.get('trades_nans_filled_with_zero', 0)}")
    print(f"- Final clean rows: {audit['final_rows']}")
    print(f"- Final columns ({audit['final_cols']}): {list(df_clean.columns)}")
    print(f"- Clean date range: {audit['date_range']}")
    print(f"- Saved clean data to: {audit['saved_to']}")
    print("\nFirst 3 rows of processed data:")
    print(df_clean[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].head(3))
    print("\nLast 3 rows of processed data:")
    print(df_clean[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
    print("==================================================")
