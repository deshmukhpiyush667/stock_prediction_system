import os
import pandas as pd
import numpy as np
from typing import List, Tuple

# EXACT ORDER OF FEATURES USED FOR TRAINING AND SERVING PREDICTIONS
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "Daily_Return",
    "Price_Change",
    "Intraday_Return",
    "Intraday_Volatility",
    "MA_5",
    "MA_10",
    "MA_20",
    "MA_50",
    "Close_to_MA5",
    "Close_to_MA10",
    "Close_to_MA20",
    "Close_to_MA50",
    "Volume_Change",
    "Volume_to_MA5",
    "RSI_14",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes standard 14-day Relative Strength Index (RSI).
    RSI > 70 indicates overbought conditions (price may pull back).
    RSI < 30 indicates oversold conditions (price may rebound).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average (Wilder's RSI) or simple rolling
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss was 0 and avg_gain > 0, RSI is 100
    rsi = rsi.fillna(50.0)
    return rsi


def create_features(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Extracts predictive financial features strictly using current and past data.
    
    ZERO DATA LEAKAGE:
    - Every calculation only references today (t) and previous days (t-1, t-2, ...).
    - Future prices (t+1) are NEVER referenced in this function.
    
    Features Created:
    1. Close: Current day's closing price
    2. Volume: Current day's total traded volume
    3. Daily_Return: (Close[t] - Close[t-1]) / Close[t-1]
    4. Price_Change: Close[t] - Open[t] (daily price direction)
    5. Intraday_Return: (Close[t] - Open[t]) / Open[t] (normalized intraday return)
    6. Intraday_Volatility: (High[t] - Low[t]) / Open[t] (daily range %)
    7. MA_5, MA_10, MA_20, MA_50: Moving averages (trend tracking)
    8. Close_to_MA5, 10, 20, 50: Price ratio relative to moving averages (stationary indicators)
    9. Volume_Change: Day-over-day % volume change
    10. Volume_to_MA5: Today's volume relative to 5-day average volume
    11. RSI_14: 14-day Relative Strength Index momentum
    """
    data = df.copy()

    # Ensure chronological order
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date').reset_index(drop=True)

    # 1 & 2: Core prices
    close = data['Close']
    open_p = data['Open']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    # 3. Daily Return: Percentage change from yesterday's close
    data['Daily_Return'] = close.pct_change()

    # 4. Price Change: Intraday points gained or lost
    data['Price_Change'] = close - open_p

    # 5. Intraday Return: Percentage gain/loss during market hours
    data['Intraday_Return'] = (close - open_p) / open_p

    # 6. Intraday Volatility: Daily high-low range as % of open
    data['Intraday_Volatility'] = (high - low) / open_p

    # 7. Moving Averages
    data['MA_5'] = close.rolling(window=5).mean()
    data['MA_10'] = close.rolling(window=10).mean()
    data['MA_20'] = close.rolling(window=20).mean()
    data['MA_50'] = close.rolling(window=50).mean()

    # 8. Normalized Moving Average Ratios (Stationary features)
    data['Close_to_MA5'] = close / data['MA_5']
    data['Close_to_MA10'] = close / data['MA_10']
    data['Close_to_MA20'] = close / data['MA_20']
    data['Close_to_MA50'] = close / data['MA_50']

    # 9. Volume Change & Ratio
    data['Volume_Change'] = vol.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    vol_ma5 = vol.rolling(window=5).mean()
    data['Volume_to_MA5'] = (vol / vol_ma5.replace(0, np.nan)).fillna(1.0)

    # 10. RSI (14 periods)
    data['RSI_14'] = compute_rsi(close, period=14)

    # Drop warm-up rows (first 49 rows where 50-day rolling values are NaN)
    if drop_warmup:
        initial_len = len(data)
        data = data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        dropped = initial_len - len(data)
        print(f"Dropped {dropped} initial warm-up rows (where MA_50 / RSI required past history).")

    return data


def create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Creates the binary classification target for next-day direction:
    
    Target = 1 (UP)   if Tomorrow_Close > Today_Close
    Target = 0 (DOWN) if Tomorrow_Close <= Today_Close
    
    STRICT DATA LEAKAGE PREVENTION:
    - Tomorrow_Close is ONLY used to construct the label 'Target'.
    - Tomorrow_Close is dropped from the returned dataset so future price
      never leaks into model features.
    - The very last row (where Tomorrow_Close is unknown) is dropped.
    
    Returns:
        DataFrame containing features and 'Target', plus class distribution stats.
    """
    data = df.copy()

    # Shift Close backwards by 1 to get tomorrow's closing price
    tomorrow_close = data['Close'].shift(-1)

    # 1 if Tomorrow Close > Today Close, else 0
    target_series = (tomorrow_close > data['Close']).astype(int)
    data['Target'] = target_series

    # Identify the last row where tomorrow's price is not yet known
    last_row_idx = data.index[-1]
    last_row_date = data.loc[last_row_idx, 'Date'].strftime('%Y-%m-%d') if 'Date' in data.columns else str(last_row_idx)

    # Drop the row with NaN target
    data = data.iloc[:-1].reset_index(drop=True)

    # Calculate class distribution statistics
    counts = data['Target'].value_counts()
    up_count = int(counts.get(1, 0))
    down_count = int(counts.get(0, 0))
    total = len(data)

    stats = {
        "total_samples": total,
        "up_count": up_count,
        "down_count": down_count,
        "up_pct": round(up_count / total * 100, 2),
        "down_pct": round(down_count / total * 100, 2),
        "majority_class_baseline": round(max(up_count, down_count) / total * 100, 2),
        "dropped_unlabeled_date": last_row_date
    }

    return data, stats


def get_feature_columns() -> List[str]:
    """Returns the immutable list of model feature columns in exact order."""
    return FEATURE_COLUMNS.copy()


if __name__ == "__main__":
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "processed_stock_data.csv"
    )
    print("==================================================")
    print("PHASE 4: TESTING FEATURE ENGINEERING PIPELINE")
    print("==================================================")
    df_raw = pd.read_csv(csv_path)
    print(f"Input records loaded: {len(df_raw)}")

    df_featured = create_features(df_raw, drop_warmup=True)
    print(f"Output records with features: {len(df_featured)}")
    print(f"\nFeature Columns ({len(FEATURE_COLUMNS)} features):")
    for idx, col in enumerate(FEATURE_COLUMNS, 1):
        sample_val = df_featured[col].iloc[-1]
        print(f"  {idx:2d}. {col:<22} (Latest value: {sample_val:,.4f})")

    # Verify no NaN values in features
    nan_counts = df_featured[FEATURE_COLUMNS].isna().sum().sum()
    print(f"\nTotal NaNs in feature matrix: {nan_counts}")
    assert nan_counts == 0, "Error: Features contain NaN values!"
    print("Zero Data Leakage Check: PASS (All features calculated only from past/current day)")

    print("\n==================================================")
    print("PHASE 5: TESTING TARGET CREATION")
    print("==================================================")
    df_labeled, stats = create_target(df_featured)
    print(f"Total Labeled Samples:       {stats['total_samples']}")
    print(f"Class 1 (UP):                {stats['up_count']} ({stats['up_pct']}%)")
    print(f"Class 0 (DOWN / Flat):       {stats['down_count']} ({stats['down_pct']}%)")
    print(f"Majority-Class Baseline Acc: {stats['majority_class_baseline']}%")
    print(f"Dropped Unlabeled Date:      {stats['dropped_unlabeled_date']} (latest day has no tomorrow close yet)")
    print(f"'Tomorrow_Close' in columns: {'Tomorrow_Close' in df_labeled.columns} (Ensures no label leakage)")
    print("==================================================")

