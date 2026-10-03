"""
Executes the EDA analysis and generates all required charts into notebooks/plots/
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "processed_stock_data.csv")
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print("==================================================")
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("==================================================")
    print("\n1. DATASET OVERVIEW:")
    print(f"Total Rows:    {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Date Range:    {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().sum() > 0 else '0 missing values across all columns'}")

    # 2. Closing Price Chart
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Close'], color='#2563eb', linewidth=1.8, label='Daily Close Price')
    plt.title(f"{df['Symbol'].iloc[0]} - Historical Closing Price (2010 - 2020)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    chart1 = os.path.join(plots_dir, "closing_price_trend.png")
    plt.savefig(chart1, dpi=200)
    plt.close()
    print(f"\n2. Closing price chart saved to: {chart1}")

    # 3. Trading Volume Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df['Date'], df['Close'], color='#2563eb', linewidth=1.5, label='Close Price')
    ax1.set_ylabel("Close Price (INR)", fontsize=11)
    ax1.set_title(f"{df['Symbol'].iloc[0]} Price & Volume Over Time", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    ax2.bar(df['Date'], df['Volume'], color='#10b981', width=2.5, alpha=0.8, label='Trading Volume')
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Volume (Shares)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    chart2 = os.path.join(plots_dir, "price_volume_combined.png")
    plt.savefig(chart2, dpi=200)
    plt.close()
    print(f"3. Trading volume chart saved to: {chart2}")

    # 4. OHLC Analysis
    df['Intraday_Spread'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['Intraday_Spread'] / df['Open']) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    sns.boxplot(data=df[ohlc_cols], palette="Blues", ax=axes[0])
    axes[0].set_title("OHLC Price Distributions", fontweight='bold')
    axes[0].set_ylabel("Price (INR)")
    sns.histplot(df['Daily_Range_Pct'], bins=50, kde=True, color='#f59e0b', ax=axes[1])
    axes[1].set_title("Distribution of Daily Trading Range % ((High-Low)/Open)", fontweight='bold')
    axes[1].set_xlabel("Daily Range %")
    plt.tight_layout()
    chart3 = os.path.join(plots_dir, "ohlc_analysis.png")
    plt.savefig(chart3, dpi=200)
    plt.close()
    print(f"4. OHLC analysis chart saved to: {chart3}")

    # 5. Correlation Heatmap
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable_Volume', 'Deliverble_Pct']
    available_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[available_cols].corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Numerical Market Features", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    chart4 = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(chart4, dpi=200)
    plt.close()
    print(f"5. Correlation heatmap saved to: {chart4}")

    # 6. Basic Statistics
    stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
    available_stats_cols = [c for c in stats_cols if c in df.columns]
    stats_summary = pd.DataFrame({
        'Mean': df[available_stats_cols].mean(),
        'Median': df[available_stats_cols].median(),
        'Std Dev': df[available_stats_cols].std(),
        'Min': df[available_stats_cols].min(),
        'Max': df[available_stats_cols].max()
    })
    print("\n6. BASIC DESCRIPTIVE STATISTICS:")
    print(stats_summary.to_string())
    print("==================================================")

if __name__ == "__main__":
    run_eda()
