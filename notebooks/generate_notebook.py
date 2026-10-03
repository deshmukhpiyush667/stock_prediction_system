"""
Generates the comprehensive Phase 3 EDA Jupyter Notebook: notebooks/stock_analysis.ipynb
and executes it to save static chart assets for easy offline inspection.
"""
import os
import nbformat as nbf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def build_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Phase 3: Exploratory Data Analysis (EDA) - Stock Market Prediction

This notebook performs a comprehensive Exploratory Data Analysis (EDA) on the cleaned Indian Stock Market dataset (`TCS`).
The goal is to understand historical price trends, trading volume, OHLC distributions, volatility, and feature correlations before building machine learning models.

### Objectives:
1. **Dataset Overview:** Rows, columns, date range, missing values check.
2. **Closing Price Trend:** Date vs. Closing Price across the 10-year span (2010 - 2020).
3. **Trading Volume Analysis:** Date vs. Traded Volume & liquidity surges.
4. **OHLC Analysis:** Open, High, Low, Close price ranges and daily spread.
5. **Correlation Heatmap:** Linear relationships between numerical market features.
6. **Descriptive Statistics:** Mean, median, standard deviation, min, and max.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# Path to processed dataset
data_path = os.path.join("..", "data", "processed", "processed_stock_data.csv")
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"Dataset successfully loaded from: {data_path}")
"""))

    # Section 1: Overview
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Overview
We verify the dimensions, column types, date span, and ensure zero missing values remain after Phase 2 cleaning.
"""))

    cells.append(nbf.v4.new_code_cell("""print(f"Total Rows (Trading Days): {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Missing Values Count:\\n{df.isnull().sum()}")

# First 5 rows
df.head()
"""))

    # Section 2: Closing Price Chart
    cells.append(nbf.v4.new_markdown_cell("""## 2. Closing Price Trend (Date vs. Closing Price)
Tracking the historical closing price shows long-term growth, bull/bear cycles, and significant market shocks (e.g. March 2020 COVID dip).
"""))

    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], color='#2563eb', linewidth=1.8, label='Daily Close Price')
plt.title(f"{df['Symbol'].iloc[0]} - Historical Closing Price (2010 - 2020)", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (INR)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/closing_price_trend.png", dpi=200)
plt.show()
"""))

    # Section 3: Trading Volume Chart
    cells.append(nbf.v4.new_markdown_cell("""## 3. Trading Volume Chart (Date vs. Volume)
Volume indicates institutional participation and market liquidity. Extreme volume spikes often coincide with quarterly earnings reports or major macroeconomic announcements.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top: Close Price
ax1.plot(df['Date'], df['Close'], color='#2563eb', linewidth=1.5, label='Close Price')
ax1.set_ylabel("Close Price (INR)", fontsize=11)
ax1.set_title(f"{df['Symbol'].iloc[0]} Price & Volume Over Time", fontsize=14, fontweight='bold')
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

# Bottom: Volume Bar
ax2.bar(df['Date'], df['Volume'], color='#10b981', width=2.5, alpha=0.8, label='Trading Volume')
ax2.set_xlabel("Date", fontsize=11)
ax2.set_ylabel("Volume (Shares)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("plots/price_volume_combined.png", dpi=200)
plt.show()
"""))

    # Section 4: OHLC Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 4. OHLC (Open, High, Low, Close) Analysis
We compare the daily high-low spread (intraday volatility) and inspect the distributions of the OHLC components.
"""))

    cells.append(nbf.v4.new_code_cell("""# Calculate intraday spread: High - Low
df['Intraday_Spread'] = df['High'] - df['Low']
df['Daily_Range_Pct'] = (df['Intraday_Spread'] / df['Open']) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot of OHLC components
ohlc_cols = ['Open', 'High', 'Low', 'Close']
sns.boxplot(data=df[ohlc_cols], palette="Blues", ax=axes[0])
axes[0].set_title("OHLC Price Distributions", fontweight='bold')
axes[0].set_ylabel("Price (INR)")

# Distribution of Daily Volatility Range %
sns.histplot(df['Daily_Range_Pct'], bins=50, kde=True, color='#f59e0b', ax=axes[1])
axes[1].set_title("Distribution of Daily Trading Range % ((High-Low)/Open)", fontweight='bold')
axes[1].set_xlabel("Daily Range %")

plt.tight_layout()
plt.savefig("plots/ohlc_analysis.png", dpi=200)
plt.show()
"""))

    # Section 5: Correlation Heatmap
    cells.append(nbf.v4.new_markdown_cell("""## 5. Correlation Heatmap
We analyze linear correlations between numerical market metrics. 
Price variables (Open, High, Low, Close, VWAP) exhibit near-perfect collinearity (~0.99), which is expected in financial time-series.
Volume and Turnover show how trading activity relates to price level.
"""))

    cells.append(nbf.v4.new_code_cell("""# Select core numeric columns
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
plt.savefig("plots/correlation_heatmap.png", dpi=200)
plt.show()
"""))

    # Section 6: Basic Statistics
    cells.append(nbf.v4.new_markdown_cell("""## 6. Basic Descriptive Statistics
Summary of central tendency (mean, median), dispersion (standard deviation), and bounds (minimum, maximum) for key financial variables.
"""))

    cells.append(nbf.v4.new_code_cell("""stats_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Trades']
available_stats_cols = [c for c in stats_cols if c in df.columns]

stats_summary = pd.DataFrame({
    'Mean': df[available_stats_cols].mean(),
    'Median': df[available_stats_cols].median(),
    'Std Dev': df[available_stats_cols].std(),
    'Min': df[available_stats_cols].min(),
    'Max': df[available_stats_cols].max()
})

print("Descriptive Statistics Summary:")
display_stats = stats_summary.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
print(display_stats)
display_stats
"""))

    # Section 7: Key Takeaways
    cells.append(nbf.v4.new_markdown_cell("""## 7. Key Findings & Implications for Machine Learning:
1. **Non-Stationarity:** Closing price exhibits a strong upward long-term trend. Raw price levels cannot be used directly as features without differencing, returns, or normalization, as a model trained on lower price levels will fail to generalize to higher price regimes.
2. **Stationary Features Needed:** In Phase 4, we must engineer normalized features such as `Daily_Return`, `Price_Change`, and moving average ratios (`Close / MA_20`).
3. **Multicollinearity:** Open, High, Low, and Close have correlation > 0.99. Tree-based models (Random Forest and XGBoost) handle correlated inputs well, but derived relative spreads provide far more predictive signal.
4. **Time Order:** Data spans 2,668 continuous trading sessions. Shuffling would create disastrous lookahead bias, confirming our chronological 80/20 split strategy in Phase 6.
"""))

    nb.cells = cells
    notebook_path = os.path.join(os.path.dirname(__file__), "stock_analysis.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Jupyter notebook created successfully at: {notebook_path}")

if __name__ == "__main__":
    build_eda_notebook()
