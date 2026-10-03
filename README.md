# 📈 Stock Market Direction Prediction System

An end-to-end Machine Learning and Full-Stack quantitative finance application that predicts whether next-day stock prices will close **UP** or **DOWN** using historical National Stock Exchange of India (NSE) market data.

Built with **Python**, **Scikit-Learn**, **XGBoost**, **FastAPI**, and **React + Vite**.

---

## 📌 1. Project Overview

Predicting exact numerical future stock prices is notoriously noisy and unreliable. Instead, this system frames next-day market forecasting as a **binary classification problem**:

$$\text{Target}_t = \begin{cases} 1 & \text{if } Close_{t+1} > Close_t \quad (\textbf{UP}) \\ 0 & \text{if } Close_{t+1} \le Close_t \quad (\textbf{DOWN / FLAT}) \end{cases}$$

The objective is to discover statistical and momentum edges from historical price action and volume dynamics to outperform a naive majority-class baseline on unseen future market sessions.

### Key Highlights
* **Zero Fake / Mock Data:** Trained strictly on historical NSE blue-chip data (100 stocks across a 10-year period: 2010 to 2020).
* **User Authentication System:** Complete Sign Up (registration) and Sign In (login) system backed by SQLite and NIST-approved PBKDF2 salted password hashing.
* **Strict Anti-Leakage Pipeline:** Future prices are isolated solely to create the target label and are strictly excluded from the feature space.
* **Chronological Time-Series Validation:** No random shuffling; 80% chronological train (2010–2018) and 20% true out-of-sample forward test (2018–2020).
* **Dual ML Ensemble:** Compares **Random Forest** (bagging) and **XGBoost** (boosting) against a naive benchmark.
* **Production REST API:** High-speed, in-memory model caching with FastAPI and session authentication.
* **Interactive Financial Terminal:** React dashboard featuring live predictions, model confidence bars, moving average toggles, user profile badges, and performance comparisons.


---

## 🏗️ 2. System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   KAGGLE RAW DATA                      │
│      100 NSE Stocks + 3 Indices (2010 to 2020)         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                PREPROCESSING PIPELINE                  │
│   • Duplicate resolution   • Chronological sorting     │
│   • Numeric validation     • Non-destructive imputation│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                    │
│   • Stationary MA ratios   • Intraday returns & range  │
│   • Volume surge metrics   • 14-day RSI Momentum       │
│   • Target: Tomorrow Direction (UP=1, DOWN=0)          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│               TIME-SERIES SPLIT & ML                   │
│   • Train (2010 - 2018)    • Test (2018 - 2020)        │
│   • Random Forest (54.2%)  • XGBoost (53.8%)           │
│   • Joblib Serialization   • Immutable feature schema  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND                       │
│   • In-memory model cache  • Dynamic feature pipeline  │
│   • Endpoints: /stocks, /stock/{ticker}, /predict      │
└──────────────────────────┬─────────────────────────────┘
                           │  HTTP / JSON / CORS
                           ▼
┌────────────────────────────────────────────────────────┐
│               REACT + VITE DASHBOARD                   │
│   • Stock dropdown (100)   • Model selector toggle     │
│   • Real probability bar   • Interactive SVG chart     │
│   • Out-of-sample metrics  • Financial disclaimer      │
└────────────────────────────────────────────────────────┘
```

---

## 📁 3. Project Structure

```text
stock-prediction-system/
│
├── data/
│   ├── raw/                           # Raw Excel/CSV stock datasets from Kaggle
│   │   ├── Stock_TCS.xlsx
│   │   ├── Stock_RELIANCE.xlsx
│   │   └── ... (100 stocks + 3 indices)
│   └── processed/
│       ├── processed_stock_data.csv   # Cleaned 2,668 rows of OHLCV data
│       └── featured_stock_data.csv    # 17 features + directional Target
│
├── notebooks/
│   ├── stock_analysis.ipynb           # Interactive EDA Jupyter notebook
│   ├── generate_notebook.py           # Notebook builder
│   ├── run_eda.py                     # Standalone CLI runner for EDA
│   └── plots/                         # High-res chart exports
│       ├── closing_price_trend.png
│       ├── price_volume_combined.png
│       ├── ohlc_analysis.png
│       ├── correlation_heatmap.png
│       └── confusion_matrices.png
│
├── models/
│   ├── random_forest_model.pkl        # Serialized Random Forest model
│   ├── xgboost_model.pkl              # Serialized XGBoost model
│   ├── model_metadata.json            # Exact 17-feature order & class mappings
│   └── evaluation_results.json        # Test performance metrics & confusion matrix
│
├── backend/
│   ├── main.py                        # FastAPI production application
│   ├── config.py                      # Application configuration & CORS settings
│   ├── requirements.txt               # Pinned Python dependencies
│   ├── clean_data.py                  # CLI runner for Phase 2 data cleaning
│   ├── prepare_dataset.py             # CLI runner for Phase 4 & 5 feature prep
│   ├── split_data.py                  # CLI runner for Phase 6 time-series split
│   ├── train_rf.py                    # CLI runner for Phase 7 Random Forest
│   ├── train_xgb.py                   # CLI runner for Phase 8 XGBoost
│   ├── evaluate_models.py             # CLI runner for Phase 9 model comparison
│   ├── save_models.py                 # CLI runner for Phase 10 model serialization
│   ├── test_prediction.py             # CLI runner for Phase 11 inference service
│   ├── test_api.py                    # Automated FastAPI endpoint test suite
│   ├── test_system.py                 # Phase 14 master end-to-end verification suite
│   │
│   ├── services/
│   │   ├── data_service.py            # Loading, cleaning, and validation
│   │   ├── feature_service.py         # 17 technical features & target creation
│   │   ├── model_service.py           # Time-series splitting, RF & XGB training
│   │   └── prediction_service.py      # In-memory inference engine
│   │
│   └── utils/
│       └── helpers.py                 # Ticker scanning and helper functions
│
├── frontend/                          # React 18 + Vite frontend
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx                   # React entry point
│       ├── App.jsx                    # Master financial dashboard
│       ├── index.css                  # Professional dark terminal design system
│       └── components/
│           ├── StockChart.jsx         # Interactive SVG chart with MA 20/50 toggles
│           └── ModelEvaluation.jsx    # Out-of-sample benchmark table
│
├── README.md
└── .gitignore
```

---

## ⚙️ 4. Installation & Setup

### Prerequisites
* **Python 3.10+** (tested on Python 3.13)
* **Node.js 18+** and **npm**

### Step 1: Clone or Navigate to the Repository
```bash
cd stock-prediction-system
```

### Step 2: Install Python Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 5. Running the Application

### Option A: Start the Complete System (Backend + Frontend)

#### 1. Launch FastAPI Backend (Terminal 1):
```bash
uvicorn backend.main:app --reload --port 8000
```
* **API Documentation (Swagger UI):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### 2. Launch React Dashboard (Terminal 2):
```bash
cd frontend
npm run dev
```
* **Interactive Dashboard:** [http://localhost:5173](http://localhost:5173)

---

### Option B: Run Standalone Pipeline Steps

Each phase can be independently executed and verified via dedicated CLI tools:

```bash
# Phase 1: Inspect Kaggle dataset
python backend/inspect_dataset.py

# Phase 2: Run data cleaning pipeline
python backend/clean_data.py

# Phase 3: Run EDA & generate charts
python notebooks/run_eda.py

# Phase 4 & 5: Generate features and next-day target
python backend/prepare_dataset.py

# Phase 6: Verify chronological time-series split
python backend/split_data.py

# Phase 7: Train Random Forest model
python backend/train_rf.py

# Phase 8: Train XGBoost model
python backend/train_xgb.py

# Phase 9: Run model evaluation & comparison
python backend/evaluate_models.py

# Phase 10: Serialize models with Joblib
python backend/save_models.py

# Phase 11: Run CLI prediction inference
python backend/test_prediction.py --ticker TCS --model xgboost

# Phase 12: Run automated API endpoint test suite
python backend/test_api.py

# Phase 14: Run Master End-to-End System Verification Suite
python backend/test_system.py
```

---

## 🔬 6. Machine Learning Methodology

### 1. The 17 Engineered Features
Raw prices are **non-stationary** (a price of ₹2,500 in 2020 has a completely different distribution from ₹700 in 2010). If raw prices were passed into decision trees, the model would fail to extrapolate to higher price ranges. We engineered 17 relative, percentage-based, and stationary features:

| Feature | Type | Financial Interpretation |
|---|---|---|
| `Daily_Return` | Stationary | Day-over-day price velocity |
| `Price_Change` | Intraday | Intraday points gained or lost |
| `Intraday_Return` | Stationary | Intraday percentage change $(Close - Open) / Open$ |
| `Intraday_Volatility` | Stationary | Daily trading spread $(High - Low) / Open$ |
| `MA_5, 10, 20, 50` | Trend | Moving averages capturing short-, medium-, and long-term trend lines |
| `Close_to_MA (5, 10, 20, 50)` | **Stationary Ratio** | Ratio of price relative to trend ($Close / MA_n$). Indicates premium or discount |
| `Volume_Change` | Stationary | Day-over-day trading volume percentage change |
| `Volume_to_MA5` | Stationary Ratio | Volume relative to its 5-day rolling average (identifies institutional volume surges) |
| `RSI_14` | Momentum | 14-day Relative Strength Index (overbought > 70, oversold < 30) |

### 2. Why Random Shuffling is Invalid
In standard tabular ML, `train_test_split(..., shuffle=True)` is routine. In financial time-series, random shuffling introduces severe **lookahead bias**:
* If August 2020 data were randomly shuffled into training while May 2020 was left in testing, the model would "peek into the future" to forecast the past.
* Markets exhibit **volatility clustering and regime shifts**. Shuffling destroys temporal continuity.
* **Our Strategy:** Strict chronological split:
  * **Train Set (80%):** March 2010 to August 2018 (2,094 sessions)
  * **Test Set (20% Out-of-Sample):** August 2018 to September 2020 (524 forward sessions)

### 3. Out-of-Sample Benchmark Comparison

| Model | Test Accuracy | Precision (UP) | Recall (UP) | F1-Score | Strategy |
|---|---|---|---|---|---|
| **Baseline (Majority Class)** | 51.91% | 0.5191 | 1.0000 | 0.6834 | Naive heuristic: always predict "UP" |
| **Random Forest** | **54.20%** | **0.5423** | 0.7537 | 0.6308 | **+2.29% accuracy lift** over baseline |
| **XGBoost** | **53.82%** | **0.5381** | **0.7794** | **0.6366** | **Highest directional recall** (212/272 UP days) |

---

## ⚠️ 7. Limitations & Real-World Trading Disclaimers

1. **Efficient Market Hypothesis (EMH):** Financial markets reflect incoming information almost instantaneously. In quantitative hedge funds, a sustained directional accuracy of **53% - 55%** is considered realistic and statistically significant over large sample sizes.
2. **Transaction Costs & Slippage:** A 54% directional hit rate can easily become unprofitable once exchange transaction fees, brokerage fees, and bid-ask slippage are deducted.
3. **Asymmetric Payoffs:** Accuracy measures *frequency*, not *magnitude*. Six small +0.5% wins can be wiped out by one unexpected -8% market crash. Real-world systems require position sizing and stop-loss mechanisms.
4. **Educational Purpose:** This project demonstrates quantitative machine learning and full-stack integration and does not constitute financial advice.

---

## 🐙 8. Git & GitHub Setup

Initialize your repository and commit the project using the commands below:

```bash
# 1. Initialize Git repository
git init

# 2. Add files (respecting .gitignore which excludes node_modules and logs)
git add .

# 3. Create initial commit
git commit -m "Complete stock market prediction system (ML + FastAPI + React)"

# 4. Set default branch to main
git branch -M main

# 5. Connect your GitHub remote repository (replace with your actual URL)
git remote add origin YOUR_GITHUB_REPOSITORY_URL

# 6. Push to GitHub
git push -u origin main
```

> **Note on Large Datasets:** If your `data/raw/` directory exceeds GitHub's 100MB file limit, you can track raw files using [Git LFS](https://git-lfs.github.com/) or uncomment the `# data/raw/*.xlsx` line in `.gitignore` so users can download the dataset directly from Kaggle into `data/raw/`.

---

## 👨‍💻 Author & Pair Programming
Built as a complete end-to-end AI/ML and Full-Stack quantitative project. Clean, modular, and interview-ready.
