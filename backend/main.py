import os
import sys
import json
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.dirname(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from config import settings
from utils.helpers import get_available_tickers
from services.prediction_service import predict_stock_direction, load_model_metadata
from services.data_service import get_raw_filepath, load_raw_stock_data, clean_stock_data
from services.feature_service import create_features

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production REST API for Stock Market Direction Prediction (UP/DOWN) using trained ML models."
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Root endpoint providing system information and available routes."""
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "stocks": "/stocks",
            "stock_data": "/stock/{ticker}",
            "prediction": "/predict/{ticker}",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "description": "Binary classification system predicting whether next-day stock price closes UP or DOWN."
    }


@app.get("/health")
def health_check():
    """Health check verifying presence of datasets and trained ML models."""
    rf_model_exists = os.path.exists(os.path.join(settings.MODELS_DIR, "random_forest_model.pkl"))
    xgb_model_exists = os.path.exists(os.path.join(settings.MODELS_DIR, "xgboost_model.pkl"))
    data_raw_exists = os.path.exists(settings.DATA_RAW_DIR)
    
    status = "healthy" if (rf_model_exists and xgb_model_exists and data_raw_exists) else "degraded"
    return {
        "status": status,
        "components": {
            "random_forest_model": "loaded" if rf_model_exists else "missing",
            "xgboost_model": "loaded" if xgb_model_exists else "missing",
            "data_raw_directory": "available" if data_raw_exists else "missing"
        }
    }


@app.get("/stocks")
def list_stocks():
    """Returns list of actual stock tickers available in the dataset."""
    tickers = get_available_tickers(settings.DATA_RAW_DIR)
    if not tickers:
        raise HTTPException(status_code=404, detail="No stock datasets found in data/raw/")
    return {
        "total_stocks": len(tickers),
        "stocks": tickers,
        "default_stock": settings.DEFAULT_TICKER
    }


@app.get("/stock/{ticker}")
def get_stock_details(
    ticker: str,
    days: int = Query(default=120, ge=30, le=1000, description="Number of recent trading days to return for chart")
):
    """
    Returns latest stock quote and historical series (with MA_20 and MA_50)
    for interactive chart rendering in the dashboard.
    """
    try:
        raw_path = get_raw_filepath(ticker.upper(), settings.DATA_RAW_DIR)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Stock ticker '{ticker.upper()}' does not exist in the dataset. Use /stocks to view available stocks."
        )

    try:
        raw_df = load_raw_stock_data(raw_path)
        clean_df, _ = clean_stock_data(raw_df)
        featured_df = create_features(clean_df, drop_warmup=False)

        # Slice recent 'days'
        recent_df = featured_df.tail(days).copy()

        # Format historical chart data
        chart_data = []
        for _, row in recent_df.iterrows():
            date_str = row['Date'].strftime("%Y-%m-%d") if hasattr(row['Date'], "strftime") else str(row['Date'])[:10]
            chart_data.append({
                "date": date_str,
                "close": round(float(row['Close']), 2),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "volume": int(row['Volume']),
                "ma_20": round(float(row['MA_20']), 2) if pd.notnull(row.get('MA_20')) else None,
                "ma_50": round(float(row['MA_50']), 2) if pd.notnull(row.get('MA_50')) else None,
            })

        latest = recent_df.iloc[-1]
        latest_date_str = latest['Date'].strftime("%Y-%m-%d") if hasattr(latest['Date'], "strftime") else str(latest['Date'])[:10]

        return {
            "stock": ticker.upper(),
            "latest_date": latest_date_str,
            "latest_close": round(float(latest['Close']), 2),
            "open": round(float(latest['Open']), 2),
            "high": round(float(latest['High']), 2),
            "low": round(float(latest['Low']), 2),
            "volume": int(latest['Volume']),
            "day_change": round(float(latest.get('Price_Change', 0)), 2),
            "day_return_pct": f"{float(latest.get('Daily_Return', 0)) * 100:+.2f}%",
            "historical_records_returned": len(chart_data),
            "history": chart_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process stock data: {str(e)}")


@app.get("/predict/{ticker}")
def predict_stock(
    ticker: str,
    model: str = Query(default="xgboost", regex="^(random_forest|xgboost|rf|xgb)$", description="Model to use")
):
    """
    Predicts next-day direction (UP or DOWN) with real continuous model probability.
    
    Example response:
    {
        "stock": "TCS",
        "prediction": "UP",
        "probability": 0.5466,
        "model_used": "XGBoost",
        "latest_close": 2523.45, ...
    }
    """
    try:
        result = predict_stock_direction(ticker=ticker.upper(), model_name=model)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
def get_metrics():
    """Returns evaluation metrics, confusion matrices, and baseline comparison."""
    results_path = os.path.join(settings.MODELS_DIR, "evaluation_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run evaluate_models.py first.")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
