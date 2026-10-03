import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.feature_service import create_features, get_feature_columns
from services.data_service import get_raw_filepath, load_raw_stock_data, clean_stock_data

# Global in-memory model cache for fast API response times
_MODEL_CACHE: Dict[str, Any] = {}
_METADATA_CACHE: Optional[Dict[str, Any]] = None


def get_models_dir() -> str:
    """Returns absolute path to models directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))


def load_model_metadata() -> Dict[str, Any]:
    """Loads and caches model metadata and feature list."""
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE

    meta_path = os.path.join(get_models_dir(), "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Model metadata not found at '{meta_path}'. Please run save_models.py first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        _METADATA_CACHE = json.load(f)
    return _METADATA_CACHE


def load_model(model_name: str = "random_forest") -> Any:
    """
    Loads and caches trained model from disk using Joblib.
    Supports 'random_forest' or 'xgboost'.
    """
    global _MODEL_CACHE
    normalized_name = model_name.lower().replace(" ", "_").replace("-", "_")

    if normalized_name in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_name]

    models_dir = get_models_dir()
    if normalized_name in ["random_forest", "rf"]:
        model_file = "random_forest_model.pkl"
    elif normalized_name in ["xgboost", "xgb"]:
        model_file = "xgboost_model.pkl"
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'random_forest' or 'xgboost'.")

    model_path = os.path.join(models_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file not found at '{model_path}'. Please train and save the model first."
        )

    model = joblib.load(model_path)
    _MODEL_CACHE[normalized_name] = model
    return model


def predict_stock_direction(
    ticker: str = "TCS",
    model_name: str = "random_forest"
) -> Dict[str, Any]:
    """
    Predicts whether the stock price will move UP or DOWN on the next trading day.
    
    Workflow:
    1. Loads clean historical data for the ticker.
    2. Dynamically calculates the 17 technical & momentum features.
    3. Extracts the latest trading day's feature vector.
    4. Passes features in the exact required schema to the loaded model.
    5. Computes continuous probability using model.predict_proba().
    6. Returns structured response with direction, probability, and latest prices.
    """
    # 1. Load metadata and model
    meta = load_model_metadata()
    model = load_model(model_name)
    feature_names = meta.get("feature_names", get_feature_columns())

    # 2. Get cleaned stock time-series
    # First check if processed file exists for standard benchmark ticker
    processed_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "processed_stock_data.csv"
    )
    if ticker.upper() == "TCS" and os.path.exists(processed_path):
        df_stock = pd.read_csv(processed_path)
    else:
        # Dynamically load and clean from raw files
        raw_path = get_raw_filepath(ticker)
        raw_df = load_raw_stock_data(raw_path)
        df_stock, _ = clean_stock_data(raw_df)

    if len(df_stock) < 55:
        raise ValueError(
            f"Insufficient data for ticker '{ticker}'. Need at least 55 trading days to compute 50-day moving average."
        )

    # 3. Calculate features
    df_featured = create_features(df_stock, drop_warmup=False)

    # 4. Extract latest trading day's feature vector as float64
    latest_row = df_featured.iloc[-1]
    latest_features = df_featured[feature_names].iloc[-1:].astype('float64').copy()

    # Verify no NaNs in latest features
    if latest_features.isna().any().any():
        nan_cols = latest_features.columns[latest_features.isna().any()].tolist()
        raise ValueError(f"Latest feature vector contains NaN values in: {nan_cols}")

    # 5. Make prediction & calculate probability
    prediction_num = int(model.predict(latest_features)[0])
    probabilities = model.predict_proba(latest_features)[0]
    predicted_prob = float(probabilities[prediction_num])

    direction_str = "UP" if prediction_num == 1 else "DOWN"
    prob_up = float(probabilities[1]) if len(probabilities) > 1 else predicted_prob

    # Latest date formatting
    latest_date_val = latest_row.get("Date", "Unknown")
    if hasattr(latest_date_val, "strftime"):
        latest_date_str = latest_date_val.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date_val)[:10]

    return {
        "stock": ticker.upper(),
        "latest_date": latest_date_str,
        "latest_close": round(float(latest_row.get("Close", 0.0)), 2),
        "open": round(float(latest_row.get("Open", 0.0)), 2),
        "high": round(float(latest_row.get("High", 0.0)), 2),
        "low": round(float(latest_row.get("Low", 0.0)), 2),
        "volume": int(latest_row.get("Volume", 0)),
        "model_used": "Random Forest" if "forest" in model_name.lower() or "rf" in model_name.lower() else "XGBoost",
        "prediction": direction_str,
        "direction": prediction_num,
        "probability": round(predicted_prob, 4),
        "probability_up": round(prob_up, 4),
        "confidence_pct": f"{predicted_prob * 100:.1f}%",
        "disclaimer": "Model prediction is based on historical technical features and does not guarantee future financial market performance."
    }


if __name__ == "__main__":
    print("==================================================")
    print("PHASE 11: TESTING PREDICTION SERVICE")
    print("==================================================")

    # Test with Random Forest
    rf_result = predict_stock_direction(ticker="TCS", model_name="random_forest")
    print("\n[Prediction Service - Random Forest Output]:")
    print(json.dumps(rf_result, indent=2))

    # Test with XGBoost
    xgb_result = predict_stock_direction(ticker="TCS", model_name="xgboost")
    print("\n[Prediction Service - XGBoost Output]:")
    print(json.dumps(xgb_result, indent=2))

    # Verify probability requirement
    assert isinstance(rf_result["probability"], float), "Probability must be float"
    assert 0.0 <= rf_result["probability"] <= 1.0, "Probability must be between 0 and 1"
    assert rf_result["prediction"] in ["UP", "DOWN"], "Prediction must be UP or DOWN"
    print("\nValidation: PASS (Real probabilities extracted from trained models)")
    print("==================================================")
