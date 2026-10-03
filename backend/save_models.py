"""
Phase 10: Model Serialization and Verification using Joblib.
Saves:
  - models/random_forest_model.pkl
  - models/xgboost_model.pkl
  - models/model_metadata.json
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.model_service import split_time_series_data, train_random_forest, train_xgboost
from services.feature_service import get_feature_columns

def save_and_verify_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "processed", "featured_stock_data.csv")
    models_dir = os.path.join(base_dir, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    print("==================================================")
    print("PHASE 10: MODEL SERIALIZATION & PERSISTENCE")
    print("==================================================")

    df_featured = pd.read_csv(csv_path)
    X_train, y_train, X_test, y_test, meta = split_time_series_data(df_featured, train_ratio=0.8)

    # 1. Train models
    print("Training Random Forest...")
    rf_model, rf_metrics, _ = train_random_forest(X_train, y_train, X_test, y_test)

    print("Training XGBoost...")
    xgb_model, xgb_metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)

    # 2. Serialize models with Joblib
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")

    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    print(f"\n[SAVED] Random Forest model: {rf_path} ({os.path.getsize(rf_path) / 1024:.1f} KB)")
    print(f"[SAVED] XGBoost model:       {xgb_path} ({os.path.getsize(xgb_path) / 1024:.1f} KB)")

    # 3. Save feature list and metadata
    metadata = {
        "saved_at": datetime.now().isoformat(),
        "ticker": "TCS",
        "feature_names": get_feature_columns(),
        "feature_count": len(get_feature_columns()),
        "target_mapping": {
            "0": "DOWN",
            "1": "UP"
        },
        "train_period": {
            "start": meta["train_start_date"],
            "end": meta["train_end_date"],
            "samples": meta["train_samples"]
        },
        "test_period": {
            "start": meta["test_start_date"],
            "end": meta["test_end_date"],
            "samples": meta["test_samples"]
        },
        "models": {
            "random_forest": {
                "accuracy": rf_metrics["accuracy"],
                "precision": rf_metrics["precision"],
                "recall": rf_metrics["recall"],
                "f1": rf_metrics["f1"],
                "file": "random_forest_model.pkl"
            },
            "xgboost": {
                "accuracy": xgb_metrics["accuracy"],
                "precision": xgb_metrics["precision"],
                "recall": xgb_metrics["recall"],
                "f1": xgb_metrics["f1"],
                "file": "xgboost_model.pkl"
            }
        }
    }

    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Model metadata & feature order: {meta_path}")

    # 4. Verify deserialization and inference
    print("\n--- Verifying Reloaded Models with Test Inference ---")
    loaded_rf = joblib.load(rf_path)
    loaded_xgb = joblib.load(xgb_path)

    sample_features = X_test.iloc[-1:].copy()
    print("Test Sample Features Vector (17 features):")
    print(sample_features.to_dict(orient="records")[0])

    rf_pred = int(loaded_rf.predict(sample_features)[0])
    rf_prob = float(loaded_rf.predict_proba(sample_features)[0][rf_pred])

    xgb_pred = int(loaded_xgb.predict(sample_features)[0])
    xgb_prob = float(loaded_xgb.predict_proba(sample_features)[0][xgb_pred])

    print(f"\nRandom Forest Inference:")
    print(f"  Prediction:  {rf_pred} -> {metadata['target_mapping'][str(rf_pred)]}")
    print(f"  Probability: {rf_prob:.4f} ({rf_prob*100:.2f}%)")

    print(f"\nXGBoost Inference:")
    print(f"  Prediction:  {xgb_pred} -> {metadata['target_mapping'][str(xgb_pred)]}")
    print(f"  Probability: {xgb_prob:.4f} ({xgb_prob*100:.2f}%)")

    assert rf_prob > 0 and rf_prob <= 1.0, "Invalid RF probability"
    assert xgb_prob > 0 and xgb_prob <= 1.0, "Invalid XGBoost probability"
    print("\n[SUCCESS] Both models loaded and successfully verified with continuous model probabilities.")
    print("==================================================")

if __name__ == "__main__":
    save_and_verify_models()
