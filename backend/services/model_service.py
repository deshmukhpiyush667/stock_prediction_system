import os
import sys

# Ensure backend root is in python path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from services.feature_service import get_feature_columns


def split_time_series_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Splits time-series data strictly chronologically (No random shuffling).
    
    CRITICAL ML CONCEPT - WHY NO RANDOM SHUFFLING?
    1. Lookahead Bias: If we randomly shuffled stock records, data from 2020 would
       leak into the training set, allowing the model to 'peek' into the future to
       predict past days (e.g. using 2020 market conditions to predict 2015).
    2. Temporal Autocorrelation: Financial markets have regimes, volatility clustering,
       and macroeconomic memory. Shuffling violates the independent and identically
       distributed (i.i.d.) assumption.
    3. Real-World Alignment: In live trading, an algorithm trained on past history
       must always predict on unseen future trading days.
    
    Split Strategy:
    - First 80% of chronological records -> Training set
    - Final 20% of chronological records -> Out-of-sample Testing set
    """
    # Ensure chronological order
    data = df.copy()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date').reset_index(drop=True)

    feature_cols = get_feature_columns()
    if 'Target' not in data.columns:
        raise ValueError("DataFrame must contain 'Target' column for splitting.")

    # Chronological split index
    total_samples = len(data)
    split_idx = int(total_samples * train_ratio)

    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    X_train = train_data[feature_cols]
    y_train = train_data['Target'].astype(int)

    X_test = test_data[feature_cols]
    y_test = test_data['Target'].astype(int)

    # Class balance stats
    y_train_dist = y_train.value_counts(normalize=True).to_dict()
    y_test_dist = y_test.value_counts(normalize=True).to_dict()

    metadata = {
        "total_samples": total_samples,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_ratio": train_ratio,
        "test_ratio": round(1.0 - train_ratio, 2),
        "train_start_date": train_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in train_data.columns else "N/A",
        "train_end_date": train_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in train_data.columns else "N/A",
        "test_start_date": test_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in test_data.columns else "N/A",
        "test_end_date": test_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in test_data.columns else "N/A",
        "train_up_pct": round(y_train_dist.get(1, 0) * 100, 2),
        "train_down_pct": round(y_train_dist.get(0, 0) * 100, 2),
        "test_up_pct": round(y_test_dist.get(1, 0) * 100, 2),
        "test_down_pct": round(y_test_dist.get(0, 0) * 100, 2),
        "test_baseline_accuracy": round(max(y_test_dist.get(1, 0), y_test_dist.get(0, 0)) * 100, 2),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
    }

    return X_train, y_train, X_test, y_test, metadata


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], pd.Series]:
    """
    Trains a Random Forest Classifier on time-series training data.
    
    Hyperparameters Chosen to Prevent Overfitting:
    - max_depth=6: Limits tree depth so trees capture broad trends rather than memorizing market noise.
    - min_samples_split=10, min_samples_leaf=4: Enforces statistical significance at leaf nodes.
    - random_state=42: Ensures reproducible training.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )

    # Train exclusively on training data
    model.fit(X_train, y_train)

    # Predictions & probabilities on unseen test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "model_name": "Random Forest",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": len(y_test),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        }
    }

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    return model, metrics, importances


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], pd.Series]:
    """
    Trains an XGBoost (Extreme Gradient Boosting) Classifier.
    
    Why XGBoost?
    - Sequentially trains weak decision trees, with each new tree correcting
      the residual errors of the previous ensemble.
    - Uses L1/L2 regularization to control model complexity and prevent overfitting.
    - Subsample and colsample_bytree inject stochastic noise, preventing dominant
      features from skewing the tree branches.
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric='logloss'
    )

    # Train exclusively on training data
    model.fit(X_train, y_train)

    # Predictions & probabilities on unseen test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "model_name": "XGBoost",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": len(y_test),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state
        }
    }

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    return model, metrics, importances



if __name__ == "__main__":
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "featured_stock_data.csv"
    )
    print("==================================================")
    print("PHASE 6: TESTING TIME-SERIES CHRONOLOGICAL SPLIT")
    print("==================================================")
    df_featured = pd.read_csv(csv_path)

    X_train, y_train, X_test, y_test, meta = split_time_series_data(df_featured, train_ratio=0.8)

    print(f"Total Records: {meta['total_samples']}")
    print(f"\nTRAINING SET (80%):")
    print(f"  - Samples:        {meta['train_samples']} trading sessions")
    print(f"  - Date Range:     {meta['train_start_date']} to {meta['train_end_date']}")
    print(f"  - Class Balance:  UP = {meta['train_up_pct']}%, DOWN = {meta['train_down_pct']}%")
    print(f"  - Feature Matrix: {X_train.shape}")

    print(f"\nTESTING SET (20% - Out-of-Sample Forward Window):")
    print(f"  - Samples:        {meta['test_samples']} trading sessions")
    print(f"  - Date Range:     {meta['test_start_date']} to {meta['test_end_date']}")
    print(f"  - Class Balance:  UP = {meta['test_up_pct']}%, DOWN = {meta['test_down_pct']}%")
    print(f"  - Baseline Acc:   {meta['test_baseline_accuracy']}% (Naive majority prediction)")
    print(f"  - Feature Matrix: {X_test.shape}")

    print(f"\nCHRONOLOGICAL SEPARATION CHECK:")
    print(f"  Last Training Date:   {meta['train_end_date']}")
    print(f"  First Testing Date:   {meta['test_start_date']}")
    assert meta['train_end_date'] < meta['test_start_date'], "ERROR: Training and test dates overlap!"
    print("  Temporal Boundary Check: PASS (Zero overlap, strictly forward in time)")
    print("==================================================")
