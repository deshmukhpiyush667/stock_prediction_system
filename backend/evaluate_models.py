"""
Phase 8 & 9: Model Evaluation, Baseline Comparison, and Visual Confusion Matrix
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from services.model_service import split_time_series_data, train_random_forest, train_xgboost

def evaluate_all():
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_stock_data.csv"
    )
    plots_dir = os.path.join(os.path.dirname(__file__), "..", "notebooks", "plots")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_featured = pd.read_csv(csv_path)
    X_train, y_train, X_test, y_test, meta = split_time_series_data(df_featured, train_ratio=0.8)

    # 1. Baseline: Predict Majority Class (UP = 1)
    y_baseline = np.ones(len(y_test), dtype=int)
    cm_base = confusion_matrix(y_test, y_baseline).tolist()
    base_metrics = {
        "model_name": "Baseline (Majority Class)",
        "accuracy": round(float(accuracy_score(y_test, y_baseline)), 4),
        "precision": round(float(precision_score(y_test, y_baseline, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_baseline, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_baseline, zero_division=0)), 4),
        "confusion_matrix": cm_base,
        "description": "Always predicts the market will go UP (naive heuristic)"
    }

    # 2. Random Forest
    rf_model, rf_metrics, rf_importances = train_random_forest(X_train, y_train, X_test, y_test)

    # 3. XGBoost
    xgb_model, xgb_metrics, xgb_importances = train_xgboost(X_train, y_train, X_test, y_test)

    # Build Comparison Table
    results = [base_metrics, rf_metrics, xgb_metrics]

    # Save metrics JSON for Backend & Frontend access
    results_json_path = os.path.join(models_dir, "evaluation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta,
            "models": {
                "baseline": base_metrics,
                "random_forest": rf_metrics,
                "xgboost": xgb_metrics
            }
        }, f, indent=2)

    # Visual Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Random Forest CM
    cm_rf = np.array(rf_metrics['confusion_matrix'])
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar=False,
        xticklabels=["Pred DOWN (0)", "Pred UP (1)"],
        yticklabels=["Actual DOWN (0)", "Actual UP (1)"]
    )
    axes[0].set_title(f"Random Forest (Acc: {rf_metrics['accuracy']*100:.1f}%)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("True Market Direction", fontsize=11)
    axes[0].set_xlabel("Predicted Direction", fontsize=11)

    # XGBoost CM
    cm_xgb = np.array(xgb_metrics['confusion_matrix'])
    sns.heatmap(
        cm_xgb,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        cbar=False,
        xticklabels=["Pred DOWN (0)", "Pred UP (1)"],
        yticklabels=["Actual DOWN (0)", "Actual UP (1)"]
    )
    axes[1].set_title(f"XGBoost (Acc: {xgb_metrics['accuracy']*100:.1f}%)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("True Market Direction", fontsize=11)
    axes[1].set_xlabel("Predicted Direction", fontsize=11)

    plt.suptitle("Out-of-Sample Confusion Matrices (Testing Period: Aug 2018 - Sep 2020)", fontsize=14, y=1.02)
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Print Clean Comparison Table
    print("==========================================================================")
    print("PHASE 8 & 9: MODEL EVALUATION & COMPARISON SUMMARY")
    print("==========================================================================")
    print(f"Testing Set: {meta['test_samples']} days ({meta['test_start_date']} to {meta['test_end_date']})")
    print("\nComparison Table:")
    print(f"{'Model':<26} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 72)
    for r in results:
        print(f"{r['model_name']:<26} {r['accuracy']*100:>7.2f}%   {r['precision']:>10.4f}   {r['recall']:>10.4f}   {r['f1']:>8.4f}")
    print("-" * 72)

    print(f"\nVisual Confusion Matrix saved to: {cm_plot_path}")
    print(f"Evaluation metrics JSON saved to: {results_json_path}")
    print("\n==========================================================================")

    return results, cm_plot_path

if __name__ == "__main__":
    evaluate_all()
