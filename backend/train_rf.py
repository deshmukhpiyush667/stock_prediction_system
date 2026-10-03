"""
CLI script to execute Phase 7: Train Random Forest Classifier
"""
import os
import pandas as pd
from services.model_service import split_time_series_data, train_random_forest

def main():
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_stock_data.csv"
    )
    if not os.path.exists(csv_path):
        print("Featured data not found. Preparing dataset first...")
        from prepare_dataset import prepare_and_save_featured_data
        df_featured, _ = prepare_and_save_featured_data()
    else:
        df_featured = pd.read_csv(csv_path)

    print("Splitting dataset chronologically (80% Train, 20% Test)...")
    X_train, y_train, X_test, y_test, meta = split_time_series_data(df_featured, train_ratio=0.8)

    print("\nTraining Random Forest Classifier...")
    model, metrics, importances = train_random_forest(X_train, y_train, X_test, y_test)

    print("==================================================")
    print("PHASE 7: RANDOM FOREST TRAINING & EVALUATION")
    print("==================================================")
    print(f"Model:                     {metrics['model_name']}")
    print(f"Hyperparameters:           {metrics['hyperparameters']}")
    print(f"\n--- OUT-OF-SAMPLE TEST PERFORMANCE ---")
    print(f"Naive Baseline Accuracy:   {meta['test_baseline_accuracy']}%")
    print(f"Random Forest Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (Class UP):      {metrics['precision']:.4f}")
    print(f"Recall (Class UP):         {metrics['recall']:.4f}")
    print(f"F1-Score:                  {metrics['f1']:.4f}")

    print(f"\nConfusion Matrix on Unseen Test Data ({metrics['test_samples']} days):")
    cm = metrics['confusion_matrix']
    print(f"                Predicted DOWN   Predicted UP")
    print(f"  Actual DOWN      {cm[0][0]:<15} {cm[0][1]}")
    print(f"  Actual UP        {cm[1][0]:<15} {cm[1][1]}")

    print(f"\nTop 8 Most Informative Features:")
    for idx, (feat, score) in enumerate(importances.head(8).items(), 1):
        print(f"  {idx}. {feat:<20} {score:.4f} ({score*100:.2f}%)")
    print("==================================================")

if __name__ == "__main__":
    main()
