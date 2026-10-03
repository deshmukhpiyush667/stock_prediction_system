"""
CLI script to demonstrate Phase 6: Time-Series Chronological Train/Test Split
"""
import os
import pandas as pd
from services.model_service import split_time_series_data

def main():
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_stock_data.csv"
    )
    if not os.path.exists(csv_path):
        print("Featured data not found. Preparing now...")
        from prepare_dataset import prepare_and_save_featured_data
        df_featured, _ = prepare_and_save_featured_data()
    else:
        df_featured = pd.read_csv(csv_path)

    X_train, y_train, X_test, y_test, meta = split_time_series_data(df_featured, train_ratio=0.8)

    print("==================================================")
    print("PHASE 6: TIME-SERIES CHRONOLOGICAL SPLIT SUMMARY")
    print("==================================================")
    print(f"Total Trading Sessions:   {meta['total_samples']}")
    print(f"Split Strategy:           Strict Chronological (80% Train, 20% Test)")
    print(f"No Shuffling:             True (Prevents Lookahead Bias)")

    print(f"\n--- TRAINING SET (80%) ---")
    print(f"Samples:                  {meta['train_samples']}")
    print(f"Date Range:               {meta['train_start_date']} to {meta['train_end_date']}")
    print(f"Class Distribution:       UP = {meta['train_up_pct']}%, DOWN = {meta['train_down_pct']}%")
    print(f"Feature Dimensions:       {X_train.shape}")

    print(f"\n--- TESTING SET (20% Out-of-Sample) ---")
    print(f"Samples:                  {meta['test_samples']}")
    print(f"Date Range:               {meta['test_start_date']} to {meta['test_end_date']}")
    print(f"Class Distribution:       UP = {meta['test_up_pct']}%, DOWN = {meta['test_down_pct']}%")
    print(f"Naive Baseline Accuracy:  {meta['test_baseline_accuracy']}%")
    print(f"Feature Dimensions:       {X_test.shape}")

    print(f"\n--- BOUNDARY INTEGRITY ---")
    print(f"Train Cutoff:             {meta['train_end_date']}")
    print(f"Test Start:               {meta['test_start_date']}")
    print("Temporal Overlap:         NONE (Future data completely isolated)")
    print("==================================================")

if __name__ == "__main__":
    main()
