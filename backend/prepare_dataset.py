"""
Script to prepare and save the complete featured and labeled dataset.
Output: data/processed/featured_stock_data.csv
"""
import os
import pandas as pd
from services.feature_service import create_features, create_target, get_feature_columns

def prepare_and_save_featured_data(ticker: str = "TCS"):
    input_csv = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "processed_stock_data.csv"
    )
    output_csv = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "featured_stock_data.csv"
    )

    print(f"Loading cleaned data from: {input_csv}")
    df_clean = pd.read_csv(input_csv)

    print("Generating 17 technical and statistical features...")
    df_feat = create_features(df_clean, drop_warmup=True)

    print("Creating next-day binary classification target (UP=1, DOWN=0)...")
    df_labeled, stats = create_target(df_feat)

    df_labeled.to_csv(output_csv, index=False)
    print(f"Saved featured dataset to: {output_csv}")

    print("\n--- Target & Dataset Summary ---")
    print(f"Total labeled samples:   {stats['total_samples']}")
    print(f"UP days (Class 1):       {stats['up_count']} ({stats['up_pct']}%)")
    print(f"DOWN days (Class 0):     {stats['down_count']} ({stats['down_pct']}%)")
    print(f"Baseline (Majority Acc): {stats['majority_class_baseline']}%")
    print(f"Latest unlabelled date:  {stats['dropped_unlabeled_date']}")
    print(f"Feature count:           {len(get_feature_columns())}")

    return df_labeled, stats

if __name__ == "__main__":
    prepare_and_save_featured_data()
