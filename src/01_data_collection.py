# ==============================================================================
# 01_data_collection.py
# Task 1 & 2: Data Collection and Dataset Connection
# Connects to the real 50k bug dataset (Kaggle-sourced).
# Input:  data/bug_dataset_50k.csv
# Output: Prints dataset summary to console
# ==============================================================================

import pandas as pd
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_PATH = 'data/bug_dataset_50k.csv'

def load_dataset():
    print("=" * 60)
    print("  TASK 1 & 2: Data Collection & Dataset Connection")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"  [ERROR] Dataset not found at '{DATASET_PATH}'")
        return None

    df = pd.read_csv(DATASET_PATH)

    print(f"\n  Source       : Kaggle — Real Bug Report Dataset (50k)")
    print(f"  File         : {DATASET_PATH}")
    print(f"  Total Records: {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\n  Sample (first 5 rows):")
    print(df.head(5).to_string(index=False))
    print("\n" + "=" * 60)

    return df

if __name__ == "__main__":
    load_dataset()
