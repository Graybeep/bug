# ==============================================================================
# 02_preprocessing.py
# Task 3: Data Preprocessing
# Cleans the dataset — reports nulls, duplicates, and anomalies before fixing them.
# Encodes categorical columns for ML.
# Input:  data/bug_reports.csv
# Output: data/bug_reports_processed.csv  |  models/label_encoders.pkl
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(file_path='data/bug_reports.csv'):
    print("=" * 60)
    print("  TASK 3: Data Preprocessing")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Dataset
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        return

    total_records = len(df)
    print(f"\n  Loaded '{file_path}'  |  Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. NULL VALUE REPORT
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [A] NULL VALUE ANALYSIS")
    print("-" * 60)
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / total_records * 100).round(2)
    null_report = pd.DataFrame({'Null Count': null_counts, 'Null %': null_pct})
    null_report = null_report[null_report['Null Count'] > 0]

    if null_report.empty:
        print("  No null values found in any column.")
    else:
        print(null_report.to_string())

    # Fix nulls
    if 'Description' in df.columns:
        df.fillna({'Description': ''}, inplace=True)
    critical_cols = [c for c in ['Summary', 'Status', 'Severity', 'Priority'] if c in df.columns]
    rows_before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    rows_dropped_null = rows_before - len(df)
    print(f"\n  Action: Filled null 'Description' with empty string.")
    print(f"  Action: Dropped {rows_dropped_null} row(s) with nulls in critical columns.")

    # ------------------------------------------------------------------
    # 3. DUPLICATE REPORT
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [B] DUPLICATE ANALYSIS")
    print("-" * 60)
    dup_full = df.duplicated().sum()
    dup_bugid = df.duplicated(subset=['Bug ID']).sum() if 'Bug ID' in df.columns else 0

    print(f"  Fully duplicate rows         : {dup_full}")
    print(f"  Duplicate Bug IDs            : {dup_bugid}")

    if 'Bug ID' in df.columns:
        df.drop_duplicates(subset=['Bug ID'], inplace=True)
    print(f"  Action: Removed duplicate Bug IDs. Remaining rows: {len(df)}")

    # ------------------------------------------------------------------
    # 4. ANOMALY / OUTLIER REPORT
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [C] ANOMALY & OUTLIER ANALYSIS")
    print("-" * 60)

    anomalies_found = False

    # Empty summaries
    empty_summaries = (df['Summary'].str.strip() == '').sum() if 'Summary' in df.columns else 0
    print(f"  Empty Summary values         : {empty_summaries}")
    if empty_summaries > 0:
        anomalies_found = True

    # Empty descriptions
    empty_desc = (df['Description'].str.strip() == '').sum() if 'Description' in df.columns else 0
    print(f"  Empty Description values     : {empty_desc}")
    if empty_desc > 0:
        anomalies_found = True

    # Unexpected categorical values
    expected = {
        'Status':   {'New', 'Assigned', 'In Progress', 'Resolved', 'Closed'},
        'Severity': {'Trivial', 'Minor', 'Major', 'Critical'},
        'Priority': {'P1', 'P2', 'P3', 'P4', 'P5'},
    }
    for col, valid_set in expected.items():
        if col in df.columns:
            unexpected = df[~df[col].isin(valid_set)][col].unique()
            unexpected = [v for v in unexpected if pd.notna(v)]
            if unexpected:
                print(f"  Unexpected '{col}' values    : {unexpected}")
                anomalies_found = True
            else:
                print(f"  '{col}' values               : All valid ({sorted(valid_set)})")

    # Status/Resolution consistency check
    if 'Status' in df.columns and 'Resolution' in df.columns:
        open_but_resolved = df[
            (df['Status'].isin(['New', 'Assigned', 'In Progress'])) &
            (df['Resolution'].isin(['Fixed', 'Duplicate', "Won't Fix", 'Cannot Reproduce', 'Not a Bug']))
        ]
        print(f"  Status/Resolution mismatch   : {len(open_but_resolved)} row(s) (open status with a resolution)")
        if len(open_but_resolved) > 0:
            anomalies_found = True

    if not anomalies_found:
        print("\n  No significant anomalies detected.")

    # ------------------------------------------------------------------
    # 5. ENCODING
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [D] LABEL ENCODING")
    print("-" * 60)
    encoders = {}
    categorical_cols = ['Status', 'Severity', 'Priority', 'Resolution']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}' -> classes: {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 6. SAVE
    # ------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/label_encoders.pkl')

    output_path = 'data/bug_reports_processed.csv'
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Original records : {total_records}")
    print(f"  Cleaned records  : {len(df)}")
    print(f"  Columns in output: {list(df.columns)}")
    print(f"  Output CSV       : {output_path}")
    print(f"  Encoders saved   : models/label_encoders.pkl")
    print("=" * 60)

    return df

if __name__ == "__main__":
    preprocess_data()
