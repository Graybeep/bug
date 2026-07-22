# ==============================================================================
# 02_preprocessing.py
# Task 3: Data Preprocessing
# Cleans the 50k bug dataset — reports nulls, duplicates, and anomalies.
# Encodes categorical columns for ML.
# Input:  data/bug_dataset_50k.csv
# Output: data/bug_reports_processed.csv  |  models/label_encoders.pkl
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_PATH = 'data/bug_dataset_50k.csv'

def preprocess_data(file_path=DATASET_PATH):
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

    # Fix nulls — fill text columns with empty string, drop rows missing key fields
    text_cols    = [c for c in ['title', 'description', 'root_cause', 'suggested_fix', 'explanation'] if c in df.columns]
    critical_cols = [c for c in ['severity', 'bug_category'] if c in df.columns]

    for col in text_cols:
        df.fillna({col: ''}, inplace=True)

    rows_before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    rows_dropped = rows_before - len(df)

    print(f"\n  Action: Filled null text columns with empty string.")
    print(f"  Action: Dropped {rows_dropped} row(s) with nulls in critical columns.")

    # ------------------------------------------------------------------
    # 3. DUPLICATE REPORT
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [B] DUPLICATE ANALYSIS")
    print("-" * 60)
    dup_full  = df.duplicated().sum()
    id_col    = 'bug_id' if 'bug_id' in df.columns else df.columns[0]
    dup_bugid = df.duplicated(subset=[id_col]).sum()

    print(f"  Fully duplicate rows         : {dup_full}")
    print(f"  Duplicate '{id_col}' values  : {dup_bugid}")

    df.drop_duplicates(subset=[id_col], inplace=True)
    print(f"  Action: Removed duplicate IDs. Remaining rows: {len(df):,}")

    # ------------------------------------------------------------------
    # 4. ANOMALY / OUTLIER REPORT
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [C] ANOMALY & OUTLIER ANALYSIS")
    print("-" * 60)
    anomalies_found = False

    # Empty descriptions
    if 'description' in df.columns:
        empty_desc = (df['description'].str.strip() == '').sum()
        print(f"  Empty Description values     : {empty_desc}")
        if empty_desc > 0:
            anomalies_found = True

    # Empty titles
    if 'title' in df.columns:
        empty_title = (df['title'].str.strip() == '').sum()
        print(f"  Empty Title values           : {empty_title}")
        if empty_title > 0:
            anomalies_found = True

    # Unexpected severity values
    if 'severity' in df.columns:
        valid_sev = {'Low', 'Medium', 'High', 'Critical'}
        actual    = set(df['severity'].dropna().unique())
        unexpected = actual - valid_sev
        if unexpected:
            print(f"  Unexpected 'severity' values : {unexpected}")
            anomalies_found = True
        else:
            print(f"  'severity' values            : All valid {sorted(actual)}")

    # error_code outliers (numerical)
    if 'error_code' in df.columns:
        ec = df['error_code'].dropna()
        if len(ec) > 0:
            q1, q3 = ec.quantile(0.25), ec.quantile(0.75)
            iqr     = q3 - q1
            outliers = ((ec < q1 - 1.5 * iqr) | (ec > q3 + 1.5 * iqr)).sum()
            print(f"  'error_code' outliers (IQR)  : {outliers} value(s)")
            print(f"  'error_code' range           : {ec.min():.0f} - {ec.max():.0f}")
            if outliers > 0:
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
    categorical_cols = ['severity', 'bug_category', 'bug_domain', 'tech_stack',
                        'environment', 'developer_role']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}' -> {len(le.classes_)} classes")

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
    print(f"  Original records : {total_records:,}")
    print(f"  Cleaned records  : {len(df):,}")
    print(f"  Columns in output: {list(df.columns)}")
    print(f"  Output CSV       : {output_path}")
    print(f"  Encoders saved   : models/label_encoders.pkl")
    print("=" * 60)

    return df

if __name__ == "__main__":
    preprocess_data()
