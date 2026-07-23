# ==============================================================================
# 02_preprocessing.py
# Task 3: Data Preprocessing
# Cleans the 50k bug dataset — reports nulls, duplicates, and anomalies.
# Encodes categorical columns for ML.
# Input:  data/bug_reports_enriched.csv (falls back to data/bug_dataset_50k.csv)
# Output: data/bug_reports_processed.csv  |  models/label_encoders.pkl
# ==============================================================================

import os
import sys

import _deps
_deps.check('pandas', 'numpy', 'sklearn', 'joblib')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENRICHED_PATH = 'data/bug_reports_enriched.csv'
RAW_PATH      = 'data/bug_dataset_50k.csv'

def preprocess_data(file_path=None):
    print("=" * 60)
    print("  TASK 3: Data Preprocessing")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Dataset — prefer the enriched output of 01_data_collection.py
    # ------------------------------------------------------------------
    if file_path is None:
        file_path = ENRICHED_PATH if os.path.exists(ENRICHED_PATH) else RAW_PATH
        if file_path == RAW_PATH:
            print(f"\n  [WARN] '{ENRICHED_PATH}' not found — falling back to the raw")
            print(f"         dataset. Run 01_data_collection.py first so status,")
            print(f"         priority, lifecycle_stage and resolution are available.")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  [ERROR] Dataset not found: {os.path.abspath(file_path)}")
        print(f"\n  Fix one of these:")
        print(f"    1. Run 'python src/01_data_collection.py' first — it produces")
        print(f"       {ENRICHED_PATH} from the raw dataset.")
        print(f"    2. Make sure '{RAW_PATH}' exists. Download it from Kaggle")
        print(f"       and place it in the data/ folder — see README.")
        return
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

    # error_code is numeric with ~12% missing — impute with the mode so it can be
    # used as a model feature without dropping an eighth of the dataset.
    ec_filled = 0
    if 'error_code' in df.columns:
        ec_filled = int(df['error_code'].isnull().sum())
        if ec_filled:
            df.fillna({'error_code': df['error_code'].mode()[0]}, inplace=True)

    rows_before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    rows_dropped = rows_before - len(df)

    print(f"\n  Action: Filled null text columns with empty string.")
    if 'error_code' in df.columns:
        print(f"  Action: Imputed {ec_filled:,} null 'error_code' value(s) with the mode.")
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

    # Unexpected priority values
    if 'priority' in df.columns:
        valid_pri  = {'P1', 'P2', 'P3', 'P4', 'P5'}
        actual_pri = set(df['priority'].dropna().unique())
        unexpected = actual_pri - valid_pri
        if unexpected:
            print(f"  Unexpected 'priority' values : {unexpected}")
            anomalies_found = True
        else:
            print(f"  'priority' values            : All valid {sorted(actual_pri)}")

    # Status must map to a known life cycle stage
    if 'status' in df.columns and 'lifecycle_stage' in df.columns:
        unmapped = int(df['lifecycle_stage'].isnull().sum())
        print(f"  Statuses without a stage     : {unmapped}")
        if unmapped > 0:
            anomalies_found = True

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
    categorical_cols = ['severity', 'priority', 'status', 'lifecycle_stage',
                        'resolution', 'bug_category', 'bug_domain', 'tech_stack',
                        'environment', 'developer_role']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            mapping = ', '.join(f"{c}={i}" for i, c in enumerate(le.classes_))
            if len(mapping) > 68:
                mapping = mapping[:65] + '...'
            print(f"  '{col}' -> {len(le.classes_)} classes  [{mapping}]")

    # ------------------------------------------------------------------
    # 6. CLEAN DATA PREVIEW
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  [E] CLEANED DATA PREVIEW")
    print("-" * 60)

    # Verify the cleaning actually worked
    print(f"\n  Post-clean verification:")
    print(f"    Remaining null values     : {int(df.isnull().sum().sum())}")
    print(f"    Remaining duplicate IDs   : {int(df.duplicated(subset=[id_col]).sum())}")
    print(f"    Rows x Columns            : {df.shape[0]:,} x {df.shape[1]}")

    # Sample of the cleaned, human-readable columns
    preview_cols = [c for c in ['bug_id', 'severity', 'priority', 'status',
                                'lifecycle_stage', 'resolution', 'bug_category',
                                'environment'] if c in df.columns]
    print(f"\n  Cleaned data — first 10 rows (readable columns):")
    print('\n'.join('    ' + line for line in
                    df[preview_cols].head(10).to_string(index=False).split('\n')))

    # Same rows, encoded — shows the categorical -> numerical conversion
    enc_cols = [c for c in ['severity_encoded', 'priority_encoded', 'status_encoded',
                            'lifecycle_stage_encoded', 'resolution_encoded',
                            'bug_category_encoded', 'environment_encoded']
                if c in df.columns]
    if enc_cols:
        short = {c: c.replace('_encoded', '')[:9] for c in enc_cols}
        print(f"\n  Same rows after encoding (ML-ready numerical form):")
        print('\n'.join('    ' + line for line in
                        df[enc_cols].head(10).rename(columns=short)
                          .to_string(index=False).split('\n')))

    # ------------------------------------------------------------------
    # 7. SAVE
    # ------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/label_encoders.pkl')

    output_path = 'data/bug_reports_processed.csv'
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Original records : {total_records:,}")
    print(f"  Cleaned records  : {len(df):,}  ({len(df.columns)} columns)")
    print(f"  Nulls remaining  : {int(df.isnull().sum().sum())}")
    print(f"  Output CSV       : {output_path}")
    print(f"  Encoders saved   : models/label_encoders.pkl")
    print("=" * 60)

    return df

if __name__ == "__main__":
    preprocess_data()
