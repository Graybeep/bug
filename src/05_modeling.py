# ==============================================================================
# 05_modeling.py
# Task 6: Training and Testing ML Models
# Trains 5 models per target to predict Severity, Priority and Bug Category.
#   - Severity / Bug Category : TF-IDF features from the bug description
#   - Priority                : TF-IDF + encoded operational features
# Input:  data/bug_reports_processed.csv
# Output: models/*.pkl  |  data/model_evaluation_results.json
# ==============================================================================

import json, os, sys

import _deps
_deps.check('pandas', 'numpy', 'sklearn', 'joblib')

import pandas as pd
import numpy as np
import joblib

import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

# Operational features used for Priority prediction. Priority is a triage
# decision driven by impact, not by the (boilerplate) description text — so it
# gets the structured columns as well as the text.
PRIORITY_FEATURES = ['severity_encoded', 'environment_encoded', 'error_code',
                     'bug_domain_encoded', 'tech_stack_encoded',
                     'developer_role_encoded']


RULE = "-" * 62


def build_models():
    return {
        'Naive Bayes':         MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM (Linear)':        CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000)),
    }


def print_per_class(y_test, y_pred, class_names, model_name):
    """Compact one-line-per-class breakdown, collapsed when every class is identical."""
    rep = classification_report(
        y_test, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0
    )
    rows = [(c, rep[c]['precision'], rep[c]['recall'],
             rep[c]['f1-score'], int(rep[c]['support'])) for c in class_names]

    # If every class scores the same, one line says it all
    distinct = {(round(p, 2), round(r, 2), round(f, 2)) for _, p, r, f, _ in rows}
    if len(distinct) == 1:
        p, r, f = distinct.pop()
        print(f"  Per-class ({model_name}): all {len(rows)} classes at "
              f"P={p:.2f} R={r:.2f} F1={f:.2f}")
        return

    width = min(max(len(c) for c in class_names), 24)
    print(f"  Per-class ({model_name}):")
    shown = rows if len(rows) <= 10 else rows[:8]
    for name, p, r, f, n in shown:
        print(f"    {name[:width]:<{width}}  P {p:.2f}   R {r:.2f}   F1 {f:.2f}   n={n:,}")
    if len(rows) > len(shown):
        print(f"    ... and {len(rows)-len(shown)} more class(es)")


def evaluate_target(X, y, label, model_key, index, total, class_names=None, note=None):
    """Train all 5 models on X -> y, print a compact report, save the best."""
    n_classes = len(set(y))
    print(f"\n{RULE}")
    print(f"  [{index}/{total}] {label.upper()}   "
          f"{n_classes} classes · {X.shape[1]} features")
    print(RULE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    target_results = {}
    best_f1, best_model, best_name, best_pred = -1, None, None, None
    rf_model = None

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if name == 'Random Forest':
            rf_model = model

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        target_results[name] = {
            'Accuracy':  round(acc,  4),
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1-Score':  round(f1,   4),
        }

        if f1 > best_f1:
            best_f1, best_model, best_name, best_pred = f1, model, name, y_pred

    # Print the table only once the winner is known, so the marker is accurate
    print(f"  {'Model':<21}{'Accuracy':>9}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for name, m in target_results.items():
        star = ' *' if name == best_name else ''
        print(f"  {name:<21}{m['Accuracy']:>9.4f}{m['Precision']:>11.4f}"
              f"{m['Recall']:>9.4f}{m['F1-Score']:>9.4f}{star}")

    model_path = f"models/best_{model_key}_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"  * best by F1 -> {model_path}")

    # Always persist Random Forest too — 07_bug_triage.py uses it by name.
    if rf_model is not None:
        rf_path = f"models/rf_{model_key}_model.pkl"
        joblib.dump(rf_model, rf_path)
        if best_name != 'Random Forest':
            print(f"    Random Forest also saved -> {rf_path}")

    if class_names is not None:
        print()
        print_per_class(y_test, best_pred, class_names, best_name)

    if note:
        print()
        for line in note:
            print(f"  ! {line}")

    return target_results, {'model': best_name, 'accuracy': target_results[best_name]['Accuracy'],
                            'f1': best_f1, 'path': model_path}


def train_and_evaluate():
    print("=" * 62)
    print("  TASK 6: Training and Testing ML Models")
    print("=" * 62)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv('data/bug_reports_processed.csv')
    except FileNotFoundError:
        print("  [ERROR] 'data/bug_reports_processed.csv' not found.")
        print("          Run 02_preprocessing.py first.")
        return

    try:
        encoders = joblib.load('models/label_encoders.pkl')
    except FileNotFoundError:
        encoders = {}

    # ------------------------------------------------------------------
    # 2. Feature extraction
    # ------------------------------------------------------------------
    df['description'] = df['description'].fillna('')

    # Sample for speed if very large
    SAMPLE_SIZE = 20000
    sampled = len(df) > SAMPLE_SIZE
    n_loaded = len(df)
    if sampled:
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    os.makedirs('models', exist_ok=True)

    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X_text = vectorizer.fit_transform(df['description']).toarray()
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

    # Structured features, min-max scaled to [0,1] so they sit on the same scale
    # as TF-IDF weights and stay non-negative for MultinomialNB.
    feature_cols = [c for c in PRIORITY_FEATURES if c in df.columns]
    X_full, scaler = None, None
    if feature_cols:
        scaler = MinMaxScaler()
        X_struct = scaler.fit_transform(df[feature_cols].astype(float))
        X_full = np.hstack([X_text, X_struct])
        joblib.dump({'feature_cols': feature_cols, 'scaler': scaler},
                    'models/priority_features.pkl')

    train_note = f"{len(df):,} sampled from {n_loaded:,}" if sampled else f"{n_loaded:,}"
    print(f"\n  Records   : {train_note}")
    print(f"  Text      : {X_text.shape[1]} TF-IDF terms (max_features=2000, 1-2 grams)")
    print(f"  Structured: {len(feature_cols)} encoded columns")
    print(f"  Split     : 80/20 stratified · 5 models per target")

    # ------------------------------------------------------------------
    # 3. Train per target
    # ------------------------------------------------------------------
    results, best = {}, {}

    targets = []
    if 'severity_encoded' in df.columns:
        targets.append(('Severity', 'severity', X_text, [
            "~25% is chance level for 4 balanced classes. Severity is statistically",
            "independent of every other field here — see README limitation #3.",
        ]))
    if 'priority_encoded' in df.columns and X_full is not None:
        targets.append(('Priority', 'priority', X_full, [
            "Priority is a derived field: models recover the documented scoring rule",
            "minus its ~8% jitter — see README 'Derived Fields'.",
        ]))
    if 'bug_category_encoded' in df.columns:
        targets.append(('Bug Category', 'bug_category', X_text, [
            "100% is leakage, not skill: descriptions are per-category boilerplate",
            "(16 templates for 50k rows) — see README limitation #2.",
        ]))

    for i, (label, key, X, note) in enumerate(targets, start=1):
        names = list(encoders[key].classes_) if key in encoders else None
        results[label], best[label] = evaluate_target(
            X, df[f'{key}_encoded'], label, key, i, len(targets), names, note
        )

    if 'priority_encoded' not in df.columns:
        print(f"\n  [SKIP] 'priority' not found — run 01_data_collection.py, then")
        print(f"         02_preprocessing.py, so the priority field exists.")

    json_path = 'data/model_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  SUMMARY — best model per target")
    print("=" * 62)
    print(f"  {'Target':<14}{'Best model':<21}{'Accuracy':>9}{'F1':>9}")
    for label, b in best.items():
        print(f"  {label:<14}{b['model']:<21}{b['accuracy']:>9.4f}{b['f1']:>9.4f}")
    print(f"\n  Models saved  : models/best_*_model.pkl")
    print(f"  Metrics saved : {json_path}")
    print("=" * 62)

if __name__ == "__main__":
    train_and_evaluate()
