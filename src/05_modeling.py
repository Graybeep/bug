# ==============================================================================
# 05_modeling.py
# Task 6: Training and Testing ML Models
# Trains 5 models to predict Severity from bug descriptions.
# Input:  data/bug_reports_processed.csv
# Output: models/*.pkl  |  data/model_evaluation_results.json
# ==============================================================================

import pandas as pd
import json, os, sys, joblib

import warnings
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    print("=" * 60)
    print("  TASK 6: Training and Testing ML Models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv('data/bug_reports_processed.csv')
    except FileNotFoundError:
        print("  [ERROR] Run 02_preprocessing.py first.")
        return

    print(f"\n  Loaded processed dataset  |  {len(df):,} records")

    # ------------------------------------------------------------------
    # 2. TF-IDF on description
    # ------------------------------------------------------------------
    df['description'] = df['description'].fillna('')

    # Sample for speed if very large
    SAMPLE_SIZE = 20000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"  [INFO] Sampled {SAMPLE_SIZE:,} records for training.")

    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['description']).toarray()

    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print(f"  TF-IDF vectorizer fitted  |  max_features=2000, ngram=(1,2)")

    # ------------------------------------------------------------------
    # 3. Models — using LinearSVC (faster than kernel SVC on large data)
    # ------------------------------------------------------------------
    models = {
        'Naive Bayes':         MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM (Linear)':        CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000)),
    }

    # Predict severity only (the 50k dataset has no Priority column)
    targets = {
        'severity_encoded': 'Severity',
    }
    # Add bug_category if encoded
    if 'bug_category_encoded' in df.columns:
        targets['bug_category_encoded'] = 'Bug Category'

    results = {}

    for encoded_col, label in targets.items():
        if encoded_col not in df.columns:
            print(f"\n  [SKIP] '{encoded_col}' column not found.")
            continue

        print(f"\n{'-'*60}")
        print(f"  TARGET: {label.upper()} Prediction")
        print(f"{'-'*60}")
        print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>9}")
        print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*8} {'-'*9}")

        y = df[encoded_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        target_results = {}
        best_f1, best_model, best_name = -1, None, None

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

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

            marker = '  << BEST' if f1 > best_f1 else ''
            print(f"  {name:<22} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>9.4f}{marker}")

            if f1 > best_f1:
                best_f1, best_model, best_name = f1, model, name

        results[label] = target_results
        model_key = 'severity' if 'severity' in encoded_col else encoded_col.replace('_encoded','')
        model_path = f"models/best_{model_key}_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\n  Best model for {label}: {best_name}  (F1={best_f1:.4f})")
        print(f"  Saved to: {model_path}")

        if encoded_col == 'severity_encoded':
            print(f"\n  [NOTE] Severity accuracy above is expected to sit near chance level")
            print(f"         (~25% for 4 classes). Severity in this dataset is statistically")
            print(f"         independent of bug_category/environment/error_code/developer_role")
            print(f"         and of the description text, so no model can learn real signal for it.")
        if encoded_col == 'bug_category_encoded':
            print(f"\n  [NOTE] Bug Category accuracy above is expected to be ~100%. This is")
            print(f"         leakage, not generalization: 'description' is a fixed boilerplate")
            print(f"         template unique per category (16 templates for 50k rows), so the")
            print(f"         model is matching template text back to its own label.")

    json_path = 'data/model_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print("  MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Evaluation results saved : {json_path}")
    print(f"  Models saved in          : models/")
    print("=" * 60)

if __name__ == "__main__":
    train_and_evaluate()
