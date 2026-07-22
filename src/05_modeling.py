# ==============================================================================
# 05_modeling.py
# Task 6: Training and Testing ML Models
# Trains 5 models to predict Severity and Priority from bug descriptions.
# Input:  data/bug_reports_processed.csv
# Output: models/*.pkl  |  data/model_evaluation_results.json
# ==============================================================================

import pandas as pd
import json
import os
import sys
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    print("=" * 60)
    print("  TASK 6: Training and Testing ML Models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load preprocessed data
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv('data/bug_reports_processed.csv')
    except FileNotFoundError:
        print("  [ERROR] Run 02_preprocessing.py first.")
        return

    print(f"\n  Loaded processed dataset  |  {len(df)} records")

    # ------------------------------------------------------------------
    # 2. Feature extraction — TF-IDF on Description
    # ------------------------------------------------------------------
    df['Description'] = df['Description'].fillna('')
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['Description']).toarray()

    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("  TF-IDF vectorizer fitted  |  max_features=1000")

    # ------------------------------------------------------------------
    # 3. Models
    # ------------------------------------------------------------------
    models = {
        'Naive Bayes':        MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':      DecisionTreeClassifier(random_state=42),
        'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                SVC(probability=True, random_state=42),
    }

    targets = {
        'Severity_encoded': 'Severity',
        'Priority_encoded': 'Priority',
    }

    results = {}

    for encoded_col, label in targets.items():
        if encoded_col not in df.columns:
            print(f"  [SKIP] '{encoded_col}' column not found.")
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
        best_f1        = -1
        best_model     = None
        best_name      = None

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
                best_f1    = f1
                best_model = model
                best_name  = name

        results[label] = target_results

        # Save best model
        model_path = f"models/best_{label.lower()}_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\n  Best model for {label}: {best_name}  (F1={best_f1:.4f})")
        print(f"  Saved to: {model_path}")

    # ------------------------------------------------------------------
    # 4. Save full results JSON
    # ------------------------------------------------------------------
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
