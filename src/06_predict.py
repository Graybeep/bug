# ==============================================================================
# 06_predict.py
# Task 7: Bug Severity and Priority Prediction
# Uses the best trained models to predict Severity and Priority for a new bug.
# Input:  --desc "Your bug description here"
# Output: Predicted Severity and Priority (console)
# ==============================================================================

import joblib
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def predict_bug(description: str):
    print("=" * 60)
    print("  TASK 7: Bug Severity & Priority Prediction")
    print("=" * 60)

    # Load models and encoders
    try:
        vectorizer     = joblib.load('models/tfidf_vectorizer.pkl')
        severity_model = joblib.load('models/best_severity_model.pkl')
        priority_model = joblib.load('models/best_priority_model.pkl')
        encoders       = joblib.load('models/label_encoders.pkl')
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Please run 05_modeling.py first to train the models.")
        return None

    # Vectorize input
    X_input = vectorizer.transform([description]).toarray()

    # Predict encoded labels
    sev_enc = severity_model.predict(X_input)[0]
    pri_enc = priority_model.predict(X_input)[0]

    # Decode back to original labels
    severity = encoders['Severity'].inverse_transform([sev_enc])[0]
    priority = encoders['Priority'].inverse_transform([pri_enc])[0]

    # Severity colour hints for interpretation
    sev_note = {
        'Critical': 'Immediate action required — system is unusable.',
        'Major':    'Significant impact — needs urgent resolution.',
        'Minor':    'Limited impact — schedule for next sprint.',
        'Trivial':  'Cosmetic/low impact — address when possible.',
    }.get(severity, '')

    pri_note = {
        'P1': 'Highest priority — fix immediately.',
        'P2': 'High priority — fix in current release.',
        'P3': 'Medium priority — fix in next release.',
        'P4': 'Low priority — fix when convenient.',
        'P5': 'Lowest priority — optional fix.',
    }.get(priority, '')

    print(f"\n  Input Description:")
    print(f"  {description}\n")
    print(f"  {'─'*56}")
    print(f"  Predicted Severity : {severity}")
    print(f"  Note               : {sev_note}")
    print(f"  {'─'*56}")
    print(f"  Predicted Priority : {priority}")
    print(f"  Note               : {pri_note}")
    print(f"  {'─'*56}")
    print("=" * 60)

    return {'Severity': severity, 'Priority': priority}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bug Severity and Priority")
    parser.add_argument(
        '--desc',
        type=str,
        default="The application throws a NullPointerException during checkout, causing a total failure of the payment process.",
        help="Bug description text"
    )
    args = parser.parse_args()
    predict_bug(args.desc)
