# ==============================================================================
# 06_predict.py
# Task 7: Bug Severity Prediction
# Uses the best trained model to predict Severity for a new bug description.
# Input:  --desc "Your bug description here"
# Output: Predicted Severity (console)
# ==============================================================================

import joblib
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def predict_bug(description: str):
    print("=" * 60)
    print("  TASK 7: Bug Severity Prediction")
    print("=" * 60)

    # Load models and encoders
    try:
        vectorizer     = joblib.load('models/tfidf_vectorizer.pkl')
        severity_model = joblib.load('models/best_severity_model.pkl')
        encoders       = joblib.load('models/label_encoders.pkl')
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Please run 05_modeling.py first to train the models.")
        return None

    # Vectorize and predict
    X_input  = vectorizer.transform([description]).toarray()
    sev_enc  = severity_model.predict(X_input)[0]
    severity = encoders['severity'].inverse_transform([sev_enc])[0]

    sev_note = {
        'Critical': 'Immediate action required -- system is unusable.',
        'High':     'Significant impact -- needs urgent resolution.',
        'Medium':   'Moderate impact -- schedule for next sprint.',
        'Low':      'Minor impact -- address when possible.',
    }.get(severity, '')

    print(f"\n  Input Description:")
    print(f"  {description}\n")
    print(f"  {'-'*56}")
    print(f"  Predicted Severity : {severity}")
    print(f"  Note               : {sev_note}")
    print(f"  {'-'*56}")
    print("=" * 60)

    return {'Severity': severity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bug Severity")
    parser.add_argument(
        '--desc',
        type=str,
        default="The application throws a NullPointerException during checkout, causing a total failure of the payment process.",
        help="Bug description text"
    )
    args = parser.parse_args()
    predict_bug(args.desc)
