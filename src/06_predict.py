# ==============================================================================
# 06_predict.py
# Task 7: Bug Severity & Priority Prediction
# Uses the best trained models to predict Severity and Priority (P1-P5) for a
# newly reported bug, so developers can triage it.
# Input:  --desc "Your bug description here"  (+ optional context flags)
# Output: Predicted Severity and Priority (console)
# ==============================================================================

import argparse
import os
import sys

import _deps
_deps.check('pandas', 'numpy', 'sklearn', 'joblib')

import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEVERITY_NOTE = {
    'Critical': 'Immediate action required -- system is unusable.',
    'High':     'Significant impact -- needs urgent resolution.',
    'Medium':   'Moderate impact -- schedule for next sprint.',
    'Low':      'Minor impact -- address when possible.',
}

PRIORITY_NOTE = {
    'P1': 'Highest priority -- fix now, blocks release.',
    'P2': 'High priority -- fix in the current sprint.',
    'P3': 'Medium priority -- schedule in an upcoming sprint.',
    'P4': 'Low priority -- backlog, fix when capacity allows.',
    'P5': 'Lowest priority -- cosmetic or deferred.',
}


def encode_value(encoders, col, value, fallback_index=0):
    """Map a raw category label to its encoded integer, tolerating unseen values."""
    le = encoders.get(col)
    if le is None:
        return fallback_index
    if value is not None and value in list(le.classes_):
        return int(le.transform([value])[0])
    return fallback_index


def predict_bug(description: str, severity=None, environment='Production',
                error_code=500.0, bug_domain=None, tech_stack=None,
                developer_role=None):
    print("=" * 60)
    print("  TASK 7: Bug Severity & Priority Prediction")
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

    # ------------------------------------------------------------------
    # Severity — predicted from the description text
    # ------------------------------------------------------------------
    X_text = vectorizer.transform([description]).toarray()

    if severity is not None and severity in list(encoders['severity'].classes_):
        predicted_severity = severity
        severity_source = 'provided by reporter'
    else:
        sev_enc = severity_model.predict(X_text)[0]
        predicted_severity = encoders['severity'].inverse_transform([sev_enc])[0]
        severity_source = 'predicted from description'

    # ------------------------------------------------------------------
    # Priority — predicted from the description + operational context
    # ------------------------------------------------------------------
    priority = None
    try:
        priority_model = joblib.load('models/best_priority_model.pkl')
        meta           = joblib.load('models/priority_features.pkl')
    except FileNotFoundError:
        priority_model, meta = None, None

    if priority_model is not None:
        raw = {
            'severity_encoded':       encode_value(encoders, 'severity', predicted_severity),
            'environment_encoded':    encode_value(encoders, 'environment', environment),
            'error_code':             float(error_code),
            'bug_domain_encoded':     encode_value(encoders, 'bug_domain', bug_domain),
            'tech_stack_encoded':     encode_value(encoders, 'tech_stack', tech_stack),
            'developer_role_encoded': encode_value(encoders, 'developer_role', developer_role),
        }
        cols   = meta['feature_cols']
        struct = pd.DataFrame([[raw[c] for c in cols]], columns=cols, dtype=float)
        struct = meta['scaler'].transform(struct)

        X_input  = np.hstack([X_text, struct])
        pri_enc  = priority_model.predict(X_input)[0]
        priority = encoders['priority'].inverse_transform([pri_enc])[0]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n  Input Description:")
    print(f"  {description}\n")
    print(f"  Context: environment={environment}, error_code={error_code:.0f}"
          f"{', domain=' + bug_domain if bug_domain else ''}"
          f"{', tech=' + tech_stack if tech_stack else ''}")
    print(f"\n  {'-'*56}")
    print(f"  Predicted Severity : {predicted_severity}  ({severity_source})")
    print(f"  Note               : {SEVERITY_NOTE.get(predicted_severity, '')}")
    if priority is not None:
        print(f"  {'-'*56}")
        print(f"  Predicted Priority : {priority}")
        print(f"  Note               : {PRIORITY_NOTE.get(priority, '')}")
    else:
        print(f"  {'-'*56}")
        print(f"  Predicted Priority : [unavailable -- no priority model trained]")
        print(f"  Run 01 -> 02 -> 05 so the derived priority field exists.")
    print(f"  {'-'*56}")
    print("=" * 60)

    return {'Severity': predicted_severity, 'Priority': priority}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Bug Severity and Priority")
    parser.add_argument('--desc', type=str,
        default="The application throws a NullPointerException during checkout, causing a total failure of the payment process.",
        help="Bug description text")
    parser.add_argument('--severity', type=str, default=None,
        choices=['Low', 'Medium', 'High', 'Critical'],
        help="Known severity, if the reporter already triaged it (otherwise predicted)")
    parser.add_argument('--environment', type=str, default='Production',
        choices=['Development', 'Staging', 'Production'],
        help="Environment where the bug was observed")
    parser.add_argument('--error-code', dest='error_code', type=float, default=500.0,
        help="HTTP-style error code observed (400/401/403/404/500/502/503)")
    parser.add_argument('--domain', dest='bug_domain', type=str, default=None,
        help="Bug domain, e.g. 'Backend Systems'")
    parser.add_argument('--tech-stack', dest='tech_stack', type=str, default=None,
        help="Technology involved, e.g. 'Django'")
    parser.add_argument('--role', dest='developer_role', type=str, default=None,
        help="Developer role responsible, e.g. 'Backend Developer'")
    args = parser.parse_args()

    predict_bug(args.desc, severity=args.severity, environment=args.environment,
                error_code=args.error_code, bug_domain=args.bug_domain,
                tech_stack=args.tech_stack, developer_role=args.developer_role)
