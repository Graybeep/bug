# ==============================================================================
# model_bridge.py
# The one place the dashboard talks to the trained models.
#
# Everything that needs a prediction goes through here:
#   - 08_dashboard.py   scores the whole dataset for the console KPI report, so
#                       it can report predicted-vs-recorded priority per split
#   - streamlit_app.py  does the same thing interactively, and calls
#                       predict_one() for the live triage console
#
# Keeping both on one code path means the console and the charts can never
# disagree about how a feature vector is built.
#
# Nothing here trains anything — run src/05_modeling.py for that.
# ==============================================================================

import os

import numpy as np
import pandas as pd

MODEL_DIR = 'models'

# Mirrors PRIORITY_FEATURES in 05_modeling.py. The saved priority_features.pkl
# carries the authoritative column list; this is only the fallback ordering.
PRIORITY_FEATURES = ['severity_encoded', 'environment_encoded', 'error_code',
                     'bug_domain_encoded', 'tech_stack_encoded',
                     'developer_role_encoded']

SAMPLE_SIZE = 20000     # 05_modeling.py's sampling cap
SEED = 42

SEVERITY_NOTE = {
    'Critical': 'System unusable — immediate action required.',
    'High':     'Significant impact — needs urgent resolution.',
    'Medium':   'Moderate impact — schedule for the next sprint.',
    'Low':      'Minor impact — address when capacity allows.',
}
PRIORITY_NOTE = {
    'P1': 'Highest priority — fix now, blocks the release.',
    'P2': 'High priority — fix in the current sprint.',
    'P3': 'Medium priority — schedule in an upcoming sprint.',
    'P4': 'Low priority — backlog, fix when capacity allows.',
    'P5': 'Lowest priority — cosmetic or deferred.',
}


class Bundle:
    """Everything loaded off disk, plus what it can and cannot do."""

    def __init__(self):
        self.vectorizer = None
        self.severity_model = None
        self.priority_model = None
        self.priority_meta = None
        self.encoders = {}
        self.errors = []

    @property
    def ready(self):
        return self.vectorizer is not None and self.priority_model is not None

    @property
    def can_predict_severity(self):
        return self.vectorizer is not None and self.severity_model is not None

    def describe(self):
        name = type(self.priority_model).__name__ if self.priority_model is not None else None
        return {
            'ready': self.ready,
            'priority_model': name,
            'severity_model': type(self.severity_model).__name__ if self.severity_model is not None else None,
            'errors': self.errors,
        }


def load(model_dir=MODEL_DIR, quiet=False):
    """Load the saved models. Never raises — a partial bundle reports why."""
    import joblib

    b = Bundle()

    def grab(filename, label, required):
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            if required:
                b.errors.append(f"{label} not found at {path} — run 'python src/05_modeling.py'.")
            return None
        try:
            return joblib.load(path)
        except Exception as exc:                                  # noqa: BLE001
            b.errors.append(f"{label} could not be loaded ({exc.__class__.__name__}: {exc}).")
            return None

    b.vectorizer     = grab('tfidf_vectorizer.pkl',     'TF-IDF vectorizer',  True)
    b.priority_model = grab('best_priority_model.pkl',  'Priority model',     True)
    b.severity_model = grab('best_severity_model.pkl',  'Severity model',     False)
    b.priority_meta  = grab('priority_features.pkl',    'Priority feature metadata', True)
    b.encoders       = grab('label_encoders.pkl',       'Label encoders',     True) or {}

    if not quiet and b.errors:
        for err in b.errors:
            print(f"  [WARN] {err}")
    return b


# ──────────────────────────────────────────────────────────────────────────────
#  Feature construction
# ──────────────────────────────────────────────────────────────────────────────
def encode_value(encoders, column, value, fallback=0):
    """Raw label -> encoded integer, tolerating values the encoder never saw."""
    le = encoders.get(column)
    if le is None or value is None:
        return fallback
    classes = list(le.classes_)
    return int(classes.index(value)) if value in classes else fallback


def _feature_columns(bundle):
    meta = bundle.priority_meta or {}
    return meta.get('feature_cols') or PRIORITY_FEATURES


def _structured(bundle, frame):
    """Scale the structured block exactly as 05_modeling.py fitted it."""
    cols = _feature_columns(bundle)
    block = pd.DataFrame({c: frame[c].astype(float) for c in cols}, columns=cols)
    scaler = (bundle.priority_meta or {}).get('scaler')
    return scaler.transform(block) if scaler is not None else block.to_numpy()


# ──────────────────────────────────────────────────────────────────────────────
#  Single-bug prediction — what the triage console calls
# ──────────────────────────────────────────────────────────────────────────────
def predict_one(bundle, description, severity=None, environment='Production',
                error_code=500.0, bug_domain=None, tech_stack=None,
                developer_role=None):
    """Predict severity (when not supplied) and priority for one new bug.

    Returns the label, the note, and the full class-probability vector, so the
    console can show how confident the model is rather than just its argmax.
    """
    if not bundle.ready:
        raise RuntimeError('; '.join(bundle.errors) or 'models are not loaded')

    X_text = bundle.vectorizer.transform([description or '']).toarray()

    if severity and severity in list(bundle.encoders.get('severity').classes_ if bundle.encoders.get('severity') else []):
        predicted_severity, severity_source = severity, 'provided by reporter'
    elif bundle.can_predict_severity:
        enc = bundle.severity_model.predict(X_text)[0]
        predicted_severity = str(bundle.encoders['severity'].inverse_transform([enc])[0])
        severity_source = 'predicted from the description (chance-level on this dataset — see the caveat)'
    else:
        predicted_severity, severity_source = (severity or 'Medium'), 'assumed — no severity model available'

    raw = pd.DataFrame([{
        'severity_encoded':       encode_value(bundle.encoders, 'severity', predicted_severity),
        'environment_encoded':    encode_value(bundle.encoders, 'environment', environment),
        'error_code':             float(error_code or 0),
        'bug_domain_encoded':     encode_value(bundle.encoders, 'bug_domain', bug_domain),
        'tech_stack_encoded':     encode_value(bundle.encoders, 'tech_stack', tech_stack),
        'developer_role_encoded': encode_value(bundle.encoders, 'developer_role', developer_role),
    }])
    X = np.hstack([X_text, _structured(bundle, raw)])

    encoded = int(bundle.priority_model.predict(X)[0])
    priority = str(bundle.encoders['priority'].inverse_transform([encoded])[0])

    probs, classes = None, list(bundle.encoders['priority'].classes_)
    if hasattr(bundle.priority_model, 'predict_proba'):
        row = bundle.priority_model.predict_proba(X)[0]
        # predict_proba columns follow the model's own classes_, which need not
        # cover every label the encoder knows about.
        order = list(getattr(bundle.priority_model, 'classes_', range(len(row))))
        probs = [0.0] * len(classes)
        for col, code in enumerate(order):
            if 0 <= int(code) < len(classes):
                probs[int(code)] = float(row[col])

    return {
        'source':          'model',
        'model':           type(bundle.priority_model).__name__,
        'severity':        predicted_severity,
        'severity_source': severity_source,
        'severity_note':   SEVERITY_NOTE.get(predicted_severity, ''),
        'priority':        priority,
        'priority_note':   PRIORITY_NOTE.get(priority, ''),
        'priority_classes': classes,
        'priority_probs':  probs,
        'confidence':      max(probs) if probs else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Batch scoring — what the dashboard build calls
# ──────────────────────────────────────────────────────────────────────────────
def score_dataset(bundle, df, chunk=5000, progress=None):
    """Re-score every row in df with the saved priority model.

    Returns (labels, confidence) where labels are priority strings and
    confidence is the model's own probability for the class it chose (None when
    the model has no predict_proba).
    """
    if not bundle.ready:
        raise RuntimeError('; '.join(bundle.errors) or 'models are not loaded')

    cols = _feature_columns(bundle)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"processed data is missing {', '.join(missing)} — "
                           f"re-run 02_preprocessing.py")

    classes = list(bundle.encoders['priority'].classes_)
    text = df['description'].fillna('').to_numpy()
    n = len(df)
    labels = np.empty(n, dtype=object)
    conf = np.zeros(n, dtype=float)
    has_proba = hasattr(bundle.priority_model, 'predict_proba')

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        X_text = bundle.vectorizer.transform(text[start:stop]).toarray()
        X = np.hstack([X_text, _structured(bundle, df.iloc[start:stop])])

        if has_proba:
            proba = bundle.priority_model.predict_proba(X)
            best = proba.argmax(axis=1)
            order = np.asarray(getattr(bundle.priority_model, 'classes_', np.arange(proba.shape[1])), dtype=int)
            encoded = order[best]
            conf[start:stop] = proba[np.arange(stop - start), best]
        else:
            encoded = bundle.priority_model.predict(X).astype(int)
            conf[start:stop] = np.nan

        labels[start:stop] = [classes[e] if 0 <= e < len(classes) else classes[0] for e in encoded]
        if progress:
            progress(stop, n)

    return labels, conf


def training_split(df, target='priority_encoded', sample_size=SAMPLE_SIZE, seed=SEED):
    """Which rows 05_modeling.py trained on, tested on, and never sampled.

    05_modeling.py draws a `sample_size` sample with `random_state=seed` from the
    same CSV, then does a stratified 80/20 split with the same seed. Both are
    deterministic, so the exact partition is reproducible here — which is what
    lets the dashboard separate honest held-out agreement from memorised rows.

    Returns an int array over df's rows: 0 never sampled, 1 train, 2 held-out test.
    """
    out = np.zeros(len(df), dtype=np.uint8)
    if target not in df.columns:
        return out
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return out

    if len(df) > sample_size:
        sampled = df.sample(sample_size, random_state=seed).index
    else:
        sampled = df.index

    y = df.loc[sampled, target].reset_index(drop=True)
    positions = np.arange(len(sampled))
    try:
        train_pos, test_pos = train_test_split(
            positions, test_size=0.2, random_state=seed, stratify=y)
    except ValueError:
        # A class with a single member makes stratification impossible; fall
        # back to the unstratified split rather than reporting no split at all.
        train_pos, test_pos = train_test_split(positions, test_size=0.2, random_state=seed)

    labels = np.asarray(sampled)
    out[df.index.get_indexer(labels[train_pos])] = 1
    out[df.index.get_indexer(labels[test_pos])] = 2
    return out
