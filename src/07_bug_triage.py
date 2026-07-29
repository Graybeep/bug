# ==============================================================================
# 07_bug_triage.py
# Bug triage: take real bug reports from the preprocessing output, predict
# severity/priority with Random Forest, assign the right developer, and chart
# where the bugs went.
#
#   Triage a sample from the preprocessed data (default):
#     python src/07_bug_triage.py            # 15 bugs
#     python src/07_bug_triage.py --n 30     # 30 bugs
#
#   Or report a single bug by hand:
#     python src/07_bug_triage.py --title "Checkout fails" \
#         --desc "Payment page crashes after submit" \
#         --error-code 500 --category "Backend Logic Bug" --environment Production
#
# Input:  data/bug_reports_processed.csv (from 02_preprocessing.py)
# Output: data/tracked_bugs.json  |  visualizations/bug_assignments.png
# ==============================================================================

import argparse
import json
import os
import re
import sys
from datetime import datetime

import _deps
_deps.check('pandas', 'numpy', 'sklearn', 'joblib', 'matplotlib', 'seaborn')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACKER_PATH   = 'data/tracked_bugs.json'
KB_PATH        = 'data/bug_knowledge_base.json'
PROCESSED_PATH = 'data/bug_reports_processed.csv'
CHART_PATH     = 'visualizations/bug_assignments.png'

DOMAIN_OVERRIDE = {'Mobile': 'Mobile Developer'}


def load_bugs_from_data(n, seed=42):
    """Draw a random sample of real bug reports from the preprocessing output."""
    if not os.path.exists(PROCESSED_PATH):
        print(f"  [ERROR] '{PROCESSED_PATH}' not found — run 02_preprocessing.py first.")
        return None

    df = pd.read_csv(PROCESSED_PATH)
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=seed)

    bugs = []
    for _, row in sample.iterrows():
        bugs.append({
            'title':       row.get('title'),
            'desc':        row.get('description'),
            'error_code':  row.get('error_code'),
            'category':    row.get('bug_category'),
            'environment': row.get('environment'),
            'domain':      row.get('bug_domain'),
            'tech_stack':  row.get('tech_stack'),
            'severity':    None,   # let the Random Forest predict it
        })
    return bugs


# ──────────────────────────────────────────────────────────────────────────────
#  Clean the reported bug information
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    if not text:
        return ''
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    return text


def clean_report(title, description, error_code, category, environment):
    title       = clean_text(title)
    description = clean_text(description) or title
    try:
        error_code = float(error_code)
    except (TypeError, ValueError):
        error_code = 500.0
    return {
        'title':       title,
        'description': description,
        'error_code':  error_code,
        'category':    category,
        'environment': environment,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Predict severity and priority with Random Forest
# ──────────────────────────────────────────────────────────────────────────────
def load_models():
    required = {
        'vectorizer': 'models/tfidf_vectorizer.pkl',
        'encoders':   'models/label_encoders.pkl',
        'severity':   'models/rf_severity_model.pkl',
        'priority':   'models/rf_priority_model.pkl',
        'features':   'models/priority_features.pkl',
    }
    missing = [p for p in required.values() if not os.path.exists(p)]
    if missing:
        print("  [ERROR] Missing trained model files. Run the pipeline first "
              "(01_data_collection.py, 02_preprocessing.py, 05_modeling.py).")
        return None
    return {k: joblib.load(p) for k, p in required.items()}


def encode_value(encoders, column, value):
    le = encoders.get(column)
    if le is None or value is None:
        return 0
    classes = list(le.classes_)
    return int(le.transform([value])[0]) if value in classes else 0


def predict_severity_priority(m, record, bug_domain=None, tech_stack=None,
                              developer_role=None, known_severity=None):
    X_text = m['vectorizer'].transform([record['description']]).toarray()

    if known_severity:
        severity = known_severity
    else:
        sev_enc  = m['severity'].predict(X_text)[0]
        severity = m['encoders']['severity'].inverse_transform([sev_enc])[0]

    raw = {
        'severity_encoded':       encode_value(m['encoders'], 'severity', severity),
        'environment_encoded':    encode_value(m['encoders'], 'environment', record['environment']),
        'error_code':             record['error_code'],
        'bug_domain_encoded':     encode_value(m['encoders'], 'bug_domain', bug_domain),
        'tech_stack_encoded':     encode_value(m['encoders'], 'tech_stack', tech_stack),
        'developer_role_encoded': encode_value(m['encoders'], 'developer_role', developer_role),
    }
    cols   = m['features']['feature_cols']
    struct = pd.DataFrame([[raw[c] for c in cols]], columns=cols, dtype=float)
    struct = m['features']['scaler'].transform(struct)

    pri_enc  = m['priority'].predict(np.hstack([X_text, struct]))[0]
    priority = m['encoders']['priority'].inverse_transform([pri_enc])[0]

    return severity, priority


# ──────────────────────────────────────────────────────────────────────────────
#  Assign the developer
# ──────────────────────────────────────────────────────────────────────────────
def load_kb():
    if not os.path.exists(KB_PATH):
        print(f"  [ERROR] '{KB_PATH}' not found — run 01_data_collection.py first.")
        return None
    with open(KB_PATH) as f:
        return json.load(f)


def assign_developer(kb, category, bug_domain):
    entry = kb.get(category)
    if entry is None:
        return 'Full-Stack Developer'
    return DOMAIN_OVERRIDE.get(bug_domain, entry['assigned_role'])


# ──────────────────────────────────────────────────────────────────────────────
#  Persist the tracked bugs
# ──────────────────────────────────────────────────────────────────────────────
def load_tracker():
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH) as f:
            return json.load(f)
    return []


def save_tracker(tickets):
    os.makedirs('data', exist_ok=True)
    with open(TRACKER_PATH, 'w') as f:
        json.dump(tickets, f, indent=4)


# ──────────────────────────────────────────────────────────────────────────────
#  Triage one bug and print a clean summary
# ──────────────────────────────────────────────────────────────────────────────
def triage_bug(args, models, kb, tickets):
    record = clean_report(args.title, args.desc, args.error_code,
                          args.category, args.environment)

    severity, priority = predict_severity_priority(
        models, record, bug_domain=args.domain, tech_stack=args.tech_stack,
        known_severity=args.severity)

    assigned_to = assign_developer(kb, record['category'], args.domain)

    ticket = {
        'ticket_id':   f"TKT_{len(tickets) + 1:04d}",
        'title':       record['title'],
        'description': record['description'],
        'category':    record['category'],
        'environment': record['environment'],
        'severity':    severity,
        'priority':    priority,
        'assigned_to': assigned_to,
        'created_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    tickets.append(ticket)

    # Clean summary: what the bug is, and where it went.
    print(f"  {ticket['ticket_id']}  {record['title']}")
    print(f"      Description : {record['description']}")
    print(f"      Severity    : {severity}   Priority: {priority}")
    print(f"      Assigned to : {assigned_to}")
    print()
    return ticket


# ──────────────────────────────────────────────────────────────────────────────
#  Graph: all bugs by who they were assigned to
# ──────────────────────────────────────────────────────────────────────────────
def chart_assignments(tickets, show=True):
    if not tickets:
        return None

    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.05)

    counts = pd.Series([t['assigned_to'] for t in tickets]).value_counts()

    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * len(counts) + 1.5)))
    palette = sns.color_palette('viridis', len(counts))
    bars = ax.barh(counts.index, counts.values, color=palette,
                   edgecolor='white', linewidth=1.0)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f'{int(w)}',
                va='center', fontsize=10, fontweight='bold')

    ax.set_title('Bugs by Assigned Developer', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Number of bugs', fontsize=11)
    ax.set_xlim(0, counts.max() * 1.18)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
    # plt.show() is modal: it blocks until the window is closed by hand. That is
    # fine when this script is run on its own, but it would stall run_pipeline.py
    # part-way through, so the runner passes --no-open.
    if show:
        plt.show()      # pop the chart window up after the run
    plt.close(fig)
    return CHART_PATH


# ──────────────────────────────────────────────────────────────────────────────
#  Entry points
# ──────────────────────────────────────────────────────────────────────────────
def run(bugs, models, kb, tickets, show_chart=True):
    new_tickets = []
    for bug in bugs:
        a = argparse.Namespace(
            title=bug['title'], desc=bug['desc'], error_code=bug['error_code'],
            category=bug['category'], environment=bug['environment'],
            domain=bug.get('domain'), tech_stack=bug.get('tech_stack'),
            severity=bug.get('severity'))
        new_tickets.append(triage_bug(a, models, kb, tickets))

    save_tracker(tickets)
    # Chart only the bugs triaged in this run.
    chart = chart_assignments(new_tickets, show=show_chart)
    if chart:
        print(f"  Chart saved: {chart}")


def main():
    p = argparse.ArgumentParser(
        description="Bug triage: predict, assign, and chart bug assignments.")
    p.add_argument('--title', type=str, help="Short title of the bug")
    p.add_argument('--desc', type=str, help="Detailed description of the bug")
    p.add_argument('--error-code', dest='error_code', default=500,
                   help="Error code observed (e.g. 400, 404, 500, 503)")
    p.add_argument('--category', type=str, default='Backend Logic Bug',
                   help="Bug category, e.g. 'Memory Leak', 'UI Bug'")
    p.add_argument('--environment', type=str, default='Production',
                   choices=['Development', 'Staging', 'Production'])
    p.add_argument('--domain', type=str, default=None,
                   help="Bug domain, e.g. 'Backend Systems', 'Mobile'")
    p.add_argument('--tech-stack', dest='tech_stack', type=str, default=None)
    p.add_argument('--severity', type=str, default=None,
                   choices=['Low', 'Medium', 'High', 'Critical'],
                   help="Override the predicted severity with the reporter's own")
    p.add_argument('--n', type=int, default=15,
                   help="How many bugs to sample from the preprocessed data")
    p.add_argument('--seed', type=int, default=42,
                   help="Random seed for the data sample")
    p.add_argument('--reset', action='store_true',
                   help="Clear the tracker before running")
    p.add_argument('--no-open', dest='show_chart', action='store_false',
                   help="Only save the assignment chart; do not pop up its window")
    p.set_defaults(show_chart=True)
    args = p.parse_args()

    if args.reset and os.path.exists(TRACKER_PATH):
        os.remove(TRACKER_PATH)

    models = load_models()
    if models is None:
        return 1
    kb = load_kb()
    if kb is None:
        return 1

    tickets = load_tracker()


    if args.title or args.desc:
        bugs = [{'title': args.title, 'desc': args.desc, 'error_code': args.error_code,
                 'category': args.category, 'environment': args.environment,
                 'domain': args.domain, 'tech_stack': args.tech_stack,
                 'severity': args.severity}]
    else:
        bugs = load_bugs_from_data(args.n, args.seed)
        if bugs is None:
            return 1

    run(bugs, models, kb, tickets, show_chart=args.show_chart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
