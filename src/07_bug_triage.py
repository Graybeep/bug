# ==============================================================================
# 07_bug_triage.py
# End-to-end Bug Management workflow.
#
# A user reports a bug (title, description, error code, category, environment).
# The system cleans and analyses it, uses RANDOM FOREST to predict severity and
# priority, assigns the appropriate developer, attaches the identified root
# cause and suggested fix, then tracks and visualizes the bug through its life
# cycle until it is resolved.
#
#   Report a bug:
#     python src/07_bug_triage.py --title "Checkout fails" \
#         --desc "Payment page crashes after submit" \
#         --error-code 500 --category "Backend Logic Bug" --environment Production
#
#   Track progress:
#     python src/07_bug_triage.py --list
#     python src/07_bug_triage.py --advance TKT_0001
#     python src/07_bug_triage.py --resolve TKT_0001
#
# Input:  models/rf_*.pkl, models/label_encoders.pkl, data/bug_knowledge_base.json
# Output: data/tracked_bugs.json  |  visualizations/tracked_bug_lifecycle.png
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
import matplotlib.ticker as mticker
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACKER_PATH = 'data/tracked_bugs.json'
KB_PATH      = 'data/bug_knowledge_base.json'

# The bug life cycle a ticket walks through, in order.
LIFECYCLE = ['New', 'Assigned', 'In Progress', 'Fixed',
             'Pending Retest', 'Verified', 'Closed']

STAGE_OF = {
    'New': 'Reported',        'Assigned': 'Reported',
    'In Progress': 'In Progress',
    'Fixed': 'Resolved',      'Pending Retest': 'Resolved',
    'Verified': 'Verification',
    'Closed': 'Closed',
}

SEVERITY_NOTE = {
    'Critical': 'System unusable — immediate action required.',
    'High':     'Significant impact — needs urgent resolution.',
    'Medium':   'Moderate impact — schedule for next sprint.',
    'Low':      'Minor impact — address when possible.',
}

PRIORITY_NOTE = {
    'P1': 'Fix now — blocks release.',
    'P2': 'Fix in the current sprint.',
    'P3': 'Schedule in an upcoming sprint.',
    'P4': 'Backlog — fix when capacity allows.',
    'P5': 'Lowest — cosmetic or deferred.',
}

DOMAIN_OVERRIDE = {'Mobile': 'Mobile Developer'}

# Representative bugs used when the script is run without arguments, chosen to
# exercise different categories, environments and severities so the routing and
# priority logic are visible side by side.
DEMO_BUGS = [
    {'title': 'Checkout page crashes on payment',
     'desc':  'Users report the checkout page freezes and the application crashes '
              'after submitting payment on the live site.',
     'error_code': 500, 'category': 'Backend Logic Bug',
     'environment': 'Production', 'domain': 'Backend Systems', 'severity': 'Critical'},

    {'title': 'Login API times out under peak load',
     'desc':  'Authentication endpoint stops responding during peak traffic hours.',
     'error_code': 503, 'category': 'Authentication Bug',
     'environment': 'Production', 'domain': 'Backend Systems', 'severity': 'High'},

    {'title': 'Customer records load slowly',
     'desc':  'Reports page takes over 30 seconds to return customer records.',
     'error_code': 500, 'category': 'Database Bug',
     'environment': 'Staging', 'domain': 'Data', 'severity': 'Medium'},

    {'title': 'Android app crashes on startup',
     'desc':  'Mobile application closes immediately after launch on Android 14.',
     'error_code': 500, 'category': 'Memory Leak',
     'environment': 'Production', 'domain': 'Mobile', 'severity': 'High'},

    {'title': 'Save button label overflows',
     'desc':  'Text on the settings page save button runs outside its border.',
     'error_code': 400, 'category': 'UI Bug',
     'environment': 'Development', 'domain': 'Web Development', 'severity': 'Low'},
]


# ──────────────────────────────────────────────────────────────────────────────
#  Step 1: Clean the reported bug information
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    """Normalize free-text input the same way the training data was handled."""
    if not text:
        return ''
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace/newlines
    text = re.sub(r'[^\x20-\x7E]', '', text)  # drop non-printable characters
    return text


def clean_report(title, description, error_code, category, environment):
    """Validate and normalize the user's bug report; returns (record, warnings)."""
    warnings = []

    title       = clean_text(title)
    description = clean_text(description)

    if not description:
        description = title
        warnings.append("No description given — using the title as the description.")

    try:
        error_code = float(error_code)
        if not (100 <= error_code <= 599):
            warnings.append(f"Error code {error_code:.0f} is outside the HTTP range; kept as-is.")
    except (TypeError, ValueError):
        error_code = 500.0
        warnings.append("Error code missing or non-numeric — defaulted to 500.")

    return {
        'title':       title,
        'description': description,
        'error_code':  error_code,
        'category':    category,
        'environment': environment,
    }, warnings


# ──────────────────────────────────────────────────────────────────────────────
#  Step 2: Predict severity and priority with Random Forest
# ──────────────────────────────────────────────────────────────────────────────
def load_models():
    """Load the Random Forest models and the fitted transformers."""
    required = {
        'vectorizer': 'models/tfidf_vectorizer.pkl',
        'encoders':   'models/label_encoders.pkl',
        'severity':   'models/rf_severity_model.pkl',
        'priority':   'models/rf_priority_model.pkl',
        'features':   'models/priority_features.pkl',
    }
    missing = [p for p in required.values() if not os.path.exists(p)]
    if missing:
        print("  [ERROR] Missing trained model files:")
        for p in missing:
            print(f"            {p}")
        print("\n  Run the pipeline first:")
        print("    python src/01_data_collection.py")
        print("    python src/02_preprocessing.py")
        print("    python src/05_modeling.py")
        return None
    return {k: joblib.load(p) for k, p in required.items()}


def encode_value(encoders, column, value):
    """Map a category label to its trained integer code (0 if unseen)."""
    le = encoders.get(column)
    if le is None or value is None:
        return 0
    classes = list(le.classes_)
    return int(le.transform([value])[0]) if value in classes else 0


def predict_severity_priority(m, record, bug_domain=None, tech_stack=None,
                              developer_role=None, known_severity=None):
    """Random Forest predictions for severity, then priority.

    If the reporter supplied a severity, it is used as-is *and* fed into the
    priority features — priority depends on severity, so the two must agree.
    """
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
#  Step 3: Assign developer + attach root cause and suggested fix
# ──────────────────────────────────────────────────────────────────────────────
def load_kb():
    if not os.path.exists(KB_PATH):
        print(f"  [ERROR] '{KB_PATH}' not found — run 01_data_collection.py first.")
        return None
    with open(KB_PATH) as f:
        return json.load(f)


def assign_and_diagnose(kb, category, bug_domain, priority):
    """Look up who owns this category, plus its root cause and suggested fix."""
    entry = kb.get(category)
    if entry is None:
        return {
            'assigned_to':   'Full-Stack Developer',
            'root_cause':    'Unrecognized category — needs manual investigation.',
            'suggested_fix': 'Triage manually and add this category to the knowledge base.',
            'known_category': False,
            'escalated':     False,
        }

    role = DOMAIN_OVERRIDE.get(bug_domain, entry['assigned_role'])
    return {
        'assigned_to':    role,
        'root_cause':     entry['root_cause'],
        'suggested_fix':  entry['suggested_fix'],
        'known_category': True,
        'escalated':      priority in ('P1', 'P2'),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Step 4: Track the bug through its life cycle
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


def new_ticket_id(tickets):
    return f"TKT_{len(tickets) + 1:04d}"


def log_event(ticket, status, note=''):
    ticket['status'] = status
    ticket['stage']  = STAGE_OF[status]
    ticket.setdefault('history', []).append({
        'status':    status,
        'stage':     STAGE_OF[status],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note':      note,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  Step 5: Visualize the tracked bugs
# ──────────────────────────────────────────────────────────────────────────────
def chart_tracker(tickets):
    if not tickets:
        return None

    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.05)

    counts = pd.Series([t['status'] for t in tickets]).value_counts()
    counts = counts.reindex(LIFECYCLE).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={'width_ratios': [1.25, 1]})

    # Left: how many tickets sit at each life cycle status
    palette = sns.color_palette('viridis', len(counts))
    bars = ax1.bar(counts.index, counts.values, color=palette,
                   edgecolor='white', linewidth=1.0)
    for bar in bars:
        h = bar.get_height()
        if h:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.06, f'{int(h)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Tracked Bugs by Life Cycle Status', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Tickets', fontsize=11)
    ax1.set_ylim(0, max(counts.max() * 1.25, 1))
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(axis='x', rotation=30)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha('right')
    ax1.spines[['top', 'right']].set_visible(False)

    # Right: each ticket's progress along the life cycle
    ax2.set_title('Progress per Ticket', fontsize=13, fontweight='bold', pad=12)
    recent = tickets[-10:]
    for row, t in enumerate(recent):
        pos = LIFECYCLE.index(t['status'])
        pct = (pos + 1) / len(LIFECYCLE)
        colour = '#43A047' if t['status'] == 'Closed' else '#1E88E5'
        ax2.barh(row, pct, color=colour, edgecolor='white', height=0.62)
        ax2.text(pct + 0.02, row, t['status'], va='center', fontsize=9)
    ax2.set_yticks(range(len(recent)))
    ax2.set_yticklabels([f"{t['ticket_id']} ({t['priority']})" for t in recent], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.45)
    ax2.set_xticks([])
    ax2.set_xlabel(f"{LIFECYCLE[0]}  ->  {LIFECYCLE[-1]}", fontsize=10)
    ax2.spines[['top', 'right', 'bottom']].set_visible(False)

    fig.tight_layout()
    path = 'visualizations/tracked_bug_lifecycle.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Report a new bug — the full workflow
# ──────────────────────────────────────────────────────────────────────────────
def report_bug(args, models=None, kb=None, compact=False):
    if not compact:
        print("=" * 66)
        print("  BUG MANAGEMENT SYSTEM — NEW BUG REPORT")
        print("=" * 66)

    # -- Step 1: clean ------------------------------------------------
    record, warnings = clean_report(args.title, args.desc, args.error_code,
                                    args.category, args.environment)

    if compact:
        return report_bug_compact(args, record, warnings, models, kb)

    print(f"\n  [1] REPORTED BUG DETAILS")
    print(f"      Title       : {record['title']}")
    print(f"      Description : {record['description'][:64]}"
          f"{'...' if len(record['description']) > 64 else ''}")
    print(f"      Error code  : {record['error_code']:.0f}")
    print(f"      Category    : {record['category']}")
    print(f"      Environment : {record['environment']}")
    if args.domain:
        print(f"      Domain      : {args.domain}")

    print(f"\n  [2] CLEANING & VALIDATION")
    if warnings:
        for w in warnings:
            print(f"      ! {w}")
    else:
        print(f"      All fields present and valid — no cleaning issues.")

    # -- Step 2: predict ----------------------------------------------
    m = load_models()
    if m is None:
        return 1
    kb = load_kb()
    if kb is None:
        return 1

    severity, priority = predict_severity_priority(
        m, record, bug_domain=args.domain, tech_stack=args.tech_stack,
        known_severity=args.severity)

    source = 'given by reporter' if args.severity else 'predicted from description'
    print(f"\n  [3] RANDOM FOREST PREDICTION")
    print(f"      Severity    : {severity:<10} {SEVERITY_NOTE.get(severity, '')}")
    print(f"                    ({source})")
    print(f"      Priority    : {priority:<10} {PRIORITY_NOTE.get(priority, '')}")

    # -- Step 3: assign + diagnose ------------------------------------
    triage = assign_and_diagnose(kb, record['category'], args.domain, priority)

    print(f"\n  [4] ASSIGNMENT & DIAGNOSIS")
    print(f"      Assigned to : {triage['assigned_to']}")
    if triage['escalated']:
        print(f"      Escalation  : {priority} — flagged to the team lead for immediate triage.")
    if not triage['known_category']:
        print(f"      ! Category '{record['category']}' is not in the knowledge base.")
    print(f"      Root cause  : {triage['root_cause']}")
    print(f"      Suggested   : {triage['suggested_fix']}")

    # -- Step 4: track -------------------------------------------------
    tickets = load_tracker()
    ticket = {
        'ticket_id':     new_ticket_id(tickets),
        'title':         record['title'],
        'description':   record['description'],
        'error_code':    record['error_code'],
        'category':      record['category'],
        'environment':   record['environment'],
        'bug_domain':    args.domain,
        'severity':      severity,
        'priority':      priority,
        'assigned_to':   triage['assigned_to'],
        'root_cause':    triage['root_cause'],
        'suggested_fix': triage['suggested_fix'],
        'escalated':     triage['escalated'],
        'created_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    log_event(ticket, 'New', 'Bug reported by user.')
    log_event(ticket, 'Assigned', f"Routed to {triage['assigned_to']}.")
    tickets.append(ticket)
    save_tracker(tickets)

    print(f"\n  [5] LIFE CYCLE TRACKING")
    print(f"      Ticket ID   : {ticket['ticket_id']}")
    print_progress(ticket)

    chart = chart_tracker(tickets)
    print(f"\n      Tracker saved : {TRACKER_PATH}  ({len(tickets)} ticket(s))")
    if chart:
        print(f"      Chart saved   : {chart}")

    print(f"\n      Next: python src/07_bug_triage.py --advance {ticket['ticket_id']}")
    print("\n" + "=" * 66)
    return 0


def create_ticket(record, args, severity, priority, triage):
    """Build the ticket, log its opening events, and persist it."""
    tickets = load_tracker()
    ticket = {
        'ticket_id':     new_ticket_id(tickets),
        'title':         record['title'],
        'description':   record['description'],
        'error_code':    record['error_code'],
        'category':      record['category'],
        'environment':   record['environment'],
        'bug_domain':    args.domain,
        'severity':      severity,
        'priority':      priority,
        'assigned_to':   triage['assigned_to'],
        'root_cause':    triage['root_cause'],
        'suggested_fix': triage['suggested_fix'],
        'escalated':     triage['escalated'],
        'created_at':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    log_event(ticket, 'New', 'Bug reported by user.')
    log_event(ticket, 'Assigned', f"Routed to {triage['assigned_to']}.")
    tickets.append(ticket)
    save_tracker(tickets)
    return ticket, tickets


def report_bug_compact(args, record, warnings, models, kb):
    """One tight block per bug — used by the multi-bug demo."""
    severity, priority = predict_severity_priority(
        models, record, bug_domain=args.domain, tech_stack=args.tech_stack,
        known_severity=args.severity)

    triage = assign_and_diagnose(kb, record['category'], args.domain, priority)
    ticket, _ = create_ticket(record, args, severity, priority, triage)

    print(f"  {ticket['ticket_id']}  {record['title']}")
    print(f"      Reported : {record['category']} · {record['environment']} · "
          f"error {record['error_code']:.0f}")
    if warnings:
        for w in warnings:
            print(f"      Cleaning : ! {w}")
    print(f"      Predicted: Severity {severity}  ->  Priority {priority}"
          f"{'   [ESCALATED]' if triage['escalated'] else ''}")
    print(f"      Assigned : {triage['assigned_to']}")
    print(f"      Cause    : {triage['root_cause']}")
    print(f"      Fix      : {triage['suggested_fix']}")
    print(f"      Status   : {ticket['status']}  (stage: {ticket['stage']})")
    print()
    return ticket


def run_demo(args):
    """Triage several representative bugs so the routing logic is visible."""
    print("=" * 78)
    print("  BUG MANAGEMENT SYSTEM — END-TO-END DEMONSTRATION")
    print("=" * 78)
    print(f"\n  Triaging {len(DEMO_BUGS)} bugs through the full workflow:")
    print(f"  report -> clean -> Random Forest predict -> assign -> diagnose -> track\n")

    models = load_models()
    if models is None:
        return 1
    kb = load_kb()
    if kb is None:
        return 1

    print("-" * 78)
    for i, bug in enumerate(DEMO_BUGS, start=1):
        print(f"  [{i}/{len(DEMO_BUGS)}]")
        demo_args = argparse.Namespace(
            title=bug['title'], desc=bug['desc'], error_code=bug['error_code'],
            category=bug['category'], environment=bug['environment'],
            domain=bug.get('domain'), tech_stack=bug.get('tech_stack'),
            severity=bug.get('severity'),
        )
        record, warnings = clean_report(
            demo_args.title, demo_args.desc, demo_args.error_code,
            demo_args.category, demo_args.environment)
        report_bug_compact(demo_args, record, warnings, models, kb)

    print("-" * 78)
    tickets = load_tracker()
    print(f"\n  RESULT — {len(DEMO_BUGS)} bugs triaged and now tracked:\n")
    print(f"  {'Ticket':<10}{'Pri':<5}{'Severity':<10}{'Assigned to':<22}"
          f"{'Status':<12}Title")
    print("  " + "-" * 74)
    for t in tickets[-len(DEMO_BUGS):]:
        print(f"  {t['ticket_id']:<10}{t['priority']:<5}{t['severity']:<10}"
              f"{t['assigned_to']:<22}{t['status']:<12}{t['title'][:24]}")

    chart = chart_tracker(tickets)
    print(f"\n  Tracker : {TRACKER_PATH}  ({len(tickets)} ticket(s) total)")
    if chart:
        print(f"  Chart   : {chart}")
    print(f"\n  Notice how the same system routes each bug differently — the category")
    print(f"  decides the owner, and severity + environment together decide priority.")
    print(f"\n  Track any of them:")
    print(f"    python src/07_bug_triage.py --list")
    print(f"    python src/07_bug_triage.py --advance {tickets[-1]['ticket_id']}")
    print(f"    python src/07_bug_triage.py --resolve {tickets[-1]['ticket_id']}")
    print("\n" + "=" * 78)
    return 0


def print_progress(ticket):
    """Render the ticket's position along the life cycle."""
    pos = LIFECYCLE.index(ticket['status'])
    trail = []
    for i, stage in enumerate(LIFECYCLE):
        mark = '[x]' if i < pos else ('[>]' if i == pos else '[ ]')
        trail.append(f"{mark} {stage}")
    print(f"      Progress    : {'  ->  '.join(trail[:4])}")
    print(f"                    {'  ->  '.join(trail[4:])}")
    print(f"      Current     : {ticket['status']}  (stage: {ticket['stage']})")


# ──────────────────────────────────────────────────────────────────────────────
#  Tracking commands
# ──────────────────────────────────────────────────────────────────────────────
def list_bugs():
    tickets = load_tracker()
    print("=" * 66)
    print("  TRACKED BUGS")
    print("=" * 66)
    if not tickets:
        print("\n  No bugs tracked yet. Report one with --title/--desc/--category.")
        return 0

    print(f"\n  {'Ticket':<10}{'Pri':<5}{'Severity':<10}{'Status':<16}"
          f"{'Assigned to':<22}Title")
    print("  " + "-" * 90)
    for t in tickets:
        print(f"  {t['ticket_id']:<10}{t['priority']:<5}{t['severity']:<10}"
              f"{t['status']:<16}{t['assigned_to']:<22}{t['title'][:28]}")

    open_n = sum(1 for t in tickets if t['status'] != 'Closed')
    print(f"\n  {len(tickets)} ticket(s) — {open_n} open, {len(tickets)-open_n} closed.")
    chart = chart_tracker(tickets)
    if chart:
        print(f"  Chart saved: {chart}")
    return 0


def advance_bug(ticket_id, to_end=False):
    tickets = load_tracker()
    match = next((t for t in tickets if t['ticket_id'] == ticket_id), None)

    print("=" * 66)
    print(f"  UPDATE TICKET {ticket_id}")
    print("=" * 66)

    if match is None:
        print(f"\n  [ERROR] No ticket '{ticket_id}'. Use --list to see tracked bugs.")
        return 1

    if match['status'] == 'Closed':
        print(f"\n  {ticket_id} is already Closed — the life cycle is complete.")
        print_progress(match)
        return 0

    pos = LIFECYCLE.index(match['status'])
    targets = LIFECYCLE[pos + 1:] if to_end else [LIFECYCLE[pos + 1]]
    for status in targets:
        log_event(match, status, 'Advanced via 07_bug_triage.py')

    save_tracker(tickets)
    print(f"\n  {match['title']}")
    print(f"  Assigned to : {match['assigned_to']}   |   {match['severity']} / {match['priority']}")
    print()
    print_progress(match)

    if match['status'] == 'Closed':
        print(f"\n  Life cycle complete — bug resolved and closed.")
        print(f"  Fix applied : {match['suggested_fix']}")

    chart = chart_tracker(tickets)
    if chart:
        print(f"\n  Chart saved: {chart}")
    print("\n" + "=" * 66)
    return 0


def main():
    p = argparse.ArgumentParser(
        description="End-to-end bug triage: predict, assign, diagnose, track.")
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

    p.add_argument('--list', action='store_true', help="List all tracked bugs")
    p.add_argument('--advance', type=str, metavar='TICKET_ID',
                   help="Move a ticket to the next life cycle status")
    p.add_argument('--resolve', type=str, metavar='TICKET_ID',
                   help="Drive a ticket all the way through to Closed")
    p.add_argument('--categories', action='store_true',
                   help="Show the knowledge base: category -> owner / root cause / fix")
    p.add_argument('--demo', action='store_true',
                   help="Triage several representative bugs (default with no arguments)")
    p.add_argument('--reset', action='store_true',
                   help="Clear the ticket tracker before running")

    args = p.parse_args()

    if args.reset and os.path.exists(TRACKER_PATH):
        os.remove(TRACKER_PATH)
        print(f"  Tracker cleared: {TRACKER_PATH}\n")

    if args.demo:
        return run_demo(args)
    if args.list:
        return list_bugs()
    if args.advance:
        return advance_bug(args.advance)
    if args.resolve:
        return advance_bug(args.resolve, to_end=True)
    if args.categories:
        kb = load_kb()
        if kb is None:
            return 1
        print("=" * 66)
        print("  KNOWLEDGE BASE — category -> owner / root cause / fix")
        print("=" * 66)
        for cat, e in kb.items():
            print(f"\n  {cat}")
            print(f"    Owner : {e['assigned_role']}")
            print(f"    Cause : {e['root_cause']}")
            print(f"    Fix   : {e['suggested_fix']}")
        return 0

    # No bug supplied -> run the multi-bug demonstration
    if not args.title and not args.desc:
        return run_demo(args)

    return report_bug(args)


if __name__ == "__main__":
    sys.exit(main())
