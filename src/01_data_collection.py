# ==============================================================================
# 01_data_collection.py
# Task 1 & 2: Data Collection and Dataset Connection
# Connects to the real 50k bug dataset (Kaggle-sourced) and derives the bug
# lifecycle fields the milestone requires but the source dataset does not carry
# (status, lifecycle_stage, priority, resolution).
# Input:  data/bug_dataset_50k.csv
# Output: data/bug_reports_enriched.csv  +  dataset summary on console
# ==============================================================================

import json
import os
import sys

import _deps
_deps.check('pandas', 'numpy')

import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_PATH = 'data/bug_dataset_50k.csv'
ENRICHED_PATH = 'data/bug_reports_enriched.csv'
KB_PATH = 'data/bug_knowledge_base.json'
SEED = 42

# ── Bug life cycle definition ─────────────────────────────────────────────────
# Standard defect life cycle: New -> Assigned -> In Progress -> Fixed ->
# Pending Retest -> Verified -> Closed, with Reopened / Duplicate / Rejected /
# Deferred as terminal or looping side states.
STATUS_WEIGHTS = {
    'New':            0.10,
    'Assigned':       0.10,
    'In Progress':    0.14,
    'Fixed':          0.14,
    'Pending Retest': 0.09,
    'Verified':       0.10,
    'Closed':         0.20,
    'Reopened':       0.05,
    'Duplicate':      0.04,
    'Rejected':       0.03,
    'Deferred':       0.01,
}

STATUS_TO_STAGE = {
    'New':            'Reported',
    'Assigned':       'Reported',
    'In Progress':    'In Progress',
    'Reopened':       'In Progress',
    'Fixed':          'Resolved',
    'Pending Retest': 'Resolved',
    'Verified':       'Verification',
    'Closed':         'Closed',
    'Duplicate':      'Closed',
    'Rejected':       'Closed',
    'Deferred':       'Closed',
}

STATUS_TO_RESOLUTION = {
    'New':            'Unresolved',
    'Assigned':       'Unresolved',
    'In Progress':    'Unresolved',
    'Reopened':       'Unresolved',
    'Fixed':          'Fixed',
    'Pending Retest': 'Fixed',
    'Verified':       'Fixed',
    'Closed':         'Fixed',
    'Duplicate':      'Duplicate',
    'Rejected':       'Invalid',
    'Deferred':       "Won't Fix",
}

# Priority is derived from operational impact, not from the (uninformative)
# description text — see README "Derived Fields" for the full rationale.
SEVERITY_SCORE    = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
ENVIRONMENT_SCORE = {'Production': 2, 'Staging': 1, 'Development': 0}
BLOCKING_ERRORS   = {500.0, 502.0, 503.0}   # server-side failures block users

PRIORITY_LEVELS = ['P1', 'P2', 'P3', 'P4', 'P5']

# ── Developer routing ─────────────────────────────────────────────────────────
# Which specialist should own each bug category. This is a documented routing
# policy, NOT learned from the data: `developer_role` in the source dataset is
# uniformly random (~11.1% for each of the 9 roles inside every single
# category), so there is no assignment pattern for a model to learn.
CATEGORY_TO_ROLE = {
    'API Bug':                 'Backend Developer',
    'Authentication Bug':      'Security Engineer',
    'Authorization Bug':       'Security Engineer',
    'Backend Logic Bug':       'Backend Developer',
    'CI/CD Bug':               'DevOps Engineer',
    'Cloud Configuration Bug': 'Cloud Engineer',
    'Concurrency Bug':         'Backend Developer',
    'Database Bug':            'Data Engineer',
    'Deployment Bug':          'DevOps Engineer',
    'Frontend Routing Bug':    'Frontend Developer',
    'Logging Bug':             'DevOps Engineer',
    'Memory Leak':             'Backend Developer',
    'Monitoring Bug':          'DevOps Engineer',
    'Performance Bug':         'Backend Developer',
    'Security Vulnerability':  'Security Engineer',
    'UI Bug':                  'Frontend Developer',
}

# A mobile-domain bug goes to the mobile specialist regardless of category.
DOMAIN_OVERRIDE = {'Mobile': 'Mobile Developer'}


def build_knowledge_base(df):
    """Root cause + suggested fix + owning role for each bug category.

    root_cause and suggested_fix have exactly one distinct value per category
    in this dataset, so they can be looked up rather than predicted.
    """
    kb = {}
    for category, group in df.groupby('bug_category'):
        first = group.iloc[0]
        kb[category] = {
            'root_cause':    first.get('root_cause', ''),
            'suggested_fix': first.get('suggested_fix', ''),
            'assigned_role': CATEGORY_TO_ROLE.get(category, 'Full-Stack Developer'),
            'sample_title':  first.get('title', ''),
            'bug_count':     int(len(group)),
        }
    return kb


def derive_priority(df, rng):
    """Score-based P1..P5 assignment from severity + environment + error_code."""
    score = (
        df['severity'].map(SEVERITY_SCORE).fillna(1)
        + df['environment'].map(ENVIRONMENT_SCORE).fillna(0)
        + df['error_code'].isin(BLOCKING_ERRORS).astype(int)
    )

    # score range is 1..7 -> P1 (most urgent) .. P5 (least urgent)
    priority_idx = np.select(
        [score >= 6, score == 5, score == 4, score == 3],
        [0, 1, 2, 3],
        default=4,
    )

    # Real triage is not perfectly rule-driven: nudge ~8% of rows by one level
    # so the target is learnable but not a closed-form lookup.
    jitter_mask = rng.random(len(df)) < 0.08
    nudge = rng.choice([-1, 1], size=len(df))
    priority_idx = np.where(jitter_mask, priority_idx + nudge, priority_idx)
    priority_idx = np.clip(priority_idx, 0, len(PRIORITY_LEVELS) - 1)

    return pd.Series([PRIORITY_LEVELS[i] for i in priority_idx], index=df.index)


def derive_lifecycle_fields(df):
    """Add status, lifecycle_stage, priority and resolution columns."""
    rng = np.random.default_rng(SEED)

    statuses = list(STATUS_WEIGHTS.keys())
    weights  = np.array(list(STATUS_WEIGHTS.values()), dtype=float)
    weights /= weights.sum()

    df['status']          = rng.choice(statuses, size=len(df), p=weights)
    df['lifecycle_stage'] = df['status'].map(STATUS_TO_STAGE)
    df['resolution']      = df['status'].map(STATUS_TO_RESOLUTION)
    df['priority']        = derive_priority(df, rng)

    return df


def load_dataset():
    print("=" * 60)
    print("  TASK 1 & 2: Data Collection & Dataset Connection")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"  [ERROR] Dataset not found at '{DATASET_PATH}'")
        return None

    df = pd.read_csv(DATASET_PATH)

    print(f"\n  Source       : Kaggle — Real Bug Report Dataset (50k)")
    print(f"  File         : {DATASET_PATH}")
    print(f"  Total Records: {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")

    # ------------------------------------------------------------------
    # Required-field coverage check (milestone spec)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  REQUIRED FIELD COVERAGE (before enrichment)")
    print("-" * 60)
    required = {
        'Bug ID':      'bug_id',
        'Summary':     'title',
        'Description': 'description',
        'Status':      'status',
        'Severity':    'severity',
        'Priority':    'priority',
        'Resolution':  'resolution',
    }
    missing = []
    for label, col in required.items():
        present = col in df.columns
        print(f"  {label:<12} -> {f'{col!r}':<15} {'PRESENT' if present else 'MISSING'}")
        if not present:
            missing.append(col)

    # ------------------------------------------------------------------
    # Derive the missing lifecycle fields
    # ------------------------------------------------------------------
    if missing:
        print("\n" + "-" * 60)
        print("  DERIVING MISSING LIFECYCLE FIELDS")
        print("-" * 60)
        print(f"  The source dataset does not ship {', '.join(missing)}.")
        print(f"  Deriving them deterministically (seed={SEED}) so the bug life")
        print(f"  cycle, status, priority and resolution stages can be analysed.")

        df = derive_lifecycle_fields(df)

        print(f"\n  status          -> {df['status'].nunique()} states")
        print(f"  lifecycle_stage -> {df['lifecycle_stage'].nunique()} stages "
              f"({', '.join(df['lifecycle_stage'].unique())})")
        print(f"  priority        -> {df['priority'].nunique()} levels (P1-P5)")
        print(f"  resolution      -> {df['resolution'].nunique()} outcomes")
        print(f"\n  [NOTE] status/resolution are drawn from a realistic life cycle")
        print(f"         distribution; priority is a documented rule over severity +")
        print(f"         environment + error_code. Both are derived, not observed —")
        print(f"         see README 'Derived Fields'.")

    os.makedirs('data', exist_ok=True)
    df.to_csv(ENRICHED_PATH, index=False)

    # ------------------------------------------------------------------
    # Knowledge base: root cause / suggested fix / owning role per category
    # ------------------------------------------------------------------
    kb = build_knowledge_base(df)
    with open(KB_PATH, 'w') as f:
        json.dump(kb, f, indent=4)

    print("\n" + "-" * 60)
    print("  KNOWLEDGE BASE (root cause / fix / owner per category)")
    print("-" * 60)
    print(f"  {'Category':<26}{'Assigned role':<22}{'Bugs':>7}")
    for cat, entry in list(kb.items())[:5]:
        print(f"  {cat:<26}{entry['assigned_role']:<22}{entry['bug_count']:>7,}")
    print(f"  ... {len(kb)} categories total  ->  {KB_PATH}")

    print(f"\n  Sample (first 5 rows):")
    preview_cols = ['bug_id', 'title', 'severity', 'priority', 'status',
                    'lifecycle_stage', 'resolution']
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print(f"  Enriched dataset saved: {ENRICHED_PATH}  ({len(df):,} rows)")
    print("=" * 60)

    return df

if __name__ == "__main__":
    load_dataset()
