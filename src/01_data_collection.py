# ==============================================================================
# 01_data_collection.py
# Task 1 & 2: Data Collection and Dataset Connection
# Connects to the real 50k bug dataset (Kaggle-sourced) and derives the bug
# lifecycle fields the milestone requires but the source dataset does not carry
# (status, lifecycle_stage, priority, resolution) plus the delivery-tracking
# fields the dashboard task needs (sprint, release_version, module, feature,
# component, date_closed, resolution_days).
# Input:  data/bug_dataset_50k.csv
# Output: data/bug_reports_enriched.csv  |  data/module_catalog.json
#         +  dataset summary on console
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
CATALOG_PATH = 'data/module_catalog.json'
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

# ── Product taxonomy: Module -> Feature -> Component ──────────────────────────
# The source dataset describes each bug technically (domain / category / stack)
# but carries no delivery taxonomy. These three maps project those technical
# columns onto the product breakdown the dashboard reports against. They are a
# documented, deterministic mapping — not something inferred from the data.
#
#   module    <- bug_domain   (which product area owns the defect)
#   feature   <- bug_category (which user-facing capability it breaks)
#   component <- tech_stack   (which deployable/code unit it lives in)
DOMAIN_TO_MODULE = {
    'Web Development':  'Web Portal',
    'Mobile':           'Mobile App',
    'Backend Systems':  'Core Services',
    'Data':             'Data Platform',
    'Cloud':            'Cloud Infrastructure',
    'DevOps':           'Delivery Pipeline',
}

CATEGORY_TO_FEATURE = {
    'API Bug':                 'Public API',
    'Authentication Bug':      'Login & SSO',
    'Authorization Bug':       'Roles & Permissions',
    'Backend Logic Bug':       'Business Rules Engine',
    'CI/CD Bug':               'Build & Release',
    'Cloud Configuration Bug': 'Environment Provisioning',
    'Concurrency Bug':         'Job Scheduler',
    'Database Bug':            'Persistence Layer',
    'Deployment Bug':          'Release Rollout',
    'Frontend Routing Bug':    'Navigation & Routing',
    'Logging Bug':             'Audit Logging',
    'Memory Leak':             'Runtime Memory',
    'Monitoring Bug':          'Health Monitoring',
    'Performance Bug':         'Response Latency',
    'Security Vulnerability':  'Threat Protection',
    'UI Bug':                  'User Interface',
}

TECH_TO_COMPONENT = {
    'React':       'web-ui-react',
    'Angular':     'web-ui-angular',
    'Vue':         'web-ui-vue',
    'Node.js':     'svc-node-api',
    'Spring Boot': 'svc-spring-core',
    'Django':      'svc-django-app',
    'Flask':       'svc-flask-api',
    'Laravel':     'svc-laravel-web',
    'PostgreSQL':  'db-postgres',
    'MySQL':       'db-mysql',
    'MongoDB':     'db-mongo',
    'Docker':      'infra-docker',
    'Kubernetes':  'infra-k8s',
    'AWS':         'cloud-aws',
    'Azure':       'cloud-azure',
    'GCP':         'cloud-gcp',
}

# Module size in thousands of lines of code. The dataset ships no code-size
# metric, so defect density (bugs per KLOC) needs a documented size baseline;
# these are fixed project constants, published in data/module_catalog.json so
# every downstream KPI uses the same denominator.
MODULE_KLOC = {
    'Core Services':        142.5,
    'Web Portal':            96.2,
    'Data Platform':         88.6,
    'Mobile App':            74.8,
    'Cloud Infrastructure':  45.3,
    'Delivery Pipeline':     31.7,
}

# ── Sprint / release calendar ─────────────────────────────────────────────────
SPRINT_LENGTH_DAYS  = 14   # two-week sprints, numbered from the first bug date
SPRINTS_PER_RELEASE = 3    # three sprints ship as one release
MINORS_PER_MAJOR    = 3    # v1.0 v1.1 v1.2 then v2.0 ...

# ── Closure timing ────────────────────────────────────────────────────────────
# Only these statuses represent a bug that has actually left the board, so only
# they get a date_closed. Everything else is still in flight (date_closed null).
CLOSED_STATUSES = {'Closed', 'Duplicate', 'Rejected', 'Deferred'}

# Median days-to-close by priority: the urgent queue is worked first.
PRIORITY_MEDIAN_DAYS = {'P1': 2.0, 'P2': 5.0, 'P3': 12.0, 'P4': 25.0, 'P5': 45.0}

# Multiplier on that median by how the bug ended: duplicates and invalid reports
# are dismissed quickly, deferred work sits far longer than a real fix.
RESOLUTION_SPEED = {'Fixed': 1.0, 'Duplicate': 0.30, 'Invalid': 0.40, "Won't Fix": 2.0}

# Days a bug of each priority is allowed to stay open before it breaches SLA.
SLA_TARGET_DAYS = {'P1': 3, 'P2': 7, 'P3': 14, 'P4': 30, 'P5': 60}


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


def release_name(sprint_index):
    """Map a 0-based sprint number onto the release that shipped it."""
    release_idx = sprint_index // SPRINTS_PER_RELEASE
    major = 1 + release_idx // MINORS_PER_MAJOR
    minor = release_idx % MINORS_PER_MAJOR
    return f"v{major}.{minor}"


def derive_sprint_calendar(df):
    """Add sprint and release_version from created_at.

    Sprint 1 starts on the earliest bug date; every SPRINT_LENGTH_DAYS opens the
    next one. Bugs with an unparseable date fall into sprint 1 rather than being
    dropped.

    The data ends mid-sprint, so the last bucket is a stub. When it is shorter
    than half a sprint its bugs are folded into the sprint before it: left on
    its own, a 2-day bucket plots next to 14-day ones and reads as a collapse in
    intake rather than as a short window.
    """
    created = pd.to_datetime(df['created_at'], errors='coerce')
    start   = created.min()
    offset  = (created - start).dt.days.fillna(0).astype(int)

    sprint_index = offset // SPRINT_LENGTH_DAYS
    last_index   = int(sprint_index.max())
    tail_days    = int(offset.max()) - last_index * SPRINT_LENGTH_DAYS + 1
    if last_index > 0 and tail_days < SPRINT_LENGTH_DAYS / 2:
        sprint_index = sprint_index.clip(upper=last_index - 1)

    df['sprint'] = sprint_index.map(lambda i: f"SPR-{i + 1:02d}")
    df['release_version'] = sprint_index.map(release_name)

    return df, created, int(sprint_index.max()) + 1


def derive_closure_dates(df, created, rng):
    """Add date_closed and resolution_days for bugs that have left the board.

    Days-to-close is drawn from a log-normal spread around a median set by the
    bug's priority and scaled by how it was resolved, so the KPI stage sees a
    realistic right-skewed distribution instead of a flat one.
    """
    n = len(df)
    is_closed = df['status'].isin(CLOSED_STATUSES).to_numpy()

    median = df['priority'].map(PRIORITY_MEDIAN_DAYS).fillna(12.0).to_numpy()
    speed  = df['resolution'].map(RESOLUTION_SPEED).fillna(1.0).to_numpy()

    # log-normal noise: median stays put, the tail stretches to the right
    days = median * speed * np.exp(rng.normal(0.0, 0.55, size=n))
    days = np.clip(np.rint(days), 1, 365)

    # The dataset is a snapshot, so a bug cannot close after the last recorded
    # day. Where the drawn duration would overrun the snapshot, redraw it
    # uniformly inside the days that are actually left: truncating to the
    # snapshot instead would stack every one of those closures onto the final
    # date and invent a closure spike in the last sprint.
    snapshot  = created.max()
    available = (snapshot - created).dt.days.fillna(0).to_numpy()
    overruns  = days > available
    days = np.where(overruns,
                    np.rint(rng.random(n) * np.maximum(available, 0)),
                    days)
    days = np.clip(days, 0, np.maximum(available, 0))

    closed = created + pd.to_timedelta(np.where(is_closed, days, 0), unit='D')

    df['date_closed']     = closed.where(pd.Series(is_closed, index=df.index))
    df['resolution_days'] = (df['date_closed'] - created).dt.days
    df['date_closed']     = df['date_closed'].dt.strftime('%Y-%m-%d')

    return df


def derive_delivery_fields(df, rng):
    """Add the product taxonomy (module/feature/component) and closure dates."""
    df['module']    = df['bug_domain'].map(DOMAIN_TO_MODULE).fillna('Unassigned Module')
    df['feature']   = df['bug_category'].map(CATEGORY_TO_FEATURE).fillna('Unassigned Feature')
    df['component'] = df['tech_stack'].map(TECH_TO_COMPONENT).fillna('unassigned-component')

    df, created, sprint_count = derive_sprint_calendar(df)
    df = derive_closure_dates(df, created, rng)

    return df, sprint_count


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

    # Delivery-tracking fields draw from the same generator afterwards, so the
    # life cycle columns above keep the exact values earlier runs produced.
    df, sprint_count = derive_delivery_fields(df, rng)

    return df, sprint_count


def build_module_catalog(df):
    """Module -> size, features and components, for the defect-density KPI."""
    catalog = {}
    for module, group in df.groupby('module'):
        catalog[module] = {
            'size_kloc':  MODULE_KLOC.get(module, float(round(len(group) / 100.0, 1))),
            'bug_count':  int(len(group)),
            'features':   sorted(group['feature'].unique().tolist()),
            'components': sorted(group['component'].unique().tolist()),
        }
    return {
        'modules':          catalog,
        'sla_target_days':  SLA_TARGET_DAYS,
        'sprint_length_days':  SPRINT_LENGTH_DAYS,
        'sprints_per_release': SPRINTS_PER_RELEASE,
        'minors_per_major':    MINORS_PER_MAJOR,
    }


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
        'Bug ID':          'bug_id',
        'Summary':         'title',
        'Description':     'description',
        'Status':          'status',
        'Severity':        'severity',
        'Priority':        'priority',
        'Resolution':      'resolution',
        'Root Cause':      'root_cause',
        'Sprint':          'sprint',
        'Release Version': 'release_version',
        'Module':          'module',
        'Feature':         'feature',
        'Component':       'component',
        'Date Closed':     'date_closed',
    }
    missing = []
    for label, col in required.items():
        present = col in df.columns
        print(f"  {label:<16} -> {f'{col!r}':<18} {'PRESENT' if present else 'MISSING'}")
        if not present:
            missing.append(col)

    # ------------------------------------------------------------------
    # Derive the missing lifecycle fields
    # ------------------------------------------------------------------
    if missing:
        print("\n" + "-" * 60)
        print("  DERIVING MISSING LIFECYCLE & DELIVERY FIELDS")
        print("-" * 60)
        print(f"  The source dataset does not ship {', '.join(missing)}.")
        print(f"  Deriving them deterministically (seed={SEED}) so the bug life")
        print(f"  cycle and the sprint/module dashboards can be analysed.")

        df, sprint_count = derive_lifecycle_fields(df)

        closed_n = int(df['date_closed'].notna().sum())
        print(f"\n  status          -> {df['status'].nunique()} states")
        print(f"  lifecycle_stage -> {df['lifecycle_stage'].nunique()} stages "
              f"({', '.join(df['lifecycle_stage'].unique())})")
        print(f"  priority        -> {df['priority'].nunique()} levels (P1-P5)")
        print(f"  resolution      -> {df['resolution'].nunique()} outcomes")
        print(f"  sprint          -> {sprint_count} sprints of {SPRINT_LENGTH_DAYS} days "
              f"(SPR-01 .. SPR-{sprint_count:02d}); the trailing part-sprint is")
        print(f"                     folded into SPR-{sprint_count:02d} so it is not read as an")
        print(f"                     intake collapse")
        print(f"  release_version -> {df['release_version'].nunique()} releases "
              f"({SPRINTS_PER_RELEASE} sprints each)")
        print(f"  module          -> {df['module'].nunique()} product modules")
        print(f"  feature         -> {df['feature'].nunique()} features")
        print(f"  component       -> {df['component'].nunique()} components")
        print(f"  date_closed     -> {closed_n:,} closed bugs "
              f"({closed_n / len(df) * 100:.1f}%); open bugs stay null")
        print(f"  resolution_days -> median {df['resolution_days'].median():.0f} d, "
              f"mean {df['resolution_days'].mean():.1f} d")
        print(f"\n  [NOTE] status/resolution are drawn from a realistic life cycle")
        print(f"         distribution; priority is a documented rule over severity +")
        print(f"         environment + error_code; module/feature/component are a")
        print(f"         fixed mapping of domain/category/tech_stack. All derived,")
        print(f"         not observed — see README 'Derived Fields'.")

    os.makedirs('data', exist_ok=True)
    df.to_csv(ENRICHED_PATH, index=False)

    # ------------------------------------------------------------------
    # Module catalog: size baseline for defect density + SLA targets
    # ------------------------------------------------------------------
    catalog = build_module_catalog(df)
    with open(CATALOG_PATH, 'w') as f:
        json.dump(catalog, f, indent=4)

    print("\n" + "-" * 60)
    print("  MODULE CATALOG (size baseline for defect density)")
    print("-" * 60)
    print(f"  {'Module':<24}{'KLOC':>8}{'Bugs':>9}{'Bugs/KLOC':>12}")
    for name, entry in sorted(catalog['modules'].items(),
                              key=lambda kv: -kv[1]['bug_count'] / kv[1]['size_kloc']):
        density = entry['bug_count'] / entry['size_kloc']
        print(f"  {name:<24}{entry['size_kloc']:>8.1f}{entry['bug_count']:>9,}{density:>12.1f}")
    print(f"  -> {CATALOG_PATH}")

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

    print(f"\n  Sample — life cycle columns (first 5 rows):")
    preview_cols = ['bug_id', 'severity', 'priority', 'status',
                    'lifecycle_stage', 'resolution']
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(5).to_string(index=False))

    print(f"\n  Sample — delivery columns (same 5 rows):")
    delivery_cols = ['bug_id', 'sprint', 'release_version', 'module', 'feature',
                     'component', 'date_closed', 'resolution_days']
    delivery_cols = [c for c in delivery_cols if c in df.columns]
    print(df[delivery_cols].head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print(f"  Enriched dataset saved: {ENRICHED_PATH}  ({len(df):,} rows)")
    print("=" * 60)

    return df

if __name__ == "__main__":
    load_dataset()
