# ==============================================================================
# 08_dashboard.py
# Task 9: Interactive Dashboard, KPI Reporting & Actionable Insights
#
# Builds ONE self-contained interactive dashboard over the bug records, keyed on
# Bug ID, Sprint, Release Version, Module, Feature, Component, Priority,
# Resolution, Root Cause and Date Closed — plus bug trends and resolution time.
#
# The whole bug table is embedded in the page as base64 typed arrays (one byte
# per categorical value), so every filter, chart, KPI and table recomputes in
# the browser from the same slice. No server and no CDN are needed to read it.
#
# The trained models are wired in twice:
#   - at build time, model_bridge re-scores every bug with the saved priority
#     model, and reproduces 05_modeling.py's train/test split so the dashboard
#     can separate held-out agreement from memorised rows;
#   - at run time, src/09_serve.py exposes the same models over a local API so
#     the dashboard's triage console does live inference.
#
#   python src/08_dashboard.py              # build + open it
#   python src/08_dashboard.py --no-open    # build only
#   python src/08_dashboard.py --no-model   # skip model scoring (faster)
#   python src/08_dashboard.py --sample 20000   # embed fewer rows (smaller file)
#   python src/08_dashboard.py --top 15     # widen the console "top N" tables
#
# Input:  data/bug_reports_processed.csv  |  data/module_catalog.json
#         data/bug_knowledge_base.json    |  data/model_evaluation_results.json
#         models/*.pkl                    (optional — for prediction columns)
# Output: dashboards/index.html           (the interactive dashboard, offline)
#         data/kpi_report.json            (machine-readable KPIs)
#         visualizations/kpi_*.png        (static companions)
#         + the KPI report and actionable insights on the console
# ==============================================================================

import argparse
import base64
import json
import os
import sys
import webbrowser
from datetime import datetime

import _deps
_deps.check('pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'joblib')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # charts are written to disk, never shown here
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder. SRC_DIR is captured
# before the chdir so the dashboard assets stay locatable afterwards.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(SRC_DIR))
sys.path.insert(0, SRC_DIR)

import model_bridge

PROCESSED_PATH  = 'data/bug_reports_processed.csv'
CATALOG_PATH    = 'data/module_catalog.json'
KB_PATH         = 'data/bug_knowledge_base.json'
MODEL_EVAL_PATH = 'data/model_evaluation_results.json'
KPI_PATH        = 'data/kpi_report.json'
DASHBOARD_PATH  = 'dashboards/index.html'

# Superseded by the single interactive dashboard; removed on build so a stale
# copy can't be opened by mistake from an old bookmark.
LEGACY_PAGES = ['dashboards/ops_dashboard.html', 'dashboards/model_report.html']

# Fixed model -> categorical color slot, so a model keeps its identity across
# every chart rather than being recolored per target.
MODEL_ORDER = ['Naive Bayes', 'Logistic Regression', 'Decision Tree',
               'Random Forest', 'SVM (Linear)']

# Mirrors 01_data_collection.py — kept here so the KPI stage can be re-run
# against an older enriched CSV without silently changing the definitions.
SPRINT_LENGTH_DAYS = 14
DEFAULT_SLA_DAYS   = {'P1': 3, 'P2': 7, 'P3': 14, 'P4': 30, 'P5': 60}
DOMAIN_OVERRIDE    = {'Mobile': 'Mobile Developer'}

PRIORITY_ORDER   = ['P1', 'P2', 'P3', 'P4', 'P5']
SEVERITY_ORDER   = ['Critical', 'High', 'Medium', 'Low']
STATUS_ORDER     = ['New', 'Assigned', 'In Progress', 'Fixed', 'Pending Retest',
                    'Verified', 'Closed', 'Reopened', 'Duplicate', 'Rejected', 'Deferred']
LIFECYCLE_ORDER  = ['Reported', 'In Progress', 'Resolved', 'Verification', 'Closed']
RESOLUTION_ORDER = ['Fixed', 'Unresolved', 'Duplicate', 'Invalid', "Won't Fix"]
ENVIRONMENT_ORDER = ['Development', 'Staging', 'Production']

RULE = "-" * 68


# ──────────────────────────────────────────────────────────────────────────────
#  Load
# ──────────────────────────────────────────────────────────────────────────────
def load_inputs():
    if not os.path.exists(PROCESSED_PATH):
        print(f"  [ERROR] '{PROCESSED_PATH}' not found.")
        print(f"  Run 'python src/01_data_collection.py' then "
              f"'python src/02_preprocessing.py' first.")
        return None, None, None, None

    df = pd.read_csv(PROCESSED_PATH)

    required = ['sprint', 'release_version', 'module', 'feature', 'component',
                'priority', 'resolution', 'root_cause', 'date_closed']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [ERROR] '{PROCESSED_PATH}' is missing: {', '.join(missing)}")
        print(f"  Re-run 01_data_collection.py and 02_preprocessing.py — they add")
        print(f"  the sprint/release/module/feature/component/closure columns.")
        return None, None, None, None

    catalog = {}
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            catalog = json.load(f)
    else:
        print(f"  [WARN] '{CATALOG_PATH}' not found — defect density will fall back")
        print(f"         to bugs-per-module without a size baseline.")

    kb = {}
    if os.path.exists(KB_PATH):
        with open(KB_PATH) as f:
            kb = json.load(f)

    model_eval = {}
    if os.path.exists(MODEL_EVAL_PATH):
        with open(MODEL_EVAL_PATH) as f:
            model_eval = json.load(f)
    else:
        print(f"  [WARN] '{MODEL_EVAL_PATH}' not found — run 'python src/05_modeling.py' "
              f"first; the model verdicts will be skipped.")

    return df, catalog, kb, model_eval


def prepare(df, kb):
    """Normalise types and add the derived columns the KPIs are computed from."""
    df['created_at']  = pd.to_datetime(df['created_at'], errors='coerce')
    df['date_closed'] = pd.to_datetime(df['date_closed'], errors='coerce')

    origin = df['created_at'].min()
    snapshot = max(df['created_at'].max(), df['date_closed'].max())

    df['created_day'] = (df['created_at'] - origin).dt.days.fillna(0).astype(int)

    # A bug closes in the sprint its close date falls in. Clamped to the last
    # sprint stage 01 assigned: it folded the trailing part-sprint away, so an
    # unclamped index would name a sprint that no longer exists and those
    # closures would be dropped when the series is reindexed onto the labels.
    last_sprint = max_sprint_index(df)
    closed_day = (df['date_closed'] - origin).dt.days
    df['close_sprint_index'] = (closed_day // SPRINT_LENGTH_DAYS).clip(upper=last_sprint)

    df['is_closed'] = df['date_closed'].notna()
    df['is_open']   = ~df['is_closed']

    # Age of a still-open bug, measured to the snapshot date.
    df['open_age_days'] = np.where(
        df['is_open'], (snapshot - df['created_at']).dt.days, np.nan)

    df['month']       = df['created_at'].dt.strftime('%Y-%m')
    df['close_month'] = df['date_closed'].dt.strftime('%Y-%m')

    # The dataset's own `developer_role` is uniformly random across categories
    # (see README "Known Data Limitations"), so team KPIs are reported against
    # the routing policy owner — the specialist the bug would actually go to.
    if kb:
        role_by_category = {c: e.get('assigned_role', 'Full-Stack Developer')
                            for c, e in kb.items()}
        team = df['bug_category'].map(role_by_category).fillna('Full-Stack Developer')
        df['assigned_team'] = np.where(
            df['bug_domain'].isin(DOMAIN_OVERRIDE),
            df['bug_domain'].map(DOMAIN_OVERRIDE).fillna(''), team)
    else:
        df['assigned_team'] = df.get('developer_role', 'Unassigned')

    return df, origin, snapshot


def sla_targets(catalog):
    targets = dict(DEFAULT_SLA_DAYS)
    targets.update(catalog.get('sla_target_days', {}) or {})
    return targets


def module_kloc(catalog, df):
    """Module -> size in KLOC, falling back to a flat 1.0 when unknown."""
    modules = (catalog.get('modules') or {})
    sizes = {name: float(entry.get('size_kloc', 1.0))
             for name, entry in modules.items()}
    for name in df['module'].dropna().unique():
        sizes.setdefault(str(name), 1.0)
    return sizes


def max_sprint_index(df):
    """Highest 0-based sprint index present in the 'SPR-nn' labels."""
    numbers = df['sprint'].astype(str).str.extract(r'(\d+)')[0].dropna().astype(int)
    return int(numbers.max()) - 1 if len(numbers) else 0


# ──────────────────────────────────────────────────────────────────────────────
#  KPI computation
# ──────────────────────────────────────────────────────────────────────────────
def ordered_levels(series, preferred):
    """Preferred order first, then anything else the data actually contains."""
    present = set(series.dropna().unique())
    ordered = [v for v in preferred if v in present]
    ordered += sorted(str(v) for v in present if v not in preferred)
    return ordered


def compute_kpis(df, catalog, snapshot):
    targets = sla_targets(catalog)
    kloc    = module_kloc(catalog, df)

    total   = len(df)
    closed  = df[df['is_closed']]
    open_df = df[df['is_open']]

    met_sla   = closed['resolution_days'] <= closed['priority'].map(targets)
    open_late = open_df['open_age_days'] > open_df['priority'].map(targets)

    total_kloc = sum(kloc.get(m, 1.0) for m in df['module'].dropna().unique())

    headline = {
        'total_bugs':          int(total),
        'closed_bugs':         int(len(closed)),
        'open_bugs':           int(len(open_df)),
        'close_rate_pct':      round(len(closed) / total * 100, 1) if total else 0.0,
        'avg_resolution_days': round(float(closed['resolution_days'].mean()), 1) if len(closed) else 0.0,
        'median_resolution_days': round(float(closed['resolution_days'].median()), 1) if len(closed) else 0.0,
        'p90_resolution_days': round(float(closed['resolution_days'].quantile(0.90)), 1) if len(closed) else 0.0,
        'sla_compliance_pct':  round(float(met_sla.mean()) * 100, 1) if len(closed) else 0.0,
        'open_past_sla':       int(open_late.sum()),
        'avg_open_age_days':   round(float(open_df['open_age_days'].mean()), 1) if len(open_df) else 0.0,
        'defect_density':      round(total / total_kloc, 1) if total_kloc else 0.0,
        'total_kloc':          round(total_kloc, 1),
        'reopen_rate_pct':     round(float((df['status'] == 'Reopened').mean()) * 100, 2),
        'snapshot_date':       snapshot.strftime('%Y-%m-%d'),
    }

    # ── Sprint burn: opened vs closed vs carried-over backlog ─────────────────
    sprint_labels = sorted(df['sprint'].unique())
    opened = df.groupby('sprint').size().reindex(sprint_labels, fill_value=0)
    closed_per_sprint = (closed.groupby(closed['close_sprint_index']
                                        .map(lambda i: f"SPR-{int(i) + 1:02d}"))
                         .size().reindex(sprint_labels, fill_value=0))
    backlog = (opened - closed_per_sprint).cumsum()
    sprint_avg_days = (closed.groupby('sprint')['resolution_days'].mean()
                       .reindex(sprint_labels).round(1))

    by_sprint = [{
        'sprint':   s,
        'opened':   int(opened[s]),
        'closed':   int(closed_per_sprint[s]),
        'backlog':  int(backlog[s]),
        'avg_days': None if pd.isna(sprint_avg_days[s]) else float(sprint_avg_days[s]),
    } for s in sprint_labels]

    # ── Module quality ───────────────────────────────────────────────────────
    by_module = []
    for module, group in df.groupby('module'):
        g_closed = group[group['is_closed']]
        size = kloc.get(module, 1.0)
        by_module.append({
            'module':         module,
            'bugs':           int(len(group)),
            'closed':         int(len(g_closed)),
            'open':           int(len(group) - len(g_closed)),
            'critical':       int((group['severity'] == 'Critical').sum()),
            'urgent':         int(group['priority'].isin(['P1', 'P2']).sum()),
            'size_kloc':      round(size, 1),
            'defect_density': round(len(group) / size, 1) if size else 0.0,
            'avg_days':       round(float(g_closed['resolution_days'].mean()), 1) if len(g_closed) else None,
            'close_rate_pct': round(len(g_closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    by_module.sort(key=lambda r: -r['defect_density'])

    # ── Release quality ──────────────────────────────────────────────────────
    by_release = []
    for release, group in df.groupby('release_version'):
        g_closed = group[group['is_closed']]
        by_release.append({
            'release':        release,
            'bugs':           int(len(group)),
            'closed':         int(len(g_closed)),
            'open':           int(len(group) - len(g_closed)),
            'urgent':         int(group['priority'].isin(['P1', 'P2']).sum()),
            'urgent_open':    int((group['is_open'] & group['priority'].isin(['P1', 'P2'])).sum()),
            'avg_days':       round(float(g_closed['resolution_days'].mean()), 1) if len(g_closed) else None,
            'close_rate_pct': round(len(g_closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    by_release.sort(key=lambda r: r['release'])

    # ── Priority: volume, speed, SLA ─────────────────────────────────────────
    by_priority = []
    for priority in ordered_levels(df['priority'], PRIORITY_ORDER):
        group = df[df['priority'] == priority]
        g_closed = group[group['is_closed']]
        within = (g_closed['resolution_days'] <= targets.get(priority, 30)).mean() \
            if len(g_closed) else 0.0
        by_priority.append({
            'priority':       priority,
            'bugs':           int(len(group)),
            'closed':         int(len(g_closed)),
            'sla_target_days': targets.get(priority),
            'avg_days':       round(float(g_closed['resolution_days'].mean()), 1) if len(g_closed) else None,
            'median_days':    round(float(g_closed['resolution_days'].median()), 1) if len(g_closed) else None,
            'sla_pct':        round(float(within) * 100, 1),
        })

    # ── Root cause ───────────────────────────────────────────────────────────
    by_root_cause = []
    for cause, group in df.groupby('root_cause'):
        g_closed = group[group['is_closed']]
        by_root_cause.append({
            'root_cause': cause,
            'bugs':       int(len(group)),
            'share_pct':  round(len(group) / total * 100, 1),
            'avg_days':   round(float(g_closed['resolution_days'].mean()), 1) if len(g_closed) else None,
            'urgent':     int(group['priority'].isin(['P1', 'P2']).sum()),
        })
    by_root_cause.sort(key=lambda r: -r['bugs'])

    # ── Team performance (routed owner, not the random raw column) ───────────
    by_team = []
    for team, group in df.groupby('assigned_team'):
        g_closed = group[group['is_closed']]
        met = (g_closed['resolution_days'] <= g_closed['priority'].map(targets)).mean() \
            if len(g_closed) else 0.0
        by_team.append({
            'team':           team,
            'assigned':       int(len(group)),
            'closed':         int(len(g_closed)),
            'open':           int(len(group) - len(g_closed)),
            'urgent_open':    int((group['is_open'] & group['priority'].isin(['P1', 'P2'])).sum()),
            'avg_days':       round(float(g_closed['resolution_days'].mean()), 1) if len(g_closed) else None,
            'sla_pct':        round(float(met) * 100, 1),
            'close_rate_pct': round(len(g_closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    by_team.sort(key=lambda r: -r['assigned'])

    # ── Monthly trend (bug reporting vs closure over time) ────────────────────
    months = sorted(set(df['month'].dropna()) | set(df['close_month'].dropna()))
    opened_m = df.groupby('month').size().reindex(months, fill_value=0)
    closed_m = closed.groupby('close_month').size().reindex(months, fill_value=0)
    by_month = [{'month': m, 'opened': int(opened_m[m]), 'closed': int(closed_m[m])}
                for m in months]

    # ── Simple distributions ─────────────────────────────────────────────────
    def distribution(column, preferred=None):
        counts = df[column].value_counts()
        order = ordered_levels(df[column], preferred or []) if preferred \
            else counts.index.tolist()
        return [{'label': str(k), 'value': int(counts.get(k, 0))} for k in order]

    return {
        'generated_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'headline':      headline,
        'sla_targets':   targets,
        'by_sprint':     by_sprint,
        'by_month':      by_month,
        'by_module':     by_module,
        'by_release':    by_release,
        'by_priority':   by_priority,
        'by_root_cause': by_root_cause,
        'by_team':       by_team,
        'status':        distribution('status', STATUS_ORDER),
        'severity':      distribution('severity', SEVERITY_ORDER),
        'resolution':    distribution('resolution'),
        'feature':       distribution('feature'),
        'component':     distribution('component'),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Actionable insights
# ──────────────────────────────────────────────────────────────────────────────
def build_insights(kpis):
    """Data-driven recommendations — every number below comes from the KPIs."""
    head = kpis['headline']
    insights = []

    worst_density = kpis['by_module'][0]
    best_density  = kpis['by_module'][-1]
    insights.append((
        'Concentrate hardening effort on ' + worst_density['module'],
        f"Raw bug counts barely separate the modules, but size does: "
        f"{worst_density['module']} packs {worst_density['bugs']:,} defects into only "
        f"{worst_density['size_kloc']} KLOC — {worst_density['defect_density']} per KLOC, "
        f"{worst_density['defect_density'] / max(best_density['defect_density'], 0.1):.1f}x "
        f"the density of {best_density['module']} ({best_density['defect_density']} per KLOC "
        f"over {best_density['size_kloc']} KLOC), with {worst_density['open']:,} still open. "
        f"Per line of code touched, a refactor and test-coverage push there removes more "
        f"defect mass than the same effort spread across all six modules."))

    worst_sla = min(kpis['by_priority'], key=lambda r: r['sla_pct'])
    insights.append((
        f"{worst_sla['priority']} is the weakest link in the SLA",
        f"Only {worst_sla['sla_pct']}% of closed {worst_sla['priority']} bugs met their "
        f"{worst_sla['sla_target_days']}-day target (average {worst_sla['avg_days']} days, "
        f"median {worst_sla['median_days']}). Overall SLA compliance is "
        f"{head['sla_compliance_pct']}%. Either the {worst_sla['priority']} target is "
        f"unrealistic for the work it covers, or triage is under-staffing that queue — "
        f"re-baseline the target or reserve sprint capacity for it."))

    late = head['open_past_sla']
    insights.append((
        'Clear the aged backlog before it becomes technical debt',
        f"{late:,} of the {head['open_bugs']:,} open bugs ({late / max(head['open_bugs'], 1) * 100:.1f}%) "
        f"have already passed their SLA target, and the average open bug is "
        f"{head['avg_open_age_days']} days old. Run an explicit aging sweep each sprint: "
        f"close, defer with a reason, or escalate — bugs that age silently distort "
        f"every downstream estimate."))

    growth = [r for r in kpis['by_sprint'] if r['opened'] > r['closed']]
    worst_sprint = max(kpis['by_sprint'], key=lambda r: r['opened'] - r['closed'])
    last_backlog = kpis['by_sprint'][-1]['backlog']
    insights.append((
        'Intake outruns closure — cap work-in-progress',
        f"The backlog grew in {len(growth)} of {len(kpis['by_sprint'])} sprints and ends at "
        f"{last_backlog:,} open bugs. {worst_sprint['sprint']} was the worst "
        f"({worst_sprint['opened']:,} opened vs {worst_sprint['closed']:,} closed). "
        f"Holding a fixed closure quota per sprint — rather than a fixed intake — is the "
        f"only way this curve flattens."))

    top_cause = kpis['by_root_cause'][0]
    top3 = kpis['by_root_cause'][:3]
    top3_share = sum(r['share_pct'] for r in top3)
    insights.append((
        'Three root causes drive a third of all defects',
        f"'{top_cause['root_cause']}' alone accounts for {top_cause['bugs']:,} bugs "
        f"({top_cause['share_pct']}%), and the top three causes together account for "
        f"{top3_share:.1f}%. These are systemic, not incidental: a single guard rail "
        f"(lint rule, contract test, review checklist) against each removes recurring "
        f"work from every future sprint."))

    # Ranked on the absolute urgent queue rather than the closure rate: closure
    # is drawn independently of release, so rate differences are within noise
    # while the size of the unresolved P1/P2 queue is a real operational load.
    worst_release = max(kpis['by_release'], key=lambda r: r['urgent_open'])
    insights.append((
        f"Release {worst_release['release']} carries the largest urgent queue",
        f"{worst_release['release']} still has {worst_release['urgent_open']:,} unresolved "
        f"P1/P2 bugs, out of {worst_release['open']:,} open and {worst_release['bugs']:,} "
        f"total. Closure rates sit near {worst_release['close_rate_pct']}% across every "
        f"release, so the differentiator is volume, not discipline. Gate the next release "
        f"on the urgent queue being drained rather than on the calendar."))

    teams = kpis['by_team']
    busiest = max(teams, key=lambda r: r['assigned'])
    share = busiest['assigned'] / max(head['total_bugs'], 1) * 100
    lightest = min(teams, key=lambda r: r['assigned'])
    insights.append((
        'The routing policy concentrates a quarter of all work on one role',
        f"{busiest['team']} is routed {busiest['assigned']:,} bugs ({share:.1f}% of "
        f"everything) and is sitting on {busiest['urgent_open']:,} open P1/P2, while "
        f"{lightest['team']} receives {lightest['assigned']:,}. Turnaround is nearly "
        f"identical across all {len(teams)} roles ({min(r['avg_days'] for r in teams if r['avg_days']):.1f}"
        f"–{max(r['avg_days'] for r in teams if r['avg_days']):.1f} days), so the queue length "
        f"— not individual speed — is what delays urgent fixes. Re-route one or two bug "
        f"categories, or add capacity where the policy points."))

    return [{'title': t, 'detail': d} for t, d in insights]


# ──────────────────────────────────────────────────────────────────────────────
#  Console KPI report
# ──────────────────────────────────────────────────────────────────────────────
def print_kpi_report(kpis, insights, top_n):
    head = kpis['headline']

    print("\n" + "=" * 68)
    print("  KPI REPORT")
    print("=" * 68)
    print(f"  Snapshot date : {head['snapshot_date']}")

    print("\n  [A] HEADLINE KPIs")
    print(f"  {RULE}")
    rows = [
        ('Total bugs',                f"{head['total_bugs']:,}"),
        ('Closed bugs',               f"{head['closed_bugs']:,}  ({head['close_rate_pct']}%)"),
        ('Open bugs',                 f"{head['open_bugs']:,}"),
        ('Avg resolution time',       f"{head['avg_resolution_days']} days"),
        ('Median resolution time',    f"{head['median_resolution_days']} days"),
        ('90th pct resolution time',  f"{head['p90_resolution_days']} days"),
        ('SLA compliance (closed)',   f"{head['sla_compliance_pct']}%"),
        ('Open bugs past SLA',        f"{head['open_past_sla']:,}"),
        ('Avg age of an open bug',    f"{head['avg_open_age_days']} days"),
        ('Defect density',            f"{head['defect_density']} bugs / KLOC "
                                      f"over {head['total_kloc']} KLOC"),
        ('Reopen rate',               f"{head['reopen_rate_pct']}%"),
    ]
    for label, value in rows:
        print(f"    {label:<28}{value}")

    print("\n  [B] MODULE-WISE QUALITY (sorted by defect density)")
    print(f"  {RULE}")
    print(f"    {'Module':<22}{'Bugs':>8}{'Open':>8}{'KLOC':>8}{'/KLOC':>8}{'Avg d':>8}{'Closed%':>9}")
    for r in kpis['by_module']:
        avg = '-' if r['avg_days'] is None else f"{r['avg_days']:.1f}"
        print(f"    {r['module']:<22}{r['bugs']:>8,}{r['open']:>8,}{r['size_kloc']:>8.1f}"
              f"{r['defect_density']:>8.1f}{avg:>8}{r['close_rate_pct']:>9.1f}")

    print("\n  [C] PRIORITY-WISE RESOLUTION TIME vs SLA")
    print(f"  {RULE}")
    print(f"    {'Priority':<10}{'Bugs':>9}{'Closed':>9}{'Target':>9}{'Avg d':>9}{'Median':>9}{'SLA %':>9}")
    for r in kpis['by_priority']:
        avg = '-' if r['avg_days'] is None else f"{r['avg_days']:.1f}"
        med = '-' if r['median_days'] is None else f"{r['median_days']:.1f}"
        print(f"    {r['priority']:<10}{r['bugs']:>9,}{r['closed']:>9,}"
              f"{r['sla_target_days']:>9}{avg:>9}{med:>9}{r['sla_pct']:>9.1f}")

    print("\n  [D] TEAM PERFORMANCE (by routed owner)")
    print(f"  {RULE}")
    print(f"    {'Team':<24}{'Assigned':>10}{'Closed':>9}{'Open':>8}{'Avg d':>8}{'SLA %':>8}")
    for r in kpis['by_team']:
        avg = '-' if r['avg_days'] is None else f"{r['avg_days']:.1f}"
        print(f"    {r['team']:<24}{r['assigned']:>10,}{r['closed']:>9,}"
              f"{r['open']:>8,}{avg:>8}{r['sla_pct']:>8.1f}")
    print(f"\n    [NOTE] The dataset's own 'developer_role' column is uniformly")
    print(f"           random, so these rows use the documented routing policy")
    print(f"           owner (category -> specialist) instead.")

    print("\n  [E] RELEASE-WISE QUALITY")
    print(f"  {RULE}")
    print(f"    {'Release':<10}{'Bugs':>9}{'Closed':>9}{'Open':>8}{'P1/P2':>8}{'Avg d':>8}{'Closed%':>9}")
    for r in kpis['by_release']:
        avg = '-' if r['avg_days'] is None else f"{r['avg_days']:.1f}"
        print(f"    {r['release']:<10}{r['bugs']:>9,}{r['closed']:>9,}{r['open']:>8,}"
              f"{r['urgent']:>8,}{avg:>8}{r['close_rate_pct']:>9.1f}")

    print(f"\n  [F] TOP {top_n} ROOT CAUSES")
    print(f"  {RULE}")
    print(f"    {'Root cause':<52}{'Bugs':>8}{'Share':>8}{'Avg d':>8}")
    for r in kpis['by_root_cause'][:top_n]:
        avg = '-' if r['avg_days'] is None else f"{r['avg_days']:.1f}"
        print(f"    {shorten(r['root_cause'], 50):<52}{r['bugs']:>8,}"
              f"{r['share_pct']:>7.1f}%{avg:>8}")

    print("\n" + "=" * 68)
    print("  ACTIONABLE INSIGHTS")
    print("=" * 68)
    for i, item in enumerate(insights, 1):
        print(f"\n  {i}. {item['title']}")
        for line in wrap(item['detail'], 64):
            print(f"     {line}")
    print()


def shorten(text, width):
    """Trim the middle, not the tail.

    Root causes in this dataset all start 'Misconfiguration or logic issue
    related to ...' — cutting the end would leave every row looking identical.
    """
    text = str(text)
    if len(text) <= width:
        return text
    head = (width - 3) * 2 // 5
    return text[:head] + '...' + text[-(width - 3 - head):]


def wrap(text, width):
    words, lines, current = text.split(), [], ''
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ──────────────────────────────────────────────────────────────────────────────
#  Model scoring — the build-time half of the model pipeline
# ──────────────────────────────────────────────────────────────────────────────
def score_with_models(df):
    """Re-score every bug with the saved priority model, and recover which rows
    05_modeling.py trained on. Returns None when the models aren't available —
    the dashboard degrades to its non-model views rather than failing."""
    bundle = model_bridge.load(quiet=True)
    if not bundle.ready:
        for err in bundle.errors:
            print(f"    [SKIP] {err}")
        return None

    print(f"    Model        : {type(bundle.priority_model).__name__} "
          f"(models/best_priority_model.pkl)")

    def progress(done, total):
        pct = done / total * 100
        print(f"\r    Scoring      : {done:,}/{total:,} rows ({pct:.0f}%)", end='', flush=True)

    try:
        labels, conf = model_bridge.score_dataset(bundle, df, progress=progress)
    except Exception as exc:                                       # noqa: BLE001
        print(f"\r    [SKIP] Scoring failed: {exc.__class__.__name__}: {exc}")
        return None
    print()

    split = model_bridge.training_split(df)
    n_train = int((split == 1).sum())
    n_test  = int((split == 2).sum())
    print(f"    Split recall : {n_train:,} training rows, {n_test:,} held-out test rows, "
          f"{len(df) - n_train - n_test:,} never sampled")

    agree_all  = float((labels == df['priority'].to_numpy()).mean() * 100)
    test_mask  = split == 2
    agree_test = float((labels[test_mask] == df['priority'].to_numpy()[test_mask]).mean() * 100) \
        if test_mask.any() else None
    print(f"    Agreement    : {agree_all:.1f}% over all rows"
          + (f", {agree_test:.1f}% on the held-out test rows" if agree_test is not None else ""))

    return {
        'labels': labels,
        'confidence': conf,
        'split': split,
        'model_name': type(bundle.priority_model).__name__,
        'agreement_all': round(agree_all, 1),
        'agreement_test': None if agree_test is None else round(agree_test, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Embedded payload
#
#  The browser gets one byte per categorical value per bug, base64'd. 50k rows
#  across ~20 columns is about 1 MB of binary — small enough to inline, and it
#  buys genuine cross-filtering: every recompute is a single linear scan over
#  typed arrays rather than a round trip to a server that isn't there.
# ──────────────────────────────────────────────────────────────────────────────
def enc_u8(values):
    return base64.b64encode(np.asarray(values, dtype=np.uint8).tobytes()).decode('ascii')


def enc_u16(values):
    return base64.b64encode(np.asarray(values, dtype='<u2').tobytes()).decode('ascii')


def enc_i16(values):
    return base64.b64encode(np.asarray(values, dtype='<i2').tobytes()).decode('ascii')


def enc_u32(values):
    return base64.b64encode(np.asarray(values, dtype='<u4').tobytes()).decode('ascii')


def codes(series, order):
    """Categorical codes against a fixed level order.

    Unknown/missing values fall back to level 0 rather than -1: a negative code
    would index past the front of the JS label array and render as 'undefined'.
    """
    cat = pd.Categorical(series.astype(str), categories=[str(v) for v in order])
    arr = cat.codes.astype(np.int16)
    if (arr < 0).any():
        arr = np.where(arr < 0, 0, arr)
    return arr.astype(np.uint8)


def levels(series, preferred=None):
    """Fixed level order for one column: preferred first, then the rest sorted."""
    present = sorted({str(v) for v in series.dropna().unique()})
    if not preferred:
        return present
    ordered = [str(v) for v in preferred if str(v) in present]
    return ordered + [v for v in present if v not in ordered]


VERDICTS = [
    {
        'target': 'Bug Category',
        'status': 'critical',
        'label': 'Leakage, not a result',
        'narrative': [
            'Every model scores near-perfect accuracy on Bug Category — and that is not generalization. ',
            {'code': 'title'}, ', ', {'code': 'description'}, ', ', {'code': 'root_cause'}, ' and ',
            {'code': 'suggested_fix'},
            ' are boilerplate templates: only 16 unique strings across the whole dataset, one fixed '
            'template per category, so the model is matching a copied template back to the label it was '
            'copied from. Treat this target as unlearnable until the source data carries genuine free text.',
        ],
    },
    {
        'target': 'Severity',
        'status': 'critical',
        'label': 'No predictive signal',
        'narrative': [
            'Every model lands at chance level for four classes (~25%). Severity was tested against '
            'bug_category, bug_domain, environment, error_code, developer_role and the description text '
            'itself: none carries usable signal, and every split lands within a point or two of uniform. '
            'Severity appears to be assigned independently at random in the source data — ',
            {'strong': 'no model can beat chance on it from this dataset'},
            ', text-based or feature-based.',
        ],
    },
    {
        'target': 'Priority',
        'status': 'good',
        'label': 'Learnable — with a caveat',
        'narrative': [
            'Unlike the other two targets, the five models genuinely spread out here, and Random Forest '
            'separates from the rest — a real, learnable relationship. The caveat: priority is a ',
            {'strong': 'derived field'},
            ', computed from the structured features by a documented scoring rule plus ~8% seeded jitter, '
            'so the models recover that rule rather than predicting real-world triage priority. Read it as '
            'proof the pipeline trains and ranks models correctly on a learnable target — not as evidence '
            'priority is predictable from the raw Kaggle data, which has no priority field at all.',
        ],
    },
]


def build_payload(df, kpis, catalog, kb, model_eval, scores, origin, snapshot):
    targets = sla_targets(catalog)
    kloc    = module_kloc(catalog, df)

    dim_specs = [
        ('module',      'module',          None),
        ('feature',     'feature',         None),
        ('component',   'component',       None),
        ('priority',    'priority',        PRIORITY_ORDER),
        ('severity',    'severity',        SEVERITY_ORDER),
        ('status',      'status',          STATUS_ORDER),
        ('resolution',  'resolution',      RESOLUTION_ORDER),
        ('sprint',      'sprint',          None),
        ('release',     'release_version', None),
        ('category',    'bug_category',    None),
        ('domain',      'bug_domain',      None),
        ('environment', 'environment',     ENVIRONMENT_ORDER),
        ('tech',        'tech_stack',      None),
        ('errorCode',   'error_code',      None),
        ('team',        'assigned_team',   None),
        ('lifecycle',   'lifecycle_stage', LIFECYCLE_ORDER),
        ('month',       'month',           None),
    ]

    dims, cols = {}, {}
    for name, column, preferred in dim_specs:
        if column == 'error_code':
            series = df[column].fillna(0).astype(float).astype(int).astype(str)
        else:
            series = df[column].astype(str)
        order = levels(series, preferred)
        dims[name] = order
        cols[name] = enc_u8(codes(series, order))

    # Closure position: which sprint / month the bug actually closed in.
    close_sprint = (df['close_sprint_index'].fillna(0).astype(int)
                    .clip(0, len(dims['sprint']) - 1))
    cols['closeSprint'] = enc_u8(close_sprint)

    month_index = {label: i for i, label in enumerate(dims['month'])}
    close_month = df['close_month'].map(month_index).fillna(255).astype(int).clip(0, 255)
    cols['closeMonth'] = enc_u8(close_month)

    # Predictions. Absent models leave the columns flat and modelScored false,
    # so the model view can say so rather than showing a fake diagonal.
    if scores is not None:
        pri_index = {label: i for i, label in enumerate(dims['priority'])}
        cols['predPriority'] = enc_u8(pd.Series(scores['labels']).map(pri_index).fillna(0).astype(int))
        conf = np.nan_to_num(scores['confidence'], nan=0.0)
        cols['predConf'] = enc_u8(np.clip(np.round(conf * 100), 0, 100).astype(int))
        cols['split'] = enc_u8(scores['split'])
    else:
        zeros = np.zeros(len(df), dtype=np.uint8)
        cols['predPriority'] = enc_u8(zeros)
        cols['predConf'] = enc_u8(zeros)
        cols['split'] = enc_u8(zeros)

    res_days = df['resolution_days'].fillna(-1).astype(float).clip(-1, 32000).astype(int)

    # Bug IDs are BUG_nnnnnn. When they run 1..N in row order — which they do
    # for an untouched pipeline — the page rebuilds them from the row index
    # instead of carrying a 200 KB id column.
    numeric_ids = (df['bug_id'].astype(str).str.extract(r'(\d+)')[0]
                   .astype('Int64').fillna(0).astype(int).to_numpy())
    sequential = bool(np.array_equal(numeric_ids, np.arange(1, len(df) + 1)))
    ids = {'seq': True} if sequential else {'seq': False, 'data': enc_u32(numeric_ids)}

    # Per-category lookups: 16 templates for the whole dataset, so the page
    # stores 16 strings and an index rather than 50,000 repeated ones.
    def by_category(field, kb_key):
        out = []
        for cat in dims['category']:
            entry = (kb or {}).get(cat, {})
            value = entry.get(kb_key)
            if not value:
                sub = df.loc[df['bug_category'].astype(str) == cat, field]
                value = str(sub.iloc[0]) if len(sub) else cat
            out.append(str(value))
        return out

    routing = {cat: (kb or {}).get(cat, {}).get('assigned_role', 'Full-Stack Developer')
               for cat in dims['category']}

    payload = {
        'meta': {
            'generated':  kpis['generated_at'],
            'snapshot':   kpis['headline']['snapshot_date'],
            'origin':     origin.strftime('%Y-%m-%d'),
            'originMs':   int(pd.Timestamp(origin).timestamp() * 1000),
            'snapshotDay': int((snapshot - origin).days),
            'rows':       int(len(df)),
            'sprintDays': SPRINT_LENGTH_DAYS,
            # The reporting window starts and ends mid-month, so the first and
            # last points of a monthly series cover fewer days than the rest.
            # Flagged rather than dropped, so the chart can say why they dip.
            'monthFirstPartial': bool(pd.Timestamp(origin).day > 1),
            'monthLastPartial':  bool(pd.Timestamp(snapshot).day
                                      < pd.Timestamp(snapshot).days_in_month),
            'monthFirst': dims['month'][0] if dims['month'] else None,
            'monthLast':  dims['month'][-1] if dims['month'] else None,
            'modelScored': scores is not None,
            'modelName':  scores['model_name'] if scores else None,
            'agreementAll':  scores['agreement_all'] if scores else None,
            'agreementTest': scores['agreement_test'] if scores else None,
        },
        'dims': dims,
        'cols': cols,
        'u16':  {'createdDay': enc_u16(df['created_day'].clip(0, 65535))},
        'i16':  {'resDays': enc_i16(res_days)},
        'ids':  ids,
        'lookups': {
            'slaByPriority':       [int(targets.get(p, 30)) for p in dims['priority']],
            'klocByModule':        [round(float(kloc.get(m, 1.0)), 1) for m in dims['module']],
            'rootCauseByCategory': by_category('root_cause', 'root_cause'),
            'fixByCategory':       by_category('suggested_fix', 'suggested_fix'),
            'titleByCategory':     by_category('title', 'sample_title'),
        },
        'kpis':      kpis,
        'insights':  kpis['insights'],
        'modelEval': model_eval,
        'modelOrder': MODEL_ORDER,
        'verdicts':  [v for v in VERDICTS if model_eval.get(v['target'])],
        'routing':   {'by_category': routing, 'domain_override': DOMAIN_OVERRIDE},
        'priorityNote': model_bridge.PRIORITY_NOTE,
        'severityNote': model_bridge.SEVERITY_NOTE,
    }
    return payload


def read_asset(name):
    """Load one of the front-end files that get inlined into the dashboard."""
    path = os.path.join(SRC_DIR, 'dashboard', name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dashboard asset '{name}' not found at {path}. The src/dashboard/ "
            f"folder ships index_template.html, theme.css and app.js.")
    with open(path, encoding='utf-8') as f:
        return f.read()


def build_dashboard(payload):
    os.makedirs('dashboards', exist_ok=True)

    # '<' is escaped so a label can never terminate the <script> block early.
    data = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')

    html = (read_asset('index_template.html')
            .replace('/*__CSS__*/', read_asset('theme.css'))
            .replace('/*__JS__*/', read_asset('app.js'))
            .replace('__SNAPSHOT__', payload['meta']['snapshot'])
            .replace('__TOTAL__', f"{payload['meta']['rows']:,}")
            .replace('__DATA__', data))

    with open(DASHBOARD_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    return DASHBOARD_PATH, len(data)


# ──────────────────────────────────────────────────────────────────────────────
#  Static companion charts (the printable/report versions)
# ──────────────────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = f"visualizations/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def chart_sprint_trend(kpis):
    """Intake vs closure, with the backlog in its own panel below.

    Two measures on two y-scales in one plot invents a correlation the data
    doesn't contain, so the backlog gets a stacked panel on a shared x-axis
    instead of a secondary axis.
    """
    rows = kpis['by_sprint']
    labels  = [r['sprint'] for r in rows]
    opened  = [r['opened'] for r in rows]
    closed  = [r['closed'] for r in rows]
    backlog = [r['backlog'] for r in rows]
    x = np.arange(len(labels))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 7.5), sharex=True,
        gridspec_kw={'height_ratios': [2.1, 1], 'hspace': 0.12})

    ax.bar(x - 0.21, opened, width=0.4, label='Opened', color='#2a78d6')
    ax.bar(x + 0.21, closed, width=0.4, label='Closed', color='#1baf7a')
    ax.set_ylabel('Bugs in sprint', fontsize=12)
    ax.set_title('Sprint-wise Bug Intake vs Closure, with Carried-over Backlog',
                 fontsize=15, fontweight='bold', pad=34)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.005), ncol=2, frameon=False)

    ax2.fill_between(x, backlog, color='#4a3aa7', alpha=0.10)
    ax2.plot(x, backlog, color='#4a3aa7', linewidth=2, marker='o', markersize=3.5)
    ax2.set_ylabel('Open backlog', fontsize=12)
    ax2.set_xlabel('Sprint', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.annotate(f"{backlog[-1]:,}", xy=(x[-1], backlog[-1]),
                 xytext=(-6, 8), textcoords='offset points',
                 ha='right', fontsize=10, fontweight='bold')

    return save_fig(fig, 'kpi_sprint_trend')


def chart_module_priority_heatmap(df):
    pivot = pd.crosstab(df['module'], df['priority'])
    pivot = pivot.reindex(columns=[p for p in PRIORITY_ORDER if p in pivot.columns])
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(pivot, annot=True, fmt=',d', cmap='Blues', linewidths=2,
                linecolor='white', cbar_kws={'label': 'Bugs'}, ax=ax,
                annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    ax.set_title('Bug Distribution — Module x Priority', fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xlabel('Priority', fontsize=12)
    ax.set_ylabel('Module', fontsize=12)
    fig.tight_layout()
    return save_fig(fig, 'kpi_module_priority_heatmap')


def chart_resolution_time(kpis):
    rows = kpis['by_priority']
    labels  = [r['priority'] for r in rows]
    avg     = [r['avg_days'] or 0 for r in rows]
    targets = [r['sla_target_days'] for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg, width=0.5, color='#2a78d6', label='Actual average')

    for xi, target in zip(x, targets):
        ax.hlines(target, xi - 0.3, xi + 0.3, colors='#52514e',
                  linestyles='--', linewidth=2)

    # The SLA marker sits above the bar whenever the team is beating its target,
    # so the value label clears whichever of the two is higher.
    ceiling = max(max(avg), max(targets))
    for bar, value, target in zip(bars, avg, targets):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), target) + ceiling * 0.02,
                f'{value:.1f} d', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.plot([], [], color='#52514e', linestyle='--', linewidth=2, label='SLA target')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Average Resolution Time vs SLA Target, by Priority',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Priority (P1 = highest)', fontsize=12)
    ax.set_ylabel('Days to close', fontsize=12)
    ax.set_ylim(0, ceiling * 1.2)
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, 'kpi_resolution_time_by_priority')


def chart_defect_density(kpis):
    rows = sorted(kpis['by_module'], key=lambda r: r['defect_density'])
    labels  = [r['module'] for r in rows]
    density = [r['defect_density'] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, density, color='#2a78d6')
    for bar, row in zip(bars, rows):
        ax.text(bar.get_width() + max(density) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{row['defect_density']:.1f}   ({row['bugs']:,} bugs / {row['size_kloc']:.1f} KLOC)",
                va='center', fontsize=9, fontweight='bold')
    ax.set_title('Defect Density by Module (bugs per KLOC)', fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xlabel('Bugs per KLOC', fontsize=12)
    ax.set_xlim(0, max(density) * 1.45)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, 'kpi_defect_density_by_module')


def chart_release_quality(kpis):
    rows = kpis['by_release']
    labels = [r['release'] for r in rows]
    closed = [r['closed'] for r in rows]
    still_open = [r['open'] for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, closed, width=0.58, label='Closed', color='#1baf7a')
    ax.bar(x, still_open, width=0.58, bottom=closed, label='Still open',
           color='#e34948', edgecolor='white', linewidth=2)
    for xi, row in zip(x, rows):
        ax.text(xi, row['bugs'] + max(r['bugs'] for r in rows) * 0.015,
                f"{row['close_rate_pct']:.0f}% closed", ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Release-wise Bug Load and Closure', fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xlabel('Release version', fontsize=12)
    ax.set_ylabel('Number of bugs', fontsize=12)
    ax.set_ylim(0, max(r['bugs'] for r in rows) * 1.18)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, 'kpi_release_quality')


def build_static_charts(df, kpis):
    sns.set_theme(style='whitegrid', font_scale=1.05)
    os.makedirs('visualizations', exist_ok=True)
    return [
        chart_sprint_trend(kpis),
        chart_module_priority_heatmap(df),
        chart_resolution_time(kpis),
        chart_defect_density(kpis),
        chart_release_quality(kpis),
    ]


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build the interactive dashboard, the KPI report and the companion charts")
    parser.add_argument('--no-open', dest='open_dashboard', action='store_false',
                        help="Build without opening the dashboard in a browser")
    parser.add_argument('--no-model', dest='use_models', action='store_false',
                        help="Skip model scoring — faster, but the model view loses its predictions")
    parser.add_argument('--sample', type=int, default=0,
                        help="Embed only N rows in the page (0 = all). Shrinks the file for sharing.")
    parser.add_argument('--top', type=int, default=10,
                        help="How many rows the 'top N' console tables show")
    parser.set_defaults(open_dashboard=True, use_models=True)
    args = parser.parse_args()

    print("=" * 68)
    print("  TASK 9: Interactive Dashboard & KPI Reporting")
    print("=" * 68)

    df, catalog, kb, model_eval = load_inputs()
    if df is None:
        return 1

    print(f"\n  Loaded '{PROCESSED_PATH}'  |  {len(df):,} bug records")

    df, origin, snapshot = prepare(df, kb)
    print(f"  Reporting window : {origin.date()} to {snapshot.date()} "
          f"({(snapshot - origin).days} days, {df['sprint'].nunique()} sprints)")
    print(f"  Dimensions       : {df['release_version'].nunique()} releases, "
          f"{df['module'].nunique()} modules, {df['feature'].nunique()} features, "
          f"{df['component'].nunique()} components")

    kpis = compute_kpis(df, catalog, snapshot)
    insights = build_insights(kpis)
    kpis['insights'] = insights

    print_kpi_report(kpis, insights, args.top)

    print("=" * 68)
    print("  BUILDING OUTPUTS")
    print("=" * 68)

    os.makedirs('data', exist_ok=True)
    with open(KPI_PATH, 'w', encoding='utf-8') as f:
        json.dump(kpis, f, indent=4)
    print(f"\n  KPI report (machine-readable):")
    print(f"    Saved: {KPI_PATH}")

    print(f"\n  Static companion charts:")
    charts = build_static_charts(df, kpis)

    # The embedded table can be trimmed for sharing; the KPIs above always
    # cover the full dataset, so only the interactive slice shrinks.
    embed = df
    if args.sample and args.sample < len(df):
        embed = df.sample(args.sample, random_state=42).sort_index()
        print(f"\n  Embedding a {len(embed):,}-row sample "
              f"(--sample {args.sample}); KPIs still cover all {len(df):,} rows.")

    print(f"\n  Model pipeline:")
    scores = None
    if args.use_models:
        scores = score_with_models(embed)
    else:
        print(f"    Skipped (--no-model).")

    print(f"\n  Interactive dashboard:")
    payload = build_payload(embed, kpis, catalog, kb, model_eval, scores, origin, snapshot)
    path, data_bytes = build_dashboard(payload)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"    Embedded     : {len(embed):,} rows, {len(payload['cols'])} columns "
          f"({data_bytes / (1024 * 1024):.2f} MB of encoded data)")
    print(f"    Saved: {path}  ({size_mb:.2f} MB, fully self-contained)")

    for legacy in LEGACY_PAGES:
        if os.path.exists(legacy):
            os.remove(legacy)
            print(f"    Removed superseded page: {legacy}")

    print("\n" + "=" * 68)
    print("  DASHBOARD STAGE COMPLETE")
    print("=" * 68)
    print(f"  Dashboard : {path}")
    print(f"  KPI JSON  : {KPI_PATH}")
    print(f"  Charts    : {len(charts)} PNGs in visualizations/")
    print(f"  Live mode : python src/09_serve.py   (adds model-backed triage)")
    print("=" * 68)

    if args.open_dashboard:
        print("\n  Opening the dashboard in your browser...")
        try:
            webbrowser.open(f"file:///{os.path.abspath(path).replace(os.sep, '/')}")
        except Exception as e:                                     # noqa: BLE001
            print(f"  [WARN] Could not open a browser automatically: {e}")
            print(f"  Open this file manually: {os.path.abspath(path)}")
        print("  (Run with --no-open to skip this.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
