# ==============================================================================
# dashboard_data.py
# The data layer behind src/streamlit_app.py.
#
# Everything the app reads off disk goes through here, and everything is cached:
#   - load_bugs()      the prepared 50k-row frame (once per process)
#   - load_artifacts() the JSON side-cars written by stages 5-8
#   - apply_filters()  a filtered slice + its KPIs, keyed on the filter values
#   - load_models()    the trained models, loaded lazily on first prediction
#
# The KPI definitions themselves are NOT redefined here. 08_dashboard.py is
# imported by path (its name starts with a digit, so it cannot be a normal
# import) and its prepare/compute_kpis/build_insights are reused verbatim, so
# the app and the printed KPI report can never drift apart.
# ==============================================================================

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(SRC_DIR)

PROCESSED_PATH  = os.path.join(ROOT, 'data', 'bug_reports_processed.csv')
CATALOG_PATH    = os.path.join(ROOT, 'data', 'module_catalog.json')
KB_PATH         = os.path.join(ROOT, 'data', 'bug_knowledge_base.json')
MODEL_EVAL_PATH = os.path.join(ROOT, 'data', 'model_evaluation_results.json')
KPI_PATH        = os.path.join(ROOT, 'data', 'kpi_report.json')
LIFECYCLE_PATH  = os.path.join(ROOT, 'data', 'lifecycle_analysis.json')
DUPLICATES_PATH = os.path.join(ROOT, 'data', 'potential_duplicates.json')
TRACKED_PATH    = os.path.join(ROOT, 'data', 'tracked_bugs.json')

PRIORITY_ORDER  = ['P1', 'P2', 'P3', 'P4', 'P5']
SEVERITY_ORDER  = ['Critical', 'High', 'Medium', 'Low']
STATUS_ORDER    = ['New', 'Assigned', 'In Progress', 'Fixed', 'Pending Retest',
                   'Verified', 'Closed', 'Reopened', 'Duplicate', 'Rejected', 'Deferred']
LIFECYCLE_ORDER = ['Reported', 'In Progress', 'Resolved', 'Verification', 'Closed']
RESOLUTION_ORDER = ['Fixed', 'Unresolved', 'Duplicate', 'Invalid', "Won't Fix"]
MODEL_ORDER     = ['Naive Bayes', 'Logistic Regression', 'Decision Tree',
                   'Random Forest', 'SVM (Linear)']


class DataMissing(Exception):
    """Raised when a pipeline artifact the app needs has not been produced yet."""


# ──────────────────────────────────────────────────────────────────────────────
#  Shared definitions — imported from the KPI stage rather than restated
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def kpi_module():
    """Import src/08_dashboard.py as a module, by path.

    Its filename starts with a digit so `import` cannot reach it. Loading it
    here means prepare(), compute_kpis() and build_insights() have exactly one
    definition, shared by the console report and this app.
    """
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    spec = importlib.util.spec_from_file_location(
        'bug_kpi_stage', os.path.join(SRC_DIR, '08_dashboard.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def model_bridge():
    """The one module that talks to the trained models (see src/model_bridge.py)."""
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import model_bridge as bridge
    return bridge


# ──────────────────────────────────────────────────────────────────────────────
#  Load
# ──────────────────────────────────────────────────────────────────────────────
def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


@st.cache_data(show_spinner=False)
def load_artifacts():
    """The JSON side-cars: module catalog, knowledge base, model scores, KPIs."""
    return {
        'catalog':    _read_json(CATALOG_PATH, {}) or {},
        'kb':         _read_json(KB_PATH, {}) or {},
        'model_eval': _read_json(MODEL_EVAL_PATH, {}) or {},
        'kpi_report': _read_json(KPI_PATH, {}) or {},
        'lifecycle':  _read_json(LIFECYCLE_PATH, {}) or {},
        'duplicates': _read_json(DUPLICATES_PATH, []) or [],
        'tracked':    _read_json(TRACKED_PATH, []) or [],
    }


@st.cache_data(show_spinner="Loading 50,000 bug records…")
def load_bugs():
    """The prepared bug frame, plus the reporting window it covers.

    Returns (df, origin, snapshot). Cached for the life of the server process —
    the CSV is ~24 MB and the derived columns cost a couple of seconds.
    """
    if not os.path.exists(PROCESSED_PATH):
        raise DataMissing(
            f"'{os.path.relpath(PROCESSED_PATH, ROOT)}' not found. Run "
            f"`python src/01_data_collection.py` then `python src/02_preprocessing.py` "
            f"(or `python run_pipeline.py`) to produce it.")

    stage = kpi_module()
    artifacts = load_artifacts()

    df = pd.read_csv(PROCESSED_PATH)
    missing = [c for c in ('sprint', 'release_version', 'module', 'feature',
                           'component', 'priority', 'resolution', 'root_cause',
                           'date_closed') if c not in df.columns]
    if missing:
        raise DataMissing(
            f"'{os.path.relpath(PROCESSED_PATH, ROOT)}' is missing {', '.join(missing)}. "
            f"Re-run 01_data_collection.py and 02_preprocessing.py — they add the "
            f"sprint/release/module/feature/component/closure columns.")

    df, origin, snapshot = stage.prepare(df, artifacts['kb'])

    # Ordering is applied per chart from the *_ORDER constants rather than by
    # making these columns categorical: compute_kpis() maps priority -> SLA days
    # and compares the result numerically, and a categorical would turn that map
    # back into a categorical the comparison cannot use.
    df['urgent'] = df['priority'].isin(['P1', 'P2'])
    return df, origin, snapshot


@st.cache_data(show_spinner=False)
def filter_options():
    """Every value each global filter can take, in display order."""
    df, _, _ = load_bugs()

    def levels(column, order=None):
        present = list(df[column].dropna().unique())
        if order:
            return [v for v in order if v in present] + \
                   sorted(str(v) for v in present if v not in order)
        return sorted(str(v) for v in present)

    return {
        'sprint':      levels('sprint'),
        'release':     levels('release_version'),
        'module':      levels('module'),
        'feature':     levels('feature'),
        'component':   levels('component'),
        'priority':    levels('priority', PRIORITY_ORDER),
        'severity':    levels('severity', SEVERITY_ORDER),
        'status':      levels('status', STATUS_ORDER),
        'lifecycle':   levels('lifecycle_stage', LIFECYCLE_ORDER),
        'resolution':  levels('resolution', RESOLUTION_ORDER),
        'category':    levels('bug_category'),
        'domain':      levels('bug_domain'),
        'environment': levels('environment'),
        'tech_stack':  levels('tech_stack'),
        'team':        levels('assigned_team'),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Filtering
#
#  Filters arrive as a tuple of hashable values so the slice and its KPIs cache
#  on the filter state itself — moving between pages with the same filters is
#  free, and only a filter change pays for a recompute.
# ──────────────────────────────────────────────────────────────────────────────
FILTER_FIELDS = ('sprint', 'release', 'module', 'priority', 'severity',
                 'category', 'environment', 'team', 'state', 'days')


def empty_filters():
    return {f: () for f in FILTER_FIELDS if f not in ('state', 'days')} | \
           {'state': 'All', 'days': None}


def _as_key(filters):
    """Filter dict -> hashable cache key."""
    return tuple((f, tuple(filters.get(f)) if isinstance(filters.get(f), (list, tuple, set))
                  else filters.get(f)) for f in FILTER_FIELDS)


@st.cache_data(show_spinner=False)
def _slice(key):
    df, origin, snapshot = load_bugs()
    filters = dict(key)

    mask = pd.Series(True, index=df.index)
    for field, column in (('sprint', 'sprint'), ('release', 'release_version'),
                          ('module', 'module'), ('priority', 'priority'),
                          ('severity', 'severity'), ('category', 'bug_category'),
                          ('environment', 'environment'), ('team', 'assigned_team')):
        chosen = filters.get(field)
        if chosen:
            mask &= df[column].astype(str).isin([str(v) for v in chosen])

    state = filters.get('state') or 'All'
    if state == 'Open':
        mask &= df['is_open']
    elif state == 'Closed':
        mask &= df['is_closed']
    elif state == 'Urgent open (P1/P2)':
        mask &= df['is_open'] & df['urgent']
    elif state == 'Reopened':
        mask &= df['status'].astype(str).eq('Reopened')

    days = filters.get('days')
    if days:
        cutoff = pd.Timestamp(snapshot) - pd.Timedelta(days=int(days))
        mask &= df['created_at'] >= cutoff

    return df[mask]


def apply_filters(filters):
    """The filtered slice for the current filter state."""
    return _slice(_as_key(filters))


@st.cache_data(show_spinner=False)
def _kpis(key):
    stage = kpi_module()
    artifacts = load_artifacts()
    _, _, snapshot = load_bugs()
    scope = _slice(key)
    if scope.empty:
        return None
    kpis = stage.compute_kpis(scope, artifacts['catalog'], snapshot)
    try:
        kpis['insights'] = stage.build_insights(kpis)
    except (ValueError, IndexError, KeyError, ZeroDivisionError, TypeError):
        # The insights read superlatives off the KPI tables ("worst SLA band",
        # "slowest team"). A narrow enough filter can leave a table with no
        # closed bugs at all and no superlative to name — the numbers above are
        # still valid, so drop the prose rather than the whole page.
        kpis['insights'] = []
    return kpis


def compute_kpis(filters):
    """KPIs for the current filter state — None when nothing is in scope."""
    return _kpis(_as_key(filters))


@st.cache_data(show_spinner=False)
def baseline_kpis():
    """KPIs over the whole dataset, so a filtered view can show a delta."""
    return _kpis(_as_key(empty_filters()))


def sla_targets():
    stage = kpi_module()
    return stage.sla_targets(load_artifacts()['catalog'])


# ──────────────────────────────────────────────────────────────────────────────
#  Aggregations the charts ask for
#
#  These take the already-filtered slice, so they are plain functions rather
#  than cached ones — every one is a single groupby over at most 50k rows.
# ──────────────────────────────────────────────────────────────────────────────
def counts(scope, column, order=None, top=None, label=None):
    """Value counts for one column as a tidy frame: label | bugs | open | share."""
    if scope.empty:
        return pd.DataFrame(columns=[label or column, 'bugs', 'open', 'share'])

    grouped = scope.groupby(column, observed=True)
    frame = pd.DataFrame({
        'bugs': grouped.size(),
        'open': grouped['is_open'].sum(),
        'avg_days': grouped['resolution_days'].mean().round(1),
    })
    if order:
        frame = frame.reindex([v for v in order if v in frame.index])
    else:
        frame = frame.sort_values('bugs', ascending=False)
    if top:
        frame = frame.head(top)

    frame = frame.reset_index().rename(columns={column: label or column})
    frame['share'] = (frame['bugs'] / len(scope) * 100).round(1)
    return frame


def crosstab(scope, rows, cols, row_order=None, col_order=None, top_rows=None):
    """Two-dimensional count matrix for the heatmaps."""
    if scope.empty:
        return pd.DataFrame()
    table = pd.crosstab(scope[rows], scope[cols])
    if row_order:
        table = table.reindex([v for v in row_order if v in table.index])
    else:
        table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    if col_order:
        table = table.reindex(columns=[v for v in col_order if v in table.columns])
    if top_rows:
        table = table.head(top_rows)
    return table.fillna(0).astype(int)


def sprint_flow(scope, all_sprints):
    """Opened vs closed vs carried-over backlog, per sprint."""
    if scope.empty:
        return pd.DataFrame(columns=['sprint', 'opened', 'closed', 'backlog', 'avg_days'])

    closed = scope[scope['is_closed']]
    opened = scope.groupby('sprint', observed=True).size().reindex(all_sprints, fill_value=0)
    closed_in = (closed.groupby(closed['close_sprint_index']
                                .map(lambda i: f"SPR-{int(i) + 1:02d}"), observed=True)
                 .size().reindex(all_sprints, fill_value=0))
    avg_days = (closed.groupby('sprint', observed=True)['resolution_days']
                .mean().reindex(all_sprints).round(1))

    return pd.DataFrame({
        'sprint':  all_sprints,
        'opened':  opened.to_numpy(),
        'closed':  closed_in.to_numpy(),
        'backlog': (opened - closed_in).cumsum().to_numpy(),
        'avg_days': avg_days.to_numpy(),
    })


def monthly_flow(scope):
    """Bugs reported vs bugs closed, by calendar month."""
    if scope.empty:
        return pd.DataFrame(columns=['month', 'opened', 'closed', 'net'])

    closed = scope[scope['is_closed']]
    months = sorted(set(scope['month'].dropna()) | set(closed['close_month'].dropna()))
    opened = scope.groupby('month').size().reindex(months, fill_value=0)
    closed_m = closed.groupby('close_month').size().reindex(months, fill_value=0)
    return pd.DataFrame({
        'month':  months,
        'opened': opened.to_numpy(),
        'closed': closed_m.to_numpy(),
        'net':    (opened - closed_m).to_numpy(),
    })


def resolution_by_priority(scope, targets):
    """Speed and SLA compliance per priority band, against its target."""
    rows = []
    for priority in PRIORITY_ORDER:
        group = scope[scope['priority'].astype(str) == priority]
        if group.empty:
            continue
        closed = group[group['is_closed']]
        target = targets.get(priority)
        within = float((closed['resolution_days'] <= target).mean() * 100) if len(closed) else 0.0
        rows.append({
            'priority':   priority,
            'bugs':       len(group),
            'closed':     len(closed),
            'target':     target,
            'median':     round(float(closed['resolution_days'].median()), 1) if len(closed) else None,
            'avg':        round(float(closed['resolution_days'].mean()), 1) if len(closed) else None,
            'p90':        round(float(closed['resolution_days'].quantile(0.90)), 1) if len(closed) else None,
            'sla_pct':    round(within, 1),
            'open_late':  int((group['is_open'] & (group['open_age_days'] > target)).sum())
                          if target else 0,
        })
    return pd.DataFrame(rows)


def team_table(scope, targets):
    """Routed-owner performance. Uses assigned_team, not the random raw column."""
    rows = []
    for team, group in scope.groupby('assigned_team', observed=True):
        closed = group[group['is_closed']]
        met = float((closed['resolution_days'] <= closed['priority'].map(targets)).mean() * 100) \
            if len(closed) else 0.0
        rows.append({
            'team':        team,
            'assigned':    len(group),
            'closed':      len(closed),
            'open':        int(group['is_open'].sum()),
            'urgent_open': int((group['is_open'] & group['urgent']).sum()),
            'avg_days':    round(float(closed['resolution_days'].mean()), 1) if len(closed) else None,
            'sla_pct':     round(met, 1),
            'close_pct':   round(len(closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    return pd.DataFrame(rows).sort_values('assigned', ascending=False)


def module_table(scope, kloc):
    """Per-module volume, density and closure — the defect-density view."""
    rows = []
    for module, group in scope.groupby('module', observed=True):
        closed = group[group['is_closed']]
        size = float(kloc.get(module, 1.0))
        rows.append({
            'module':   module,
            'bugs':     len(group),
            'open':     int(group['is_open'].sum()),
            'critical': int((group['severity'].astype(str) == 'Critical').sum()),
            'urgent':   int(group['urgent'].sum()),
            'kloc':     round(size, 1),
            'density':  round(len(group) / size, 1) if size else 0.0,
            'avg_days': round(float(closed['resolution_days'].mean()), 1) if len(closed) else None,
            'close_pct': round(len(closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    return pd.DataFrame(rows).sort_values('density', ascending=False)


def release_table(scope):
    """Per-release risk — urgent bugs still open is the column that matters."""
    rows = []
    for release, group in scope.groupby('release_version', observed=True):
        closed = group[group['is_closed']]
        rows.append({
            'release':     release,
            'bugs':        len(group),
            'open':        int(group['is_open'].sum()),
            'urgent_open': int((group['is_open'] & group['urgent']).sum()),
            'avg_days':    round(float(closed['resolution_days'].mean()), 1) if len(closed) else None,
            'close_pct':   round(len(closed) / len(group) * 100, 1) if len(group) else 0.0,
        })
    return pd.DataFrame(rows).sort_values('release')


def root_cause_table(scope):
    rows = []
    for cause, group in scope.groupby('root_cause', observed=True):
        closed = group[group['is_closed']]
        rows.append({
            'root_cause': cause,
            'bugs':       len(group),
            'share':      round(len(group) / len(scope) * 100, 1) if len(scope) else 0.0,
            'urgent':     int(group['urgent'].sum()),
            'avg_days':   round(float(closed['resolution_days'].mean()), 1) if len(closed) else None,
        })
    return pd.DataFrame(rows).sort_values('bugs', ascending=False)


def aging_buckets(scope, targets):
    """Open bugs by how far past their SLA target they are."""
    open_df = scope[scope['is_open']]
    if open_df.empty:
        return pd.DataFrame(columns=['bucket', 'bugs'])

    target = open_df['priority'].map(targets).astype(float)
    over = open_df['open_age_days'] - target
    edges = [(-np.inf, 0, 'Within SLA'), (0, 7, '1-7 days over'),
             (7, 30, '8-30 days over'), (30, 90, '31-90 days over'),
             (90, np.inf, '90+ days over')]
    return pd.DataFrame([
        {'bucket': label, 'bugs': int(((over > low) & (over <= high)).sum())}
        for low, high, label in edges])


# ──────────────────────────────────────────────────────────────────────────────
#  Models — loaded on demand, not at startup
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the trained models…")
def load_models():
    """The saved model bundle. models/best_priority_model.pkl is ~110 MB, so
    this only runs when a page actually needs a prediction."""
    return model_bridge().load(quiet=True)


@st.cache_data(show_spinner="Scoring every bug with the saved priority model…")
def score_all():
    """Re-score the whole dataset with the saved priority model.

    Returns (labels, confidence, split) where split is 0 never sampled,
    1 training row, 2 held-out test row — reproducing 05_modeling.py's own
    partition so the app can separate honest agreement from memorised rows.
    Returns None when the models are unavailable.
    """
    bridge = model_bridge()
    bundle = load_models()
    if not bundle.ready:
        return None

    df, _, _ = load_bugs()
    try:
        labels, conf = bridge.score_dataset(bundle, df)
    except Exception:                                              # noqa: BLE001
        return None
    return {
        'labels':     np.asarray(labels),
        'confidence': np.asarray(conf),
        'split':      bridge.training_split(df),
        'model_name': type(bundle.priority_model).__name__,
        'index':      df.index.to_numpy(),
    }


def predict_one(**kwargs):
    """One live prediction through the same bridge the pipeline uses."""
    return model_bridge().predict_one(load_models(), **kwargs)
