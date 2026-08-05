# ==============================================================================
# streamlit_app.py
# Task 9: the interactive dashboard, KPI reporting and actionable insights.
#
#   streamlit run src/streamlit_app.py          (or: python run_dashboard.py)
#
# Six views over the same filtered slice of the bug table — overview, trends,
# distribution, quality, models & triage, and a record-level explorer. One set
# of global filters in the sidebar drives every one of them.
#
# The trained models are wired in live: the triage console calls
# model_bridge.predict_one() in-process, and the model view can re-score the
# whole dataset with models/best_priority_model.pkl and split the result by
# 05_modeling.py's own train/test partition. There is no separate API server —
# Streamlit *is* the server.
#
# The KPI definitions are not restated here; src/dashboard_data.py imports them
# from 08_dashboard.py, which still writes data/kpi_report.json and the static
# companion charts for the console/report path.
# ==============================================================================

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import plotly.graph_objects as go

import dashboard_data as data
import dashboard_ui as ui

STATE_OPTIONS = ['All', 'Open', 'Closed', 'Urgent open (P1/P2)', 'Reopened']
WINDOW_OPTIONS = {'All time': None, 'Last 30 days': 30, 'Last 90 days': 90,
                  'Last 180 days': 180, 'Last 12 months': 365}

MULTI_FILTERS = [
    ('module',      'Module',       'module'),
    ('priority',    'Priority',     'priority'),
    ('severity',    'Severity',     'severity'),
    ('sprint',      'Sprint',       'sprint'),
    ('release',     'Release',      'release'),
    ('category',    'Bug category', 'category'),
    ('environment', 'Environment',  'environment'),
    ('team',        'Routed owner', 'team'),
]

FOOTER = (
    "<b>Read the numbers with their caveats.</b> Sprint, release version, module, feature, component, "
    "status, resolution, priority and closure dates are <i>derived</i> fields — the source Kaggle dataset "
    "carries no workflow state and no delivery taxonomy. See the README under “Derived Fields” and "
    "“Known Data Limitations” before treating any aging, velocity or team-performance figure as real-world "
    "behaviour. Team KPIs are reported against the <i>routed owner</i> (the specialist the routing policy "
    "sends the bug to), because the dataset's own <code>developer_role</code> column is uniformly random."
)


# ──────────────────────────────────────────────────────────────────────────────
#  Sidebar — the global filters every view reads
# ──────────────────────────────────────────────────────────────────────────────
def reset_filters():
    for key in list(st.session_state):
        if key.startswith('f_'):
            del st.session_state[key]


def sidebar_filters(snapshot, total):
    options = data.filter_options()

    with st.sidebar:
        st.markdown(
            f"<div style='font-size:.78rem;color:var(--ink-faint);line-height:1.5;"
            f"margin:.2rem 0 .6rem'>Snapshot <b style='color:var(--ink-soft)'>"
            f"{snapshot:%d %b %Y}</b> · {total:,} records</div>",
            unsafe_allow_html=True)

        st.markdown("###### Filters")
        state = st.selectbox('Bug state', STATE_OPTIONS, key='f_state')
        window = st.selectbox('Reported within', list(WINDOW_OPTIONS), key='f_window')

        chosen = {}
        for field, label, option_key in MULTI_FILTERS[:3]:
            chosen[field] = st.multiselect(label, options[option_key], key=f'f_{field}',
                                           placeholder='All')

        with st.expander('More filters'):
            for field, label, option_key in MULTI_FILTERS[3:]:
                chosen[field] = st.multiselect(label, options[option_key], key=f'f_{field}',
                                               placeholder='All')

        filters = data.empty_filters()
        filters.update({k: tuple(v) for k, v in chosen.items()})
        filters['state'] = state
        filters['days'] = WINDOW_OPTIONS[window]

        active = sum(1 for v in chosen.values() if v) + \
            (state != 'All') + (WINDOW_OPTIONS[window] is not None)
        st.button('Reset filters', on_click=reset_filters, width='stretch',
                  disabled=active == 0, type='secondary')

    return filters, active


def scope_banner(scope, total, active):
    """One line under the page header: what is actually in view."""
    share = len(scope) / total * 100 if total else 0
    if active == 0:
        text = f"All <b>{total:,}</b> bug records in scope."
    else:
        text = (f"<b>{len(scope):,}</b> of {total:,} records in scope "
                f"({share:.1f}%) · {active} filter{'s' if active != 1 else ''} applied")
    st.markdown(
        f"<div style='font-size:.85rem;color:var(--ink-soft);margin:-.4rem 0 .9rem'>{text}</div>",
        unsafe_allow_html=True)


def empty_scope():
    st.warning("No bugs match the current filters. Widen them, or press "
               "**Reset filters** in the sidebar.", icon=':material/filter_alt_off:')
    st.stop()


def context():
    """The (scope, kpis, baseline, snapshot) tuple every page opens with."""
    ctx = st.session_state['ctx']
    scope = data.apply_filters(ctx['filters'])
    if scope.empty:
        scope_banner(scope, ctx['total'], ctx['active'])
        empty_scope()
    kpis = data.compute_kpis(ctx['filters'])
    return scope, kpis, ctx


def footer():
    st.markdown(f'<div class="page-foot">{FOOTER}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  View 1 — Operations overview
# ──────────────────────────────────────────────────────────────────────────────
def page_overview():
    scope, kpis, ctx = context()
    base = data.baseline_kpis()['headline']
    head = kpis['headline']

    ui.page_header(
        'Operations overview', 'Triage & ops',
        'Where the backlog stands right now: volume, speed against SLA, and the '
        'sprints where intake outran closure.')
    scope_banner(scope, ctx['total'], ctx['active'])

    sla_status = 'good' if head['sla_compliance_pct'] >= 90 else \
        'warning' if head['sla_compliance_pct'] >= 75 else 'critical'
    open_share = head['open_bugs'] / max(head['total_bugs'], 1) * 100
    reopen_status = 'good' if head['reopen_rate_pct'] < 5 else \
        'warning' if head['reopen_rate_pct'] < 10 else 'critical'

    res_foot, res_state = ui.delta(head['avg_resolution_days'], base['avg_resolution_days'],
                                   unit='d', invert=True)
    density_foot, density_state = ui.delta(head['defect_density'], base['defect_density'],
                                           invert=True)

    ui.tiles([
        dict(label='Bugs in scope', value=ui.nf(head['total_bugs']),
             foot=f"<b>{head['closed_bugs']:,}</b> closed · {head['close_rate_pct']}% close rate"),
        dict(label='Open bugs', value=ui.nf(head['open_bugs']),
             foot=f"<b>{open_share:.1f}%</b> of the slice still open",
             status='serious' if open_share > 60 else 'warning' if open_share > 40 else None),
        dict(label='SLA compliance', value=ui.pct(head['sla_compliance_pct']),
             foot=f"{sla_status.capitalize()} · <b>{head['open_past_sla']:,}</b> open past target",
             status=sla_status),
        dict(label='Avg resolution time', value=ui.nf1(head['avg_resolution_days']), unit='d',
             foot=res_foot or f"median {head['median_resolution_days']}d · p90 {head['p90_resolution_days']}d",
             status=res_state),
    ])
    ui.tiles([
        dict(label='Defect density', value=ui.nf1(head['defect_density']),
             unit=' /KLOC', foot=density_foot or f"over {head['total_kloc']:,.0f} KLOC",
             status=density_state),
        dict(label='Reopen rate', value=ui.pct(head['reopen_rate_pct'], 2),
             foot=f"{reopen_status.capitalize()} · fixes that came back", status=reopen_status),
        dict(label='Avg age of an open bug', value=ui.nf1(head['avg_open_age_days']), unit='d',
             foot=f"measured to {head['snapshot_date']}",
             status='critical' if head['avg_open_age_days'] > 120 else None),
        dict(label='Open backlog at last sprint',
             value=ui.nf(kpis['by_sprint'][-1]['backlog'] if kpis['by_sprint'] else 0),
             foot=f"carried across {len(kpis['by_sprint'])} sprints"),
    ])

    # ── Sprint flow ──────────────────────────────────────────────────────────
    ui.section('Sprint flow', 'opened vs closed, and the backlog it leaves behind')
    sprints = data.filter_options()['sprint']
    flow = data.sprint_flow(scope, sprints)

    left, right = st.columns([1.35, 1], gap='medium')
    with left:
        colors = ui.series_colors(['opened', 'closed'], ui.FLOW_SLOT)
        fig = go.Figure()
        fig.add_bar(x=flow['sprint'], y=flow['opened'], name='Opened',
                    marker_color=colors['opened'],
                    hovertemplate='%{x}<br>Opened %{y:,}<extra></extra>')
        fig.add_bar(x=flow['sprint'], y=flow['closed'], name='Closed',
                    marker_color=colors['closed'],
                    hovertemplate='%{x}<br>Closed %{y:,}<extra></extra>')
        fig.update_layout(title='Bugs opened vs closed, per sprint', barmode='group',
                          yaxis_title='Bugs', hovermode='x unified')
        ui.show(fig, height=330)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=flow['sprint'], y=flow['backlog'], mode='lines',
                        line=dict(width=2, color=ui.series_colors(['backlog'], ui.FLOW_SLOT)['backlog']),
                        fill='tozeroy', fillcolor='rgba(57,135,229,.12)',
                        name='Carried backlog',
                        hovertemplate='%{x}<br>%{y:,} open<extra></extra>')
        fig.update_layout(title='Backlog carried into each sprint', yaxis_title='Open bugs',
                          showlegend=False)
        ui.show(fig, height=330)

    ui.table_view(flow.rename(columns={'sprint': 'Sprint', 'opened': 'Opened',
                                       'closed': 'Closed', 'backlog': 'Backlog',
                                       'avg_days': 'Avg days to close'}))

    # ── Risk ─────────────────────────────────────────────────────────────────
    ui.section('Where the risk sits', 'priority mix and how far past SLA the open work is')
    left, right = st.columns(2, gap='medium')

    with left:
        mix = data.counts(scope, 'priority', order=data.PRIORITY_ORDER, label='Priority')
        colors = ui.series_colors(mix['Priority'].tolist(), ui.PRIORITY_SLOT)
        fig = go.Figure()
        fig.add_bar(y=mix['Priority'], x=mix['bugs'], orientation='h',
                    marker_color=[colors[p] for p in mix['Priority']],
                    text=[f"{v:,}" for v in mix['bugs']], textposition='outside',
                    textfont=dict(color=ui.palette()['ink_soft'], size=12),
                    hovertemplate='%{y}<br>%{x:,} bugs<extra></extra>')
        fig.update_layout(title='Bugs by priority', showlegend=False,
                          xaxis_title='Bugs', yaxis=dict(autorange='reversed', showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor=ui.palette()['line'],
                                     showline=False, range=[0, mix['bugs'].max() * 1.16]))
        ui.show(fig, height=300)

    with right:
        buckets = data.aging_buckets(scope, data.sla_targets())
        p = ui.palette()
        bucket_color = {'Within SLA': p['good'], '1-7 days over': p['warning'],
                        '8-30 days over': p['serious'], '31-90 days over': p['critical'],
                        '90+ days over': p['critical']}
        fig = go.Figure()
        fig.add_bar(x=buckets['bucket'], y=buckets['bugs'],
                    marker_color=[bucket_color[b] for b in buckets['bucket']],
                    text=[f"{v:,}" for v in buckets['bugs']], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    hovertemplate='%{x}<br>%{y:,} open bugs<extra></extra>')
        fig.update_layout(title='Open bugs, by time past their SLA target',
                          showlegend=False, yaxis_title='Open bugs',
                          yaxis=dict(range=[0, max(buckets['bugs'].max() * 1.18, 1)]))
        ui.show(fig, height=300)

    ui.table_view(pd.merge(
        mix.rename(columns={'bugs': 'Bugs', 'open': 'Open', 'share': 'Share %',
                            'avg_days': 'Avg days'}),
        data.resolution_by_priority(scope, data.sla_targets())
            [['priority', 'target', 'sla_pct']]
            .rename(columns={'priority': 'Priority', 'target': 'SLA target (days)',
                             'sla_pct': 'Met SLA %'}),
        on='Priority', how='left'), label='Show the numbers — priority')

    # ── Insights ─────────────────────────────────────────────────────────────
    if kpis.get('insights'):
        ui.section('What to act on', 'every figure below is computed from the slice in scope')
        for i, item in enumerate(kpis['insights']):
            ui.insight_card(item['title'], item['detail'], badge=f"{i + 1:02d}")

    footer()


# ──────────────────────────────────────────────────────────────────────────────
#  View 2 — Trends & resolution time
# ──────────────────────────────────────────────────────────────────────────────
def page_trends():
    scope, kpis, ctx = context()
    targets = data.sla_targets()
    p = ui.palette()

    ui.page_header(
        'Bug trends & resolution time', 'Trends',
        'How intake and closure move over time, and how long a bug actually '
        'takes to close against the target for its priority.')
    scope_banner(scope, ctx['total'], ctx['active'])

    # ── Monthly flow ─────────────────────────────────────────────────────────
    ui.section('Reported vs closed, by month')
    flow = data.monthly_flow(scope)
    colors = ui.series_colors(['opened', 'closed'], ui.FLOW_SLOT)

    fig = go.Figure()
    for key, label in (('opened', 'Reported'), ('closed', 'Closed')):
        fig.add_scatter(x=flow['month'], y=flow[key], name=label, mode='lines+markers',
                        line=dict(width=2, color=colors[key]),
                        marker=dict(size=8, color=colors[key],
                                    line=dict(width=2, color=p['surface'])),
                        hovertemplate=f'{label}: %{{y:,}}<extra></extra>')
    fig.update_layout(title='Bugs reported vs bugs closed', yaxis_title='Bugs',
                      hovermode='x unified')
    ui.show(fig, height=330)
    st.caption('The reporting window starts and ends mid-month, so the first and last '
               'points cover fewer days than the rest — the dip at either end is the '
               'calendar, not a change in behaviour.')

    left, right = st.columns(2, gap='medium')
    with left:
        # Diverging by meaning, not by series identity: two poles around a
        # neutral zero, and the legend names both directions in words.
        growing = flow['net'] > 0
        fig = go.Figure()
        fig.add_bar(x=flow.loc[growing, 'month'], y=flow.loc[growing, 'net'],
                    name='Backlog grew', marker_color=p['serious'],
                    hovertemplate='%{x}<br>+%{y:,} net<extra></extra>')
        fig.add_bar(x=flow.loc[~growing, 'month'], y=flow.loc[~growing, 'net'],
                    name='Backlog shrank', marker_color=p['good'],
                    hovertemplate='%{x}<br>%{y:,} net<extra></extra>')
        fig.update_layout(title='Net backlog change per month',
                          yaxis_title='Reported − closed',
                          yaxis=dict(zeroline=True, zerolinecolor=p['baseline'], zerolinewidth=1))
        ui.show(fig, height=300)
    with right:
        closed = scope[scope['is_closed']]
        fig = go.Figure()
        fig.add_histogram(x=closed['resolution_days'], nbinsx=40,
                          marker_color=p['series'][0],
                          hovertemplate='%{x} days<br>%{y:,} bugs<extra></extra>')
        if len(closed):
            median = float(closed['resolution_days'].median())
            fig.add_vline(x=median, line_width=2, line_dash='dot', line_color=p['ink_faint'],
                          annotation_text=f'median {median:.0f}d',
                          annotation_position='top right',
                          annotation_font=dict(color=p['ink_soft'], size=12))
        fig.update_layout(title='How long closed bugs took', showlegend=False,
                          xaxis_title='Days to close', yaxis_title='Bugs')
        ui.show(fig, height=300)

    ui.table_view(flow.rename(columns={'month': 'Month', 'opened': 'Reported',
                                       'closed': 'Closed', 'net': 'Net change'}))

    # ── Speed vs SLA ─────────────────────────────────────────────────────────
    ui.section('Speed against the SLA target', 'per priority band')
    speed = data.resolution_by_priority(scope, targets)
    if speed.empty:
        st.info('No priority bands in scope.')
        footer()
        return

    left, right = st.columns(2, gap='medium')
    with left:
        colors = ui.series_colors(speed['priority'].tolist(), ui.PRIORITY_SLOT)
        fig = go.Figure()
        fig.add_bar(x=speed['priority'], y=speed['median'], name='Median days',
                    marker_color=[colors[v] for v in speed['priority']],
                    text=[ui.days(v) for v in speed['median']], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    hovertemplate='%{x}<br>median %{y:.1f} days<extra></extra>')
        fig.add_scatter(x=speed['priority'], y=speed['target'], name='SLA target',
                        mode='markers', marker=dict(symbol='line-ew', size=26,
                                                    line=dict(width=2, color=p['ink_faint'])),
                        hovertemplate='%{x}<br>target %{y} days<extra></extra>')
        fig.update_layout(title='Median days to close vs target',
                          yaxis_title='Days',
                          yaxis=dict(range=[0, max(speed['median'].max(skipna=True) or 0,
                                                   speed['target'].max()) * 1.25]))
        ui.show(fig, height=320)
    with right:
        met = speed['sla_pct']
        bar_status = [p['good'] if v >= 90 else p['warning'] if v >= 75 else p['critical']
                      for v in met]
        fig = go.Figure()
        fig.add_bar(x=speed['priority'], y=met, marker_color=bar_status,
                    text=[ui.pct(v) for v in met], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    hovertemplate='%{x}<br>%{y:.1f}% met target<extra></extra>')
        fig.add_hline(y=90, line_width=1, line_dash='dot', line_color=p['ink_faint'],
                      annotation_text='90% goal', annotation_position='top left',
                      annotation_font=dict(color=p['ink_faint'], size=11))
        fig.update_layout(title='Closed within target (green ≥90%, amber ≥75%, red below)',
                          showlegend=False, yaxis_title='% of closed bugs',
                          yaxis=dict(range=[0, 112]))
        ui.show(fig, height=320)

    ui.table_view(speed.rename(columns={
        'priority': 'Priority', 'bugs': 'Bugs', 'closed': 'Closed',
        'target': 'SLA target (days)', 'median': 'Median days', 'avg': 'Avg days',
        'p90': 'p90 days', 'sla_pct': 'Met SLA %', 'open_late': 'Open past target'}))

    # ── Distribution of resolution time ──────────────────────────────────────
    ui.section('Spread of resolution time', 'a median hides the tail — this does not')
    closed = scope[scope['is_closed']]
    if len(closed):
        colors = ui.series_colors(data.PRIORITY_ORDER, ui.PRIORITY_SLOT)
        fig = go.Figure()
        for priority in [v for v in data.PRIORITY_ORDER
                         if v in set(closed['priority'].astype(str))]:
            values = closed.loc[closed['priority'].astype(str) == priority, 'resolution_days']
            fig.add_box(y=values, name=priority, marker_color=colors[priority],
                        line=dict(width=2), fillcolor='rgba(0,0,0,0)',
                        boxpoints=False, hovertemplate='%{y:.0f} days<extra></extra>')
        fig.update_layout(title='Days to close, by priority', showlegend=False,
                          yaxis_title='Days to close')
        ui.show(fig, height=330)
    else:
        st.info('Nothing in scope has closed yet, so there is no resolution time to spread.')

    footer()


# ──────────────────────────────────────────────────────────────────────────────
#  View 3 — Distribution
# ──────────────────────────────────────────────────────────────────────────────
DIMENSIONS = {
    'Module':      ('module', None),
    'Feature':     ('feature', None),
    'Component':   ('component', None),
    'Category':    ('bug_category', None),
    'Domain':      ('bug_domain', None),
    'Tech stack':  ('tech_stack', None),
    'Environment': ('environment', None),
    'Status':      ('status', data.STATUS_ORDER),
    'Lifecycle':   ('lifecycle_stage', data.LIFECYCLE_ORDER),
    'Resolution':  ('resolution', data.RESOLUTION_ORDER),
}


def page_distribution():
    scope, kpis, ctx = context()
    p = ui.palette()

    ui.page_header(
        'Distribution', 'Where the bugs are',
        'The same slice cut by any dimension in the taxonomy — and the '
        'module × priority matrix that says which squares are actually hot.')
    scope_banner(scope, ctx['total'], ctx['active'])

    choice = st.segmented_control('Break down by', list(DIMENSIONS), default='Module',
                                  key='dist_dim') or 'Module'
    column, order = DIMENSIONS[choice]
    frame = data.counts(scope, column, order=order, top=18, label=choice)

    left, right = st.columns([1.45, 1], gap='medium')
    with left:
        closed_counts = frame['bugs'] - frame['open']
        fig = go.Figure()
        fig.add_bar(y=frame[choice], x=closed_counts, orientation='h', name='Closed',
                    marker=dict(color=p['series'][2],
                                line=dict(width=1.5, color=p['page'])),
                    hovertemplate='%{y}<br>Closed %{x:,}<extra></extra>')
        fig.add_bar(y=frame[choice], x=frame['open'], orientation='h', name='Open',
                    marker=dict(color=p['series'][1],
                                line=dict(width=1.5, color=p['page'])),
                    hovertemplate='%{y}<br>Open %{x:,}<extra></extra>')
        fig.update_layout(title=f'Bugs by {choice.lower()} — open vs closed',
                          barmode='stack', xaxis_title='Bugs',
                          yaxis=dict(autorange='reversed', showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor=p['line'], showline=False),
                          hovermode='y unified')
        ui.show(fig, height=max(300, 34 * len(frame) + 90))
    with right:
        top = frame.head(9)
        rest = frame['bugs'].sum() - top['bugs'].sum()
        labels = top[choice].tolist() + (['Other'] if rest > 0 else [])
        values = top['bugs'].tolist() + ([rest] if rest > 0 else [])
        colors = [p['series'][i % len(p['series'])] for i in range(len(top))] + \
                 ([p['ink_faint']] if rest > 0 else [])
        fig = go.Figure()
        fig.add_pie(labels=labels, values=values, hole=0.62, sort=False,
                    marker=dict(colors=colors, line=dict(width=2, color=p['page'])),
                    textinfo='none',
                    hovertemplate='%{label}<br>%{value:,} bugs (%{percent})<extra></extra>')
        fig.update_layout(title=f'Share of the slice ({choice.lower()})',
                          legend=dict(orientation='v', x=1.0, xanchor='right', y=0.5,
                                      yanchor='middle', font=dict(size=11)))
        ui.show(fig, height=max(300, 34 * len(frame) + 90))

    ui.table_view(frame.rename(columns={'bugs': 'Bugs', 'open': 'Open',
                                        'avg_days': 'Avg days', 'share': 'Share %'}))

    # ── Module × priority heat ───────────────────────────────────────────────
    ui.section('Module × priority', 'one hue, light to dark — darker means more bugs')
    matrix = data.crosstab(scope, 'module', 'priority', col_order=data.PRIORITY_ORDER)
    if not matrix.empty:
        fig = go.Figure(go.Heatmap(
            z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
            colorscale=[[i / (len(ui.seq_scale()) - 1), c]
                        for i, c in enumerate(ui.seq_scale())],
            xgap=2, ygap=2,
            colorbar=dict(title=dict(text='Bugs', font=dict(size=11)),
                          thickness=10, outlinewidth=0, tickfont=dict(size=11)),
            hovertemplate='%{y} · %{x}<br>%{z:,} bugs<extra></extra>'))
        fig.update_layout(title='Bugs per module and priority band',
                          xaxis=dict(side='top', showline=False, ticks=''),
                          yaxis=dict(autorange='reversed', showgrid=False),
                          margin=dict(t=60))
        ui.show(fig, height=max(280, 42 * len(matrix) + 110))
        ui.table_view(matrix.reset_index(), hide_index=True,
                      label='Show the numbers — module × priority')

    # ── Sprint & release composition ─────────────────────────────────────────
    ui.section('Composition over the delivery calendar')
    left, right = st.columns(2, gap='medium')
    with left:
        by_sprint = data.crosstab(scope, 'sprint', 'severity',
                                  row_order=data.filter_options()['sprint'],
                                  col_order=data.SEVERITY_ORDER)
        colors = ui.series_colors(data.SEVERITY_ORDER, ui.SEVERITY_SLOT)
        fig = go.Figure()
        for level in by_sprint.columns:
            fig.add_bar(x=list(by_sprint.index), y=by_sprint[level], name=level,
                        marker=dict(color=colors[level], line=dict(width=1.5, color=p['page'])),
                        hovertemplate=f'{level}: %{{y:,}}<extra></extra>')
        fig.update_layout(title='Severity mix per sprint', barmode='stack',
                          yaxis_title='Bugs', hovermode='x unified')
        ui.show(fig, height=330)
    with right:
        releases = data.release_table(scope)
        fig = go.Figure()
        fig.add_bar(x=releases['release'], y=releases['bugs'] - releases['open'],
                    name='Closed', marker=dict(color=p['series'][2],
                                               line=dict(width=1.5, color=p['page'])),
                    hovertemplate='Closed %{y:,}<extra></extra>')
        fig.add_bar(x=releases['release'], y=releases['open'], name='Open',
                    marker=dict(color=p['series'][1], line=dict(width=1.5, color=p['page'])),
                    hovertemplate='Open %{y:,}<extra></extra>')
        fig.update_layout(title='Bugs per release version', barmode='stack',
                          yaxis_title='Bugs', hovermode='x unified')
        ui.show(fig, height=330)

    ui.table_view(releases.rename(columns={
        'release': 'Release', 'bugs': 'Bugs', 'open': 'Open',
        'urgent_open': 'Urgent open (P1/P2)', 'avg_days': 'Avg days', 'close_pct': 'Closed %'}))

    footer()


# ──────────────────────────────────────────────────────────────────────────────
#  View 4 — Quality & team
# ──────────────────────────────────────────────────────────────────────────────
def page_quality():
    scope, kpis, ctx = context()
    targets = data.sla_targets()
    p = ui.palette()

    ui.page_header(
        'Root cause, team & release risk', 'Quality',
        'What keeps causing bugs, who the routing policy sends them to, and '
        'which release is carrying the largest unresolved urgent queue.')
    scope_banner(scope, ctx['total'], ctx['active'])

    # ── Defect density ───────────────────────────────────────────────────────
    ui.section('Defect density by module', 'bugs per KLOC — size-adjusted, unlike raw counts')
    modules = data.module_table(scope, data.kpi_module().module_kloc(
        data.load_artifacts()['catalog'], scope))

    left, right = st.columns([1.2, 1], gap='medium')
    with left:
        fig = go.Figure()
        fig.add_bar(y=modules['module'], x=modules['density'], orientation='h',
                    marker_color=p['series'][0],
                    text=[f"{v:,.1f}" for v in modules['density']], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    customdata=np.stack([modules['bugs'], modules['kloc']], axis=-1),
                    hovertemplate='%{y}<br>%{x:.1f} bugs / KLOC<br>'
                                  '%{customdata[0]:,} bugs over %{customdata[1]:.1f} KLOC<extra></extra>')
        fig.update_layout(title='Bugs per KLOC', showlegend=False, xaxis_title='Bugs / KLOC',
                          yaxis=dict(autorange='reversed', showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor=p['line'], showline=False,
                                     range=[0, modules['density'].max() * 1.18]))
        ui.show(fig, height=max(260, 40 * len(modules) + 80))
    with right:
        fig = go.Figure()
        fig.add_scatter(x=modules['kloc'], y=modules['bugs'], mode='markers+text',
                        text=modules['module'], textposition='top center',
                        textfont=dict(size=11, color=p['ink_faint']),
                        marker=dict(size=14, color=p['series'][0],
                                    line=dict(width=2, color=p['surface'])),
                        hovertemplate='%{text}<br>%{y:,} bugs over %{x:.1f} KLOC<extra></extra>')
        fig.update_layout(title='Volume against module size', showlegend=False,
                          xaxis_title='KLOC', yaxis_title='Bugs',
                          xaxis=dict(showgrid=True, gridcolor=p['line']))
        ui.show(fig, height=max(260, 40 * len(modules) + 80))

    ui.table_view(modules.rename(columns={
        'module': 'Module', 'bugs': 'Bugs', 'open': 'Open', 'critical': 'Critical',
        'urgent': 'Urgent', 'kloc': 'KLOC', 'density': 'Bugs / KLOC',
        'avg_days': 'Avg days', 'close_pct': 'Closed %'}))

    # ── Root cause ───────────────────────────────────────────────────────────
    ui.section('Root cause', 'systemic, not incidental — each one is a candidate guard rail')
    causes = data.root_cause_table(scope).head(12)
    fig = go.Figure()
    fig.add_bar(y=causes['root_cause'], x=causes['bugs'], orientation='h',
                marker_color=p['series'][6],
                text=[f"{v:,} · {s}%" for v, s in zip(causes['bugs'], causes['share'])],
                textposition='outside', textfont=dict(color=p['ink_soft'], size=12),
                customdata=causes['urgent'],
                hovertemplate='%{y}<br>%{x:,} bugs<br>%{customdata:,} urgent<extra></extra>')
    fig.update_layout(title='Top root causes in scope', showlegend=False, xaxis_title='Bugs',
                      yaxis=dict(autorange='reversed', showgrid=False),
                      xaxis=dict(showgrid=True, gridcolor=p['line'], showline=False,
                                 range=[0, causes['bugs'].max() * 1.28]))
    ui.show(fig, height=max(300, 30 * len(causes) + 90))
    ui.table_view(causes.rename(columns={
        'root_cause': 'Root cause', 'bugs': 'Bugs', 'share': 'Share %',
        'urgent': 'Urgent (P1/P2)', 'avg_days': 'Avg days'}))

    # ── Team ─────────────────────────────────────────────────────────────────
    ui.section('Routed owner', 'the specialist the routing policy sends the bug to')
    teams = data.team_table(scope, targets)

    left, right = st.columns(2, gap='medium')
    with left:
        fig = go.Figure()
        fig.add_bar(y=teams['team'], x=teams['assigned'] - teams['open'], orientation='h',
                    name='Closed', marker=dict(color=p['series'][2],
                                               line=dict(width=1.5, color=p['page'])),
                    hovertemplate='%{y}<br>Closed %{x:,}<extra></extra>')
        fig.add_bar(y=teams['team'], x=teams['open'], orientation='h', name='Open',
                    marker=dict(color=p['series'][1], line=dict(width=1.5, color=p['page'])),
                    hovertemplate='%{y}<br>Open %{x:,}<extra></extra>')
        fig.update_layout(title='Queue length per routed owner', barmode='stack',
                          xaxis_title='Bugs', hovermode='y unified',
                          yaxis=dict(autorange='reversed', showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor=p['line'], showline=False))
        ui.show(fig, height=max(280, 40 * len(teams) + 90))
    with right:
        colors = [p['good'] if v >= 90 else p['warning'] if v >= 75 else p['critical']
                  for v in teams['sla_pct']]
        fig = go.Figure()
        fig.add_bar(y=teams['team'], x=teams['sla_pct'], orientation='h',
                    marker_color=colors,
                    text=[ui.pct(v) for v in teams['sla_pct']], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    customdata=teams['urgent_open'],
                    hovertemplate='%{y}<br>%{x:.1f}% met SLA<br>'
                                  '%{customdata:,} urgent still open<extra></extra>')
        fig.update_layout(title='SLA compliance (green ≥90%, amber ≥75%, red below)',
                          showlegend=False, xaxis_title='% of closed bugs within target',
                          yaxis=dict(autorange='reversed', showgrid=False),
                          xaxis=dict(showgrid=True, gridcolor=p['line'], showline=False,
                                     range=[0, 118]))
        ui.show(fig, height=max(280, 40 * len(teams) + 90))

    ui.table_view(teams.rename(columns={
        'team': 'Routed owner', 'assigned': 'Assigned', 'closed': 'Closed', 'open': 'Open',
        'urgent_open': 'Urgent open', 'avg_days': 'Avg days', 'sla_pct': 'Met SLA %',
        'close_pct': 'Closed %'}))

    # ── Release risk ─────────────────────────────────────────────────────────
    ui.section('Release risk', 'unresolved P1/P2 is the number that gates a release')
    releases = data.release_table(scope)
    fig = go.Figure()
    fig.add_bar(x=releases['release'], y=releases['urgent_open'],
                marker_color=p['critical'],
                text=[f"{v:,}" for v in releases['urgent_open']], textposition='outside',
                textfont=dict(color=p['ink_soft'], size=12),
                customdata=np.stack([releases['open'], releases['bugs']], axis=-1),
                hovertemplate='%{x}<br>%{y:,} urgent open<br>'
                              'of %{customdata[0]:,} open / %{customdata[1]:,} total<extra></extra>')
    fig.update_layout(title='Unresolved P1/P2 bugs per release', showlegend=False,
                      yaxis_title='Urgent open bugs',
                      yaxis=dict(range=[0, max(releases['urgent_open'].max() * 1.2, 1)]))
    ui.show(fig, height=310)
    ui.table_view(releases.rename(columns={
        'release': 'Release', 'bugs': 'Bugs', 'open': 'Open',
        'urgent_open': 'Urgent open (P1/P2)', 'avg_days': 'Avg days', 'close_pct': 'Closed %'}))

    footer()


# ──────────────────────────────────────────────────────────────────────────────
#  View 5 — Models & live triage
# ──────────────────────────────────────────────────────────────────────────────
VERDICTS = {
    'Bug Category': ('critical', 'Leakage, not a result',
        "Every model scores near-perfect accuracy on bug category — and that is not "
        "generalization. `title`, `description`, `root_cause` and `suggested_fix` are "
        "boilerplate templates: only 16 unique strings across the whole dataset, one fixed "
        "template per category, so the model is matching a copied template back to the label "
        "it was copied from. Treat this target as unlearnable until the source data carries "
        "genuine free text."),
    'Severity': ('critical', 'No predictive signal',
        "Every model lands at chance level for four classes (~25%). Severity was tested "
        "against bug_category, bug_domain, environment, error_code, developer_role and the "
        "description text itself: none carries usable signal. Severity appears to be assigned "
        "independently at random in the source data — **no model can beat chance on it from "
        "this dataset**, text-based or feature-based."),
    'Priority': ('good', 'Learnable — with a caveat',
        "Unlike the other two targets the five models genuinely spread out here, and Random "
        "Forest separates from the rest — a real, learnable relationship. The caveat: priority "
        "is a **derived field**, computed from the structured features by a documented scoring "
        "rule plus ~8% seeded jitter, so the models recover that rule rather than predicting "
        "real-world triage priority. Read it as proof the pipeline trains and ranks models "
        "correctly on a learnable target."),
}

SEVERITY_CHOICES = ['Let the model predict it'] + data.SEVERITY_ORDER


def page_models():
    scope, kpis, ctx = context()
    artifacts = data.load_artifacts()
    p = ui.palette()

    ui.page_header(
        'Model performance & live triage', 'Models',
        'How the five trained models compare per target, how the saved priority '
        'model agrees with the recorded labels, and a console that predicts a '
        'brand-new bug in-process.')

    tab_eval, tab_agree, tab_triage = st.tabs(
        ['Model comparison', 'Agreement on this data', 'Live triage console'])

    # Each tab body is its own function so an early exit skips that tab only —
    # st.stop() inside a tab would abandon the other two as well.
    with tab_eval:
        _tab_comparison(artifacts['model_eval'])
    with tab_agree:
        _tab_agreement(scope, p)
    with tab_triage:
        _tab_triage(artifacts, p)

    footer()


def _tab_comparison(model_eval):
    if not model_eval:
        st.info('`data/model_evaluation_results.json` not found — run '
                '`python src/05_modeling.py` to produce it.')
        return

    targets = [t for t in ('Priority', 'Severity', 'Bug Category') if t in model_eval]
    target = st.segmented_control('Target', targets, default=targets[0],
                                  key='eval_target') or targets[0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    frame = (pd.DataFrame(model_eval[target]).T
             .reindex([m for m in data.MODEL_ORDER if m in model_eval[target]]))

    status, label, narrative = VERDICTS.get(target, (None, None, None))
    if label:
        st.markdown(ui.chip(label, status), unsafe_allow_html=True)
        st.markdown(f"<div class='note'>{narrative}</div>", unsafe_allow_html=True)

    colors = ui.series_colors(metrics, {m: i for i, m in enumerate(metrics)})
    fig = go.Figure()
    for metric in metrics:
        fig.add_bar(x=list(frame.index), y=frame[metric] * 100, name=metric,
                    marker_color=colors[metric],
                    hovertemplate=f'%{{x}}<br>{metric} %{{y:.1f}}%<extra></extra>')
    fig.update_layout(title=f'{target}: five models, four metrics',
                      barmode='group', yaxis_title='%',
                      yaxis=dict(range=[0, 105]), hovermode='x unified')
    ui.show(fig, height=360)

    best = frame['F1-Score'].idxmax()
    st.caption(f"Best F1 on {target.lower()}: **{best}** "
               f"({frame.loc[best, 'F1-Score'] * 100:.1f}%), saved as "
               f"`models/best_{target.lower().replace(' ', '_')}_model.pkl`.")
    ui.table_view((frame * 100).round(1).reset_index().rename(columns={'index': 'Model'}),
                  label='Show the numbers — all metrics, %')


def _tab_agreement(scope, p):
    st.markdown(
        "<div class='note'>The saved priority model can be re-run over every row in "
        "the dataset here. <b>Read the held-out split</b> — the training rows are "
        "memorised, so their agreement is not evidence of anything. Scoring 50,000 rows "
        "takes about a minute the first time; the result is cached for the session."
        "</div>", unsafe_allow_html=True)

    if st.button('Score the dataset with the saved priority model',
                 type='primary', icon=':material/play_arrow:'):
        st.session_state['scored'] = True

    if not st.session_state.get('scored'):
        return

    scores = data.score_all()
    if scores is None:
        bundle = data.load_models()
        st.error('The models could not be loaded:\n\n' +
                 ('\n'.join(f'- {e}' for e in bundle.errors) or
                  'Run `python src/05_modeling.py` first.'))
        return

    full, _, _ = data.load_bugs()
    predicted = pd.Series(scores['labels'], index=scores['index'])
    confidence = pd.Series(scores['confidence'], index=scores['index'])
    split = pd.Series(scores['split'], index=scores['index'])

    rows = scope.index
    split_choice = st.segmented_control(
        'Split', ['Held-out test', 'Training rows', 'Never sampled', 'All rows'],
        default='Held-out test', key='split_choice') or 'Held-out test'
    wanted = {'Held-out test': 2, 'Training rows': 1, 'Never sampled': 0}.get(split_choice)
    if wanted is not None:
        rows = rows[split.loc[rows] == wanted]

    if len(rows) == 0:
        st.info('No rows in this split under the current filters.')
        return

    recorded = full.loc[rows, 'priority'].astype(str)
    model_says = predicted.loc[rows]
    agreement = float((recorded.to_numpy() == model_says.to_numpy()).mean() * 100)
    mean_conf = float(confidence.loc[rows].mean() * 100)

    ui.tiles([
        dict(label='Model', value=scores['model_name'],
             foot='models/best_priority_model.pkl'),
        dict(label='Rows in this split', value=ui.nf(len(rows)),
             foot=f"{split_choice.lower()}, inside the current filters"),
        dict(label='Agreement with recorded priority', value=ui.pct(agreement),
             foot='Held-out — the honest number' if wanted == 2
                  else 'Memorised rows — not evidence' if wanted == 1
                  else 'rows the model never saw',
             status='good' if wanted == 2 and agreement >= 60 else
                    'warning' if wanted == 1 else None),
        dict(label='Mean confidence', value=ui.pct(mean_conf),
             foot="the model's own probability for the class it chose"),
    ])

    ui.section('Recorded vs predicted', 'the diagonal is agreement')
    matrix = pd.crosstab(recorded, model_says).reindex(
        index=[v for v in data.PRIORITY_ORDER if v in set(recorded)],
        columns=[v for v in data.PRIORITY_ORDER if v in set(model_says)]).fillna(0).astype(int)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        colorscale=[[i / (len(ui.seq_scale()) - 1), c]
                    for i, c in enumerate(ui.seq_scale())],
        xgap=2, ygap=2, colorbar=dict(thickness=10, outlinewidth=0,
                                      tickfont=dict(size=11)),
        hovertemplate='Recorded %{y} · model said %{x}<br>%{z:,} bugs<extra></extra>'))
    fig.update_layout(title='Recorded priority (rows) vs model prediction (columns)',
                      xaxis=dict(title='Model says', side='top', showline=False, ticks=''),
                      yaxis=dict(title='Recorded', autorange='reversed', showgrid=False),
                      margin=dict(t=70))
    ui.show(fig, height=360)
    ui.table_view(matrix.reset_index(), label='Show the numbers — confusion matrix')

    ui.section('Where the model disagrees most confidently',
               'the rows worth a human second look')
    disagree = pd.DataFrame({
        'Bug': full.loc[rows, 'bug_id'],
        'Module': full.loc[rows, 'module'],
        'Recorded': recorded,
        'Model says': model_says,
        'Confidence %': (confidence.loc[rows] * 100).round(1),
        'Age (days)': full.loc[rows, 'open_age_days'].round(0),
    })
    disagree = (disagree[disagree['Recorded'] != disagree['Model says']]
                .sort_values('Confidence %', ascending=False).head(50))
    st.dataframe(disagree, width='stretch', hide_index=True, height=340)


def _tab_triage(artifacts, p):
    options = data.filter_options()
    st.markdown(
        "<div class='note'>This runs <code>model_bridge.predict_one()</code> "
        "in-process against <code>models/*.pkl</code> — the same code path the pipeline "
        "uses, with no API server in between. Severity is predicted from the description "
        "text when you leave it to the model, but that model is at chance level on this "
        "dataset (see the caveat on the comparison tab), so supplying severity yourself "
        "gives the priority model a far better feature.</div>", unsafe_allow_html=True)

    roles = sorted(data.load_bugs()[0]['developer_role'].dropna().unique())

    with st.form('triage'):
        description = st.text_area(
            'Describe the bug', height=110,
            placeholder='Checkout API returns 500 under load after the v2.1 deploy; '
                        'the retry path leaks connections.')
        row1 = st.columns(3)
        severity = row1[0].selectbox('Severity', SEVERITY_CHOICES)
        environment = row1[1].selectbox('Environment', options['environment'],
                                        index=min(2, len(options['environment']) - 1))
        error_code = row1[2].number_input('Error code', value=500, step=1, min_value=0)
        row2 = st.columns(3)
        domain = row2[0].selectbox('Domain', options['domain'])
        tech = row2[1].selectbox('Tech stack', options['tech_stack'])
        role = row2[2].selectbox('Developer role', roles)
        submitted = st.form_submit_button('Predict priority', type='primary',
                                          icon=':material/bolt:')

    if not submitted:
        return
    if not description.strip():
        st.warning('Enter a description first — the text is half of the feature vector.')
        return

    bundle = data.load_models()
    if not bundle.ready:
        st.error('The models could not be loaded:\n\n' +
                 ('\n'.join(f'- {e}' for e in bundle.errors) or
                  'Run `python src/05_modeling.py` first.'))
        return

    result = data.predict_one(
        description=description,
        severity=None if severity == SEVERITY_CHOICES[0] else severity,
        environment=environment, error_code=float(error_code),
        bug_domain=domain, tech_stack=tech, developer_role=role)

    ui.section('Prediction')
    ui.tiles([
        dict(label='Predicted priority', value=result['priority'],
             foot=result['priority_note'],
             status='critical' if result['priority'] in ('P1', 'P2') else None),
        dict(label='Severity used', value=result['severity'],
             foot=result['severity_source']),
        dict(label='Confidence',
             value=ui.pct(result['confidence'] * 100) if result['confidence'] else '—',
             foot=f"{result['model']} · models/best_priority_model.pkl"),
    ], per_row=3)

    probs = result.get('priority_probs')
    if probs:
        colors = ui.series_colors(result['priority_classes'], ui.PRIORITY_SLOT)
        fig = go.Figure()
        fig.add_bar(x=result['priority_classes'], y=[v * 100 for v in probs],
                    marker_color=[colors[c] for c in result['priority_classes']],
                    text=[f"{v * 100:.1f}%" for v in probs], textposition='outside',
                    textfont=dict(color=p['ink_soft'], size=12),
                    hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>')
        fig.update_layout(title='How the model spread its probability',
                          showlegend=False, yaxis_title='%', yaxis=dict(range=[0, 112]))
        ui.show(fig, height=280)

    # The knowledge base is what the triage stage routes on — show the same
    # recommendation the pipeline would have attached to this bug.
    kb = artifacts['kb'] or {}
    text = description.lower()
    guess = next((cat for cat in kb if cat.lower().replace(' bug', '') in text), None)
    if guess:
        entry = kb[guess]
        st.markdown(
            f"**Routing policy** — a *{guess}* goes to **{entry.get('assigned_role')}**. "
            f"{entry.get('suggested_fix', '')}")


# ──────────────────────────────────────────────────────────────────────────────
#  View 6 — Bug explorer
# ──────────────────────────────────────────────────────────────────────────────
EXPLORER_COLUMNS = ['bug_id', 'title', 'module', 'feature', 'component', 'sprint',
                    'release_version', 'priority', 'severity', 'status', 'resolution',
                    'root_cause', 'assigned_team', 'created_at', 'date_closed',
                    'resolution_days', 'open_age_days', 'environment', 'bug_category',
                    'bug_domain', 'tech_stack', 'error_code']

COLUMN_LABELS = {
    'bug_id': 'Bug ID', 'title': 'Title', 'module': 'Module', 'feature': 'Feature',
    'component': 'Component', 'sprint': 'Sprint', 'release_version': 'Release',
    'priority': 'Priority', 'severity': 'Severity', 'status': 'Status',
    'resolution': 'Resolution', 'root_cause': 'Root cause', 'assigned_team': 'Routed owner',
    'created_at': 'Reported', 'date_closed': 'Closed', 'resolution_days': 'Days to close',
    'open_age_days': 'Age (open)', 'environment': 'Environment',
    'bug_category': 'Category', 'bug_domain': 'Domain', 'tech_stack': 'Tech stack',
    'error_code': 'Error code',
}

DEFAULT_COLUMNS = ['bug_id', 'title', 'module', 'priority', 'severity', 'status',
                   'assigned_team', 'created_at', 'resolution_days', 'open_age_days']


def page_explorer():
    scope, kpis, ctx = context()

    ui.page_header(
        'Bug explorer', 'Records',
        'Every record behind the charts, at row level. The sidebar filters apply '
        'here too — search, sort, pick your columns, and export the slice.')
    scope_banner(scope, ctx['total'], ctx['active'])

    left, right = st.columns([2, 3], gap='medium')
    query = left.text_input('Search title, description or bug ID', placeholder='e.g. memory leak',
                            key='explorer_query')
    columns = right.multiselect('Columns', EXPLORER_COLUMNS, default=DEFAULT_COLUMNS,
                                format_func=lambda c: COLUMN_LABELS.get(c, c),
                                key='explorer_columns') or DEFAULT_COLUMNS

    view = scope
    if query.strip():
        needle = query.strip().lower()
        match = (view['bug_id'].str.lower().str.contains(needle, na=False) |
                 view['title'].str.lower().str.contains(needle, na=False) |
                 view['description'].str.lower().str.contains(needle, na=False))
        view = view[match]

    st.caption(f"{len(view):,} record{'s' if len(view) != 1 else ''} "
               f"{'matching' if query.strip() else 'in scope'}"
               f"{f' “{query.strip()}”' if query.strip() else ''}.")

    if view.empty:
        st.info('Nothing matches that search inside the current filters.')
        footer()
        return

    display = view[columns].rename(columns=COLUMN_LABELS)
    st.dataframe(
        display, width='stretch', hide_index=True, height=520,
        column_config={
            'Reported': st.column_config.DateColumn(format='YYYY-MM-DD'),
            'Closed': st.column_config.DateColumn(format='YYYY-MM-DD'),
            'Days to close': st.column_config.NumberColumn(format='%.0f d'),
            'Age (open)': st.column_config.NumberColumn(format='%.0f d'),
            'Error code': st.column_config.NumberColumn(format='%d'),
            'Title': st.column_config.TextColumn(width='large'),
        })

    export = view[columns].rename(columns=COLUMN_LABELS)
    st.download_button(
        'Download this slice as CSV', export.to_csv(index=False).encode('utf-8'),
        file_name=f'bug_slice_{len(view)}_rows.csv', mime='text/csv',
        icon=':material/download:')

    # ── One record in full ───────────────────────────────────────────────────
    ui.section('Inspect one bug')
    choice = st.selectbox('Bug ID', view['bug_id'].head(500).tolist(),
                          key='explorer_pick',
                          help='The first 500 records of the current slice.')
    record = view[view['bug_id'] == choice].iloc[0]

    ui.tiles([
        dict(label='Priority', value=str(record['priority']),
             foot=str(record['status']),
             status='critical' if str(record['priority']) in ('P1', 'P2') else None),
        dict(label='Severity', value=str(record['severity']), foot=str(record['environment'])),
        dict(label='Module', value=str(record['module']), foot=str(record['component'])),
        dict(label='Age' if pd.isna(record['date_closed']) else 'Days to close',
             value=ui.nf(record['open_age_days'] if pd.isna(record['date_closed'])
                         else record['resolution_days']), unit='d',
             foot=f"reported {record['created_at']:%d %b %Y}"),
    ])

    left, right = st.columns(2, gap='medium')
    with left:
        st.markdown(f"**{record['title']}**")
        st.write(record['description'])
        st.markdown(f"**Root cause** — {record['root_cause']}")
    with right:
        st.markdown(f"**Suggested fix** — {record['suggested_fix']}")
        st.markdown(f"**Routed to** — {record['assigned_team']}")
        st.write(record['explanation'])

    footer()


# ──────────────────────────────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='Bug Management — Dashboard',
        page_icon=':material/bug_report:',
        layout='wide',
        initial_sidebar_state='expanded')

    ui.inject_css()
    ui.install_template()

    try:
        df, origin, snapshot = data.load_bugs()
    except data.DataMissing as exc:
        st.title('Bug Management System')
        st.error(str(exc), icon=':material/error:')
        st.stop()
        return

    pages = [
        st.Page(page_overview,     title='Overview',        icon=':material/space_dashboard:',
                url_path='overview', default=True),
        st.Page(page_trends,       title='Trends',          icon=':material/trending_up:',
                url_path='trends'),
        st.Page(page_distribution, title='Distribution',    icon=':material/donut_large:',
                url_path='distribution'),
        st.Page(page_quality,      title='Quality & team',  icon=':material/verified:',
                url_path='quality'),
        st.Page(page_models,       title='Models & triage', icon=':material/smart_toy:',
                url_path='models'),
        st.Page(page_explorer,     title='Bug explorer',    icon=':material/table_rows:',
                url_path='explorer'),
    ]
    # The nav is drawn first so the view links sit at the top of the sidebar and
    # the filters below them; the page bodies only run at .run().
    navigation = st.navigation(pages, position='sidebar')

    filters, active = sidebar_filters(snapshot, len(df))
    st.session_state['ctx'] = {'filters': filters, 'active': active, 'total': len(df),
                               'snapshot': snapshot, 'origin': origin}
    navigation.run()


main()
