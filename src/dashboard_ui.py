# ==============================================================================
# dashboard_ui.py
# The look of the Streamlit dashboard: palette, Plotly template, CSS and the
# handful of composite components the pages reuse (stat tiles, section headers,
# chart cards, table views).
#
# Colour rules, carried over from the retired HTML build and unchanged:
#   - the categorical set is assigned in FIXED order, never cycled, so a series
#     keeps its colour when a filter changes the series count;
#   - sequential heat is ONE hue, light -> dark;
#   - status colours (good/warning/serious/critical) are reserved and never
#     reused as a series colour, and always ship with a label, not colour alone;
#   - text wears ink tokens, never a series colour — identity comes from a
#     coloured mark beside the text.
#
# Dark and light are two deliberate palettes, not an inverted flip; both pass
# the lightness band, chroma floor, CVD separation and contrast checks. Light
# sits slightly under the 3:1 surface-contrast bar on three hues, which is why
# every chart here carries a legend and a "Show the numbers" table view.
# ==============================================================================

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
#  Palette
# ──────────────────────────────────────────────────────────────────────────────
DARK = {
    'name':      'dark',
    'page':      '#0d0d0d',
    'surface':   '#1a1a19',
    'surface_2': '#232322',
    'ink':       '#ffffff',
    'ink_soft':  '#c3c2b7',
    'ink_faint': '#898781',
    'line':      '#2c2c2a',
    'baseline':  '#383835',
    'series':    ['#3987e5', '#d95926', '#199e70', '#c98500',
                  '#d55181', '#008300', '#9085e9', '#e66767'],
    'seq':       ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
                  '#256abf', '#184f95', '#0d366b'],
    'good':      '#0ca30c',
    'warning':   '#fab219',
    'serious':   '#ec835a',
    'critical':  '#e66767',
}

LIGHT = {
    'name':      'light',
    'page':      '#f9f9f7',
    'surface':   '#fcfcfb',
    'surface_2': '#f2f1ed',
    'ink':       '#0b0b0b',
    'ink_soft':  '#52514e',
    'ink_faint': '#898781',
    'line':      '#e1e0d9',
    'baseline':  '#c3c2b7',
    'series':    ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                  '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
    'seq':       ['#e3eefd', '#b7d3f6', '#86b6ef', '#3987e5',
                  '#256abf', '#184f95', '#0d366b'],
    'good':      '#0a7d0a',
    'warning':   '#b07800',
    'serious':   '#c25c2c',
    'critical':  '#c93b3b',
}

# Fixed slot per recurring entity, so priority P1 (or a given model) wears the
# same colour on every page rather than being recoloured per chart.
PRIORITY_SLOT = {'P1': 7, 'P2': 1, 'P3': 3, 'P4': 0, 'P5': 2}
SEVERITY_SLOT = {'Critical': 7, 'High': 1, 'Medium': 3, 'Low': 0}
FLOW_SLOT     = {'opened': 1, 'closed': 2, 'backlog': 0, 'net': 6}


def palette():
    """The palette for the viewer's current theme."""
    try:
        return DARK if st.context.theme.type == 'dark' else LIGHT
    except Exception:                                              # noqa: BLE001
        return DARK


def series_colors(keys, slots):
    """Fixed colour per key, so a filtered-out series never repaints the rest."""
    p = palette()
    return {k: p['series'][slots.get(k, i) % len(p['series'])] for i, k in enumerate(keys)}


def seq_scale():
    return palette()['seq']


# ──────────────────────────────────────────────────────────────────────────────
#  Plotly template
# ──────────────────────────────────────────────────────────────────────────────
def install_template():
    """Register and select the project's Plotly template for this rerun."""
    p = palette()
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=p['series'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='"Source Sans Pro", -apple-system, "Segoe UI", sans-serif',
                  size=13, color=p['ink_soft']),
        title=dict(font=dict(size=15, color=p['ink']), x=0, xanchor='left',
                   y=0.97, yanchor='top', pad=dict(b=12)),
        margin=dict(l=8, r=12, t=44, b=8),
        # Recessive axes: a horizontal rule set only, no vertical grid, no
        # zero line competing with the baseline.
        xaxis=dict(showgrid=False, zeroline=False, showline=True,
                   linecolor=p['baseline'], ticks='outside', ticklen=4,
                   tickcolor=p['line'], tickfont=dict(size=12, color=p['ink_faint']),
                   title=dict(font=dict(size=12, color=p['ink_faint']))),
        yaxis=dict(showgrid=True, gridcolor=p['line'], gridwidth=1, zeroline=False,
                   showline=False, tickfont=dict(size=12, color=p['ink_faint']),
                   title=dict(font=dict(size=12, color=p['ink_faint']))),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0,
                    font=dict(size=12, color=p['ink_soft']), itemsizing='constant',
                    bgcolor='rgba(0,0,0,0)', title=dict(text='')),
        hoverlabel=dict(bgcolor=p['surface_2'], bordercolor=p['line'],
                        font=dict(size=12, color=p['ink'])),
        hovermode='closest',
        barcornerradius=4,          # 4px rounded data-end, anchored to baseline
        bargap=0.28,
        bargroupgap=0.12,
        dragmode=False,
        colorscale=dict(sequential=[[i / (len(p['seq']) - 1), c]
                                    for i, c in enumerate(p['seq'])]),
    )
    pio.templates['bugmgmt'] = template
    pio.templates.default = 'bugmgmt'
    return template


PLOT_CONFIG = {
    'displaylogo': False,
    'displayModeBar': False,
    'scrollZoom': False,
    'responsive': True,
}


def show(fig, height=320, key=None):
    """Render a figure at a fixed height with the project's chart config."""
    fig.update_layout(height=height, autosize=True)
    st.plotly_chart(fig, width='stretch', config=PLOT_CONFIG, key=key)


def table_view(frame, label="Show the numbers", column_config=None, hide_index=True):
    """The table behind a chart. Present on every chart — identity and value are
    never colour-only, which is also what the light palette's contrast WARN
    requires as relief."""
    with st.expander(label):
        st.dataframe(frame, width='stretch', hide_index=hide_index,
                     column_config=column_config)


# ──────────────────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
  /* Tighter page frame — the default padding wastes a third of the fold. */
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

  /* Page header ------------------------------------------------------------ */
  .page-head { margin: 0 0 1.1rem 0; }
  .page-head .eyebrow {
    display: inline-block; font-size: .72rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase;
    color: var(--ink-faint); margin-bottom: .35rem;
  }
  .page-head h1 {
    font-size: 1.65rem; font-weight: 700; line-height: 1.2;
    margin: 0 0 .3rem 0; color: var(--ink);
  }
  .page-head p { font-size: .92rem; color: var(--ink-soft); margin: 0; max-width: 78ch; }

  /* Section rule ----------------------------------------------------------- */
  .section {
    display: flex; align-items: baseline; gap: .6rem;
    margin: 1.9rem 0 .7rem 0; padding-bottom: .45rem;
    border-bottom: 1px solid var(--line);
  }
  .section h2 { font-size: 1.02rem; font-weight: 650; margin: 0; color: var(--ink); }
  .section span { font-size: .82rem; color: var(--ink-faint); }

  /* Stat tiles ------------------------------------------------------------- */
  .tiles { display: grid; gap: .7rem; margin-bottom: .3rem; }
  .tile {
    position: relative; overflow: hidden;
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: .85rem .95rem .8rem 1.05rem;
  }
  .tile::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent);
  }
  .tile .t-label {
    font-size: .74rem; font-weight: 600; letter-spacing: .04em;
    text-transform: uppercase; color: var(--ink-faint);
    display: block; margin-bottom: .3rem; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .tile .t-value {
    font-size: 1.72rem; font-weight: 700; line-height: 1.05;
    color: var(--ink); font-variant-numeric: tabular-nums;
  }
  .tile .t-unit { font-size: .95rem; font-weight: 600; color: var(--ink-soft); margin-left: .12rem; }
  .tile .t-foot {
    display: block; margin-top: .35rem; font-size: .76rem;
    color: var(--ink-faint); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .tile .t-foot b { font-weight: 650; }

  /* Status chips ----------------------------------------------------------- */
  .chip {
    display: inline-flex; align-items: center; gap: .35rem;
    font-size: .76rem; font-weight: 600; border-radius: 999px;
    padding: .12rem .5rem; border: 1px solid var(--line);
    background: var(--surface-2); color: var(--ink-soft);
  }
  .chip .dot { width: .5rem; height: .5rem; border-radius: 50%; flex: none; }

  /* Insight cards ---------------------------------------------------------- */
  .insight {
    display: flex; gap: .75rem; align-items: flex-start;
    background: var(--surface); border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 10px; padding: .75rem .9rem; margin-bottom: .55rem;
  }
  .insight .i-body { min-width: 0; }
  .insight .i-title { font-weight: 650; color: var(--ink); font-size: .93rem; margin-bottom: .15rem; }
  .insight .i-text { color: var(--ink-soft); font-size: .87rem; line-height: 1.45; }
  .insight .i-badge {
    flex: none; font-size: .68rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; padding: .18rem .45rem; border-radius: 5px;
    background: var(--surface-2); color: var(--accent); border: 1px solid var(--line);
  }

  /* Caveat note ------------------------------------------------------------ */
  .note {
    border: 1px dashed var(--line); border-radius: 10px;
    padding: .7rem .9rem; margin: .8rem 0 .2rem 0;
    font-size: .84rem; line-height: 1.5; color: var(--ink-soft);
    background: color-mix(in srgb, var(--surface) 60%, transparent);
  }
  .note b { color: var(--ink); }
  .note code {
    font-size: .8rem; background: var(--surface-2); border-radius: 4px;
    padding: .05rem .3rem; color: var(--ink-soft);
  }

  /* Streamlit chrome ------------------------------------------------------- */
  [data-testid="stSidebarHeader"] { padding-bottom: .3rem; }
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
  [data-testid="stSidebarUserContent"] hr { margin: .8rem 0; }

  div[data-baseweb="tab-list"] { gap: .1rem; border-bottom: 1px solid var(--line); }
  button[data-baseweb="tab"] { padding-left: .85rem; padding-right: .85rem; }

  /* Plotly leaves its own margin; the card already provides the breathing room. */
  [data-testid="stPlotlyChart"] { margin-top: -.35rem; }

  [data-testid="stExpander"] details { border-color: var(--line); border-radius: 10px; }
  [data-testid="stExpander"] summary { font-size: .84rem; color: var(--ink-faint); }

  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

  /* The footer's caveats matter as much as the numbers — keep them legible. */
  .page-foot {
    margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid var(--line);
    font-size: .8rem; line-height: 1.55; color: var(--ink-faint);
  }
  .page-foot b { color: var(--ink-soft); }
</style>
"""


def inject_css():
    """Theme tokens + the component CSS. Runs once per rerun, before anything
    is drawn, so the tokens exist by the time a tile references them."""
    p = palette()
    tokens = f"""
    <style>
      :root {{
        --page: {p['page']}; --surface: {p['surface']}; --surface-2: {p['surface_2']};
        --ink: {p['ink']}; --ink-soft: {p['ink_soft']}; --ink-faint: {p['ink_faint']};
        --line: {p['line']}; --baseline: {p['baseline']};
        --accent: {p['series'][0]};
        --good: {p['good']}; --warning: {p['warning']};
        --serious: {p['serious']}; --critical: {p['critical']};
      }}
    </style>
    """
    st.markdown(tokens + CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Components
# ──────────────────────────────────────────────────────────────────────────────
def page_header(title, eyebrow=None, sub=None):
    parts = ['<div class="page-head">']
    if eyebrow:
        parts.append(f'<span class="eyebrow">{eyebrow}</span>')
    parts.append(f'<h1>{title}</h1>')
    if sub:
        parts.append(f'<p>{sub}</p>')
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def section(title, note=None):
    extra = f'<span>{note}</span>' if note else ''
    st.markdown(f'<div class="section"><h2>{title}</h2>{extra}</div>',
                unsafe_allow_html=True)


def _tile_html(label, value, unit=None, foot=None, status=None):
    accent = f'var(--{status})' if status else 'var(--accent)'
    unit_html = f'<span class="t-unit">{unit}</span>' if unit else ''
    foot_html = f'<span class="t-foot">{foot}</span>' if foot else ''
    return (f'<div class="tile" style="--accent:{accent}">'
            f'<span class="t-label">{label}</span>'
            f'<span class="t-value">{value}{unit_html}</span>'
            f'{foot_html}</div>')


def tiles(items, per_row=4):
    """A row of stat tiles. Each item: label, value, unit, foot, status.

    A stat tile is the right form for a single headline number — a one-bar
    chart would be worse. Status colours carry a word in the footer too, so
    state is never colour-alone.
    """
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        for column, item in zip(st.columns(len(chunk), gap='small'), chunk):
            with column:
                st.markdown(_tile_html(**item), unsafe_allow_html=True)


def chip(text, status=None):
    color = f'var(--{status})' if status else 'var(--ink-faint)'
    return (f'<span class="chip"><span class="dot" style="background:{color}"></span>'
            f'{text}</span>')


def insight_card(title, text, badge=None, status=None):
    accent = f'var(--{status})' if status else 'var(--accent)'
    badge_html = f'<span class="i-badge">{badge}</span>' if badge else ''
    st.markdown(
        f'<div class="insight" style="--accent:{accent}">{badge_html}'
        f'<div class="i-body"><div class="i-title">{title}</div>'
        f'<div class="i-text">{text}</div></div></div>',
        unsafe_allow_html=True)


def note(html):
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Formatting
# ──────────────────────────────────────────────────────────────────────────────
def nf(value):
    return '—' if value is None else f"{value:,.0f}"


def nf1(value):
    return '—' if value is None else f"{value:,.1f}"


def pct(value, digits=1):
    return '—' if value is None else f"{value:.{digits}f}%"


def days(value):
    return '—' if value is None else f"{value:,.1f}d"


def delta(current, baseline, unit='', invert=False, digits=1):
    """'+2.3 vs all bugs' — the footer line under a filtered tile.

    Returns (text, status). invert=True means lower is better.
    """
    if current is None or baseline is None:
        return None, None
    diff = current - baseline
    if abs(diff) < 10 ** -digits:
        return 'same as the full dataset', None
    better = (diff < 0) if invert else (diff > 0)
    sign = '+' if diff > 0 else '−'
    return (f"<b>{sign}{abs(diff):,.{digits}f}{unit}</b> vs all bugs",
            'good' if better else 'serious')
