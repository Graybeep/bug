# ==============================================================================
# 09_bug_analysis.py
# Module / Root-Cause / Resolution-Time analysis
#
#   1. Which modules and root causes generate the most bugs.
#   2. Resolution_Time_Hours derived from Date_Reported -> Date_Closed, compared
#      across modules and root causes to surface where the delays are.
#   3. Charts: module-wise bug volume, root-cause distribution, average
#      resolution time, a module x root-cause delay heatmap, and the shape of
#      the resolution-time distribution itself.
#
# Input:  data/bug_reports_enriched.csv (falls back to bug_reports_processed.csv)
# Output: visualizations/analysis_*.png  |  data/bug_analysis_report.json
#         + console Observations
# ==============================================================================

import json
import os
import sys

import _deps
_deps.check('pandas', 'numpy', 'matplotlib', 'seaborn')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENRICHED_PATH  = 'data/bug_reports_enriched.csv'
PROCESSED_PATH = 'data/bug_reports_processed.csv'
REPORT_PATH    = 'data/bug_analysis_report.json'

# The source columns carry the project's snake_case names; the analysis speaks
# in the report's names. One map keeps the two vocabularies in one place.
COLUMN_MAP = {
    'bug_id':      'Bug_ID',
    'module':      'Module',
    'root_cause':  'Root_Cause',
    'created_at':  'Date_Reported',
    'date_closed': 'Date_Closed',
}

HOURS_PER_DAY = 24

# Chart ink — single sequential hue for magnitude, status palette for the
# SLA verdict on resolution time. Never a hue per bar.
BLUE        = '#2a78d6'
BLUE_LIGHT  = '#9ec5f4'
BLUE_DARK   = '#184f95'
SEQ_CMAP    = 'Blues'
STATUS      = {'good': '#0ca30c', 'warning': '#fab219', 'critical': '#d03b3b'}
INK         = '#0b0b0b'
INK_MUTED   = '#898781'
GRID        = '#e1e0d9'


def setup():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        'axes.edgecolor': GRID,
        'grid.color':     GRID,
        'text.color':     INK,
        'axes.labelcolor': INK,
    })
    os.makedirs('visualizations', exist_ok=True)


def save(fig, name):
    path = f"visualizations/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# .title() would render these as 'Ui', 'Api', 'Ci/Cd' — restore the acronyms.
ACRONYMS = {'Ui': 'UI', 'Api': 'API', 'Ci': 'CI', 'Cd': 'CD', 'Ci/Cd': 'CI/CD'}


def short_cause(text):
    """The dataset states root causes as a boilerplate sentence
    ('Misconfiguration or logic issue related to memory leak'). Charts and
    tables only need the distinguishing tail, title-cased."""
    marker = 'related to '
    tail = text.split(marker, 1)[1] if marker in text else text
    words = tail.strip().title().split()
    return ' '.join(ACRONYMS.get(w, w) for w in words)


def block(title):
    print("\n" + "-" * 72)
    print(f"  {title}")
    print("-" * 72)


def table(df, indent=4):
    print('\n'.join(' ' * indent + line
                    for line in df.to_string().split('\n')))


# ── Load & derive ─────────────────────────────────────────────────────────────
def load():
    path = ENRICHED_PATH if os.path.exists(ENRICHED_PATH) else PROCESSED_PATH
    if not os.path.exists(path):
        print(f"  [ERROR] No dataset found. Expected '{ENRICHED_PATH}' or "
              f"'{PROCESSED_PATH}'.")
        print(f"          Run 'python src/01_data_collection.py' first.")
        return None

    df = pd.read_csv(path, usecols=lambda c: c in
                     set(COLUMN_MAP) | {'severity', 'priority', 'status'})
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"  [ERROR] '{path}' is missing required column(s): {missing}")
        print(f"          Run 'python src/01_data_collection.py' to rebuild it.")
        return None

    df = df.rename(columns=COLUMN_MAP)
    print(f"\n  Loaded '{path}'  |  {len(df):,} bugs x {df.shape[1]} columns")

    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], errors='coerce')
    df['Date_Closed']   = pd.to_datetime(df['Date_Closed'],   errors='coerce')
    df['Root_Cause_Short'] = df['Root_Cause'].astype(str).map(short_cause)

    # Resolution_Time_Hours — only defined for bugs that actually closed.
    # Both dates are recorded at day granularity in this dataset, so the hour
    # figure is exact to the day (x24) and not to the clock time of closure.
    delta = df['Date_Closed'] - df['Date_Reported']
    df['Resolution_Time_Hours'] = delta.dt.total_seconds() / 3600.0

    # A close date before the report date is impossible — drop those from the
    # timing population rather than letting a negative age pull the means down.
    backwards = int((df['Resolution_Time_Hours'] < 0).sum())
    if backwards:
        print(f"  [WARN] {backwards} bug(s) closed before they were reported "
              f"— excluded from timing analysis.")
        df.loc[df['Resolution_Time_Hours'] < 0, 'Resolution_Time_Hours'] = np.nan

    # Age of the bugs still on the board, measured to the latest activity in
    # the dataset. These carry no resolution time but are the real backlog.
    as_of = max(df['Date_Reported'].max(), df['Date_Closed'].max())
    open_mask = df['Date_Closed'].isna()
    df['Open_Age_Hours'] = np.where(
        open_mask, (as_of - df['Date_Reported']).dt.total_seconds() / 3600.0, np.nan)
    df.attrs['as_of'] = as_of
    return df


# ── [A] Volume: which modules and root causes generate the most bugs ──────────
def analyse_volume(df):
    block("[A] BUG VOLUME BY MODULE AND ROOT CAUSE")
    total = len(df)

    mod = (df.groupby('Module')
             .agg(Bugs=('Bug_ID', 'count'))
             .sort_values('Bugs', ascending=False))
    mod['Share_%'] = (mod['Bugs'] / total * 100).round(2)

    cause = (df.groupby('Root_Cause_Short')
               .agg(Bugs=('Bug_ID', 'count'))
               .sort_values('Bugs', ascending=False))
    cause['Share_%'] = (cause['Bugs'] / total * 100).round(2)

    print(f"\n  Bugs per module ({len(mod)} modules, {total:,} bugs):")
    table(mod)
    print(f"\n  Bugs per root cause ({len(cause)} distinct causes):")
    table(cause)

    top_mod = mod.index[0]
    top_cause = cause.index[0]
    print(f"\n  Observation:")
    print(f"    - Highest-volume module     : {top_mod} "
          f"({mod.loc[top_mod, 'Bugs']:,} bugs, {mod.loc[top_mod, 'Share_%']}%)")
    print(f"    - Highest-volume root cause : {top_cause} "
          f"({cause.loc[top_cause, 'Bugs']:,} bugs, {cause.loc[top_cause, 'Share_%']}%)")
    print(f"    - Top 3 modules hold {mod['Share_%'].head(3).sum():.1f}% of all bugs; "
          f"top 5 causes hold {cause['Share_%'].head(5).sum():.1f}%.")

    spread = mod['Bugs'].max() / mod['Bugs'].min()
    if spread < 1.25:
        print(f"    - Volume is near-uniform across modules (max/min = {spread:.2f}x),")
        print(f"      so raw counts do NOT single out a problem area. The signal in")
        print(f"      this dataset is resolution time, not bug count — see [B].")
    return mod, cause


# ── [B] Timing: Resolution_Time_Hours and where the delays are ────────────────
def analyse_timing(df):
    block("[B] RESOLUTION TIME (Date_Closed - Date_Reported)")

    closed = df[df['Resolution_Time_Hours'].notna()]
    n_open = len(df) - len(closed)
    hrs = closed['Resolution_Time_Hours']

    print(f"\n  Population:")
    print(f"    Closed bugs (timed)   : {len(closed):,}  ({len(closed)/len(df)*100:.1f}%)")
    print(f"    Still open (untimed)  : {n_open:,}  ({n_open/len(df)*100:.1f}%)")
    print(f"    Dataset as of         : {df.attrs['as_of'].date()}")

    overall = {
        'mean':   hrs.mean(),
        'median': hrs.median(),
        'p75':    hrs.quantile(0.75),
        'p90':    hrs.quantile(0.90),
        'max':    hrs.max(),
    }
    print(f"\n  Resolution_Time_Hours across all closed bugs:")
    print(f"    Mean   : {overall['mean']:>9,.1f} h  ({overall['mean']/HOURS_PER_DAY:>6.1f} days)")
    print(f"    Median : {overall['median']:>9,.1f} h  ({overall['median']/HOURS_PER_DAY:>6.1f} days)")
    print(f"    P75    : {overall['p75']:>9,.1f} h  ({overall['p75']/HOURS_PER_DAY:>6.1f} days)")
    print(f"    P90    : {overall['p90']:>9,.1f} h  ({overall['p90']/HOURS_PER_DAY:>6.1f} days)")
    print(f"    Max    : {overall['max']:>9,.1f} h  ({overall['max']/HOURS_PER_DAY:>6.1f} days)")
    print(f"\n    Mean sits {overall['mean']/overall['median']:.1f}x the median — the"
          f" distribution is right-skewed,")
    print(f"    so a tail of long-running bugs, not the typical bug, drives the average.")

    # Per-module timing, compared against the overall mean.
    mod = (closed.groupby('Module')['Resolution_Time_Hours']
                 .agg(Closed='count', Mean_Hours='mean', Median_Hours='median',
                      P90_Hours=lambda s: s.quantile(0.90), Max_Hours='max')
                 .sort_values('Mean_Hours', ascending=False))
    mod['vs_Overall_h'] = (mod['Mean_Hours'] - overall['mean'])
    mod = mod.round(1)
    print(f"\n  Resolution time by module (hours, sorted slowest first):")
    table(mod)

    # Per-root-cause timing.
    cause = (closed.groupby('Root_Cause_Short')['Resolution_Time_Hours']
                   .agg(Closed='count', Mean_Hours='mean', Median_Hours='median',
                        P90_Hours=lambda s: s.quantile(0.90))
                   .sort_values('Mean_Hours', ascending=False))
    cause['vs_Overall_h'] = (cause['Mean_Hours'] - overall['mean'])
    cause = cause.round(1)
    print(f"\n  Resolution time by root cause (hours, sorted slowest first):")
    table(cause)

    slow_mod, fast_mod = mod.index[0], mod.index[-1]
    slow_cause = cause.index[0]
    print(f"\n  Observation — where the delay is:")
    print(f"    - Slowest module  : {slow_mod} at {mod.loc[slow_mod,'Mean_Hours']:,.0f} h mean "
          f"({mod.loc[slow_mod,'vs_Overall_h']:+,.0f} h vs overall)")
    print(f"    - Fastest module  : {fast_mod} at {mod.loc[fast_mod,'Mean_Hours']:,.0f} h mean "
          f"({mod.loc[fast_mod,'vs_Overall_h']:+,.0f} h vs overall)")
    print(f"    - Module spread   : {mod['Mean_Hours'].max() - mod['Mean_Hours'].min():,.0f} h "
          f"between slowest and fastest")
    print(f"    - Slowest cause   : {slow_cause} at {cause.loc[slow_cause,'Mean_Hours']:,.0f} h mean "
          f"({cause.loc[slow_cause,'vs_Overall_h']:+,.0f} h vs overall)")

    # The long tail is the actionable delay: bugs past P90.
    tail = closed[closed['Resolution_Time_Hours'] >= overall['p90']]
    tail_mod = (tail.groupby('Module').size() / closed.groupby('Module').size() * 100
                ).sort_values(ascending=False).round(1)
    print(f"\n  Long-tail delays — bugs at or beyond the P90 mark "
          f"({overall['p90']:,.0f} h / {overall['p90']/HOURS_PER_DAY:.0f} days):")
    print(f"    {len(tail):,} bugs. Share of each module's closed bugs that land in the tail:")
    table(tail_mod.to_frame('Tail_%'), indent=6)

    # Ageing backlog — open bugs carry no resolution time but are the real risk.
    open_bugs = df[df['Open_Age_Hours'].notna()]
    if len(open_bugs):
        age = (open_bugs.groupby('Module')['Open_Age_Hours']
                        .agg(Open='count', Mean_Age_Hours='mean',
                             Median_Age_Hours='median', Max_Age_Hours='max')
                        .sort_values('Mean_Age_Hours', ascending=False).round(1))
        stale = int((open_bugs['Open_Age_Hours'] > overall['p90']).sum())
        print(f"\n  Ageing backlog — open bugs, age as of {df.attrs['as_of'].date()}:")
        table(age)
        print(f"\n    - {stale:,} open bugs ({stale/len(open_bugs)*100:.1f}% of the backlog) have")
        print(f"      already aged past the P90 closure mark and are the overdue set.")

    return overall, mod, cause, tail_mod


# ── Chart 1: Module-wise bug volume ───────────────────────────────────────────
def plot_module_volume(mod):
    counts = mod['Bugs']
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.62,
                  edgecolor='#fcfcfb', linewidth=2)
    total = counts.sum()
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + total * 0.004,
                f'{int(h):,}\n{h/total*100:.1f}%', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=INK)

    ax.axhline(counts.mean(), color=INK_MUTED, linestyle='--', linewidth=1.2, zorder=0)
    ax.text(len(counts) - 0.45, counts.mean(), f'  mean {counts.mean():,.0f}',
            va='center', ha='left', fontsize=9, color=INK_MUTED)

    ax.set_title('Bugs by Module', fontsize=15, fontweight='bold', pad=16)
    ax.set_xlabel('')
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.set_ylim(0, counts.max() * 1.16)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='x', labelrotation=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(False)
    fig.tight_layout()
    return save(fig, 'analysis_bugs_by_module')


# ── Chart 2: Root-cause distribution ──────────────────────────────────────────
def plot_root_cause(cause):
    counts = cause['Bugs'].sort_values()          # ascending -> largest on top
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(counts.index, counts.values, color=BLUE, height=0.68,
                   edgecolor='#fcfcfb', linewidth=2)
    # Only the leaders earn the deeper step — rank is read from position, and
    # the fill stays one hue.
    for bar in bars.patches[-3:]:
        bar.set_color(BLUE_DARK)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f'{int(w):,}  ({w/total*100:.1f}%)', va='center', ha='left',
                fontsize=9.5, color=INK)

    ax.set_title('Root-Cause Distribution', fontsize=15, fontweight='bold', pad=16)
    ax.set_xlabel('Number of Bugs', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(0, counts.max() * 1.20)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(False)
    fig.tight_layout()
    return save(fig, 'analysis_root_cause_distribution')


# ── Chart 3: Average resolution time by module, against the overall mean ──────
def plot_resolution_by_module(mod_time, overall):
    means = mod_time['Mean_Hours'].sort_values()
    ref   = overall['mean']

    # Polarity, not identity: the only thing separating these modules is which
    # side of the overall mean they fall on, so the fill is the diverging pair
    # (slower = red, faster = blue). Every bar is labelled with its own figure
    # and its signed gap, so colour never carries the reading alone.
    SLOWER, FASTER = '#d03b3b', BLUE
    colors = [SLOWER if v > ref else FASTER for v in means.values]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(means.index, means.values, color=colors, height=0.62,
                   edgecolor='#fcfcfb', linewidth=2)

    # Labels sit in one column clear of the longest bar and of the mean line,
    # so they never collide with the marks.
    label_x = means.max() * 1.06
    for bar, val in zip(bars, means.values):
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f} h  ({val/HOURS_PER_DAY:.1f} d)   {val-ref:+,.0f} h vs mean',
                va='center', ha='left', fontsize=10, fontweight='bold', color=INK)

    ax.axvline(ref, color=INK, linestyle='--', linewidth=1.4, zorder=3)
    ax.text(ref, len(means) - 0.42, f' overall mean {ref:,.0f} h',
            va='bottom', ha='left', fontsize=9.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (FASTER, SLOWER)]
    ax.legend(handles, ['Faster than overall mean', 'Slower than overall mean'],
              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=10)

    ax.set_title('Average Resolution Time by Module',
                 fontsize=15, fontweight='bold', pad=16)
    ax.set_xlabel('Mean Resolution_Time_Hours', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(0, means.max() * 1.62)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(False)
    fig.tight_layout()
    return save(fig, 'analysis_avg_resolution_by_module')


# ── Chart 4: Module x root-cause delay heatmap ────────────────────────────────
def plot_delay_heatmap(df):
    closed = df[df['Resolution_Time_Hours'].notna()]
    pivot = closed.pivot_table(index='Root_Cause_Short', columns='Module',
                               values='Resolution_Time_Hours', aggfunc='mean')
    # Order both axes by their own mean so the slow corner lands top-left.
    pivot = (pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index,
                       pivot.mean(axis=0).sort_values(ascending=False).index])

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(pivot, cmap=SEQ_CMAP, annot=True, fmt='.0f', linewidths=2,
                linecolor='#fcfcfb', ax=ax,
                cbar_kws={'label': 'Mean Resolution_Time_Hours'},
                annot_kws={'fontsize': 8.5})
    ax.set_title('Mean Resolution Time — Module x Root Cause\n'
                 '(hours; darker = slower)',
                 fontsize=14, fontweight='bold', pad=16)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=25)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.tick_params(axis='y', labelrotation=0)
    fig.tight_layout()
    return save(fig, 'analysis_resolution_heatmap')


# ── Chart 5: Shape of the resolution-time distribution ────────────────────────
def plot_resolution_distribution(df, overall):
    hrs = df['Resolution_Time_Hours'].dropna()

    # The extreme tail runs to ~7,500 h on a handful of bugs and would flatten
    # the body of the distribution into an unreadable spike. Cut the view at
    # P99 and account for the remainder in the annotation.
    cutoff = hrs.quantile(0.99)
    shown  = hrs[hrs <= cutoff]
    beyond = int((hrs > cutoff).sum())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(shown, bins=60, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.8)
    top = ax.get_ylim()[1]

    # Median and mean sit close together on this scale, so their labels are
    # staggered in height and hung on opposite sides of their own lines.
    for value, label, color, style, height, side in (
            (overall['median'], f"median {overall['median']:,.0f} h", STATUS['good'],     '-',  0.97, 'right'),
            (overall['mean'],   f"mean {overall['mean']:,.0f} h",     INK,                '--', 0.80, 'left'),
            (overall['p90'],    f"P90 {overall['p90']:,.0f} h",       STATUS['critical'], '-',  0.97, 'left')):
        ax.axvline(value, color=color, linestyle=style, linewidth=1.8)
        offset = cutoff * (0.008 if side == 'left' else -0.008)
        ax.text(value + offset, top * height, label,
                va='top', ha=side, fontsize=9.5, fontweight='bold', color=color)

    tail = int((hrs >= overall['p90']).sum())
    ax.annotate(f'{tail:,} bugs ({tail/len(hrs)*100:.0f}%) sit beyond P90 — the delay tail.\n'
                f'{beyond:,} run past this view, out to '
                f'{overall["max"]:,.0f} h / {overall["max"]/HOURS_PER_DAY:.0f} days.',
                xy=(overall['p90'] * 1.12, top * 0.58),
                fontsize=10, color=INK,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fcfcfb',
                          edgecolor=GRID))

    ax.set_title('Distribution of Resolution_Time_Hours (closed bugs)',
                 fontsize=15, fontweight='bold', pad=16)
    ax.set_xlim(0, cutoff)
    ax.set_xlabel(f'Resolution_Time_Hours  (view cut at P99 = {cutoff:,.0f} h)',
                  fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(False)
    fig.tight_layout()
    return save(fig, 'analysis_resolution_time_distribution')


# ── Report ────────────────────────────────────────────────────────────────────
def write_report(df, mod_vol, cause_vol, overall, mod_time, cause_time, charts):
    report = {
        'generated_from':   ENRICHED_PATH if os.path.exists(ENRICHED_PATH) else PROCESSED_PATH,
        'as_of':            str(df.attrs['as_of'].date()),
        'total_bugs':       int(len(df)),
        'closed_bugs':      int(df['Resolution_Time_Hours'].notna().sum()),
        'open_bugs':        int(df['Open_Age_Hours'].notna().sum()),
        'resolution_time_hours_overall': {k: round(float(v), 2) for k, v in overall.items()},
        'bugs_by_module':      mod_vol.to_dict(orient='index'),
        'bugs_by_root_cause':  cause_vol.to_dict(orient='index'),
        'resolution_by_module':     mod_time.to_dict(orient='index'),
        'resolution_by_root_cause': cause_time.to_dict(orient='index'),
        'charts': charts,
        'notes': [
            'Resolution_Time_Hours = (Date_Closed - Date_Reported) in hours.',
            'Both dates are stored at day granularity, so hours are exact to '
            'the day (days x 24), not to the clock time of closure.',
            'Timing statistics cover closed bugs only; open bugs are reported '
            'separately as an ageing backlog.',
        ],
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n  Saved: {REPORT_PATH}")


def main():
    print("=" * 72)
    print("  MODULE / ROOT-CAUSE / RESOLUTION-TIME ANALYSIS")
    print("=" * 72)

    df = load()
    if df is None:
        return

    setup()
    mod_vol, cause_vol = analyse_volume(df)
    overall, mod_time, cause_time, _ = analyse_timing(df)

    block("[C] VISUALIZATIONS")
    charts = {
        'bugs_by_module':      plot_module_volume(mod_vol),
        'root_cause':          plot_root_cause(cause_vol),
        'avg_resolution':      plot_resolution_by_module(mod_time, overall),
        'delay_heatmap':       plot_delay_heatmap(df),
        'resolution_spread':   plot_resolution_distribution(df, overall),
    }
    write_report(df, mod_vol, cause_vol, overall, mod_time, cause_time, charts)

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
