# ==============================================================================
# 03_visualization.py
# Task 4: Data Visualization
# Generates 6 charts from the processed dataset and prints key Observations.
# Input:  data/bug_reports_processed.csv
# Output: visualizations/*.png  +  console Observations
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE_MAIN   = 'viridis'
PALETTE_STATUS = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
PALETTE_PIE    = ['#E53935', '#FB8C00', '#FDD835', '#43A047']   # Critical → Trivial

def setup():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    os.makedirs('visualizations', exist_ok=True)

def save(fig, name):
    path = f"visualizations/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Chart 1: Bug Status Distribution ──────────────────────────────────────────
def plot_status(df):
    counts = df['Status'].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=PALETTE_STATUS[:len(counts)], edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Bug Status Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Status', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_status_distribution')
    return counts

# ── Chart 2: Bug Priority Distribution ────────────────────────────────────────
def plot_priority(df):
    order  = ['P1', 'P2', 'P3', 'P4', 'P5']
    order  = [p for p in order if p in df['Priority'].unique()]
    counts = df['Priority'].value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette('magma', len(order))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Bug Priority Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Priority', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_priority_distribution')
    return counts

# ── Chart 3: Bug Severity Distribution (Pie) ──────────────────────────────────
def plot_severity(df):
    order  = ['Critical', 'Major', 'Minor', 'Trivial']
    counts = df['Severity'].value_counts().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=PALETTE_PIE[:len(counts)],
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    ax.set_title('Bug Severity Distribution', fontsize=15, fontweight='bold', pad=20)
    fig.tight_layout()
    save(fig, 'bug_severity_distribution')
    return counts

# ── Chart 4: Bugs Assigned to Developers ──────────────────────────────────────
def plot_developers(df):
    if 'Assigned_To' not in df.columns:
        print("  [SKIP] 'Assigned_To' column not found.")
        return None
    counts = df['Assigned_To'].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('crest', len(counts))
    bars = ax.barh(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width())}', va='center', fontsize=10, fontweight='bold')
    ax.set_title('Bugs Assigned to Developers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Bugs', fontsize=12)
    ax.set_ylabel('Developer', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bugs_assigned_to_developers')
    return counts

# ── Chart 5: Bug Reporting Trend Over Time ────────────────────────────────────
def plot_trend(df):
    if 'Reported_Date' not in df.columns:
        print("  [SKIP] 'Reported_Date' column not found.")
        return None
    df['Reported_Date'] = pd.to_datetime(df['Reported_Date'], errors='coerce')
    monthly = df.groupby(df['Reported_Date'].dt.to_period('M')).size()
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, color='#1565C0', linewidth=2.5, marker='o', markersize=6)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#1565C0')
    ax.set_title('Bug Reporting Trend Over Time', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Bugs Reported', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_reporting_trend')
    return monthly

# ── Chart 6: Bugs by Module ───────────────────────────────────────────────────
def plot_modules(df):
    if 'Module' not in df.columns:
        print("  [SKIP] 'Module' column not found.")
        return None
    counts = df['Module'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('flare', len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Bugs by Module', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Module', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bugs_by_module')
    return counts

# ── Observations ──────────────────────────────────────────────────────────────
def print_observations(status_c, priority_c, severity_c, dev_c, trend_c, module_c):
    print("\n" + "=" * 60)
    print("  OBSERVATIONS")
    print("=" * 60)

    if status_c is not None:
        top_status = status_c.idxmax()
        print(f"\n  1. Bug Status Distribution")
        print(f"     - Most bugs are in '{top_status}' status ({status_c.max()} bugs).")
        closed = status_c.get('Closed', 0) + status_c.get('Resolved', 0)
        total  = status_c.sum()
        print(f"     - {closed}/{total} ({100*closed//total}%) bugs are resolved/closed.")

    if priority_c is not None:
        top_pri = priority_c.idxmax()
        print(f"\n  2. Bug Priority Distribution")
        print(f"     - '{top_pri}' is the most common priority ({priority_c.max()} bugs).")
        p1 = priority_c.get('P1', 0)
        print(f"     - {p1} critical P1 bugs need immediate attention.")

    if severity_c is not None:
        top_sev = severity_c.idxmax()
        crit    = severity_c.get('Critical', 0)
        total   = severity_c.sum()
        print(f"\n  3. Bug Severity Distribution")
        print(f"     - '{top_sev}' is the most frequent severity level ({severity_c.max()} bugs).")
        print(f"     - {crit} Critical bugs ({100*crit//total}%) require urgent resolution.")

    if dev_c is not None:
        top_dev = dev_c.idxmax()
        low_dev = dev_c.idxmin()
        print(f"\n  4. Bugs Assigned to Developers")
        print(f"     - '{top_dev}' has the highest bug load ({dev_c.max()} bugs).")
        print(f"     - '{low_dev}' has the fewest assigned bugs ({dev_c.min()} bugs).")
        print(f"     - Consider rebalancing workload across the team.")

    if trend_c is not None:
        peak_month = trend_c.idxmax()
        low_month  = trend_c.idxmin()
        print(f"\n  5. Bug Reporting Trend Over Time")
        print(f"     - Peak reporting month: {peak_month} ({trend_c.max()} bugs).")
        print(f"     - Lowest reporting month: {low_month} ({trend_c.min()} bugs).")

    if module_c is not None:
        top_mod = module_c.idxmax()
        low_mod = module_c.idxmin()
        print(f"\n  6. Bugs by Module")
        print(f"     - '{top_mod}' module has the most bugs ({module_c.max()}) — needs priority attention.")
        print(f"     - '{low_mod}' module is the most stable ({module_c.min()} bugs).")

    print("\n" + "=" * 60)

# ── Main ──────────────────────────────────────────────────────────────────────
def visualize_data(file_path='data/bug_reports_processed.csv'):
    print("=" * 60)
    print("  TASK 4: Data Visualization")
    print("=" * 60)

    try:
        df = pd.read_csv(file_path)
        print(f"\n  Loaded '{file_path}'  |  {len(df):,} records\n")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    setup()

    print("  Generating charts...")
    status_c   = plot_status(df)
    priority_c = plot_priority(df)
    severity_c = plot_severity(df)
    dev_c      = plot_developers(df)
    trend_c    = plot_trend(df)
    module_c   = plot_modules(df)

    print_observations(status_c, priority_c, severity_c, dev_c, trend_c, module_c)

    print("\n  All 6 charts saved to 'visualizations/' directory.")
    print("  Opening charts...")

    # Auto-open all saved charts in the default image viewer
    charts = [
        'visualizations/bug_status_distribution.png',
        'visualizations/bug_priority_distribution.png',
        'visualizations/bug_severity_distribution.png',
        'visualizations/bugs_assigned_to_developers.png',
        'visualizations/bug_reporting_trend.png',
        'visualizations/bugs_by_module.png',
    ]
    for chart in charts:
        if os.path.exists(chart):
            os.startfile(os.path.abspath(chart))

if __name__ == "__main__":
    visualize_data()
