# ==============================================================================
# 03_visualization.py
# Task 4: Data Visualization
# Generates 6 charts from the 50k processed dataset + prints Observations.
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

def setup():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    os.makedirs('visualizations', exist_ok=True)

def save(fig, name):
    path = f"visualizations/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

# ── Chart 1: Bug Severity Distribution (Pie) ──────────────────────────────────
def plot_severity(df):
    col = 'severity'
    order  = ['Critical', 'High', 'Medium', 'Low']
    order  = [o for o in order if o in df[col].unique()]
    counts = df[col].value_counts().reindex(order).dropna()
    colors = ['#E53935', '#FB8C00', '#FDD835', '#43A047']

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=colors[:len(counts)], autopct='%1.1f%%',
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight('bold')
    ax.set_title('Bug Severity Distribution', fontsize=15, fontweight='bold', pad=20)
    fig.tight_layout()
    save(fig, 'bug_severity_distribution')
    return counts

# ── Chart 2: Bug Category Distribution ────────────────────────────────────────
def plot_category(df):
    col    = 'bug_category'
    counts = df[col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette('viridis', len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Bug Category Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_category_distribution')
    return counts

# ── Chart 3: Bug Domain Distribution ──────────────────────────────────────────
def plot_domain(df):
    col    = 'bug_domain'
    counts = df[col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette('magma', len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Bug Domain Distribution', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=30, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_domain_distribution')
    return counts

# ── Chart 4: Bugs by Developer Role ───────────────────────────────────────────
def plot_developer_role(df):
    col    = 'developer_role'
    counts = df[col].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette('crest', len(counts))
    bars = ax.barh(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width()):,}', va='center', fontsize=9, fontweight='bold')
    ax.set_title('Bugs Assigned by Developer Role', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Bugs', fontsize=12)
    ax.set_ylabel('Developer Role', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bugs_assigned_to_developers')
    return counts

# ── Chart 5: Bug Reporting Trend Over Time ────────────────────────────────────
def plot_trend(df):
    date_col = 'created_at'
    if date_col not in df.columns:
        print("  [SKIP] 'created_at' column not found.")
        return None
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    monthly = df.groupby(df[date_col].dt.to_period('M')).size()
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly.values, color='#1565C0', linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.12, color='#1565C0')
    ax.set_title('Bug Reporting Trend Over Time', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Bugs Reported', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    step = max(1, len(monthly) // 12)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly.index[::step], rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bug_reporting_trend')
    return monthly

# ── Chart 6: Bugs by Tech Stack ───────────────────────────────────────────────
def plot_tech_stack(df):
    col    = 'tech_stack'
    counts = df[col].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette('flare', len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_title('Bugs by Tech Stack', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Tech Stack', fontsize=12)
    ax.set_ylabel('Number of Bugs', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xticks(rotation=35, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, 'bugs_by_module')
    return counts

# ── Observations ──────────────────────────────────────────────────────────────
def print_observations(sev_c, cat_c, dom_c, dev_c, trend_c, tech_c):
    print("\n" + "=" * 60)
    print("  OBSERVATIONS")
    print("=" * 60)

    if sev_c is not None:
        top = sev_c.idxmax()
        total = sev_c.sum()
        crit = sev_c.get('Critical', sev_c.get('High', 0))
        print(f"\n  1. Bug Severity Distribution")
        print(f"     - '{top}' is the most common severity ({sev_c.max():,} bugs).")
        print(f"     - {crit:,} bugs ({100*crit//total}%) are at the highest severity level.")

    if cat_c is not None:
        print(f"\n  2. Bug Category Distribution")
        print(f"     - '{cat_c.idxmax()}' is the most frequent category ({cat_c.max():,} bugs).")
        print(f"     - '{cat_c.idxmin()}' category has the fewest bugs ({cat_c.min():,}).")

    if dom_c is not None:
        print(f"\n  3. Bug Domain Distribution")
        print(f"     - '{dom_c.idxmax()}' domain has the most bugs ({dom_c.max():,}).")
        print(f"     - Spread across {len(dom_c)} domains — indicating wide system impact.")

    if dev_c is not None:
        print(f"\n  4. Bugs by Developer Role")
        print(f"     - '{dev_c.idxmax()}' role handles the most bugs ({dev_c.max():,}).")
        print(f"     - '{dev_c.idxmin()}' role handles the fewest ({dev_c.min():,}).")

    if trend_c is not None:
        print(f"\n  5. Bug Reporting Trend Over Time")
        print(f"     - Peak month: {trend_c.idxmax()} ({trend_c.max():,} bugs reported).")
        print(f"     - Lowest month: {trend_c.idxmin()} ({trend_c.min():,} bugs reported).")

    if tech_c is not None:
        print(f"\n  6. Bugs by Tech Stack")
        print(f"     - '{tech_c.idxmax()}' has the highest bug count ({tech_c.max():,}).")
        print(f"     - Represents {len(tech_c)} distinct tech stacks in the dataset.")

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

    sev_c   = plot_severity(df)
    cat_c   = plot_category(df)
    dom_c   = plot_domain(df)
    dev_c   = plot_developer_role(df)
    trend_c = plot_trend(df)
    tech_c  = plot_tech_stack(df)

    print_observations(sev_c, cat_c, dom_c, dev_c, trend_c, tech_c)

    print("\n  All 6 charts saved to 'visualizations/' directory.")
    print("  Opening charts...")

    charts = [
        'visualizations/bug_severity_distribution.png',
        'visualizations/bug_category_distribution.png',
        'visualizations/bug_domain_distribution.png',
        'visualizations/bugs_assigned_to_developers.png',
        'visualizations/bug_reporting_trend.png',
        'visualizations/bugs_by_module.png',
    ]
    for chart in charts:
        if os.path.exists(chart):
            os.startfile(os.path.abspath(chart))

if __name__ == "__main__":
    visualize_data()
