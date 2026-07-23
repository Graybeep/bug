    # ==============================================================================
# 04_duplicate_detection.py
# Task 5: Bug Identification
#   (a) Duplicate detection via TF-IDF cosine similarity on descriptions
#   (b) Bug pattern analysis and categorization by life cycle stage
# Input:  data/bug_reports_processed.csv
# Output: data/potential_duplicates.json  |  data/lifecycle_analysis.json
#         visualizations/duplicate_bugs.png
# ==============================================================================

import json, os, sys

import _deps
_deps.check('pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, so the script works the
# same whether it is launched from the root, from src/, or from an IDE that
# sets the working directory to the file's own folder.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE_ORDER  = ['Reported', 'In Progress', 'Resolved', 'Verification', 'Closed']
OPEN_STATUSES = ['New', 'Assigned', 'In Progress', 'Reopened']


def analyze_lifecycle(df):
    """Task 5(b): categorize bugs by life cycle stage and surface patterns."""
    print("\n" + "=" * 60)
    print("  BUG PATTERN ANALYSIS -- LIFE CYCLE CATEGORIZATION")
    print("=" * 60)

    if 'lifecycle_stage' not in df.columns:
        print("  [SKIP] 'lifecycle_stage' not found -- run 01_data_collection.py first.")
        return None

    total = len(df)
    analysis = {}

    # -- Stage breakdown ------------------------------------------------
    stages = [s for s in STAGE_ORDER if s in df['lifecycle_stage'].unique()]
    stage_counts = df['lifecycle_stage'].value_counts().reindex(stages).dropna()

    print(f"\n  Bugs categorized by life cycle stage:")
    print(f"  {'Stage':<16}{'Count':>10}{'Share':>10}")
    print(f"  {'-'*16}{'-'*10}{'-'*10}")
    for stage, count in stage_counts.items():
        print(f"  {stage:<16}{count:>10,}{count/total*100:>9.1f}%")
    analysis['stage_counts'] = {k: int(v) for k, v in stage_counts.items()}

    # -- Status breakdown within each stage -----------------------------
    if 'status' in df.columns:
        print(f"\n  Status detail per stage:")
        for stage in stage_counts.index:
            statuses = df.loc[df['lifecycle_stage'] == stage, 'status'].value_counts()
            detail = ', '.join(f"{s} {c:,}" for s, c in statuses.items())
            print(f"    {stage:<14} -> {detail}")
        analysis['status_counts'] = {
            k: int(v) for k, v in df['status'].value_counts().items()
        }

    # -- Open backlog ----------------------------------------------------
    if 'status' in df.columns:
        open_mask  = df['status'].isin(OPEN_STATUSES)
        open_count = int(open_mask.sum())
        print(f"\n  Open backlog (New/Assigned/In Progress/Reopened): "
              f"{open_count:,} ({open_count/total*100:.1f}%)")
        analysis['open_backlog'] = open_count

        # Highest-risk pattern: urgent bugs still sitting in the open backlog
        if 'priority' in df.columns:
            urgent_open = int((open_mask & df['priority'].isin(['P1', 'P2'])).sum())
            print(f"  Urgent bugs still open (P1/P2)                 : "
                  f"{urgent_open:,} ({urgent_open/total*100:.1f}%)  <-- triage first")
            analysis['urgent_open'] = urgent_open

        reopened = int((df['status'] == 'Reopened').sum())
        print(f"  Reopened bugs (failed verification)             : "
              f"{reopened:,} ({reopened/total*100:.1f}%)")
        analysis['reopened'] = reopened

    # -- Resolution outcomes ---------------------------------------------
    if 'resolution' in df.columns:
        print(f"\n  Resolution outcomes:")
        res_counts = df['resolution'].value_counts()
        for res, count in res_counts.items():
            print(f"    {res:<14}{count:>10,}{count/total*100:>9.1f}%")
        analysis['resolution_counts'] = {k: int(v) for k, v in res_counts.items()}

    # -- Severity x stage cross-tab ---------------------------------------
    if 'severity' in df.columns:
        print(f"\n  Critical bugs by life cycle stage:")
        crit = df[df['severity'] == 'Critical']['lifecycle_stage'].value_counts()
        crit = crit.reindex(stages).dropna()
        for stage, count in crit.items():
            print(f"    {stage:<14}{int(count):>10,}")
        analysis['critical_by_stage'] = {k: int(v) for k, v in crit.items()}

    out_path = 'data/lifecycle_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=4)
    print(f"\n  Life cycle analysis saved to: {out_path}")

    return analysis


def group_duplicates(pair_i, pair_j, n):
    """Union-find: collapse the pairwise matches into duplicate groups."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]      # path compression
            x = parent[x]
        return x

    for a, b in zip(pair_i, pair_j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    # Only groups with more than one member are duplicates
    return sorted((m for m in groups.values() if len(m) > 1), key=len, reverse=True)


def detect_duplicates(file_path='data/bug_reports_processed.csv', threshold=0.85,
                      save_all_pairs=False):
    print("=" * 60)
    print("  TASK 5: Bug Identification -- Duplicate Detection")
    print("=" * 60)

    df = pd.read_csv(file_path)
    df['description'] = df['description'].fillna('')
    total = len(df)
    id_col = 'bug_id' if 'bug_id' in df.columns else df.columns[0]

    print(f"\n  Loaded '{file_path}'  |  {total:,} records")
    print(f"  Similarity threshold : {threshold}")
    print(f"\n  Computing TF-IDF similarity matrix (this may take a moment)...")

    # Use a sample for very large datasets to keep runtime reasonable
    SAMPLE_SIZE = 5000
    if total > SAMPLE_SIZE:
        df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"  [INFO] Sampling {SAMPLE_SIZE:,} records for similarity computation.")
    else:
        df_sample = df.reset_index(drop=True)

    vectorizer   = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df_sample['description'])
    sim_matrix   = cosine_similarity(tfidf_matrix)

    # Extract the above-threshold pairs row by row (vectorized — the previous
    # nested Python loop over 12.5M pairs took ~3 minutes).
    n = len(df_sample)
    chunks_i, chunks_j, chunks_s = [], [], []
    for i in range(n):
        row  = sim_matrix[i, i + 1:]
        hits = np.nonzero(row > threshold)[0]
        if hits.size:
            chunks_i.append(np.full(hits.size, i, dtype=np.int32))
            chunks_j.append((hits + i + 1).astype(np.int32))
            chunks_s.append(row[hits])

    if chunks_i:
        pair_i = np.concatenate(chunks_i)
        pair_j = np.concatenate(chunks_j)
        pair_s = np.concatenate(chunks_s)
    else:
        pair_i = pair_j = np.array([], dtype=np.int32)
        pair_s = np.array([], dtype=float)

    num_dup = len(pair_i)

    # Column lookups for annotating each pair with what the bug actually is
    ids   = df_sample[id_col].tolist()
    ctx   = {c: df_sample[c].tolist()
             for c in ('title', 'bug_category', 'severity', 'status', 'priority')
             if c in df_sample.columns}

    def describe(idx):
        """Return the readable details for one bug in the sample."""
        d = {'bug_id': ids[idx]}
        for c, values in ctx.items():
            d[c] = values[idx]
        return d

    # ------------------------------------------------------------------
    # Group the pairwise matches into duplicate clusters
    # ------------------------------------------------------------------
    groups = group_duplicates(pair_i, pair_j, n)

    group_records = []
    for gid, members in enumerate(groups, start=1):
        cats = pd.Series([ctx['bug_category'][m] for m in members]).value_counts() \
               if 'bug_category' in ctx else pd.Series(dtype=int)
        group_records.append({
            'group_id':          gid,
            'size':              len(members),
            'dominant_category': (cats.idxmax() if len(cats) else 'n/a'),
            'category_is_pure':  bool(len(cats) == 1),
            'example_title':     (ctx['title'][members[0]] if 'title' in ctx else ''),
            'bug_ids':           [ids[m] for m in members[:20]],
        })

    # ------------------------------------------------------------------
    # Top pairs, annotated with both bugs' details
    # ------------------------------------------------------------------
    TOP_N = 200
    order = np.argsort(-pair_s)[:TOP_N] if num_dup else []
    sample_pairs = [{
        'similarity': round(float(pair_s[k]), 4),
        'bug_1':      describe(int(pair_i[k])),
        'bug_2':      describe(int(pair_j[k])),
    } for k in order]

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    os.makedirs('data', exist_ok=True)
    json_path = 'data/potential_duplicates.json'

    unique_involved = int(len(np.unique(np.concatenate([pair_i, pair_j])))) if num_dup else 0
    n_sample  = n
    total_cmp = n_sample * (n_sample - 1) // 2

    output = {
        'summary': {
            'sample_size':            n_sample,
            'similarity_threshold':   threshold,
            'total_pairs_compared':   total_cmp,
            'duplicate_pairs_found':  num_dup,
            'bugs_involved':          unique_involved,
            'duplicate_groups':       len(groups),
        },
        'duplicate_groups': group_records,
        'top_pairs':        sample_pairs,
    }
    if save_all_pairs:
        output['all_pairs'] = [
            {'Bug_1': ids[int(pair_i[k])], 'Bug_2': ids[int(pair_j[k])],
             'Similarity': round(float(pair_s[k]), 4)}
            for k in range(num_dup)
        ]

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  DUPLICATE DETECTION SUMMARY")
    print("-" * 60)
    print(f"  Sample size used             : {n_sample:,}")
    print(f"  Total pairs compared         : {total_cmp:,}")
    print(f"  Duplicate pairs found        : {num_dup:,}  "
          f"({num_dup/total_cmp*100:.1f}% of pairs)")
    print(f"  Bugs involved in duplicates  : {unique_involved:,}  "
          f"({unique_involved/n_sample*100:.1f}% of sample)")
    print(f"  Distinct duplicate groups    : {len(groups):,}")
    print(f"  Results saved to             : {json_path}")

    # ------------------------------------------------------------------
    # Top pairs — with the actual bug details, not just the score
    # ------------------------------------------------------------------
    if groups:
        # One example pair per group — showing 5 pairs that all tie at 1.0000
        # within the same group would say nothing.
        print(f"\n  Example duplicate pairs (one per group, with bug details):")
        print("  " + "-" * 58)
        for rank, members in enumerate(groups[:5], start=1):
            a, b_idx = members[0], members[1]
            print(f"   [{rank}] Similarity {float(sim_matrix[a, b_idx]):.4f}")
            for idx in (a, b_idx):
                b = describe(idx)
                meta = ' / '.join(str(b[c]) for c in
                                  ('bug_category', 'severity', 'status', 'priority')
                                  if c in b)
                print(f"       {b['bug_id']}   {meta}")
                if b.get('title'):
                    print(f"                    \"{b['title']}\"")
            print()

    # ------------------------------------------------------------------
    # Duplicate groups table
    # ------------------------------------------------------------------
    if group_records:
        print(f"  Duplicate groups (bugs mutually flagged as the same issue):")
        print(f"    {'#':>3}  {'Size':>6}  {'Dominant category':<26} {'Pure?':<6} Example IDs")
        print(f"    {'-'*3}  {'-'*6}  {'-'*26} {'-'*6} {'-'*30}")
        for g in group_records[:10]:
            examples = ', '.join(g['bug_ids'][:3])
            pure = 'yes' if g['category_is_pure'] else 'no'
            print(f"    {g['group_id']:>3}  {g['size']:>6,}  {g['dominant_category']:<26} "
                  f"{pure:<6} {examples} ...")
        if len(group_records) > 10:
            print(f"    ... and {len(group_records)-10} more group(s) — full list in {json_path}")

    print(f"\n  [NOTE] This dataset's 'description' field is a fixed boilerplate template")
    print(f"         per bug_category (16 templates total for 50k rows), not free-form text.")
    print(f"         Note the 'Pure?' column above: every group maps 1:1 onto a single")
    print(f"         bug_category, which is the giveaway -- these pairs are 'same category',")
    print(f"         not genuine duplicate bug reports, so the counts should not be read")
    print(f"         as real duplication in the underlying bugs.")

    # Chart: Duplicate vs Unique
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    dup_idx   = set(pair_i.tolist()) | set(pair_j.tolist())
    labels    = pd.Series(['Duplicate' if k in dup_idx else 'Unique' for k in range(n)])
    counts    = labels.value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#E53935', '#43A047']
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title('Duplicate vs Unique Bugs', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Bug Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    chart_path = 'visualizations/duplicate_bugs.png'
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Chart saved: {chart_path}")

    # Task 5(b): categorize the full dataset by life cycle stage
    analyze_lifecycle(df)

    print("\n" + "=" * 60)
    print("  BUG IDENTIFICATION COMPLETE")
    print("=" * 60)
    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect duplicate bugs and categorize by life cycle stage")
    parser.add_argument('--threshold', type=float, default=0.85,
                        help="Cosine similarity above which a pair is flagged (default 0.85)")
    parser.add_argument('--save-all-pairs', action='store_true',
                        help="Also write every matched pair to the JSON (large file)")
    args = parser.parse_args()
    detect_duplicates(threshold=args.threshold, save_all_pairs=args.save_all_pairs)
