# ==============================================================================
# 04_duplicate_detection.py
# Task 5: Bug Identification — Duplicate Bug Detection
# Uses TF-IDF cosine similarity on descriptions to find potential duplicates.
# Input:  data/bug_reports_processed.csv
# Output: data/potential_duplicates.json  |  visualizations/duplicate_bugs.png
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, os, sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def detect_duplicates(file_path='data/bug_reports_processed.csv', threshold=0.85):
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

    duplicates = []
    n = len(df_sample)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                duplicates.append({
                    'Bug_1':      df_sample.iloc[i][id_col],
                    'Bug_2':      df_sample.iloc[j][id_col],
                    'Similarity': round(float(sim_matrix[i, j]), 4)
                })

    num_dup = len(duplicates)

    # Save JSON
    os.makedirs('data', exist_ok=True)
    json_path = 'data/potential_duplicates.json'
    with open(json_path, 'w') as f:
        json.dump(duplicates, f, indent=4)

    # Summary
    print("\n" + "-" * 60)
    print("  DUPLICATE DETECTION SUMMARY")
    print("-" * 60)
    n_sample = len(df_sample)
    print(f"  Sample size used             : {n_sample:,}")
    print(f"  Total pairs checked          : {n_sample*(n_sample-1)//2:,}")
    print(f"  Duplicate pairs found        : {num_dup:,}")
    unique_involved = len(set(
        [d['Bug_1'] for d in duplicates] + [d['Bug_2'] for d in duplicates]
    ))
    print(f"  Unique bugs involved         : {unique_involved:,}")
    print(f"  Results saved to             : {json_path}")
    print(f"\n  [NOTE] This dataset's 'description' field is a fixed boilerplate template")
    print(f"         per bug_category (16 templates total for 50k rows), not free-form text.")
    print(f"         So every pair flagged here is really just 'same bug_category' -- these")
    print(f"         are not genuine duplicate bug reports, and this count should not be read")
    print(f"         as real duplication in the underlying bugs.")

    if duplicates:
        top5 = sorted(duplicates, key=lambda x: x['Similarity'], reverse=True)[:5]
        print(f"\n  Top 5 most similar pairs:")
        for d in top5:
            print(f"    {d['Bug_1']} <-> {d['Bug_2']}  |  Similarity: {d['Similarity']:.4f}")

    # Chart: Duplicate vs Unique
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)
    dup_ids   = set([d['Bug_1'] for d in duplicates] + [d['Bug_2'] for d in duplicates])
    labels    = df_sample[id_col].apply(lambda x: 'Duplicate' if x in dup_ids else 'Unique')
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

    print("\n" + "=" * 60)
    print("  DUPLICATE DETECTION COMPLETE")
    print("=" * 60)
    return duplicates

if __name__ == "__main__":
    detect_duplicates()
