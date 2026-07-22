# ==============================================================================
# 04_duplicate_detection.py
# Task 5: Bug Identification — Duplicate Bug Detection
# Uses TF-IDF cosine similarity to find potential duplicate bugs.
# Input:  data/bug_reports_processed.csv
# Output: data/potential_duplicates.json  |  visualizations/duplicate_bugs.png
# ==============================================================================

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def detect_duplicates(file_path='data/bug_reports_processed.csv', threshold=0.85):
    print("=" * 60)
    print("  TASK 5: Bug Identification — Duplicate Detection")
    print("=" * 60)

    # Load
    df = pd.read_csv(file_path)
    df['Description'] = df['Description'].fillna('')
    total = len(df)
    print(f"\n  Loaded '{file_path}'  |  {total} records")
    print(f"  Similarity threshold : {threshold}")

    # TF-IDF vectorization
    print("\n  Computing TF-IDF similarity matrix...")
    vectorizer     = TfidfVectorizer(stop_words='english')
    tfidf_matrix   = vectorizer.fit_transform(df['Description'])
    sim_matrix     = cosine_similarity(tfidf_matrix)

    # Find pairs above threshold
    duplicates = []
    id_col = 'Bug ID' if 'Bug ID' in df.columns else df.columns[0]

    for i in range(total):
        for j in range(i + 1, total):
            if sim_matrix[i, j] > threshold:
                duplicates.append({
                    'Bug_1':      df.iloc[i][id_col],
                    'Bug_2':      df.iloc[j][id_col],
                    'Similarity': round(float(sim_matrix[i, j]), 4)
                })

    num_dup = len(duplicates)

    # Save JSON
    os.makedirs('data', exist_ok=True)
    json_path = 'data/potential_duplicates.json'
    with open(json_path, 'w') as f:
        json.dump(duplicates, f, indent=4)

    # Summary table
    print("\n" + "-" * 60)
    print("  DUPLICATE DETECTION SUMMARY")
    print("-" * 60)
    print(f"  Total bug pairs checked      : {total*(total-1)//2}")
    print(f"  Duplicate pairs found        : {num_dup}")
    print(f"  Unique bugs involved         : {len(set([d['Bug_1'] for d in duplicates] + [d['Bug_2'] for d in duplicates]))}")
    print(f"  Results saved to             : {json_path}")

    if duplicates:
        print(f"\n  Top 5 most similar pairs:")
        top5 = sorted(duplicates, key=lambda x: x['Similarity'], reverse=True)[:5]
        for d in top5:
            print(f"    {d['Bug_1']} <-> {d['Bug_2']}  |  Similarity: {d['Similarity']:.4f}")

    # Chart: Duplicate vs Non-Duplicate
    os.makedirs('visualizations', exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)

    dup_bug_ids  = set([d['Bug_1'] for d in duplicates] + [d['Bug_2'] for d in duplicates])
    dup_label    = df[id_col].apply(lambda x: 'Duplicate' if x in dup_bug_ids else 'Unique')
    counts       = dup_label.value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#E53935', '#43A047']
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2, width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')
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
