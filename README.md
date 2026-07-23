# 🐛 Bug Management System

An end-to-end machine learning pipeline for collecting, preprocessing, visualizing, and predicting software bug **severity** and **priority**, built on a real 50k bug report dataset sourced from Kaggle.

---

## ⚠️ Dataset Setup (Required Before Running)

The dataset is **not included** in this repository (too large for GitHub). You must download it manually from Kaggle.

### Steps:
1. Go to Kaggle and search for **"50k Bug Dataset"** or use the direct link:
   👉 **[https://www.kaggle.com/datasets/search?q=50k+bug+dataset](https://www.kaggle.com/datasets/search?q=50k+bug+dataset)**

2. Download the file — it will be named **`bug_dataset_50k.csv`**

3. Place it inside the `data/` folder of this project:
   ```
   BugManagement/
   └── data/
       └── bug_dataset_50k.csv   ← place it here
   ```

4. Then run the scripts in order (see **How to Run** below)

> The large CSVs (`bug_dataset_50k.csv`, `bug_reports_enriched.csv`, `bug_reports_processed.csv`, `potential_duplicates.json`) and the whole `models/` directory are gitignored — they exceed GitHub's file-size limits. All of them are regenerated automatically when you run the pipeline. The small result files (`model_evaluation_results.json`, `lifecycle_analysis.json`) and the charts in `visualizations/` are tracked.

---

## 📁 Project Structure

```
BugManagement/
├── data/
│   ├── bug_dataset_50k.csv            # Source dataset (download from Kaggle, see above)
│   ├── bug_reports_enriched.csv       # Source + derived lifecycle fields
│   ├── bug_reports_processed.csv      # Cleaned & encoded dataset
│   ├── potential_duplicates.json      # Detected duplicate bug pairs
│   ├── lifecycle_analysis.json        # Life cycle stage / status / resolution breakdown
│   └── model_evaluation_results.json  # ML model metrics
├── models/
│   ├── label_encoders.pkl             # Fitted label encoders
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── priority_features.pkl          # Structured-feature column list + scaler
│   ├── best_severity_model.pkl        # Best model for Severity prediction
│   ├── best_priority_model.pkl        # Best model for Priority prediction
│   └── best_bug_category_model.pkl    # Best model for Bug Category prediction
├── src/
│   ├── 01_data_collection.py          # Task 1 & 2: Load dataset + derive lifecycle fields
│   ├── 02_preprocessing.py            # Task 3: Clean & encode data
│   ├── 03_visualization.py            # Task 4: Generate 9 charts + Observations
│   ├── 04_duplicate_detection.py      # Task 5: Duplicates + life cycle categorization
│   ├── 05_modeling.py                 # Task 6: Train & evaluate 5 ML models per target
│   ├── 06_predict.py                  # Task 7: Predict severity & priority
│   ├── _deps.py                       # Dependency guard (clear message if venv not active)
│   └── present_dataset.py             # Optional: rich console summary of the dataset
├── visualizations/
│   ├── bug_severity_distribution.png
│   ├── bug_priority_distribution.png
│   ├── bug_status_distribution.png
│   ├── bug_lifecycle_stages.png
│   ├── bug_category_distribution.png
│   ├── bug_domain_distribution.png
│   ├── bugs_assigned_to_developers.png
│   ├── bug_reporting_trend.png
│   ├── bugs_by_module.png
│   └── duplicate_bugs.png
├── docs/
│   ├── PROJECT_REPORT.md              # Full write-up: method, results, limitations
│   └── PROJECT_REPORT.pdf             # Same report, rendered with charts embedded
├── run_pipeline.py                    # Runs all 7 tasks in order
├── requirements.txt
└── README.md
```

📄 **Full project report:** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — detailed method, per-stage results, per-class metrics, and the data-quality analysis behind the caveats below.

---

## 🚀 How to Run

> Run each script **in order** from the project root directory.

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **You must activate the venv before running anything.** If you run `python src/...` with your system Python, the packages won't be there and you'll get a `ModuleNotFoundError`. The scripts detect this and print the exact command to fix it, but the simplest check is that your prompt shows `(venv)` first.
>
> On Windows you can also skip activation and call the venv's interpreter directly:
> ```bash
> venv\Scripts\python.exe run_pipeline.py
> ```

### Option A — Run everything at once (recommended)

```bash
python run_pipeline.py
```

Runs all 7 tasks in order, streams each stage's output to the console, and opens the 9 charts when it reaches Task 4. Takes ~35 seconds end-to-end.

```bash
python run_pipeline.py --no-open           # save charts without opening them
python run_pipeline.py --skip-duplicates   # skip Task 5 — finishes in ~40s
```

### Option B — Step-by-step Execution

```bash
# Step 1 & 2: Load the 50k dataset and derive status/priority/resolution fields
python src/01_data_collection.py

# Step 3: Preprocess — clean nulls/duplicates/anomalies, encode, preview clean data
python src/02_preprocessing.py

# Step 4: Visualize — generates and opens 9 charts + observations
python src/03_visualization.py          # add --no-open to only save them

# Step 5: Detect duplicates + categorize bugs by life cycle stage
python src/04_duplicate_detection.py

# Step 6: Train 5 ML models for Severity, Priority & Bug Category
python src/05_modeling.py

# Step 7: Predict severity and priority for a new bug
python src/06_predict.py --desc "App crashes on login with valid credentials" \
                         --environment Production --error-code 500
```

> **Working directory doesn't matter.** Every script resolves its paths from the project root, so you can launch them from the project root, from inside `src/`, from an IDE's Run button, or from anywhere else — they'll always find `data/` and `models/`.

### What each stage prints

| Stage | Console output |
|-------|----------------|
| `01` | Record count, column list, required-field coverage check, derived-field summary, sample rows |
| `02` | Null report, duplicate report, anomaly report, encoding maps, **cleaned data preview** (10 rows readable + the same 10 rows encoded), post-clean verification |
| `03` | Chart-by-chart save log, 9 numbered **Observations**, then opens each PNG |
| `04` | Duplicate pair count, **example pairs with each bug's title/category/severity/status/priority**, **duplicate group table** (size + dominant category + purity), life cycle stage table, open backlog / urgent / reopen figures, resolution mix |
| `05` | Per-target metrics table (5 models × Accuracy/Precision/Recall/F1), best model per target, per-class classification report |
| `06` | Input description + context, predicted **Severity** and **Priority** with triage notes |

---

## 📋 Task Coverage

| # | Task | Script | Output |
|---|------|--------|--------|
| 1 | Data Collection | `01_data_collection.py` | `data/bug_reports_enriched.csv` |
| 2 | Dataset Connection | `01_data_collection.py` | CSV loaded via Pandas |
| 3 | Data Preprocessing | `02_preprocessing.py` | `data/bug_reports_processed.csv` |
| 4 | Data Visualization | `03_visualization.py` | 9 PNG charts + Observations |
| 5 | Bug Identification | `04_duplicate_detection.py` | `potential_duplicates.json`, `lifecycle_analysis.json` |
| 6 | Model Training & Testing | `05_modeling.py` | 5 models × 3 targets, best saved |
| 7 | Severity & Priority Prediction | `06_predict.py` | Console prediction output |

---

## 📊 Dataset Fields (50k Kaggle Dataset)

| Field | Description |
|-------|-------------|
| `bug_id` | Unique identifier (BUG_000001 ... BUG_050000) |
| `title` | Short title/summary of the bug |
| `description` | Detailed description text (used for ML) |
| `error_code` | HTTP-style error code (400, 401, 403, 404, 500, 502, 503) |
| `bug_category` | Type of bug (Memory Leak, API Bug, Auth Bug, etc. — 16 values) |
| `bug_domain` | System domain (Backend, Mobile, DevOps, Cloud, Data, Web) |
| `tech_stack` | Technology involved (Angular, Flask, Django, etc.) |
| `severity` | Low / Medium / High / Critical |
| `environment` | Development / Staging / Production |
| `developer_role` | Role responsible (Backend, Frontend, DevOps, etc.) |
| `root_cause` | Stated cause of the bug |
| `suggested_fix` | Suggested remediation |
| `explanation` | Note on which role/skillset the bug requires |
| `created_at` | Date the bug was reported |

---

## 🧬 Derived Fields (added by `01_data_collection.py`)

The milestone requires **Status**, **Priority** and **Resolution**, plus analysis of the **bug life cycle**. The Kaggle dataset ships none of these — it has no workflow state at all. Rather than swap datasets, `01_data_collection.py` derives them deterministically (`seed=42`, so every run is reproducible) and writes `data/bug_reports_enriched.csv`.

| Field | Values | How it's derived |
|-------|--------|------------------|
| `status` | New, Assigned, In Progress, Fixed, Pending Retest, Verified, Closed, Reopened, Duplicate, Rejected, Deferred | Sampled from a realistic defect-workflow distribution |
| `lifecycle_stage` | Reported → In Progress → Resolved → Verification → Closed | Deterministic mapping from `status` |
| `resolution` | Fixed, Unresolved, Duplicate, Invalid, Won't Fix | Deterministic mapping from `status` |
| `priority` | P1 (highest) … P5 (lowest) | Impact score, below |

**Priority scoring rule:**

```
score = severity_weight + environment_weight + blocking_error_weight

  severity      Critical 4 | High 3 | Medium 2 | Low 1
  environment   Production 2 | Staging 1 | Development 0
  error_code    500/502/503 → +1  (server-side, blocks users)

  score ≥ 6 → P1     score = 5 → P2     score = 4 → P3
  score = 3 → P4     score ≤ 2 → P5
```

About 8% of rows are nudged one level up or down (seeded) to reflect the fact that real triage isn't a closed-form lookup — this keeps the target learnable without being a trivial identity function.

> **These four fields are derived, not observed.** They make the life cycle, status, priority and resolution stages of this project analysable and reproducible, but they are a modelled triage policy layered on the Kaggle data — not ground truth from a real issue tracker. Any conclusion about priority is a conclusion about that policy.

---

## 📈 Data Visualization — Charts Generated

| Chart | Description | Script |
|-------|-------------|--------|
| Bug Severity Distribution | Pie chart of Low/Medium/High/Critical bugs | `03` |
| Bug Priority Distribution | Bar chart of P1–P5 with percentage labels | `03` |
| Bug Status Distribution | All 11 workflow states, in life cycle order | `03` |
| Bug Life Cycle Stages | Funnel across Reported → … → Closed | `03` |
| Bug Category Distribution | Count of bugs per category (Memory Leak, API Bug, etc.) | `03` |
| Bug Domain Distribution | Count of bugs per system domain | `03` |
| Bugs by Developer Role | Horizontal bar — workload per developer role | `03` |
| Bug Reporting Trend Over Time | Monthly line chart of bug reports | `03` |
| Bugs by Tech Stack | Bar chart of bugs per technology | `03` |
| Duplicate vs Unique Bugs | Duplicate detection result | `04` |

All charts are saved to the `visualizations/` folder **and opened in your default image viewer** when the script runs. Pass `--no-open` to only save them.

---

## 🤖 Machine Learning Models

All 5 models are trained per target, on an 80/20 stratified train/test split:

| Model | Notes |
|-------|-------|
| Naïve Bayes | Fast, probabilistic baseline |
| Logistic Regression | Linear classifier |
| Decision Tree | Interpretable rule-based model |
| Random Forest | Ensemble of decision trees |
| SVM (Linear) | `LinearSVC` + `CalibratedClassifierCV` |

**Targets and features:**

| Target | Features |
|--------|----------|
| Severity (Low/Medium/High/Critical) | TF-IDF of `description` |
| Priority (P1–P5) | TF-IDF + scaled `severity`, `environment`, `error_code`, `bug_domain`, `tech_stack`, `developer_role` |
| Bug Category (16 classes) | TF-IDF of `description` |

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score (weighted) — plus a per-class report for the winning model. The best model per target (by F1) is saved to `models/`.

### Actual results (20k training sample, seed 42)

| Model | Severity Acc | Priority Acc | Bug Category Acc |
|---|---|---|---|
| Naïve Bayes | 0.2555 | 0.3805 | 1.0000 |
| Logistic Regression | 0.2560 | 0.4595 | 1.0000 |
| Decision Tree | 0.2592 | 0.7782 | 1.0000 |
| **Random Forest** | 0.2560 | **0.8077** | 1.0000 |
| SVM (Linear) | 0.2602 | 0.4587 | 1.0000 |

Read all three columns against the **Known Data Limitations** below before treating any of them at face value.

---

## 🔍 Key Observations (from actual run on 50k dataset)

1. **Low** severity is the most common — 12,628 bugs (25%)
2. **12,432 Critical** bugs (24%) require urgent resolution
3. **P3** is the most common priority (12,435); **17,175 bugs (34.4%)** are P1/P2 — the urgent queue
4. **19,402 bugs (38.8%)** are still open (New / Assigned / In Progress / Reopened)
5. **6,763 urgent bugs (13.5%)** are P1/P2 *and* still open — the highest-risk triage backlog
6. **2,447 bugs (4.9%)** are Reopened, i.e. failed verification after a fix
7. **Closed** is the largest life cycle stage (14,148 bugs, 28.3%); resolution outcomes are 53.2% Fixed, 38.8% Unresolved
8. **Memory Leak** is the most frequent bug category (3,220 bugs)
9. **Backend Systems** domain has the most bugs (8,477)
10. **Mobile Developer** role handles the highest bug load (5,701 bugs)
11. **January 2026** saw peak bug reporting (4,304 bugs)
12. **Angular** has the highest bug count by tech stack (3,300 bugs)
13. **780,515** potential duplicate pairs detected via cosine similarity (threshold > 0.85, 5k sample) — see the data limitation on duplicate detection below before reading this as genuine duplication.

---

## ⚠️ Known Data Limitations

These are properties of the source dataset itself, not bugs in this repo's code — documented here so results aren't misread.

1. **`title`, `description`, `root_cause`, and `suggested_fix` are boilerplate templates, not free text.**
   Each has only **16 unique values across all 50,000 rows** — one fixed template per `bug_category` (e.g. every "Memory Leak" row shares the exact same description string, verbatim). There is no per-bug information in these fields beyond the category name. TF-IDF over all 50k descriptions yields just **68 distinct features**, despite `max_features=2000`.

2. **Consequence for `05_modeling.py` — Bug Category prediction:** all 5 models score **100.0% accuracy**. This is not generalization, it's leakage: the model is matching a template string back to the label it was copied from.

3. **Consequence for `05_modeling.py` — Severity prediction:** all 5 models score **~25.5% accuracy** — chance level for 4 classes. We verified `severity` is statistically independent of `bug_category`, `bug_domain`, `environment`, `error_code`, and `developer_role` (near-uniform ~25% split within every group), and of the description text. Severity appears to be assigned independently at random in the source data, so **no model — text-based or feature-based — can predict it above chance from this dataset.**

4. **Consequence for `05_modeling.py` — Priority prediction:** Random Forest reaches **80.8% accuracy**, but `priority` is a **derived field** (see *Derived Fields* above), so the models are recovering the documented scoring rule from the structured features, minus the ~8% seeded jitter. This demonstrates that the pipeline trains, evaluates and ranks models correctly on a learnable target — it is *not* evidence that priority is predictable from the raw Kaggle data, because the raw data has no priority field at all.

5. **Consequence for `04_duplicate_detection.py`:** because every bug in a category shares identical description text, TF-IDF cosine similarity flags same-category bugs as "duplicates" (similarity = 1.0), regardless of whether they're actually the same reported issue. The reported pair counts reflect **category membership, not genuine duplicate bug reports.**

6. **Status and resolution distributions** are sampled from a plausible workflow distribution, not observed from a tracker, so the open-backlog and reopen-rate figures describe the derivation, not real-world team behaviour.

---

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'matplotlib'` (or pandas/sklearn/seaborn) | Running with system Python instead of the venv | Activate the venv (`venv\Scripts\activate`) or call `venv\Scripts\python.exe` directly. The scripts print the exact command. |
| No charts appear, no error | Charts were saved but not opened | They're in `visualizations/`. Auto-open is on by default; `--no-open` disables it. |
| `[ERROR] Dataset not found` | `bug_dataset_50k.csv` missing, or `01_data_collection.py` hasn't run yet | Download the CSV into `data/`, then run stage 01. The error prints the absolute path it looked for. |
| Stage 5 feels slow | It builds a 5,000×5,000 similarity matrix | Now vectorized — runs in ~4s. Use `--skip-duplicates` only if you want to skip it entirely. |

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Data:** Pandas, NumPy
- **ML:** scikit-learn (TF-IDF, Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM)
- **Visualization:** Matplotlib, Seaborn
- **Persistence:** joblib (model serialization)
