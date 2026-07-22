# 🐛 Bug Management System

An end-to-end machine learning pipeline for collecting, preprocessing, visualizing, and predicting software bug severity, simulating real-world data from platforms like **Bugzilla**, **JIRA**, and **GitHub**.

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

> The `data/` and `models/` directories are gitignored. All processed files and trained models are generated automatically when you run the pipeline.

---

## 📁 Project Structure

```
BugManagement/
├── data/
│   ├── bug_reports.csv                # Raw generated dataset
│   ├── bug_reports_processed.csv      # Cleaned & encoded dataset
│   ├── potential_duplicates.json      # Detected duplicate bug pairs
│   └── model_evaluation_results.json  # ML model metrics
├── models/
│   ├── label_encoders.pkl             # Fitted label encoders
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── best_severity_model.pkl        # Best model for Severity prediction
│   └── best_priority_model.pkl        # Best model for Priority prediction
├── src/
│   ├── 01_data_collection.py          # Task 1 & 2: Generate dataset
│   ├── 02_preprocessing.py            # Task 3: Clean & encode data
│   ├── 03_visualization.py            # Task 4: Generate 6 charts + Observations
│   ├── 04_duplicate_detection.py      # Task 5: Detect duplicate bugs
│   ├── 05_modeling.py                 # Task 6: Train & evaluate 5 ML models
│   └── 06_predict.py                  # Task 7: Predict severity & priority
├── visualizations/
│   ├── bug_status_distribution.png
│   ├── bug_priority_distribution.png
│   ├── bug_severity_distribution.png
│   ├── bugs_assigned_to_developers.png
│   ├── bug_reporting_trend.png
│   ├── bugs_by_module.png
│   └── duplicate_bugs.png
└── README.md
```

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
pip install pandas scikit-learn matplotlib seaborn joblib
```

### Step-by-step Execution

```bash
# Step 1 & 2: Generate dataset (1500 bug records)
python src/01_data_collection.py

# Step 3: Preprocess — clean nulls, duplicates, anomalies, encode categories
python src/02_preprocessing.py

# Step 4: Visualize — generates 6 charts + observations
python src/03_visualization.py

# Step 5: Detect duplicate bugs via TF-IDF cosine similarity
python src/04_duplicate_detection.py

# Step 6: Train 5 ML models for Severity & Priority prediction
python src/05_modeling.py

# Step 7: Predict severity & priority for a new bug description
python src/06_predict.py --desc "App crashes on login with valid credentials"
```

---

## 📋 Task Coverage

| # | Task | Script | Output |
|---|------|--------|--------|
| 1 | Data Collection | `01_data_collection.py` | `data/bug_reports.csv` |
| 2 | Dataset Connection | `01_data_collection.py` | CSV loaded via Pandas |
| 3 | Data Preprocessing | `02_preprocessing.py` | `data/bug_reports_processed.csv` |
| 4 | Data Visualization | `03_visualization.py` | 6 PNG charts + Observations |
| 5 | Bug Identification | `04_duplicate_detection.py` | `data/potential_duplicates.json` |
| 6 | Model Training & Testing | `05_modeling.py` | 5 models evaluated, best saved |
| 7 | Severity & Priority Prediction | `06_predict.py` | Console prediction output |

---

## 📊 Dataset Fields (50k Kaggle Dataset)

| Field | Description |
|-------|-------------|
| `bug_id` | Unique identifier (BUG_000001 ... BUG_050000) |
| `title` | Short title of the bug |
| `description` | Detailed description text (used for ML) |
| `error_code` | HTTP-style error code (400, 404, 500, 503) |
| `bug_category` | Type of bug (Memory Leak, API Bug, Auth Bug, etc.) |
| `bug_domain` | System domain (Backend, Mobile, DevOps, etc.) |
| `tech_stack` | Technology involved (Angular, Flask, Django, etc.) |
| `severity` | Low / Medium / High / Critical |
| `environment` | Development / Staging / Production |
| `developer_role` | Role responsible (Backend, Frontend, DevOps, etc.) |
| `created_at` | Date the bug was reported |

---

## 📈 Data Visualization — Charts Generated

| Chart | Description |
|-------|-------------|
| Bug Severity Distribution | Pie chart of Low/Medium/High/Critical bugs |
| Bug Category Distribution | Count of bugs per category (Memory Leak, API Bug, etc.) |
| Bug Domain Distribution | Count of bugs per system domain |
| Bugs by Developer Role | Horizontal bar — workload per developer role |
| Bug Reporting Trend Over Time | Monthly line chart of bug reports |
| Bugs by Tech Stack | Bar chart of bugs per technology |

All charts are saved to the `visualizations/` folder and open automatically when the script runs.

---

## 🤖 Machine Learning Models

All 5 models are trained on **TF-IDF features** extracted from bug descriptions to predict **Severity** and **Priority**:

| Model | Notes |
|-------|-------|
| Naïve Bayes | Fast, probabilistic baseline |
| Logistic Regression | Linear classifier |
| Decision Tree | Interpretable rule-based model |
| Random Forest | Ensemble of decision trees |
| SVM | Support Vector Machine classifier |

**Evaluation metrics:** Accuracy, Precision, Recall, F1-Score (weighted)

---

## 🔍 Key Observations (from actual run on 50k dataset)

1. **Low** severity is the most common — 12,628 bugs (25%)
2. **12,432 Critical/High** bugs (24%) require urgent resolution
3. **Memory Leak** is the most frequent bug category (3,220 bugs)
4. **Backend Systems** domain has the most bugs (8,477)
5. **Mobile Developer** role handles the highest bug load (5,701 bugs)
6. **January 2026** saw peak bug reporting (4,304 bugs)
7. **Angular** has the highest bug count by tech stack (3,300 bugs)
8. **780,515** potential duplicate pairs detected via cosine similarity (threshold > 0.85, 5k sample)

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Data:** Pandas, NumPy
- **ML:** scikit-learn (TF-IDF, Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM)
- **Visualization:** Matplotlib, Seaborn
- **Persistence:** joblib (model serialization)
