# 🐛 Bug Management System — Milestone 2

An end-to-end machine learning pipeline for collecting, preprocessing, visualizing, and predicting software bug severity and priority, simulating real-world data from platforms like **Bugzilla**, **JIRA**, and **GitHub**.

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

## 📋 Milestone 2 — Task Coverage

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

## 📊 Dataset Fields

| Field | Description |
|-------|-------------|
| `Bug ID` | Unique identifier (BUG-0001 ... BUG-1500) |
| `Summary` | Short title of the bug |
| `Description` | Detailed description text |
| `Status` | New / Assigned / In Progress / Resolved / Closed |
| `Severity` | Trivial / Minor / Major / Critical |
| `Priority` | P1 (highest) to P5 (lowest) |
| `Resolution` | Fixed / Duplicate / Won't Fix / Cannot Reproduce / Not a Bug |
| `Assigned_To` | Developer assigned to the bug |
| `Module` | System module where bug was found |
| `Reported_Date` | Date the bug was reported (2024) |

---

## 📈 Data Visualization — Charts Generated

| Chart | Description |
|-------|-------------|
| Bug Status Distribution | Count of bugs in each lifecycle stage |
| Bug Priority Distribution | Count of bugs per priority level (P1–P5) |
| Bug Severity Distribution | Pie chart of Trivial/Minor/Major/Critical bugs |
| Bugs Assigned to Developers | Horizontal bar — workload per developer |
| Bug Reporting Trend Over Time | Monthly line chart of bug reports in 2024 |
| Bugs by Module | Bar chart of bugs per system module |

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

## 🔍 Key Observations (from actual run)

1. **56%** of all bugs are in Resolved/Closed state
2. **P3** is the most common priority — 318 bugs
3. **379 Critical** bugs (25%) require urgent resolution
4. **Henry Wilson** has the highest bug load (201 bugs) — workload rebalancing recommended
5. **December 2024** saw the peak bug reporting (142 bugs)
6. **Notifications** module has the highest defect count (210 bugs)
7. **112,707** potential duplicate pairs detected via cosine similarity (threshold > 0.85)

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Data:** Pandas, NumPy
- **ML:** scikit-learn (TF-IDF, Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM)
- **Visualization:** Matplotlib, Seaborn
- **Persistence:** joblib (model serialization)
