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
│   ├── bug_reports_enriched.csv       # Source + derived lifecycle & delivery fields
│   ├── bug_reports_processed.csv      # Cleaned & encoded dataset
│   ├── potential_duplicates.json      # Detected duplicate bug pairs
│   ├── lifecycle_analysis.json        # Life cycle stage / status / resolution breakdown
│   ├── bug_knowledge_base.json        # Per-category root cause / fix / owning role
│   ├── module_catalog.json            # Module sizes (KLOC) + SLA targets (created by 01)
│   ├── kpi_report.json                # All KPIs + insights, machine-readable (created by 08)
│   ├── tracked_bugs.json              # Live ticket tracker (created by 07)
│   └── model_evaluation_results.json  # ML model metrics
├── models/
│   ├── label_encoders.pkl             # Fitted label encoders
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── priority_features.pkl          # Structured-feature column list + scaler
│   ├── best_severity_model.pkl        # Best model for Severity prediction
│   ├── best_priority_model.pkl        # Best model for Priority prediction
│   ├── best_bug_category_model.pkl    # Best model for Bug Category prediction
│   └── rf_*_model.pkl                 # Random Forest per target (used by 07)
├── src/
│   ├── 01_data_collection.py          # Task 1 & 2: Load dataset + derive lifecycle fields
│   ├── 02_preprocessing.py            # Task 3: Clean & encode data
│   ├── 03_visualization.py            # Task 4: Generate 9 charts + Observations
│   ├── 04_duplicate_detection.py      # Task 5: Duplicates + life cycle categorization
│   ├── 05_modeling.py                 # Task 6: Train & evaluate 5 ML models per target
│   ├── 06_predict.py                  # Task 7: Predict severity & priority
│   ├── 07_bug_triage.py               # Task 8: Full triage — assign, diagnose, track
│   ├── 08_dashboard.py                # Task 9: KPI report, insights + companion charts
│   ├── streamlit_app.py               # Task 9: THE interactive dashboard — 6 views, live models
│   ├── dashboard_data.py              #   its data layer: cached load, filter, KPI, model calls
│   ├── dashboard_ui.py                #   its look: palette, Plotly template, CSS, components
│   ├── model_bridge.py                # The one code path from the dashboard to models/*.pkl
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
│   ├── duplicate_bugs.png
│   ├── kpi_sprint_trend.png           # ── the five below are created by 08 ──
│   ├── kpi_module_priority_heatmap.png
│   ├── kpi_resolution_time_by_priority.png
│   ├── kpi_defect_density_by_module.png
│   └── kpi_release_quality.png
├── docs/
│   ├── PROJECT_REPORT.md              # Full write-up: method, results, limitations
│   └── PROJECT_REPORT.pdf             # Same report, rendered with charts embedded
├── .streamlit/
│   └── config.toml                    # Dashboard theme (validated palette, dark + light)
├── run_pipeline.py                    # Runs all 9 tasks in order
├── run_dashboard.py                   # Launches the interactive dashboard
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

Runs all 9 tasks in order, streams each stage's output to the console, and opens the 9 charts at Task 4.

Task 9 additionally re-scores all 50,000 bugs with the trained priority model, so allow ~1 minute for that stage (`python src/08_dashboard.py --no-model` skips it).

```bash
python run_pipeline.py --no-open           # save the charts without opening them
python run_pipeline.py --skip-duplicates   # skip Task 5 — finishes in ~40s
```

Then start the interactive dashboard — it is a live app, not a build artifact, so it is not part of the pipeline run:

```bash
python run_dashboard.py                    # http://localhost:8501
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

# Step 8: Full triage — predict, assign a developer, diagnose, and track
python src/07_bug_triage.py --title "Checkout page crashes" \
       --desc "Payment page freezes after submit" \
       --error-code 500 --category "Backend Logic Bug" --environment Production

# Step 9: KPI report + actionable insights + the 5 companion charts
python src/08_dashboard.py              # add --no-model to skip model scoring

# Step 9 (interactive): the dashboard — 6 cross-filtered views + live triage console
python run_dashboard.py                 # http://localhost:8501
python run_dashboard.py --port 8600     # different port
python run_dashboard.py --no-open       # start the server without opening a browser
```

> **Working directory doesn't matter.** Every script resolves its paths from the project root, so you can launch them from the project root, from inside `src/`, from an IDE's Run button, or from anywhere else — they'll always find `data/` and `models/`.

### What each stage prints

| Stage | Console output |
|-------|----------------|
| `01` | Record count, column list, required-field coverage check, derived-field summary, **module catalog table** (size + bugs + density), sample rows for both the life cycle and delivery columns |
| `02` | Null report, duplicate report, anomaly report, encoding maps, **cleaned data preview** (10 rows readable + the same 10 rows encoded), post-clean verification |
| `03` | Chart-by-chart save log, 9 numbered **Observations**, then opens each PNG |
| `04` | Duplicate pair count, **example pairs with each bug's title/category/severity/status/priority**, **duplicate group table** (size + dominant category + purity), life cycle stage table, open backlog / urgent / reopen figures, resolution mix |
| `05` | Per-target metrics table (5 models × Accuracy/Precision/Recall/F1), best model per target, per-class classification report |
| `06` | Input description + context, predicted **Severity** and **Priority** with triage notes |
| `07` | 5-step triage report: reported details → cleaning → Random Forest prediction → developer assignment + root cause + suggested fix → life cycle tracking. Pops up the assignment chart; pass `--no-open` to only save it |
| `08` | **KPI report** in six blocks (headline KPIs, module-wise quality, priority vs SLA, team performance, release-wise quality, top root causes), then **seven actionable insights**, then the model-scoring log (rows scored, train/test split recovered, agreement) and the save log for the KPI JSON and 5 charts |
| `run_dashboard.py` | Which inputs were found, which are missing and how to produce them, then the Streamlit server banner and URL |

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
| 8 | End-to-End Triage & Tracking | `07_bug_triage.py` | `data/tracked_bugs.json`, lifecycle chart |
| 9 | KPI Reporting & Actionable Insights | `08_dashboard.py` | `data/kpi_report.json`, 5 KPI charts, printed report |
| 9 | Interactive Dashboard | `streamlit_app.py` (`run_dashboard.py`) | 6 cross-filtered views + a triage console running `models/*.pkl` live |

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

The milestone requires **Status**, **Priority** and **Resolution**, plus analysis of the **bug life cycle**; the dashboard task additionally requires **Sprint**, **Release Version**, **Module**, **Feature**, **Component** and **Date Closed**. The Kaggle dataset ships none of these — it has no workflow state and no delivery taxonomy at all. Rather than swap datasets, `01_data_collection.py` derives them deterministically (`seed=42`, so every run is reproducible) and writes `data/bug_reports_enriched.csv`.

### Life cycle fields

| Field | Values | How it's derived |
|-------|--------|------------------|
| `status` | New, Assigned, In Progress, Fixed, Pending Retest, Verified, Closed, Reopened, Duplicate, Rejected, Deferred | Sampled from a realistic defect-workflow distribution |
| `lifecycle_stage` | Reported → In Progress → Resolved → Verification → Closed | Deterministic mapping from `status` |
| `resolution` | Fixed, Unresolved, Duplicate, Invalid, Won't Fix | Deterministic mapping from `status` |
| `priority` | P1 (highest) … P5 (lowest) | Impact score, below |

### Delivery-tracking fields

| Field | Values | How it's derived |
|-------|--------|------------------|
| `sprint` | `SPR-01` … `SPR-26` | 14-day buckets counted from the earliest `created_at`. The data ends mid-sprint, so the trailing 2-day stub is folded into the last full sprint — on its own it would plot beside 14-day bars and read as a collapse in intake |
| `release_version` | `v1.0` … `v3.2` (9 releases) | 3 sprints per release; major version bumps every 3 releases |
| `module` | Web Portal, Mobile App, Core Services, Data Platform, Cloud Infrastructure, Delivery Pipeline | Fixed map from `bug_domain` — which product area owns the defect |
| `feature` | Login & SSO, Public API, Persistence Layer, … (16) | Fixed map from `bug_category` — which user-facing capability it breaks |
| `component` | `svc-spring-core`, `web-ui-react`, `db-postgres`, … (16) | Fixed map from `tech_stack` — which deployable/code unit it lives in |
| `date_closed` | Date, or null while the bug is open | Set only for the four terminal statuses (Closed, Duplicate, Rejected, Deferred) — 28.3% of rows |
| `resolution_days` | 0 – 311 days, or null | `date_closed − created_at` |

Module, feature and component are three **independent axes** of the taxonomy, not a strict tree: a feature can span several components, exactly as it does in a real product. Their source columns are independent in the Kaggle data, so a Mobile-domain bug can carry a `Public API` feature.

**Days-to-close model.** Each closed bug draws a duration from a log-normal spread around a median set by its priority (P1 2d · P2 5d · P3 12d · P4 25d · P5 60d), scaled by how it was resolved (Duplicate ×0.3, Invalid ×0.4, Won't Fix ×2.0). Where a drawn duration would run past the snapshot date the duration is **redrawn uniformly inside the days actually remaining** — truncating it to the snapshot instead would stack every one of those closures onto the final date and invent a closure spike in the last sprint.

**SLA targets** (used for compliance KPIs, published in `data/module_catalog.json`): P1 3 days · P2 7 · P3 14 · P4 30 · P5 60.

**Module sizes in KLOC** are fixed project constants, also in `data/module_catalog.json` — defect density needs a size denominator and the dataset carries no code-size metric. They are an assumption, so treat density as "bugs per unit of assumed module size", not a measurement.

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

> **All of these fields are derived, not observed.** They make the life cycle, triage, sprint and KPI stages of this project analysable and reproducible, but they are a modelled delivery process layered on the Kaggle data — not ground truth from a real issue tracker. Any conclusion about priority, sprint velocity or defect density is a conclusion about that model.

---

## 🎫 End-to-End Bug Triage (`07_bug_triage.py`)

The complete workflow: a user reports a bug, the system cleans and analyses it, **Random Forest** predicts severity and priority, the bug is assigned to the right developer with a root cause and suggested fix, and its progress is tracked and visualized until resolved.

```bash
# Demo: triage 5 representative bugs at once (default with no arguments)
python src/07_bug_triage.py
python src/07_bug_triage.py --reset       # clear old tickets first

# Report a single bug
python src/07_bug_triage.py --title "Login API times out" \
       --desc "Authentication endpoint times out during peak traffic" \
       --error-code 503 --category "Authentication Bug" \
       --environment Production --severity High

# Track it
python src/07_bug_triage.py --list                # all tickets + chart
python src/07_bug_triage.py --advance TKT_0001    # move one stage forward
python src/07_bug_triage.py --resolve TKT_0001    # drive through to Closed
python src/07_bug_triage.py --categories          # show the knowledge base
```

Running with no arguments triages five bugs chosen to exercise different categories, environments and severities, so the routing is visible side by side:

```
  Ticket    Pri  Severity  Assigned to           Title
  TKT_0001  P1   Critical  Backend Developer     Checkout page crashes on payment
  TKT_0002  P2   High      Security Engineer     Login API times out under load
  TKT_0003  P3   Medium    Data Engineer         Customer records load slowly
  TKT_0004  P1   High      Mobile Developer      Android app crashes on startup
  TKT_0005  P5   Low       Frontend Developer    Save button label overflows
```

Same system, five different owners and priorities — the **category** decides the owner, and **severity + environment + error code** together decide the priority.

Output:

```
  [3] RANDOM FOREST PREDICTION
      Severity    : Critical   System unusable — immediate action required.
      Priority    : P2         Fix in the current sprint.

  [4] ASSIGNMENT & DIAGNOSIS
      Assigned to : Security Engineer
      Escalation  : P2 — flagged to the team lead for immediate triage.
      Root cause  : Misconfiguration or logic issue related to authentication bug
      Suggested   : Review and fix the authentication bug according to best practices

  [5] LIFE CYCLE TRACKING
      Ticket ID   : TKT_0002
      Progress    : [x] New  ->  [>] Assigned  ->  [ ] In Progress  ->  [ ] Fixed
                    [ ] Pending Retest  ->  [ ] Verified  ->  [ ] Closed
```

### Developer routing

`developer_role` in the source dataset is **uniformly random** — all 9 roles appear at ~11.1% inside every single category, so no model can learn assignment from it. Instead the system uses a **documented routing table** (category → specialist), with a domain override for mobile and automatic escalation for P1/P2:

| Category | Owner | | Category | Owner |
|---|---|---|---|---|
| API Bug, Backend Logic, Concurrency, Memory Leak, Performance | Backend Developer | | Database Bug | Data Engineer |
| Authentication, Authorization, Security Vulnerability | Security Engineer | | Cloud Configuration | Cloud Engineer |
| CI/CD, Deployment, Logging, Monitoring | DevOps Engineer | | UI Bug, Frontend Routing | Frontend Developer |
| *any category, `--domain Mobile`* | Mobile Developer | | | |

### Root cause & suggested fix

Looked up from `data/bug_knowledge_base.json`, built by `01_data_collection.py`. Each of the 16 categories has exactly **one** distinct `root_cause` and `suggested_fix` in the dataset, so these are a reliable lookup rather than something to predict.

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
| Tracked Bug Life Cycle | Ticket status counts + per-ticket progress bars | `07` |
| Sprint Intake vs Closure | Opened/closed bars per sprint, with the carried-over backlog in its own panel below on a shared x-axis (not a second y-axis) | `08` |
| Module × Priority Heatmap | Where the urgent work concentrates | `08` |
| Resolution Time vs SLA | Average days to close per priority against its SLA target | `08` |
| Defect Density by Module | Bugs per KLOC, volume normalised by module size | `08` |
| Release-wise Quality | Closed vs still-open bugs stacked per release | `08` |

All charts are saved to the `visualizations/` folder **and opened in your default image viewer** when the script runs. Pass `--no-open` to only save them.

---

## 📊 Interactive Dashboard & KPI Reporting (Task 9)

Stage 9 has two halves, and they share one set of definitions.

| Half | Script | What you get |
|------|--------|--------------|
| **Reporting** | `src/08_dashboard.py` | The KPI report printed to the console, `data/kpi_report.json`, and 5 static companion charts |
| **Interactive** | `src/streamlit_app.py`, launched by `run_dashboard.py` | Six cross-filtered views in the browser, plus a triage console that runs the trained models live |

```bash
python src/08_dashboard.py                # KPI report + insights + companion charts
python src/08_dashboard.py --no-model     # skip model scoring (faster)
python src/08_dashboard.py --top 15       # widen the "top N root causes" console table

python run_dashboard.py                   # the dashboard, on http://localhost:8501
python run_dashboard.py --port 8600       # different port
python run_dashboard.py --no-open         # start the server without opening a browser
streamlit run src/streamlit_app.py        # the same app, without the input pre-flight check
```

`run_dashboard.py` checks the inputs first, so a missing pipeline artifact gets you the command that produces it rather than a stack trace.

### One definition of every KPI

The app does **not** restate the KPI formulas. `src/dashboard_data.py` imports `prepare()`, `compute_kpis()` and `build_insights()` out of `08_dashboard.py` by path (its filename starts with a digit, so a normal `import` cannot reach it) and calls them on whatever slice the filters select. The console report and the dashboard therefore cannot drift apart: there is exactly one implementation, and the app just feeds it a different frame.

### The filters scope everything

The sidebar carries **Bug state · Reported within · Module · Priority · Severity**, with Sprint, Release, Category, Environment and Routed owner under *More filters*. Every view — tiles, charts, tables, model agreement, explorer — recomputes from the same filtered slice, and each page says how many of the 50,000 records are in scope. `Reset filters` clears the lot.

The filtered slice and its KPIs are cached on the filter values themselves, so switching views with the same filters is instant and only a filter change pays for a recompute.

### The six views

| View | Shows |
|------|-------|
| **Overview** | 8 stat tiles (bugs in scope, open, SLA compliance, avg resolution time, defect density, reopen rate, avg open age, carried backlog), sprint intake vs closure, the backlog it leaves behind, the priority mix, how far past SLA the open work is, and the 7 ranked insights |
| **Trends** | Reported vs closed by month, net backlog change per month, the days-to-close distribution, median days against each priority's SLA target, SLA compliance per band, and the spread of resolution time by priority |
| **Distribution** | Any dimension in the taxonomy (module, feature, component, category, domain, tech stack, environment, status, lifecycle, resolution) as open-vs-closed bars plus a share ring, the Module × Priority heatmap, the severity mix per sprint, and bugs per release |
| **Quality & team** | Defect density per module against module size, the top root causes, queue length and SLA compliance per routed owner, and release risk ranked by unresolved P1/P2 |
| **Models & triage** | The five models across four metrics per target with the verdict for each, the recorded × predicted priority confusion matrix **per data split**, the most confident disagreements, and the live triage console |
| **Bug explorer** | Every record behind the charts — search across ID/title/description, pick your columns, sort on any of them, export the slice as CSV, and open one bug in full |

**Every chart has a table view** behind a *Show the numbers* expander, so no value is reachable only by hovering.

### Model verdicts

| Target | Verdict |
|--------|---------|
| Bug Category | **Leakage, not a result** — `title`/`description` are 16 boilerplate templates, so accuracy is ~100% for the wrong reason |
| Severity | **No predictive signal** — every model sits at chance level (~25% for 4 classes); severity is independent of every feature tested |
| Priority | **Learnable — with a caveat** — Random Forest genuinely separates from the rest, but priority is a derived field, so the model is recovering the scoring rule, not real-world triage priority |

### Theming

`.streamlit/config.toml` carries the project palette for both modes; switch with Streamlit's own theme control (☰ → Settings). Dark is the default surface, so every ink role is a light value (`#ffffff` primary, `#c3c2b7` secondary on a `#1a1a19` chart surface). Light is a **separate selected set**, not an inverted flip. The 8 categorical hues pass the lightness band, chroma floor, adjacent-CVD separation (worst ΔE 8.4), normal-vision floor (worst ΔE 19.3) and ≥3:1 contrast against the dark surface. Categorical hues are assigned in fixed order — P1 keeps its colour when a filter drops P3 — sequential heat is one hue light→dark, and status colours (good/warning/serious/critical) are reserved, never reused as a series colour, and always carry a word as well as a colour.

### KPIs computed

| KPI | Definition |
|-----|------------|
| Close rate | Closed bugs ÷ total bugs |
| Avg / median / p90 resolution time | Days from `created_at` to `date_closed`, over closed bugs only |
| SLA compliance | Closed bugs whose `resolution_days` ≤ their priority's target |
| Open past SLA | Open bugs whose age at the snapshot already exceeds their target |
| Defect density | Bugs ÷ module size in KLOC |
| Reopen rate | Share of bugs currently in `Reopened` |
| Sprint backlog | Cumulative (opened − closed) per sprint |

Everything is also written to **`data/kpi_report.json`** — headline KPIs plus per-sprint, per-module, per-release, per-priority, per-root-cause and per-team breakdowns, and the insights — so the numbers can be consumed without re-parsing the CSV.

### Team performance uses the routed owner

The dataset's own `developer_role` column is **uniformly random** — every role appears ~11.1% of the time inside every category (see *Known Data Limitations*), so grouping by it produces nine near-identical rows. The team KPIs therefore group by the **routing-policy owner** (bug category → specialist, with a Mobile-domain override), the same policy `07_bug_triage.py` assigns with. That reflects the workload the policy actually creates.

---

## 🔌 How the dashboard connects to the models

`src/model_bridge.py` is the single code path from the dashboard to `models/*.pkl`. Both halves of stage 9 go through it, so the charts and the triage console can never disagree about how a feature vector is built.

### Scoring the dataset

`08_dashboard.py` — and, on demand, the *Models & triage* view — loads `best_priority_model.pkl` + `tfidf_vectorizer.pkl` + `priority_features.pkl` and re-scores all 50,000 rows in chunks, keeping the predicted priority and the model's own confidence per row.

It also **reproduces `05_modeling.py`'s train/test partition exactly** — the same `sample(20000, random_state=42)` followed by the same stratified `train_test_split(test_size=0.2, random_state=42)` — and tags every row `training` / `held-out test` / `never sampled`. That is what makes the *Models & triage* view honest: the split selector separates memorised rows from held-out ones.

```
Split recall : 16,000 training rows, 4,000 held-out test rows, 30,000 never sampled
Agreement    : 86.9% over all rows, 80.8% on the held-out test rows
```

The **80.8% held-out figure reproduces `05_modeling.py`'s reported Random Forest priority accuracy (0.8077) exactly**, which is the check that the split reconstruction is right rather than approximately right.

In the app, scoring sits behind a button on the *Agreement* tab: the priority model is ~110 MB, so it is loaded lazily and the result is cached for the session.

### The live triage console

The console on the *Models & triage* view takes the same inputs `07_bug_triage.py` takes on the command line and returns the predicted priority with its **full class-probability vector**, plus the routed owner and suggested fix from the knowledge base.

It calls `model_bridge.predict_one()` **in-process** — Streamlit is already the server, so there is no API hop, no fallback rule, and no offline mode to explain. If `models/*.pkl` cannot be loaded, the console says exactly which file is missing and which script produces it.

Severity is predicted from the description when you leave it to the model, but that model is at chance level on this dataset (see the verdict above), so supplying severity yourself gives the priority model a far better feature.

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

**KPIs (from `08_dashboard.py`, over the derived delivery fields):**

14. **28.3% close rate** — 14,148 of 50,000 bugs closed; the backlog grows in 26 of 27 sprints and ends at 35,852 open
15. **Average 16.8 days to close** (median 9, p90 42); **77.6% of closed bugs met their priority's SLA**
16. **P3 is the weakest SLA queue** at 70.5% compliance against a 14-day target; P1 is strongest at 86.9%
17. **Delivery Pipeline has the highest defect density** — 263.8 bugs per KLOC, 4.4× Core Services (59.5), on nearly identical bug counts
18. **The routing policy concentrates 26.0% of all bugs on Backend Developer** (13,004) versus 5.2% on Data Engineer (2,586), while turnaround varies by under 2 days across all seven roles (15.7–17.2)

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

7. **Consequence for `08_dashboard.py` — the open backlog skews old.** `status` is drawn independently of `created_at`, so a bug reported in the first week is exactly as likely to still be "New" as one reported yesterday. The dashboard therefore reports a very high *open past SLA* figure (~93% of the open queue) and a ~182-day average open age. Both are arithmetically correct for this data; neither is a realistic aging profile, because a real tracker closes old bugs faster than new ones.

8. **Consequence for `08_dashboard.py` — closures bunch into the final sprints.** The same independence means ~28% of bugs reported near the snapshot date are marked closed, and those closures can only land in the days that remain. The last one or two sprints therefore show more closures than a steady state would produce. The duration redraw (see *Derived Fields*) spreads them across the remaining window rather than piling them on the final day, but it cannot remove the effect.

9. **Release- and module-level *rates* are within noise.** Because closure is independent of both, every release lands within ~1.5 points of the same ~28% close rate and every module within ~1.3 points, on near-identical bug counts (8,225–8,477). What genuinely separates them is **volume** and, for modules, the **assumed KLOC baseline** — which is why the insights rank on absolute queue size and defect density rather than on closure rate.

10. **Team differences are structural, not behavioural.** Turnaround across the seven routed owners spans only 15.7–17.2 days, because priority mix is near-identical across categories. The meaningful signal is queue length: the routing policy sends 26.0% of all bugs to Backend Developer and 5.2% to Data Engineer.

---

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'matplotlib'` (or pandas/sklearn/seaborn) | Running with system Python instead of the venv | Activate the venv (`venv\Scripts\activate`) or call `venv\Scripts\python.exe` directly. The scripts print the exact command. |
| No charts appear, no error | Charts were saved but not opened | They're in `visualizations/`. Auto-open is on by default; `--no-open` disables it. |
| `ModuleNotFoundError: No module named 'streamlit'` (or plotly) | The dashboard dependencies aren't installed | `pip install -r requirements.txt` — stages 1–8 don't need them, the dashboard does. |
| Dashboard says a data file is missing | That pipeline stage hasn't run yet | `run_dashboard.py` prints the exact command that produces each missing file. |
| `Port 8501 is already in use` | Another Streamlit app is running | `python run_dashboard.py --port 8600` |
| Triage console reports the models could not be loaded | `models/*.pkl` missing or unreadable | It names the file. Usually `python src/05_modeling.py` hasn't been run. |
| The *Agreement* tab takes a minute | It is loading a ~110 MB model and scoring 50,000 rows | That is expected on first use; the result is cached for the rest of the session. |
| The dashboard feels stale after re-running the pipeline | Streamlit caches the loaded data per server process | Press `R` in the browser, or use *Rerun* → *Clear cache* from the ☰ menu. |
| Pipeline appears to freeze after Task 8 | Stage 7's chart window is modal, so a direct `python src/07_bug_triage.py` waits for you to close it | Close the chart window. `run_pipeline.py` passes `--no-open` to stage 7 for exactly this reason, so the full run never blocks. |
| `[ERROR] ... is missing: sprint, module, ...` from stage 08 | `bug_reports_processed.csv` predates the delivery fields | Re-run stages 01 and 02, then 08. |
| `[ERROR] Dataset not found` | `bug_dataset_50k.csv` missing, or `01_data_collection.py` hasn't run yet | Download the CSV into `data/`, then run stage 01. The error prints the absolute path it looked for. |
| Stage 5 feels slow | It builds a 5,000×5,000 similarity matrix | Now vectorized — runs in ~4s. Use `--skip-duplicates` only if you want to skip it entirely. |

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Data:** Pandas, NumPy
- **ML:** scikit-learn (TF-IDF, Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM)
- **Visualization:** Matplotlib, Seaborn (static companion PNGs)
- **Dashboard:** Streamlit (six views, cached slices, in-process model inference) with Plotly charts on the project's validated palette
- **Persistence:** joblib (model serialization)
#   I n t e l l i g e n t - S o f t w a r e - D e f e c t - T r a c k i n g - S y s t e m - w i t h - R e s o l u t i o n - A s s i s t a n c e  
 