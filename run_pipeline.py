# ==============================================================================
# run_pipeline.py
# Runs the complete Bug Management pipeline (Tasks 1-9) in order, streaming each
# stage's output to the console.
#
#   python run_pipeline.py                 # full pipeline, charts + dashboard open
#   python run_pipeline.py --no-open       # full pipeline, artifacts saved only
#   python run_pipeline.py --skip-duplicates   # skip the slow similarity stage
# ==============================================================================

import subprocess
import argparse
import time
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve every relative path from the project root, regardless of where this
# runner was launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

STAGES = [
    ('1 & 2', 'Data Collection & Dataset Connection', 'src/01_data_collection.py'),
    ('3',     'Data Preprocessing',                   'src/02_preprocessing.py'),
    ('4',     'Data Visualization',                   'src/03_visualization.py'),
    ('5',     'Bug Identification',                   'src/04_duplicate_detection.py'),
    ('6',     'Training and Testing Models',          'src/05_modeling.py'),
    ('7',     'Severity & Priority Prediction',       'src/06_predict.py'),
    ('8',     'End-to-End Bug Triage & Tracking',     'src/07_bug_triage.py'),
    ('9',     'Interactive Dashboards & KPI Reports', 'src/08_dashboard.py'),
]


def run_stage(task, name, script, extra_args=None):
    banner = f"  TASK {task}: {name}  "
    print("\n" + "#" * 70)
    print("#" + banner.center(68) + "#")
    print("#" * 70 + "\n", flush=True)

    cmd = [sys.executable, script] + (extra_args or [])
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  [FAILED] {script} exited with code {result.returncode}")
        return False

    print(f"\n  [OK] {script} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full Bug Management pipeline")
    parser.add_argument('--no-open', action='store_true',
                        help="Do not open the charts (step 4) or the dashboard (step 9)")
    parser.add_argument('--skip-duplicates', action='store_true',
                        help="Skip step 5's similarity matrix (the slowest stage)")
    args = parser.parse_args()

    if not os.path.exists('data/bug_dataset_50k.csv'):
        print("  [ERROR] 'data/bug_dataset_50k.csv' not found.")
        print("  Download it from Kaggle and place it in data/ — see README.")
        return 1

    print("=" * 70)
    print("  BUG MANAGEMENT SYSTEM — FULL PIPELINE")
    print("=" * 70)
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  Stages : {len(STAGES)} scripts, Tasks 1 through 9")

    overall = time.time()
    completed = []

    for task, name, script in STAGES:
        if args.skip_duplicates and script.endswith('04_duplicate_detection.py'):
            print(f"\n  [SKIPPED] {script} (--skip-duplicates)")
            continue

        extra = []
        if args.no_open and script.endswith(('03_visualization.py', '08_dashboard.py')):
            extra = ['--no-open']

        # Stage 7's chart window is modal — it blocks until closed by hand, which
        # would stall the pipeline before the dashboard stage ever runs. Its PNG
        # is still written to visualizations/. Stages 3 and 8 open their output
        # without blocking, so they keep their normal behaviour.
        if script.endswith('07_bug_triage.py'):
            extra = ['--no-open']

        if not run_stage(task, name, script, extra):
            print("\n  Pipeline stopped — fix the error above and re-run.")
            return 1
        completed.append(name)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    for name in completed:
        print(f"    [DONE] {name}")
    print(f"\n  Total runtime : {time.time() - overall:.1f}s")
    print(f"  Charts        : visualizations/")
    print(f"  Dashboards    : dashboards/ops_dashboard.html")
    print(f"                  dashboards/model_report.html")
    print(f"  Models        : models/")
    print(f"  Results       : data/model_evaluation_results.json")
    print(f"                  data/lifecycle_analysis.json")
    print(f"                  data/kpi_report.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
