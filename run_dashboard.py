# ==============================================================================
# run_dashboard.py
# Launches the Streamlit dashboard (Task 9) with the project's own config.
#
#   python run_dashboard.py               # start it and open a browser
#   python run_dashboard.py --port 8600   # pick a port
#   python run_dashboard.py --no-open     # start it without opening a browser
#
# Equivalent to `streamlit run src/streamlit_app.py`, but it checks the inputs
# first and gives a useful message when a pipeline stage has not been run yet.
# ==============================================================================

import argparse
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

APP = os.path.join('src', 'streamlit_app.py')
REQUIRED = [
    ('data/bug_reports_processed.csv',
     "python src/01_data_collection.py  then  python src/02_preprocessing.py"),
]
OPTIONAL = [
    ('data/module_catalog.json',            'src/01_data_collection.py'),
    ('data/model_evaluation_results.json',  'src/05_modeling.py'),
    ('models/best_priority_model.pkl',      'src/05_modeling.py'),
]


def main():
    parser = argparse.ArgumentParser(description="Run the Bug Management dashboard")
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--no-open', dest='open_browser', action='store_false',
                        help="Start the server without opening a browser")
    parser.set_defaults(open_browser=True)
    args = parser.parse_args()

    try:
        import streamlit                                            # noqa: F401
        import plotly                                               # noqa: F401
    except ImportError as exc:
        print(f"  [ERROR] {exc.name} is not installed.")
        print(f"  Install the dashboard dependencies:  pip install -r requirements.txt")
        return 1

    for path, how in REQUIRED:
        if not os.path.exists(path):
            print(f"  [ERROR] '{path}' not found — the dashboard has nothing to show.")
            print(f"  Produce it with:  {how}")
            print(f"  Or run the whole pipeline:  python run_pipeline.py")
            return 1

    missing = [(path, how) for path, how in OPTIONAL if not os.path.exists(path)]
    for path, how in missing:
        print(f"  [WARN] '{path}' not found — the views that need it will say so. "
              f"Produce it with: python {how}")

    print("=" * 68)
    print("  BUG MANAGEMENT — INTERACTIVE DASHBOARD")
    print("=" * 68)
    print(f"  App    : {APP}")
    print(f"  URL    : http://localhost:{args.port}")
    print(f"  Stop   : Ctrl-C")
    print("=" * 68)

    cmd = [sys.executable, '-m', 'streamlit', 'run', APP,
           '--server.port', str(args.port),
           '--server.headless', 'false' if args.open_browser else 'true']
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
