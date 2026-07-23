# ==============================================================================
# _deps.py
# Shared dependency guard. Every pipeline script calls check() before importing
# its third-party libraries, so running with the wrong interpreter produces a
# clear instruction instead of a raw ModuleNotFoundError traceback.
# ==============================================================================

import sys
import os

# import name -> pip package name
PACKAGE_NAMES = {
    'pandas':     'pandas',
    'numpy':      'numpy',
    'sklearn':    'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn':    'seaborn',
    'joblib':     'joblib',
}


def _venv_python(root):
    """Return the venv interpreter path if this project has one."""
    for candidate in (os.path.join(root, 'venv', 'Scripts', 'python.exe'),
                      os.path.join(root, 'venv', 'bin', 'python')):
        if os.path.exists(candidate):
            return candidate
    return None


def check(*modules):
    """Exit with a helpful message if any required module is missing."""
    missing = []
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(PACKAGE_NAMES.get(module, module))

    if not missing:
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv = _venv_python(root)

    print("=" * 68)
    print("  MISSING DEPENDENCIES")
    print("=" * 68)
    print(f"\n  These packages are not installed in the Python you are using:")
    for pkg in missing:
        print(f"    - {pkg}")
    print(f"\n  Interpreter in use:")
    print(f"    {sys.executable}")

    if venv and os.path.abspath(venv) != os.path.abspath(sys.executable):
        print(f"\n  This project has a virtual environment that already has them.")
        print(f"  Activate it first, then re-run:\n")
        if os.name == 'nt':
            print(f"    venv\\Scripts\\activate")
        else:
            print(f"    source venv/bin/activate")
        print(f"    python {os.path.relpath(sys.argv[0], root)}")
        print(f"\n  Or run this script with the venv's Python directly:\n")
        print(f"    {venv} {os.path.relpath(sys.argv[0], root)}")
    else:
        print(f"\n  Install them with:\n")
        print(f"    pip install -r requirements.txt")

    print("\n" + "=" * 68)
    sys.exit(1)
