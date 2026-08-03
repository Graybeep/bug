# ==============================================================================
# 09_serve.py
# Serves dashboards/index.html and puts the trained models behind a local API,
# so the dashboard's triage console runs real inference instead of the
# documented fallback rule.
#
#   python src/09_serve.py                # serve on http://127.0.0.1:8000
#   python src/09_serve.py --port 8080    # different port
#   python src/09_serve.py --no-open      # don't open a browser
#
# Endpoints:
#   GET  /                 -> dashboards/index.html
#   GET  /api/health       -> {"models_loaded": true, "priority_model": "..."}
#   POST /api/predict      -> severity + priority + class probabilities
#
# Uses only the standard library's http.server — no Flask, no new dependency.
# It binds to loopback and is meant for a developer's own machine, not for
# deployment: single process, no auth, no TLS.
# ==============================================================================

import argparse
import json
import os
import posixpath
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import _deps
_deps.check('pandas', 'numpy', 'sklearn', 'joblib')

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
os.chdir(ROOT)
sys.path.insert(0, SRC_DIR)

import model_bridge

WEB_ROOT = os.path.join(ROOT, 'dashboards')
INDEX = 'index.html'
MAX_BODY = 256 * 1024          # a triage request is a few hundred bytes

BUNDLE = None                  # loaded once at startup, shared by all threads
BUNDLE_LOCK = threading.Lock()  # sklearn predict is not guaranteed thread-safe
KB = {}


def load_kb():
    path = os.path.join(ROOT, 'data', 'bug_knowledge_base.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


DOMAIN_OVERRIDE = {'Mobile': 'Mobile Developer'}


def route_owner(category, domain):
    """Same routing policy 07_bug_triage.py and the dashboard use."""
    if domain in DOMAIN_OVERRIDE:
        return DOMAIN_OVERRIDE[domain]
    return (KB.get(category) or {}).get('assigned_role', 'Full-Stack Developer')


class Handler(BaseHTTPRequestHandler):
    server_version = 'BugManagementDashboard/1.0'

    # ── plumbing ─────────────────────────────────────────────────────────────
    def log_message(self, fmt, *args):
        if self.path.startswith('/api/'):
            print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _safe_path(self, url_path):
        """Map a URL onto dashboards/, refusing anything that escapes it."""
        rel = posixpath.normpath(urlparse(url_path).path).lstrip('/')
        if rel in ('', '.'):
            rel = INDEX
        full = os.path.normpath(os.path.join(WEB_ROOT, rel))
        if os.path.commonpath([full, WEB_ROOT]) != WEB_ROOT:
            return None
        return full

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        route = urlparse(self.path).path

        if route == '/api/health':
            info = BUNDLE.describe() if BUNDLE else {'ready': False, 'errors': ['bundle not loaded']}
            return self._send_json({
                'models_loaded': bool(BUNDLE and BUNDLE.ready),
                'priority_model': info.get('priority_model'),
                'severity_model': info.get('severity_model'),
                'errors': info.get('errors', []),
            })

        path = self._safe_path(route)
        if not path or not os.path.isfile(path):
            if not os.path.isfile(os.path.join(WEB_ROOT, INDEX)):
                return self._send_json({'error': 'dashboards/index.html not built — '
                                                 'run python src/08_dashboard.py first'}, 404)
            return self._send_json({'error': 'not found'}, 404)

        ctype = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
                 '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
                 '.png': 'image/png', '.svg': 'image/svg+xml'}.get(
            os.path.splitext(path)[1].lower(), 'application/octet-stream')

        with open(path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    # ── POST /api/predict ────────────────────────────────────────────────────
    def do_POST(self):
        if urlparse(self.path).path != '/api/predict':
            return self._send_json({'error': 'not found'}, 404)

        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return self._send_json({'error': 'empty or oversized request body'}, 400)

        try:
            req = json.loads(self.rfile.read(length).decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_json({'error': f'invalid JSON: {exc}'}, 400)

        if not (BUNDLE and BUNDLE.ready):
            return self._send_json({'error': 'models are not loaded — run python src/05_modeling.py'}, 503)

        description = str(req.get('desc') or req.get('description') or '').strip()
        title = str(req.get('title') or '').strip()
        if not description:
            description = title
        if not description:
            return self._send_json({'error': 'a description (or at least a title) is required'}, 400)

        category = req.get('category')
        domain = req.get('domain')
        try:
            with BUNDLE_LOCK:
                result = model_bridge.predict_one(
                    BUNDLE,
                    description=f"{title} {description}".strip(),
                    severity=req.get('severity'),
                    environment=req.get('environment') or 'Production',
                    error_code=req.get('error_code') or 500,
                    bug_domain=domain,
                    tech_stack=req.get('tech_stack'),
                    developer_role=None,
                )
        except Exception as exc:                                    # noqa: BLE001
            return self._send_json({'error': f'{exc.__class__.__name__}: {exc}'}, 500)

        entry = KB.get(category) or {}
        result['owner'] = route_owner(category, domain)
        result['root_cause'] = entry.get('root_cause', 'Not in the knowledge base for this category')
        result['suggested_fix'] = entry.get('suggested_fix', 'No stored remediation for this category')
        result['escalated'] = result['priority'] in ('P1', 'P2')
        return self._send_json(result)


def main():
    global BUNDLE, KB

    parser = argparse.ArgumentParser(description="Serve the dashboard with live model inference")
    parser.add_argument('--port', type=int, default=8000, help="Port to listen on (default 8000)")
    parser.add_argument('--host', default='127.0.0.1', help="Interface to bind (default loopback)")
    parser.add_argument('--no-open', dest='open_browser', action='store_false',
                        help="Don't open a browser window")
    parser.set_defaults(open_browser=True)
    args = parser.parse_args()

    print("=" * 68)
    print("  LIVE DASHBOARD SERVER")
    print("=" * 68)

    index_path = os.path.join(WEB_ROOT, INDEX)
    if not os.path.isfile(index_path):
        print(f"\n  [ERROR] {os.path.relpath(index_path, ROOT)} not found.")
        print(f"  Build it first:  python src/08_dashboard.py --no-open")
        return 1

    print(f"\n  Dashboard : {os.path.relpath(index_path, ROOT)} "
          f"({os.path.getsize(index_path) / (1024 * 1024):.2f} MB)")

    KB = load_kb()
    print(f"  Knowledge base: {len(KB)} bug categories")

    BUNDLE = model_bridge.load()
    if BUNDLE.ready:
        print(f"  Models    : priority={type(BUNDLE.priority_model).__name__}"
              f"  severity={type(BUNDLE.severity_model).__name__ if BUNDLE.severity_model is not None else 'n/a'}")
        print(f"  The triage console will run live inference.")
    else:
        print(f"  [WARN] Models unavailable — the console falls back to the documented")
        print(f"         scoring rule. Run 'python src/05_modeling.py' to fix that.")

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"\n  [ERROR] Could not bind {args.host}:{args.port} — {exc}")
        print(f"  Try a different port:  python src/09_serve.py --port 8080")
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Serving   : {url}")
    print(f"  API       : GET {url}api/health · POST {url}api/predict")
    print(f"  Stop with Ctrl+C")
    print("=" * 68)

    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
