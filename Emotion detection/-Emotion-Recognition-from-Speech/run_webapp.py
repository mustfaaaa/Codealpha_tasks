"""One-command launcher for the SER Studio web UI.

Builds the front end if needed, then serves everything from Flask on 5001.

    .venv\Scripts\python.exe run_webapp.py
    .venv\Scripts\python.exe run_webapp.py --rebuild
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "webapp" / "frontend"
STATIC = ROOT / "webapp" / "backend" / "static"


def build_frontend() -> None:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        sys.exit("npm not found on PATH — install Node.js to build the UI.")
    if not (FRONTEND / "node_modules").exists():
        print("Installing front-end dependencies...")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True, shell=False)
    print("Building front end...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True, shell=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the front end even if a bundle exists")
    a = ap.parse_args()

    if a.rebuild or not (STATIC / "index.html").exists():
        build_frontend()
    else:
        print(f"Using existing bundle in {STATIC}  (--rebuild to refresh)")

    sys.path.insert(0, str(ROOT / "webapp" / "backend"))
    from app import app, recognizer  # noqa: E402

    print("Loading model bundle...")
    recognizer()
    print("\n  SER Studio ready ->  http://127.0.0.1:5001\n")
    app.run(host="127.0.0.1", port=5001, debug=False)


if __name__ == "__main__":
    main()
