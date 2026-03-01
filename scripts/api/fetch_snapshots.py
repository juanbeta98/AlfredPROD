"""
Fetch both optimization input (services) and driver directory snapshots in a
single run, saving all outputs under the same run folder.

Usage (from repo root):
    python scripts/api/fetch_snapshots.py
    python scripts/api/fetch_snapshots.py --dry-run
    python scripts/api/fetch_snapshots.py --ignore-request-filters
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
SCRIPTS_DIR = ROOT / "scripts" / "api"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch services and driver directory snapshots into a shared run folder."
    )
    parser.add_argument(
        "--request-path",
        type=Path,
        default=ROOT / "request.json",
        help="Path to request JSON used to source filters.",
    )
    parser.add_argument(
        "--ignore-request-filters",
        action="store_true",
        help="Ignore request.json filters and only use constants in each script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print effective filters and exit without calling the API.",
    )
    args = parser.parse_args()

    run_id = str(uuid.uuid4())[:8]
    print(f"run_id={run_id}", flush=True)

    common_args = [f"--run-id={run_id}", f"--request-path={args.request_path}"]
    if args.dry_run:
        common_args.append("--dry-run")
    if args.ignore_request_filters:
        common_args.append("--ignore-request-filters")

    scripts = [
        SCRIPTS_DIR / "fetch_optimization_input.py",
        SCRIPTS_DIR / "fetch_driver_directory.py",
    ]

    for script in scripts:
        result = subprocess.run([sys.executable, str(script)] + common_args)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
