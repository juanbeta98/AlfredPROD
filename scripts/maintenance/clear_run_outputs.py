"""
Delete generated run directories under data outputs.

Targets:
    - data/api_snapshots/run-*
    - data/model_output/run-*
    - data/intermediate_exports/run-*

Usage:
    python3 scripts/maintenance/clear_run_outputs.py          # dry run
    python3 scripts/maintenance/clear_run_outputs.py --apply  # delete matching directories
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
TARGET_BASE_DIRS = [
    ROOT / "data" / "api_snapshots",
    ROOT / "data" / "model_output",
    ROOT / "data" / "intermediate_exports",
]
RUN_DIR_PREFIX = "run-"


def _collect_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    return sorted(
        [
            child
            for child in base_dir.iterdir()
            if child.is_dir() and child.name.startswith(RUN_DIR_PREFIX)
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear generated run directories under data outputs."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Without this flag, only prints what would be removed.",
    )
    args = parser.parse_args()

    all_run_dirs: list[Path] = []
    for base_dir in TARGET_BASE_DIRS:
        all_run_dirs.extend(_collect_run_dirs(base_dir))

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(f"mode: {mode}")
    print(f"root: {ROOT}")
    print(f"run_directories_found: {len(all_run_dirs)}")

    for path in all_run_dirs:
        print(f"- {path}")

    if not args.apply:
        print("")
        print("No changes made. Re-run with --apply to delete these directories.")
        return 0

    deleted = 0
    failed = 0
    for path in all_run_dirs:
        try:
            shutil.rmtree(path)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"failed_to_delete: {path} ({exc})")

    print("")
    print(f"deleted: {deleted}")
    print(f"failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
