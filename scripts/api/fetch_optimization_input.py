"""
Fetch optimization input from the ALFRED API and save a local JSON snapshot.

Usage (from repo root):
    python scripts/api/fetch_optimization_input.py
    python scripts/api/fetch_optimization_input.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Filters (edit these directly as needed)
# --------------------------------------------------
DEPARTMENT: Optional[int] = None
START_DATE: Optional[str] = None  # ISO datetime, e.g. "2024-01-01T00:00:00"
END_DATE: Optional[str] = None  # ISO datetime, e.g. "2024-01-31T23:59:59"
REQUEST_PATH: Path = ROOT / "request.json"

# Output location
OUTPUT_DIR = ROOT / "data" / "api_snapshots"


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _load_request_filters(path: Path) -> tuple[Optional[int], Optional[datetime], Optional[datetime]]:
    if not path.exists():
        logger.info("request_file_not_found path=%s", path.as_posix())
        return None, None, None

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    filters = payload.get("filters") if isinstance(payload, dict) else None
    if not isinstance(filters, dict):
        return None, None, None

    department = _parse_int(filters.get("department"))
    start_dt = _parse_datetime(filters.get("start_date"))
    end_dt = _parse_datetime(filters.get("end_date"))
    return department, start_dt, end_dt


def _require(value: str, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is required. Set it via environment variables.")
    return value


def _write_json(
    payload: object,
    *,
    prefix: str,
    run_id: str,
    request_id: str | None = None,
) -> Path:
    from src.io.artifact_naming import build_artifact_stem, build_run_subdir  # noqa: E402

    output_dir = OUTPUT_DIR / build_run_subdir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = build_artifact_stem(prefix, run_id=run_id, request_id=request_id)
    path = output_dir / f"{stem}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch optimization input from ALFRED API and write a JSON snapshot."
    )
    parser.add_argument(
        "--request-path",
        type=Path,
        default=REQUEST_PATH,
        help="Path to request JSON used to source filters when top constants are unset.",
    )
    parser.add_argument(
        "--ignore-request-filters",
        action="store_true",
        help="Ignore request.json filters and only use constants in this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print effective filters and exit without calling the API.",
    )
    args = parser.parse_args()

    request_path = _resolve_path(args.request_path)

    request_department: Optional[int] = None
    request_start_dt: Optional[datetime] = None
    request_end_dt: Optional[datetime] = None
    if not args.ignore_request_filters:
        request_department, request_start_dt, request_end_dt = _load_request_filters(request_path)

    department = DEPARTMENT if DEPARTMENT is not None else request_department
    start_dt = _parse_datetime(START_DATE) if START_DATE is not None else request_start_dt
    end_dt = _parse_datetime(END_DATE) if END_DATE is not None else request_end_dt

    if args.dry_run:
        print(
            json.dumps(
                {
                    "request_path": request_path.as_posix(),
                    "request_filters_enabled": not args.ignore_request_filters,
                    "department": department,
                    "start_date": start_dt.isoformat() if start_dt else None,
                    "end_date": end_dt.isoformat() if end_dt else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    from src.config import Config  # noqa: E402
    from src.integration.client import ALFREDAPIClient  # noqa: E402
    from src.logging_utils import set_run_id, setup_logging_context  # noqa: E402

    setup_logging_context()
    run_id = str(uuid.uuid4())[:8]
    set_run_id(run_id)
    Config.configure_logging()

    endpoint = _require(Config.SERVICES_ENDPOINT, "SERVICES_ENDPOINT or API_BASE_URL/API_ENDPOINT")
    token = _require(Config.API_TOKEN, "API_TOKEN")

    client = ALFREDAPIClient(
        endpoint_url=endpoint,
        api_token=token,
        timeout=Config.REQUEST_TIMEOUT,
        max_retries=Config.API_MAX_RETRIES,
    )

    logger.info(
        "fetching_optimization_input department=%s start_date=%s end_date=%s request_path=%s request_filters_enabled=%s",
        department,
        start_dt.isoformat() if start_dt else None,
        end_dt.isoformat() if end_dt else None,
        request_path.as_posix(),
        not args.ignore_request_filters,
    )

    payload = client.get_optimization_data(
        department=department,
        start_date=start_dt,
        end_date=end_dt,
    )

    payload_request_id = payload.get("request_id") if isinstance(payload, dict) else None
    path = _write_json(
        payload,
        prefix="optimization_input_snapshot",
        run_id=run_id,
        request_id=payload_request_id,
    )
    logger.info("snapshot_written path=%s bytes=%s", path.as_posix(), path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
