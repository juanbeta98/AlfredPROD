"""
Find the day with the highest number of scheduled services in an optimization snapshot.

Usage (from repo root):
    python3 scripts/analysis/find_peak_service_day.py
    python3 scripts/analysis/find_peak_service_day.py --snapshot-path data/api_snapshots/run-a70fb042/optimization_input_snapshot__ts-20260207T181612.json
    python3 scripts/analysis/find_peak_service_day.py --top 10
    python3 scripts/analysis/find_peak_service_day.py --ignore-request-range
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not resolve repository root from script path.")


ROOT = _find_repo_root()
DEFAULT_REQUEST_PATH = ROOT / "request.json"
DEFAULT_SNAPSHOT_GLOB = "data/api_snapshots/**/optimization_input_snapshot__*.json"


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_services(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    raise ValueError("Snapshot must be a list or an object with a 'data' list.")


def _find_latest_snapshot() -> Path:
    candidates = list(ROOT.glob(DEFAULT_SNAPSHOT_GLOB))
    if not candidates:
        raise FileNotFoundError(
            "No optimization snapshots found under data/api_snapshots/. "
            "Pass --snapshot-path explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def _get_range_from_request(request_payload: Any) -> tuple[datetime | None, datetime | None]:
    if not isinstance(request_payload, dict):
        return None, None
    filters = request_payload.get("filters")
    if not isinstance(filters, dict):
        return None, None
    start_dt = _parse_iso_datetime(filters.get("start_date"))
    end_dt = _parse_iso_datetime(filters.get("end_date"))
    return start_dt, end_dt


def _count_services_by_day(
    services: list[dict[str, Any]],
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in services:
        raw_schedule = row.get("schedule_date")
        if not isinstance(raw_schedule, str):
            continue
        schedule_dt = _parse_iso_datetime(raw_schedule)
        if schedule_dt is None:
            continue
        if start_dt and schedule_dt < start_dt:
            continue
        if end_dt and schedule_dt > end_dt:
            continue
        counts[schedule_dt.date().isoformat()] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the day with the most scheduled services in an optimization snapshot."
    )
    parser.add_argument(
        "--request-path",
        type=Path,
        default=DEFAULT_REQUEST_PATH,
        help="Path to request JSON (used for filters.start_date and filters.end_date).",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=None,
        help="Path to optimization snapshot JSON. Defaults to latest snapshot under data/api_snapshots/.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top days to print (default: 5).",
    )
    parser.add_argument(
        "--ignore-request-range",
        action="store_true",
        help="Ignore request start/end date filters and evaluate the full snapshot.",
    )
    args = parser.parse_args()

    request_path = _resolve_path(args.request_path)
    snapshot_path = _resolve_path(args.snapshot_path) if args.snapshot_path else _find_latest_snapshot()
    snapshot_payload = _load_json(snapshot_path)
    services = _extract_services(snapshot_payload)

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    if not args.ignore_request_range and request_path.exists():
        request_payload = _load_json(request_path)
        start_dt, end_dt = _get_range_from_request(request_payload)

    counts = _count_services_by_day(services, start_dt=start_dt, end_dt=end_dt)
    all_counts = _count_services_by_day(services, start_dt=None, end_dt=None)

    print(f"snapshot_path: {snapshot_path}")
    print(f"total_services_in_snapshot: {len(services)}")
    if args.ignore_request_range:
        print("applied_range: none (--ignore-request-range)")
    elif request_path.exists():
        print(
            "applied_range: "
            f"start={start_dt.isoformat() if start_dt else None}, "
            f"end={end_dt.isoformat() if end_dt else None}"
        )
    else:
        print(f"applied_range: none (request path not found: {request_path})")

    if counts:
        peak_day, peak_count = counts.most_common(1)[0]
        print(f"peak_day_in_applied_range: {peak_day} ({peak_count} services)")
        print("")
        print(f"top_{args.top}_days_in_applied_range:")
        for day, count in counts.most_common(args.top):
            print(f"- {day}: {count}")
    else:
        print("peak_day_in_applied_range: none (no services found in the applied range)")

    if all_counts:
        global_peak_day, global_peak_count = all_counts.most_common(1)[0]
        print("")
        print(f"peak_day_in_full_snapshot: {global_peak_day} ({global_peak_count} services)")
    else:
        print("peak_day_in_full_snapshot: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
