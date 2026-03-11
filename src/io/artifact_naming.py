from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config import Config

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _resolve_timezone(tz_name: Optional[str] = None) -> ZoneInfo:
    name = (tz_name or Config.ARTIFACT_TIMEZONE or "America/Bogota").strip()
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid ARTIFACT_TIMEZONE value: {name!r}") from exc


def timestamp_token(
    created_at: Optional[datetime] = None,
    *,
    timezone_name: Optional[str] = None,
) -> str:
    tz = _resolve_timezone(timezone_name)
    ts = created_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    local_ts = ts.astimezone(tz)
    return local_ts.strftime("%Y%m%dT%H%M%S")


def sanitize_token(value: Any, *, fallback: str = "na") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    text = _SANITIZE_PATTERN.sub("-", text).strip("._-")
    return text or fallback


def build_run_subdir(run_id: Any) -> str:
    return f"run-{sanitize_token(run_id, fallback='local')}"


def build_artifact_stem(
    prefix: str,
    *,
    run_id: Any = None,
    created_at: Optional[datetime] = None,
    timezone_name: Optional[str] = None,
    request_id: Any = None,
) -> str:
    """
    Build a standardized artifact stem: <prefix>
    Timestamp is stored once per run in run.json, not repeated in every filename.
    """
    _ = run_id
    _ = created_at
    _ = timezone_name
    _ = request_id
    return sanitize_token(prefix, fallback="artifact")


def write_run_manifest(
    run_dir: Path,
    run_id: str,
    *,
    created_at: Optional[datetime] = None,
    timezone_name: Optional[str] = None,
) -> Path:
    """
    Write the initial run.json manifest with status 'running'.
    Idempotent: skips writing if the file already exists.
    """
    manifest_path = run_dir / "run.json"
    if manifest_path.exists():
        return manifest_path

    tz = _resolve_timezone(timezone_name)
    ts = created_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local_ts = ts.astimezone(tz)

    manifest = {
        "run_id": run_id,
        "started_at": local_ts.isoformat(),
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def finalize_run_manifest(
    run_dir: Path,
    *,
    status: str = "success",
    duration_seconds: Optional[float] = None,
    solver: Optional[str] = None,
    services_total: int = 0,
    services_planned: int = 0,
    services_failed: int = 0,
    labors_summary: Optional[dict] = None,
    instance_config: Optional[dict] = None,
    algorithm_config: Optional[dict] = None,
    timezone_name: Optional[str] = None,
) -> Path:
    """
    Update run.json with finish time and pipeline summary.
    Reads the existing manifest (preserving started_at / run_id) and enriches it.
    """
    manifest_path = run_dir / "run.json"

    existing: dict = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            pass

    tz = _resolve_timezone(timezone_name)
    finished_at = datetime.now(timezone.utc).astimezone(tz)

    manifest: dict = {
        **existing,
        "finished_at": finished_at.isoformat(),
        "status": status,
    }

    if duration_seconds is not None:
        manifest["duration_seconds"] = round(duration_seconds, 1)

    if solver:
        manifest["solver"] = solver

    if services_total > 0 or services_planned > 0 or services_failed > 0:
        manifest["services"] = {
            "total": services_total,
            "planned": services_planned,
            "failed": services_failed,
        }

    if labors_summary:
        manifest["labors"] = labors_summary

    if instance_config:
        manifest["instance"] = instance_config

    if algorithm_config:
        manifest["algorithm"] = algorithm_config

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path
