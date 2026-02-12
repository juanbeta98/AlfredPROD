from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.config import Config

_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _resolve_timezone(tz_name: Optional[str] = None) -> ZoneInfo:
    """
    Resolve artifact timezone from explicit input or config.
    """
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
    """
    Build a compact local-time timestamp token for artifact names.
    """
    tz = _resolve_timezone(timezone_name)
    ts = created_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    local_ts = ts.astimezone(tz)
    return local_ts.strftime("%Y%m%dT%H%M%S")


def sanitize_token(value: Any, *, fallback: str = "na") -> str:
    """
    Convert an arbitrary value to a filesystem-safe token.
    """
    if value is None:
        return fallback

    text = str(value).strip()
    if not text:
        return fallback

    text = _SANITIZE_PATTERN.sub("-", text).strip("._-")
    return text or fallback


def build_run_subdir(run_id: Any) -> str:
    """
    Standard run-scoped directory name.
    """
    return f"run-{sanitize_token(run_id, fallback='local')}"


def build_artifact_stem(
    prefix: str,
    *,
    run_id: Any,
    created_at: Optional[datetime] = None,
    timezone_name: Optional[str] = None,
    request_id: Any = None,
) -> str:
    """
    Build a standardized artifact stem:
    <prefix>__ts-<local_ts>
    """
    _ = run_id  # directory already scopes artifacts by run id
    _ = request_id  # request id is intentionally excluded from filenames
    parts = [
        sanitize_token(prefix, fallback="artifact"),
        f"ts-{timestamp_token(created_at, timezone_name=timezone_name)}",
    ]
    return "__".join(parts)
