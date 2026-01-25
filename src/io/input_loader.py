import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ======================================================
# Public API
# ======================================================

def load_local_input(
    local_path: str | Path,
    *,
    write_debug_json: bool = False,
    debug_output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load local input data from a CSV file and transform it into
    an API-like JSON payload for local development/testing.

    Args:
        local_path: Path to the input CSV file
        write_debug_json: Whether to write the generated payload to disk
        debug_output_path: Optional path for debug JSON output

    Returns:
        Dict[str, Any]: API-compatible input payload
    """
    csv_path = Path(local_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    logger.debug("Loading local input from CSV", extra={"path": str(csv_path)})

    df = pd.read_csv(csv_path)

    _validate_required_columns(df)

    payload: Dict[str, Any] = {
        "count": 0,
        "next": None,
        "previous": None,
        "data": [],
    }

    for service_id, group in df.groupby("service_id"):
        try:
            record = _build_service_record(service_id, group)
            payload["data"].append(record)
            payload["count"] += 1

        except Exception as exc:
            logger.exception(
                "Failed to build service record",
                extra={"service_id": service_id},
            )
            raise

    logger.info(
        "local_input_loaded services=%s",
        payload["count"],
    )

    if write_debug_json:
        _write_debug_payload(payload, debug_output_path)

    return payload


# ======================================================
# Helpers
# ======================================================

def _validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure required columns exist in the CSV input.
    """
    required_columns = {
        "service_id",
        "schedule_date",
        "start_address_id",
        "start_address_point",
        "end_address_id",
        "end_address_point",
        "city",
        "labor_id",
        "labor_type",
        "labor_name",
        "labor_category",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")


def _build_service_record(service_id: Any, group: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a single service record from grouped rows.
    """
    schedule_date = pd.to_datetime(group["schedule_date"].iloc[0])

    record: Dict[str, Any] = {
        "id": int(service_id),
        "state": "",
        "scheduleDate": schedule_date.isoformat(),
        "startAddress": {
            "id": int(group["start_address_id"].iloc[0]),
            "point": _parse_point(group["start_address_point"].iloc[0]),
        },
        "endAddress": {
            "id": int(group["end_address_id"].iloc[0]),
            "point": _parse_point(group["end_address_point"].iloc[0]),
        },
        "city": {
            "id": int(group["city"].iloc[0]),
        },
        "serviceLabors": [],
    }

    for labor_id, labor_group in group.groupby("labor_id"):
        labor_record = {
            "id": int(labor_id),    # type: ignore
            "labor": {
                "id": int(labor_group["labor_type"].iloc[0]),
                "name": labor_group["labor_name"].iloc[0],
                "category": labor_group["labor_category"].iloc[0],
            },
        }
        record["serviceLabors"].append(labor_record)

    return record


def _parse_point(point_string: str) -> Dict[str, float]:
    """
    Convert a WKT POINT string into a longitude/latitude dict.

    Expected format:
        POINT (longitude latitude)
    """
    if not isinstance(point_string, str):
        raise ValueError(f"Invalid POINT value: {point_string!r}")

    try:
        cleaned = (
            point_string.replace("POINT", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )
        lon_str, lat_str = cleaned.split()

        return {
            "longitude": float(lon_str),
            "latitude": float(lat_str),
        }

    except Exception as exc:
        raise ValueError(f"Failed to parse POINT string: {point_string}") from exc


def _write_debug_payload(
    payload: Dict[str, Any],
    output_path: Optional[str | Path],
) -> None:
    """
    Persist the generated payload for debugging purposes.
    """
    path = Path(output_path) if output_path else Path("processed_input.json")

    logger.debug("Writing debug payload", extra={"path": str(path)})

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
