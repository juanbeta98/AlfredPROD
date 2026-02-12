from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.data.id_normalization import normalize_id_value
from src.data.parsing.driver_directory_parser import DriverDirectoryParser

logger = logging.getLogger(__name__)


def load_driver_directory_csv(path: str | Path) -> List[dict]:
    """
    Load a local driver directory CSV and convert each row into an
    API-like driver payload item.
    """
    driver_path = Path(path)
    if not driver_path.exists():
        raise FileNotFoundError(f"Driver directory file not found: {driver_path}")

    logger.debug("Loading driver directory file", extra={"path": str(driver_path)})

    df = pd.read_csv(driver_path, low_memory=False)
    if df.empty:
        return []

    required_columns = {
        "driver_id",
        "start_time",
        "end_time",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"Driver directory CSV missing required columns: {missing}"
        )
    if "point" not in df.columns and "ubicacion" not in df.columns:
        raise ValueError(
            "Driver directory CSV missing location column: expected 'point' or 'ubicacion'"
        )
    location_columns = {"city", "city_id", "city_name", "department_code", "department_name"}
    if not (location_columns & set(df.columns)):
        raise ValueError(
            "Driver directory CSV missing location identifiers: expected at least one of "
            f"{sorted(location_columns)}"
        )

    records: List[dict] = []
    for _, row in df.iterrows():
        payload = _row_to_driver_payload(row.to_dict())
        records.append(payload)
    return records


def _row_to_driver_payload(row: Dict[str, Any]) -> dict:
    lon, lat = _parse_wkt_point(row.get("point") or row.get("ubicacion"))

    legacy_city = _clean_nullable_text(row.get("city"))
    city_id = normalize_id_value(row.get("city_id"))
    if city_id is None:
        city_id = _normalize_city_code_candidate(legacy_city)

    city_name = _clean_nullable_text(row.get("city_name"))
    if city_name is None and legacy_city and not legacy_city.isdigit() and "-" not in legacy_city:
        city_name = legacy_city

    department_code = normalize_id_value(row.get("department_code"))
    department_name = _clean_nullable_text(row.get("department_name"))

    payload_name: str | None = None
    if legacy_city and not legacy_city.isdigit() and "-" in legacy_city:
        payload_name = legacy_city
    elif department_name and city_name:
        payload_name = f"{department_name}-{city_name}"
    elif city_name:
        payload_name = city_name
    elif department_name:
        payload_name = department_name

    location_data: dict[str, Any] = {}
    if city_name is not None:
        location_data["city_name"] = city_name
    if department_code is not None:
        location_data["department_code"] = department_code
    if department_name is not None:
        location_data["department_name"] = department_name

    city_payload: dict[str, Any] = {}
    if city_id is not None:
        city_payload["id"] = city_id
    if payload_name is not None:
        city_payload["name"] = payload_name
    if location_data:
        city_payload["data"] = location_data

    return {
        "id": normalize_id_value(row.get("driver_id")),
        "address": {
            "point": {
                "longitude": lon,
                "latitude": lat,
            },
            "city": city_payload or None,
        },
        "schedule": {
            "startHour": _clean_nullable_text(row.get("start_time")),
            "endHour": _clean_nullable_text(row.get("end_time")),
        },
    }


def _parse_wkt_point(value: Any) -> Tuple[float | None, float | None]:
    text = _clean_nullable_text(value)
    if text is None:
        return None, None

    match = re.search(
        r"POINT\s*\(\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None

    try:
        lon = float(match.group(1))
        lat = float(match.group(2))
    except ValueError:
        return None, None

    return lon, lat


def _clean_nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _normalize_city_code_candidate(value: Any) -> str | None:
    text = _clean_nullable_text(value)
    if text is None:
        return None
    if text.isdigit():
        return text
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return None

def load_driver_directory_df(path: str | Path) -> pd.DataFrame:
    """
    Load and parse a local CSV driver directory into a DataFrame.
    """
    raw = load_driver_directory_csv(path)
    return DriverDirectoryParser.parse(raw)
