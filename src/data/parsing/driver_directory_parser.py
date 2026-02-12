from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.data.id_normalization import normalize_id_value
from src.datetime_utils import utc_to_colombia_timestamp
from src.location import parse_service_location

logger = logging.getLogger(__name__)


class DriverDirectoryParser:
    """
    Parse driver directory payloads into a DataFrame expected by optimization.
    """

    @staticmethod
    def parse(json_data: Any) -> pd.DataFrame:
        if json_data is None:
            raise ValueError("Driver directory payload is None")

        if isinstance(json_data, dict) and "results" in json_data:
            json_data = json_data.get("results")

        if not isinstance(json_data, list):
            raise TypeError("Driver directory payload must be a list")

        records = []
        skipped = 0
        for item in json_data:
            record = DriverDirectoryParser._parse_driver(item)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        if not records:
            logger.warning("driver_directory_parsed_empty skipped=%s", skipped)
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(
            "driver_directory_parsed rows=%s skipped=%s",
            len(df),
            skipped,
        )
        return df

    @staticmethod
    def _parse_driver(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        driver_id = normalize_id_value(item.get("id"))
        if driver_id is None:
            return None

        address = item.get("address") or {}
        point = address.get("point") or {}
        lon = point.get("longitude")
        lat = point.get("latitude")

        schedule = item.get("schedule") or {}
        start_hour_raw = schedule.get("startHour")
        end_hour_raw = schedule.get("endHour")

        city = address.get("city")
        location = parse_service_location(city)

        if lon is None or lat is None:
            logger.debug("driver_missing_coordinates driver_id=%s", driver_id)
            return None

        try:
            lon_val = float(lon)
            lat_val = float(lat)
        except (TypeError, ValueError):
            logger.debug("driver_invalid_coordinates driver_id=%s", driver_id)
            return None

        start_hour = _normalize_time(start_hour_raw)
        end_hour = _normalize_time(end_hour_raw)

        if start_hour is None:
            logger.debug("driver_missing_start_hour driver_id=%s", driver_id)
            return None

        has_location = any(
            [
                location["city_code"] is not None,
                bool(location["city_name"]),
                location["department_code"] is not None,
                bool(location["department_name"]),
            ]
        )
        if not has_location:
            logger.debug("driver_missing_location driver_id=%s", driver_id)
            return None

        return {
            "driver_id": driver_id,
            "latitud": lat_val,
            "longitud": lon_val,
            "start_time": start_hour,
            "end_time": end_hour,
            "city_id": normalize_id_value(location["city_code"]),
            "city_name": location["city_name"],
            "department_code": normalize_id_value(location["department_code"]),
            "department_name": location["department_name"],
            "location_resolution_status": location["location_resolution_status"],
        }


def _normalize_time(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = utc_to_colombia_timestamp(value, errors="coerce")
        return None if pd.isna(ts) else ts.strftime("%H:%M:%S")
    if isinstance(value, str):
        val = value.strip()
        if not val:
            return None
        if any(token in val for token in ("T", "Z", "+", "-")):
            ts = utc_to_colombia_timestamp(val, errors="coerce")
            if not pd.isna(ts):
                return ts.strftime("%H:%M:%S")
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(val, fmt).time().strftime("%H:%M:%S")
            except ValueError:
                continue
        return val if len(val.split(":")) == 3 else None
    return None
