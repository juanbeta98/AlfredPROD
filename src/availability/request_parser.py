"""
request_parser.py — Parse the external API request format into ServiceRequest.

The external format the customer/API sends is address-centric and minimal:

    {
        "department_id": 25,
        "department_name": "CUNDINAMARCA",
        "date": "2026-03-09T14:30:00-05:00",
        "start_address": {
            "id": 123456,
            "name": "Calle 123",
            "city": "Bogotá",
            "department": "CUNDINAMARCA",
            "point": {"x": -74.08, "y": 4.71, "srid": 4326}
        },
        "end_address": {
            "id": 123456,
            "name": "Calle 123",
            "city": "Bogotá",
            "department": "CUNDINAMARCA",
            "point": {"x": -74.05, "y": 4.62, "srid": 4326}
        }
    }

Key mapping decisions
---------------------
- department_id (int) → department_code (str): str(department_id)
- date           → desired_slot.  No as_of_time in the external format;
                   defaults to now_colombia() so the scan covers the real
                   remainder of the day.
- start/end_address.point {x, y} → WKT "POINT (x y)".
  x = longitude, y = latitude (standard GIS convention used by the pipeline).
  srid is accepted but not used for coordinate transformation (WGS-84/4326
  assumed; warn if different).
- A single VEHICLE_TRANSPORTATION LaborRequest is auto-constructed from the
  two addresses.  service_id is auto-generated when absent.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from src.availability.models import LaborRequest, ServiceRequest
from src.utils.datetime_utils import now_colombia

logger = logging.getLogger(__name__)

_EXPECTED_SRID = 4326


def parse_api_request(
    data: Dict[str, Any],
    *,
    as_of_time: Optional[datetime] = None,
    labor_type: Optional[str] = None,
    labor_name: Optional[str] = None,
) -> ServiceRequest:
    """
    Convert an external API availability request into a ServiceRequest.

    Args:
        data        : Raw JSON dict in the external API format.
        as_of_time  : Override the "current time" for the slot scan.
                      Defaults to now_colombia() — the real current moment.
        labor_type  : Optional labor type key for the auto-created labor.
                      Defaults to None (INSERT falls back to tiempo_other_min).
        labor_name  : Optional human-readable label for the labor.

    Returns:
        ServiceRequest ready for check_availability() or scan_availability().

    Raises:
        ValueError: On missing required fields or unparseable coordinates.
    """
    # --- department ---
    dept_id = data.get("department_id")
    if dept_id is None:
        raise ValueError("Missing required field: department_id")
    department_code = str(int(dept_id))

    # --- desired slot ---
    date_raw = data.get("date")
    if not date_raw:
        raise ValueError("Missing required field: date")
    desired_slot = datetime.fromisoformat(str(date_raw))

    # --- as_of_time ---
    effective_as_of = as_of_time if as_of_time is not None else now_colombia()

    # --- service_id ---
    service_id = str(data.get("service_id") or f"avail-{uuid.uuid4().hex[:8]}")

    # --- coordinates ---
    start_point = _parse_address_point(data.get("start_address"), field="start_address")
    end_point = _parse_address_point(data.get("end_address"), field="end_address")

    # --- auto-create a single VEHICLE_TRANSPORTATION labor ---
    labor = LaborRequest(
        labor_sequence=1,
        labor_category="VEHICLE_TRANSPORTATION",
        map_start_point=start_point,
        map_end_point=end_point,
        labor_type=labor_type,
        labor_name=labor_name or "Traslado",
    )

    return ServiceRequest(
        service_id=service_id,
        department_code=department_code,
        labors=[labor],
        desired_slot=desired_slot,
        as_of_time=effective_as_of,
    )


def _parse_address_point(address: Any, *, field: str) -> str:
    """
    Extract a WKT POINT string from an address dict.

    Accepts point as {"x": lon, "y": lat, "srid": ...} or
    {"longitude": lon, "latitude": lat}.

    Returns "POINT (lon lat)" WKT string.
    """
    if not isinstance(address, dict):
        raise ValueError(f"Missing or invalid field: {field}")

    point = address.get("point")
    if not isinstance(point, dict):
        raise ValueError(f"{field}.point is missing or not an object")

    # Warn if srid is not the expected WGS-84
    srid = point.get("srid")
    if srid is not None and int(srid) != _EXPECTED_SRID:
        logger.warning(
            "address_point_unexpected_srid field=%s srid=%s expected=%s "
            "(coordinate transformation not applied; assuming WGS-84)",
            field, srid, _EXPECTED_SRID,
        )

    # Prefer x/y; fall back to longitude/latitude keys
    lon = point.get("x") if point.get("x") is not None else point.get("longitude")
    lat = point.get("y") if point.get("y") is not None else point.get("latitude")

    if lon is None or lat is None:
        raise ValueError(
            f"{field}.point must have 'x'/'y' or 'longitude'/'latitude' fields"
        )

    try:
        lon_f, lat_f = float(lon), float(lat)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field}.point coordinates are not numeric: x={lon!r}, y={lat!r}"
        ) from exc

    return f"POINT ({lon_f} {lat_f})"
