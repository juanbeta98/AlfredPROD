"""
service_builder.py — Build a synthetic labor DataFrame from a ServiceRequest.

The resulting DataFrame is passed to run_insertion_worker as new_labors_df.
All labors get schedule_date = slot_time.  The INSERT algorithm's internal
forced_start_time propagation (via curr_end_time) handles downstream timing
for labors 2, 3, ... automatically.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.availability.models import ServiceRequest

COLOMBIA_TZ = "America/Bogota"


def build_candidate_df(
    request: ServiceRequest,
    slot_time: datetime,
    day_str: str,
) -> pd.DataFrame:
    """
    Build a synthetic labor DataFrame for a single slot probe.

    Args:
        request  : ServiceRequest with labor definitions.
        slot_time: The schedule_date for the first labor (timezone-aware).
        day_str  : "YYYY-MM-DD" string for the scheduling day.

    Returns:
        DataFrame with one row per labor, ready for run_insertion_worker.
    """
    tz = ZoneInfo(COLOMBIA_TZ)

    if slot_time.tzinfo is None:
        slot_time = slot_time.replace(tzinfo=tz)

    schedule_ts = pd.Timestamp(slot_time).tz_convert(COLOMBIA_TZ)
    schedule_date = schedule_ts.date()

    rows = []
    for labor in sorted(request.labors, key=lambda l: l.labor_sequence):
        rows.append({
            # Identity
            "service_id": request.service_id,
            "labor_id": f"{request.service_id}_L{labor.labor_sequence}",
            "labor_sequence": labor.labor_sequence,
            # Classification
            "labor_category": labor.labor_category,
            "labor_type": labor.labor_type,
            "labor_name": labor.labor_name or labor.labor_type or "service",
            # Locations
            "map_start_point": labor.map_start_point,
            "map_end_point": labor.map_end_point,
            "start_address_point": labor.map_start_point,
            "end_address_point": labor.map_end_point,
            # Timing — all labors start at slot_time;
            # the INSERT algorithm propagates actual times downstream
            "schedule_date": schedule_ts,
            "date": schedule_date,
            # Geography
            "department_code": str(request.department_code),
            "city_code": None,
            "city_name": None,
            "department_name": None,
            # Duration (for non-VT labors)
            "estimated_time": labor.estimated_time,
            # Unassigned — ready for insertion
            "assigned_driver": pd.NA,
            # Placeholders
            "dist_km": pd.NA,
            "actual_start": pd.NaT,
            "actual_end": pd.NaT,
        })

    return pd.DataFrame(rows)
