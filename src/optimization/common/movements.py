"""
movements.py — shared driver movement timeline builder.

Extracted from offline_algorithms.py so that it can be reused by
preassigned.py, insert_algorithms.py, and any future algorithm or
analysis module without creating a dependency on the OFFLINE algorithm.
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .distance_utils import distance
from ...data.id_normalization import normalize_id_value

logger = logging.getLogger(__name__)

DistDict = Dict[str, Any] | None


# ---------------------------------------------------------------------------
# Internal helpers  (also importable for use in insert_algorithms.py)
# ---------------------------------------------------------------------------

def _location_fields(row: pd.Series) -> Dict[str, Any]:
    return {
        "city_code": row.get("city_code"),
        "department_code": row.get("department_code"),
        "city_name": row.get("city_name"),
        "department_name": row.get("department_name"),
    }


def _filter_drivers_by_city(directorio_df: pd.DataFrame, city: Any) -> pd.DataFrame:
    """
    Return the subset of directorio_df whose department matches *city*.

    The match is attempted against several column names in priority order
    so the function works regardless of the directory schema version.
    """
    if directorio_df is None or directorio_df.empty:
        return pd.DataFrame()

    department_key = str(city).strip()
    if not department_key:
        return directorio_df.iloc[0:0].copy()

    candidate_masks: List[pd.Series] = []
    for col in ("department_code", "department_name", "city"):
        if col in directorio_df.columns:
            candidate_masks.append(
                directorio_df[col].astype(str).str.strip() == department_key
            )

    for mask in candidate_masks:
        df_city = directorio_df.loc[mask].copy()
        if not df_city.empty:
            return df_city

    return directorio_df.iloc[0:0].copy()


def _extract_driver_key(driver_row: pd.Series) -> Optional[str]:
    """Return the canonical driver ID string from a directory row, or None."""
    for col in ("driver_id", "ALFRED'S"):
        if col in driver_row.index:
            value = driver_row.get(col)
            if pd.notna(value):
                key = str(value).strip()
                if key:
                    return key
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_driver_movements(
    labors_df: pd.DataFrame,
    directory_df: pd.DataFrame,
    day_str: str,
    dist_method: str,
    dist_dict: Optional[DistDict],
    ALFRED_SPEED: float,
    city_key: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Build a fully standardised movement timeline for each driver.

    Every labor is expanded into exactly three rows:
        1. ``<labor_id>_free``  — Waiting time before moving to next labor.
        2. ``<labor_id>_move``  — Travel from previous end-point to labor start.
        3. ``<labor_id>_labor`` — The actual labor itself.

    If the driver starts moving immediately (no wait), the ``_free`` record
    will have zero duration (``actual_start == actual_end``).
    If the next labor is at the same location, ``_move`` will have zero
    distance and possibly zero duration.

    Parameters
    ----------
    labors_df : pd.DataFrame
        DataFrame of already-assigned labors containing at minimum:
        ``assigned_driver``, ``actual_start``, ``actual_end``,
        ``map_start_point``, ``map_end_point``,
        ``labor_id``, ``labor_name``, ``labor_category``, ``schedule_date``.
    directory_df : pd.DataFrame
        Driver directory.  Must include driver start positions and
        shift start times (columns ``driver_id``/``ALFRED'S``,
        ``latitud``, ``longitud``, ``start_time``).
    day_str : str
        Planning date in ``YYYY-MM-DD`` format.
    dist_method : str
        Distance computation method (``'haversine'``, ``'osrm'``, …).
    dist_dict : dict or None
        Pre-computed distance look-up table.
    ALFRED_SPEED : float
        Average driver travel speed in km/h.
    city_key : str
        Department / city key used to filter the driver directory.
    **kwargs :
        Passed through to ``distance()`` (e.g. ``osrm_url``).

    Returns
    -------
    pd.DataFrame
        Standardised movement DataFrame with columns:
        ``service_id``, ``labor_id``, ``labor_context_id``,
        ``labor_name``, ``labor_category``, ``assigned_driver``,
        ``schedule_date``, ``actual_start``, ``actual_end``,
        ``start_point``, ``end_point``, ``distance_km``,
        ``duration_min``, ``date``,
        plus any location fields from ``_location_fields``.
    """
    driver_col, start_col, end_col = "assigned_driver", "actual_start", "actual_end"

    # --- Determine timezone from the data ---
    tz = labors_df["schedule_date"].dt.tz
    if tz is None and not labors_df[start_col].dropna().empty:
        tz = labors_df[start_col].dropna().iloc[0].tz

    day_dt = pd.to_datetime(day_str).date()

    # --- Initialise driver positions and shift-start times ---
    driver_pos: Dict[str, str] = {}
    driver_end: Dict[str, pd.Timestamp] = {}
    driver_has_departed: Dict[str, bool] = {}

    df_city = _filter_drivers_by_city(directory_df, city_key)
    for _, d in df_city.iterrows():
        if pd.isna(d["latitud"]):
            continue
        drv = _extract_driver_key(d)
        if drv is None:
            continue
        driver_pos[drv] = f"POINT ({d['longitud']} {d['latitud']})"
        start_t = datetime.strptime(d["start_time"], "%H:%M:%S").time()
        st = datetime.combine(day_dt, start_t)
        driver_end[drv] = pd.Timestamp(st, tz=tz)
        driver_has_departed[drv] = False

    # --- Build standardised movement records ---
    records: List[Dict[str, Any]] = []

    for _, row in labors_df.dropna(subset=[start_col]).sort_values(start_col).iterrows():
        driver_value = row[driver_col]
        if pd.isna(driver_value):
            continue
        drv = str(driver_value).strip()
        if not drv:
            continue

        labor_id = normalize_id_value(row.get("labor_id"))
        if labor_id is None:
            continue

        prev_end = driver_end.get(drv)
        prev_pos = driver_pos.get(drv)

        if prev_end is None or prev_pos is None:
            continue

        # Travel from previous position to this labor's start
        move_dkm, _ = distance(
            prev_pos,
            row["map_start_point"],
            method=dist_method,
            dist_dict=dist_dict,
            **kwargs,
        )
        move_dkm = 0.0 if (pd.isna(move_dkm) or math.isnan(move_dkm)) else move_dkm
        travel_time = (
            timedelta(minutes=(move_dkm / ALFRED_SPEED * 60)) if move_dkm > 0 else timedelta(0)
        )

        labor_start = row[start_col]
        move_end = row[start_col]
        move_start = row[start_col] - travel_time

        if not driver_has_departed[drv]:
            free_start = move_start if move_start <= prev_end else prev_end
            driver_has_departed[drv] = True
        else:
            free_start = prev_end

        free_end = move_start

        # (1) FREE_TIME
        records.append({
            "service_id": row.get("service_id", np.nan),
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_free",
            "labor_name": "FREE_TIME",
            "labor_category": "FREE_TIME",
            driver_col: drv,
            "schedule_date": row["schedule_date"],
            start_col: free_start,
            end_col: free_end,
            "start_point": prev_pos,
            "end_point": prev_pos,
            "distance_km": 0.0,
            **_location_fields(row),
        })

        # (2) DRIVER_MOVE
        records.append({
            "service_id": row.get("service_id", np.nan),
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_move",
            "labor_name": "DRIVER_MOVE",
            "labor_category": "DRIVER_MOVE",
            driver_col: drv,
            "schedule_date": row["schedule_date"],
            start_col: move_start,
            end_col: move_end,
            "start_point": prev_pos,
            "end_point": row["map_start_point"],
            "distance_km": move_dkm,
            **_location_fields(row),
        })

        # (3) Actual labor
        records.append({
            "service_id": row.get("service_id", np.nan),
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_labor",
            "labor_name": row["labor_name"],
            "labor_category": row["labor_category"],
            driver_col: drv,
            "schedule_date": row["schedule_date"],
            start_col: labor_start,
            end_col: row[end_col],
            "start_point": row["map_start_point"],
            "end_point": row["map_end_point"],
            "distance_km": row["dist_km"],
            **_location_fields(row),
        })

        driver_end[drv] = row[end_col]
        driver_pos[drv] = row["map_end_point"]

    # --- Final assembly ---
    df_moves = pd.DataFrame(records)
    if df_moves.empty:
        return df_moves

    df_moves[start_col] = pd.to_datetime(df_moves[start_col])
    df_moves[end_col] = pd.to_datetime(df_moves[end_col])

    df_moves["duration_min"] = (
        (df_moves[end_col] - df_moves[start_col]).dt.total_seconds() / 60
    ).round(1).fillna(0)

    df_moves["date"] = df_moves["schedule_date"].dt.date

    df_moves = df_moves.sort_values(
        ["schedule_date", driver_col, start_col, end_col]
    ).reset_index(drop=True)

    return df_moves
