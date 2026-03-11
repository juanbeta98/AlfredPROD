"""
feasibility_probe.py — Single time-slot feasibility check.

Wraps one call to run_insertion_worker (the INSERT algorithm's core) and
interprets the result to determine whether the candidate service can be
scheduled at the given slot_time.

Design notes:
- Uses seed=0 for a deterministic single probe (no random multi-iteration).
- The base schedule state is read-only; run_insertion_worker operates on
  internal copies and never mutates the ScheduleState.
- Each probe is fully independent and safe for parallel execution.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.availability.models import ScheduleState, ServiceRequest, TimeSlotResult
from src.availability.service_builder import build_candidate_df
from src.optimization.algorithms.insert.insert_algorithms import (
    get_drivers,
    run_insertion_worker,
)
from src.optimization.common.utils import compute_workday_end

logger = logging.getLogger(__name__)

COLOMBIA_TZ = "America/Bogota"


def _city_dist_slice(dist_dict_all: Any, city_key: str) -> Dict[Any, Any]:
    """Extract the distance sub-dict for a given city key."""
    if not isinstance(dist_dict_all, dict):
        return {}
    city_key_str = str(city_key)
    for key, value in dist_dict_all.items():
        if str(key) == city_key_str:
            return value if isinstance(value, dict) else {}
    return {}


def _city_day_slice(df: pd.DataFrame, city_key: str, day) -> pd.DataFrame:
    """Filter a DataFrame to the rows matching city_key and day."""
    if df is None or df.empty:
        return pd.DataFrame()
    date_series = pd.to_datetime(df["schedule_date"], errors="coerce").dt.date
    mask = (
        df["department_code"].astype(str).str.strip() == str(city_key)
    ) & (date_series == day)
    return df[mask].copy()


def _infer_reason(
    result: Optional[Dict],
    slot_time: datetime,
    workday_end_dt,
) -> str:
    """Classify why a slot is infeasible."""
    if workday_end_dt is not None:
        ts = pd.Timestamp(slot_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize(COLOMBIA_TZ)
        if ts >= workday_end_dt:
            return "workday_end"
    if result is None or result.get("num_inserted", 0) == 0:
        return "no_driver"
    return "not_inserted"


def probe_slot(
    slot_time: datetime,
    request: ServiceRequest,
    state: ScheduleState,
) -> TimeSlotResult:
    """
    Test whether the candidate service can be inserted at slot_time.

    Uses a single deterministic probe (seed=0).  Feasibility is defined as
    all labors in the service being successfully inserted:
        result["num_inserted"] == len(request.labors)

    Args:
        slot_time : The schedule_date to test for the first labor.
        request   : ServiceRequest describing the candidate service.
        state     : Frozen preassigned schedule state for the day.

    Returns:
        TimeSlotResult with feasible=True/False and an optional reason string.
    """
    city_key = state.department_code
    model_params = state.settings.model_params
    day_str = state.day_str
    day = pd.Timestamp(day_str).date()

    # Build synthetic candidate DataFrame
    new_labors_df = build_candidate_df(request, slot_time, day_str)

    # Slice base schedule to city+day
    city_base_labors = _city_day_slice(state.base_labors_df, city_key, day)
    city_base_moves = _city_day_slice(state.base_moves_df, city_key, day)

    # Get available driver IDs for this city from the directory
    drivers = get_drivers(city_base_labors, state.master_data.directorio_df, city_key)

    # Workday boundary and city distance lookup
    workday_end_dt = compute_workday_end(
        day_str=day_str,
        workday_end_str=model_params.workday_end_str,
        tzinfo=COLOMBIA_TZ,
    )
    dist_dict = _city_dist_slice(state.master_data.dist_dict, city_key)

    duraciones = (
        state.master_data.duraciones_df
        if state.master_data.duraciones_df is not None
        and not state.master_data.duraciones_df.empty
        else None
    )

    result = run_insertion_worker(
        base_labors_df=city_base_labors,
        base_moves_df=city_base_moves,
        new_labors_df=new_labors_df,
        seed=0,
        city=city_key,
        fecha=day_str,
        directorio_df=state.master_data.directorio_df,
        drivers=drivers,
        dist_dict=dist_dict,
        distance_method=state.settings.distance_method,
        alfred_speed=model_params.alfred_speed_kmh,
        vehicle_transport_speed=model_params.vehicle_transport_speed_kmh,
        tiempo_alistar=model_params.tiempo_alistar_min,
        tiempo_finalizacion=model_params.tiempo_finalizacion_min,
        tiempo_gracia=model_params.tiempo_gracia_min,
        early_buffer=model_params.tiempo_previo_min,
        workday_end_dt=workday_end_dt,
        duraciones_df=duraciones,
        time_method=state.settings.time_method,
        time_dict={},
    )

    expected = len(request.labors)
    inserted = result.get("num_inserted", 0) if result else 0
    feasible = inserted == expected

    reason = None if feasible else _infer_reason(result, slot_time, workday_end_dt)

    logger.debug(
        "probe_slot slot=%s feasible=%s inserted=%d/%d city=%s",
        pd.Timestamp(slot_time).isoformat(),
        feasible, inserted, expected, city_key,
    )

    return TimeSlotResult(slot_time=slot_time, feasible=feasible, reason=reason)
