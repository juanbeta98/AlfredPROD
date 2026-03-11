import logging
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils.datetime_utils import normalize_datetime_columns_to_colombia
from src.geo.location import series_location_key

logger = logging.getLogger(__name__)

INTERMEDIATE_DELAY_WARNING_MINUTES = 15.0
_BOGOTA_TZ_DTYPE = "datetime64[ns, America/Bogota]"


def _log_preassigned_reconstruction_summary(metrics: Dict[str, Any]) -> None:
    logger.info(
        "preassigned_reconstruction_summary total=%s failed=%s overtime=%s infeasible=%s warnings=%s",
        int(metrics.get("preassigned_total", 0)),
        int(metrics.get("preassigned_failed", 0)),
        int(metrics.get("preassigned_overtime", 0)),
        int(metrics.get("preassigned_infeasible", 0)),
        int(metrics.get("preassigned_warnings", 0)),
    )


def _initialize_preassigned_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df["is_infeasible"] = False
    df["is_warning"] = False
    df["infeasibility_cause_code"] = None
    df["infeasibility_cause_detail"] = None
    df["warning_code"] = None
    df["warning_detail"] = None
    df["payload_schedule_reference"] = pd.Series(
        pd.NaT,
        index=df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    df["computed_arrival"] = pd.Series(
        pd.NaT,
        index=df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    df["previous_labor_end"] = pd.Series(
        pd.NaT,
        index=df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    df["feasible_window_end"] = pd.Series(
        pd.NaT,
        index=df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    df["driver_available_at"] = pd.Series(
        pd.NaT,
        index=df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    df["computed_travel_min"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )
    df["minutes_late_payload"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )
    df["minutes_after_window"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Float64",
    )
    return df


def _effective_payload_schedule(row: pd.Series) -> pd.Timestamp:
    payload_value = row.get("payload_labor_schedule_date")
    payload_ts = pd.to_datetime(payload_value, errors="coerce")
    if pd.notna(payload_ts):
        return payload_ts

    schedule_value = row.get("schedule_date")
    return pd.to_datetime(schedule_value, errors="coerce")


def _delay_minutes(actual_ts: Any, planned_ts: Any) -> Optional[float]:
    actual = pd.to_datetime(actual_ts, errors="coerce", utc=True)
    planned = pd.to_datetime(planned_ts, errors="coerce", utc=True)
    if pd.isna(actual) or pd.isna(planned):
        return None
    return float((actual - planned).total_seconds() / 60.0)


def _mark_infeasible(
    assigned_df: pd.DataFrame,
    idx: Any,
    *,
    cause_code: str,
    detail: str,
) -> None:
    assigned_df.loc[idx, "is_infeasible"] = True
    assigned_df.loc[idx, "infeasibility_cause_code"] = cause_code
    assigned_df.loc[idx, "infeasibility_cause_detail"] = detail


def _mark_warning(
    assigned_df: pd.DataFrame,
    idx: Any,
    *,
    warning_code: str,
    detail: str,
) -> None:
    assigned_df.loc[idx, "is_warning"] = True
    assigned_df.loc[idx, "warning_code"] = warning_code
    assigned_df.loc[idx, "warning_detail"] = detail


def _warn_if_intermediate_delay(
    *,
    assigned_df: pd.DataFrame,
    idx: Any,
    labor_sequence: Any,
    scheduled_ts: Any,
    actual_start_ts: Any,
) -> None:
    if _is_first_labor_sequence(labor_sequence):
        return
    delay_min = _delay_minutes(actual_start_ts, scheduled_ts)
    if delay_min is None or delay_min < INTERMEDIATE_DELAY_WARNING_MINUTES:
        return

    _mark_warning(
        assigned_df,
        idx,
        warning_code="intermediate_arrival_delay",
        detail=(
            f"Intermediate labor started {delay_min:.1f} min after payload schedule "
            f"(threshold={INTERMEDIATE_DELAY_WARNING_MINUTES:.0f})."
        ),
    )


def _require_location_columns(df: pd.DataFrame) -> None:
    required_cols = {"department_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing location columns after validation: {sorted(missing)}"
        )


def _finalize_preassigned_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final projection hook for preassigned labors.
    """
    if df is None or df.empty:
        return df
    return df


def _split_service_level_preassigned(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split labors by service-level preassignment.

    A service is considered preassigned when any of its labors has a
    non-null `assigned_driver`.
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "service_id" not in df.columns:
        raise ValueError("Missing required column: service_id")

    # A service is preassigned if ANY of its labors has a non-null assigned_driver.
    assigned_driver_col = df.get("assigned_driver")
    if assigned_driver_col is None:
        assigned_service_ids = pd.Index([])
    else:
        assigned_service_ids = df.loc[assigned_driver_col.notna(), "service_id"].unique()

    assigned_mask = df["service_id"].isin(assigned_service_ids)
    assigned_df = df.loc[assigned_mask].copy()
    unassigned_df = df.loc[~assigned_mask].copy()
    return assigned_df, unassigned_df


def _city_dist_slice(dist_dict: Dict[str, Dict] | Any, city_key: Any) -> Dict:
    if not isinstance(dist_dict, dict):
        return {}
    if city_key in dist_dict:
        return dist_dict.get(city_key, {})
    city_key_txt = str(city_key)
    for key, value in dist_dict.items():
        if str(key) == city_key_txt:
            return value if isinstance(value, dict) else {}
    return {}


def _ensure_dist_km(
    labors_df: pd.DataFrame,
    *,
    dist_method: str,
    city_dist: Dict[Any, Any],
) -> Tuple[pd.DataFrame, Dict[Any, Any]]:
    """
    Ensure labor rows include `dist_km` using the configured distance method.
    """
    if labors_df is None or labors_df.empty:
        return labors_df, city_dist

    df = labors_df.copy()
    if "dist_km" not in df.columns:
        df["dist_km"] = pd.NA

    missing_mask = df["dist_km"].isna()
    if not missing_mask.any():
        return df, city_dist

    from src.optimization.common.distance_utils import distance

    for idx, row in df.loc[missing_mask].iterrows():
        dist_km, city_dist = distance(
            row.get("map_start_point"),
            row.get("map_end_point"),
            method=dist_method,
            dist_dict=city_dist,
        )
        df.at[idx, "dist_km"] = 0.0 if pd.isna(dist_km) else float(dist_km)

    df["dist_km"] = pd.to_numeric(df["dist_km"], errors="coerce").fillna(0.0)
    return df, city_dist


def _is_first_labor_sequence(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _apply_payload_schedule_fallback(
    *,
    assigned_df: pd.DataFrame,
    service_rows: pd.DataFrame,
    payload_schedule_col: str,
    city_key: str,
    workday_end_dt: pd.Timestamp,
    duraciones_df: pd.DataFrame,
    dist_method: str,
    city_dist: Dict[Any, Any],
    tiempo_alistar: int,
    tiempo_other: float,
    vehicle_transport_speed: float,
    distance_fn: Any,
) -> Tuple[bool, Optional[pd.Timestamp], Dict[Any, Any], str]:
    """
    Reconstruct one service using immutable payload labor schedule dates.
    """
    if payload_schedule_col not in service_rows.columns:
        return False, None, city_dist, "missing_payload_schedule_column"

    service_rows_sorted = service_rows.sort_values(
        ["labor_sequence", "schedule_date"],
        kind="stable",
    )
    payload_starts = service_rows_sorted[payload_schedule_col]
    if payload_starts.isna().any():
        return False, None, city_dist, "missing_payload_labor_schedule_date"

    fallback_note = "payload_labor_schedule_fallback_after_missed_schedule"
    last_end: Optional[pd.Timestamp] = None

    for pos, (idx, row) in enumerate(service_rows_sorted.iterrows()):
        start_ts = payload_starts.iloc[pos]

        if pos < len(service_rows_sorted) - 1:
            end_ts = payload_starts.iloc[pos + 1]
        else:
            if row.get("labor_category") == "VEHICLE_TRANSPORTATION":
                dist_km, city_dist = distance_fn(
                    row.get("map_start_point"),
                    row.get("map_end_point"),
                    method=dist_method,
                    dist_dict=city_dist,
                )
                duration_min = (
                    tiempo_alistar
                    + (0 if pd.isna(dist_km) else float(dist_km) / vehicle_transport_speed * 60)
                )
                end_ts = start_ts + timedelta(minutes=duration_min)
                assigned_df.at[idx, "dist_km"] = 0.0 if pd.isna(dist_km) else float(dist_km)
            else:
                est_time = row.get("estimated_time")
                if est_time is not None and pd.notna(est_time):
                    duration_min = float(est_time)
                else:
                    duration_min = _nontransport_duration(
                        duraciones_df=duraciones_df,
                        city=city_key,
                        labor_type=row.get("labor_type"),
                        fallback_min=tiempo_other,
                    )
                    _mark_warning(
                        assigned_df, idx,
                        warning_code="estimated_time_fallback",
                        detail=(
                            f"estimated_time missing for labor_type={row.get('labor_type')!r}; "
                            f"used duraciones fallback ({duration_min:.1f} min)."
                        ),
                    )
                end_ts = start_ts + timedelta(minutes=duration_min)
                if pd.isna(assigned_df.at[idx, "dist_km"]):
                    assigned_df.at[idx, "dist_km"] = 0.0

        assigned_df.at[idx, "actual_start"] = start_ts
        assigned_df.at[idx, "actual_end"] = end_ts
        assigned_df.at[idx, "actual_status"] = "COMPLETED"
        assigned_df.at[idx, "preassigned_failed"] = False
        assigned_df.at[idx, "preassigned_failure_reason"] = None
        assigned_df.at[idx, "preassigned_overtime"] = bool(end_ts > workday_end_dt)
        assigned_df.at[idx, "preassigned_fallback_applied"] = True
        assigned_df.at[idx, "preassigned_reconstruction_note"] = fallback_note
        assigned_df.at[idx, "payload_schedule_reference"] = start_ts
        if pd.isna(assigned_df.at[idx, "computed_arrival"]):
            assigned_df.at[idx, "computed_arrival"] = start_ts
        if pd.isna(assigned_df.at[idx, "minutes_late_payload"]):
            assigned_df.at[idx, "minutes_late_payload"] = 0.0
        last_end = end_ts

    return True, last_end, city_dist, ""


def build_preassigned_state(
    labors_df: pd.DataFrame,
    *,
    directorio_df: pd.DataFrame,
    dist_method: str,
    dist_dict: Dict[str, Dict],
    alfred_speed: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labors_df is None or labors_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = labors_df.copy()
    _require_location_columns(df)
    if "map_start_point" not in df.columns and "start_address_point" in df.columns:
        df["map_start_point"] = df["start_address_point"]
    if "map_end_point" not in df.columns and "end_address_point" in df.columns:
        df["map_end_point"] = df["end_address_point"]

    assigned_df, unassigned_df = _split_service_level_preassigned(df)

    if assigned_df.empty:
        return _finalize_preassigned_df(assigned_df), unassigned_df, pd.DataFrame()

    if "actual_start" in assigned_df.columns and assigned_df["actual_start"].isna().any():
        logger.warning("preassigned_missing_actual_start rows=%s", int(assigned_df["actual_start"].isna().sum()))
    if "actual_end" in assigned_df.columns and assigned_df["actual_end"].isna().any():
        logger.warning("preassigned_missing_actual_end rows=%s", int(assigned_df["actual_end"].isna().sum()))

    if "schedule_date" not in assigned_df.columns:
        return _finalize_preassigned_df(assigned_df), unassigned_df, pd.DataFrame()

    from src.optimization.common.movements import build_driver_movements

    moves_parts = []
    assigned_for_group = assigned_df.dropna(subset=["schedule_date"]).copy()
    assigned_for_group["__city_key"] = series_location_key(assigned_for_group)
    grouped = assigned_for_group.groupby(
        [assigned_for_group["__city_key"], assigned_for_group["schedule_date"].dt.date],
        sort=False,
    )
    for (city_key, day), group in grouped:
        day_str = str(day)
        city_key = str(city_key)
        city_dist = _city_dist_slice(dist_dict, city_key)
        group_with_dist, city_dist = _ensure_dist_km(
            group.drop(columns=["__city_key"], errors="ignore"),
            dist_method=dist_method,
            city_dist=city_dist,
        )
        assigned_df.loc[group_with_dist.index, "dist_km"] = group_with_dist["dist_km"]
        moves = build_driver_movements(
            labors_df=group_with_dist,
            directory_df=directorio_df,
            day_str=day_str,
            dist_method=dist_method,
            dist_dict=city_dist,
            ALFRED_SPEED=alfred_speed,
            city_key=city_key,
        )
        if isinstance(moves, pd.DataFrame) and not moves.empty:
            moves_parts.append(moves)

    moves_df = pd.concat(moves_parts, axis=0, copy=False) if moves_parts else pd.DataFrame()
    return _finalize_preassigned_df(assigned_df), unassigned_df, moves_df


def reconstruct_preassigned_state(
    labors_df: pd.DataFrame,
    *,
    directorio_df: pd.DataFrame,
    duraciones_df: pd.DataFrame,
    dist_method: str,
    dist_dict: Dict[str, Dict],
    model_params: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if labors_df is None or labors_df.empty:
        metrics = {
            "preassigned_total": 0,
            "preassigned_failed": 0,
            "preassigned_overtime": 0,
            "preassigned_infeasible": 0,
            "preassigned_warnings": 0,
        }
        _log_preassigned_reconstruction_summary(metrics)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metrics

    df = labors_df.copy()
    _require_location_columns(df)
    normalize_datetime_columns_to_colombia(
        df,
        [
            "schedule_date",
            "payload_labor_schedule_date",
            "created_at",
        ],
    )

    if "map_start_point" not in df.columns and "start_address_point" in df.columns:
        df["map_start_point"] = df["start_address_point"]
    if "map_end_point" not in df.columns and "end_address_point" in df.columns:
        df["map_end_point"] = df["end_address_point"]

    assigned_df, unassigned_df = _split_service_level_preassigned(df)

    if assigned_df.empty:
        metrics = {
            "preassigned_total": 0,
            "preassigned_failed": 0,
            "preassigned_overtime": 0,
            "preassigned_infeasible": 0,
            "preassigned_warnings": 0,
        }
        _log_preassigned_reconstruction_summary(metrics)
        return _finalize_preassigned_df(assigned_df), unassigned_df, pd.DataFrame(), metrics

    # Keep these columns tz-aware so assigning Colombia-localized timestamps
    # does not trigger pandas incompatible-dtype warnings.
    assigned_df["actual_start"] = pd.Series(
        pd.NaT,
        index=assigned_df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    assigned_df["actual_end"] = pd.Series(
        pd.NaT,
        index=assigned_df.index,
        dtype=_BOGOTA_TZ_DTYPE,
    )
    assigned_df["actual_status"] = "COMPLETED"
    assigned_df["preassigned_failed"] = False
    assigned_df["preassigned_failure_reason"] = None
    assigned_df["preassigned_overtime"] = False
    assigned_df["preassigned_fallback_applied"] = False
    assigned_df["preassigned_reconstruction_note"] = None
    if "dist_km" not in assigned_df.columns:
        assigned_df["dist_km"] = 0.0
    else:
        assigned_df["dist_km"] = pd.to_numeric(assigned_df["dist_km"], errors="coerce")
    assigned_df = _initialize_preassigned_diagnostics(assigned_df)

    from src.optimization.algorithms.offline.offline_algorithms import assign_task_to_driver, init_drivers
    from src.optimization.common.movements import build_driver_movements
    from src.optimization.common.distance_utils import distance
    from src.optimization.common.utils import compute_workday_end

    tiempo_gracia = model_params.tiempo_gracia_min
    tiempo_alistar = model_params.tiempo_alistar_min
    tiempo_finalizacion = model_params.tiempo_finalizacion_min
    tiempo_other = model_params.tiempo_other_min
    alfred_speed = model_params.alfred_speed_kmh
    vehicle_transport_speed = model_params.vehicle_transport_speed_kmh
    workday_end_str = model_params.workday_end_str

    moves_parts = []

    assigned_for_group = assigned_df.dropna(subset=["schedule_date"]).copy()
    assigned_for_group["__city_key"] = series_location_key(assigned_for_group)
    grouped = assigned_for_group.groupby(
        [assigned_for_group["__city_key"], assigned_for_group["schedule_date"].dt.date],
        sort=False,
    )

    for (city_key, day), group in grouped:
        if group.empty:
            continue
        day_str = str(day)
        city_key = str(city_key)
        city_dist = _city_dist_slice(dist_dict, city_key)

        drivers = init_drivers(
            labors_df=group,
            directorio_df=directorio_df,
            city=city_key,
        )

        service_end_times: Dict[Any, Any] = {}
        service_fallback_handled: set[Any] = set()
        workday_end_dt = compute_workday_end(
            day_str=day_str,
            workday_end_str=workday_end_str,
            tzinfo="America/Bogota",
        )

        group_sorted = group.sort_values(
            ["schedule_date", "service_id", "labor_sequence"],
            kind="stable",
        )

        for idx, row in group_sorted.iterrows():
            service_id = row.get("service_id")
            if service_id in service_fallback_handled:
                continue

            if service_end_times.get(service_id) is pd.NaT:
                assigned_df.loc[idx, "actual_status"] = "FAILED"
                assigned_df.loc[idx, "preassigned_failed"] = True
                assigned_df.loc[idx, "preassigned_failure_reason"] = "blocked_after_infeasible"
                _mark_infeasible(
                    assigned_df,
                    idx,
                    cause_code="blocked_after_infeasible",
                    detail="Service blocked because a previous labor was infeasible.",
                )
                continue

            prev_end = service_end_times.get(service_id)
            sched = _effective_payload_schedule(row)
            assigned_df.loc[idx, "payload_schedule_reference"] = sched
            if pd.notna(prev_end):
                assigned_df.loc[idx, "previous_labor_end"] = prev_end

            if row.get("labor_category") == "VEHICLE_TRANSPORTATION":
                drv = str(row.get("assigned_driver", "")).strip()
                if not drv or drv not in drivers:
                    assigned_df.loc[idx, "actual_status"] = "FAILED"
                    assigned_df.loc[idx, "preassigned_failed"] = True
                    assigned_df.loc[idx, "preassigned_failure_reason"] = "driver_not_found"
                    _mark_infeasible(
                        assigned_df,
                        idx,
                        cause_code="driver_not_found",
                        detail=f"Assigned driver '{row.get('assigned_driver')}' is unavailable in directory.",
                    )
                    service_end_times[service_id] = pd.NaT
                    continue

                driver_state = drivers[drv]
                av = driver_state["available"]
                if av.time() < driver_state["work_start"]:
                    av = pd.Timestamp(av.date()).tz_localize(av.tz) + pd.Timedelta(
                        hours=driver_state["work_start"].hour,
                        minutes=driver_state["work_start"].minute,
                        seconds=driver_state["work_start"].second,
                    )
                    driver_state["available"] = av
                assigned_df.loc[idx, "driver_available_at"] = av

                dist_km, city_dist = distance(
                    driver_state["position"],
                    row["map_start_point"],
                    method=dist_method,
                    dist_dict=city_dist,
                )
                travel_min = 0 if pd.isna(dist_km) else (dist_km / alfred_speed * 60)
                arrival = av + timedelta(minutes=travel_min)
                assigned_df.loc[idx, "computed_arrival"] = arrival
                assigned_df.loc[idx, "computed_travel_min"] = float(travel_min)

                delay_vs_payload = _delay_minutes(arrival, sched)
                if delay_vs_payload is not None:
                    assigned_df.loc[idx, "minutes_late_payload"] = delay_vs_payload

                is_first_labor = _is_first_labor_sequence(row.get("labor_sequence"))
                if is_first_labor and pd.notna(sched):
                    feasible_window_end = sched + timedelta(minutes=tiempo_gracia)
                    assigned_df.loc[idx, "feasible_window_end"] = feasible_window_end
                    minutes_after_window = _delay_minutes(arrival, feasible_window_end)
                    if minutes_after_window is not None and minutes_after_window > 0:
                        assigned_df.loc[idx, "minutes_after_window"] = minutes_after_window
                        _mark_infeasible(
                            assigned_df,
                            idx,
                            cause_code="first_labor_arrival_after_window",
                            detail=(
                                f"Computed arrival {arrival} is {minutes_after_window:.1f} min "
                                f"after feasible window end {feasible_window_end}."
                            ),
                        )

                        service_rows = group_sorted[group_sorted["service_id"] == service_id]
                        fallback_ok, fallback_end, city_dist, fallback_error = _apply_payload_schedule_fallback(
                            assigned_df=assigned_df,
                            service_rows=service_rows,
                            payload_schedule_col="payload_labor_schedule_date",
                            city_key=city_key,
                            workday_end_dt=workday_end_dt,
                            duraciones_df=duraciones_df,
                            dist_method=dist_method,
                            city_dist=city_dist,
                            tiempo_alistar=tiempo_alistar,
                            tiempo_other=tiempo_other,
                            vehicle_transport_speed=vehicle_transport_speed,
                            distance_fn=distance,
                        )
                        if fallback_ok:
                            service_end_times[service_id] = fallback_end
                            service_fallback_handled.add(service_id)
                            continue

                        assigned_df.loc[service_rows.index, "actual_status"] = "FAILED"
                        assigned_df.loc[service_rows.index, "preassigned_failed"] = True
                        assigned_df.loc[service_rows.index, "preassigned_failure_reason"] = fallback_error
                        assigned_df.loc[service_rows.index, "preassigned_reconstruction_note"] = (
                            "payload_labor_schedule_fallback_failed"
                        )
                        _mark_infeasible(
                            assigned_df,
                            service_rows.index,
                            cause_code="payload_schedule_fallback_failed",
                            detail=f"Payload schedule fallback failed with error: {fallback_error}.",
                        )
                        service_end_times[service_id] = pd.NaT
                        service_fallback_handled.add(service_id)
                        continue

                early = sched
                if pd.notna(prev_end) and (pd.isna(early) or prev_end > early):
                    early = prev_end
                if pd.isna(early):
                    early = arrival

                # assign_task_to_driver uses inverted semantics: is_last_in_service=True
                # means "has subsequent labors" (i.e. NOT last), consistent with how
                # OFFLINE computes it via `not [...].empty`.  Pass True when there ARE
                # subsequent labors so tiempo_finalizacion is added only to the last VT.
                is_last = not group_sorted[
                    (group_sorted["service_id"] == service_id)
                    & (group_sorted["labor_sequence"] > row["labor_sequence"])
                ].empty

                astart, aend, labor_dist_km = assign_task_to_driver(
                    driver_state,
                    arrival,
                    early,
                    row["map_start_point"],
                    row["map_end_point"],
                    is_last,
                    tiempo_alistar,
                    tiempo_finalizacion,
                    vehicle_transport_speed,
                    dist_method,
                    city_dist,
                )

                assigned_df.loc[idx, "actual_start"] = astart
                assigned_df.loc[idx, "actual_end"] = aend
                assigned_df.loc[idx, "dist_km"] = 0.0 if pd.isna(labor_dist_km) else float(labor_dist_km)
                assigned_df.loc[idx, "preassigned_overtime"] = aend > workday_end_dt
                _warn_if_intermediate_delay(
                    assigned_df=assigned_df,
                    idx=idx,
                    labor_sequence=row.get("labor_sequence"),
                    scheduled_ts=sched,
                    actual_start_ts=astart,
                )
                service_end_times[service_id] = aend
            else:
                arrival = prev_end if pd.notna(prev_end) else sched
                assigned_df.loc[idx, "computed_arrival"] = arrival
                delay_vs_payload = _delay_minutes(arrival, sched)
                if delay_vs_payload is not None:
                    assigned_df.loc[idx, "minutes_late_payload"] = delay_vs_payload
                if pd.notna(prev_end) and (pd.isna(sched) or prev_end > sched):
                    astart = prev_end
                else:
                    astart = sched

                est_time = row.get("estimated_time")
                if est_time is not None and pd.notna(est_time):
                    duration_min = float(est_time)
                else:
                    duration_min = _nontransport_duration(
                        duraciones_df=duraciones_df,
                        city=city_key,
                        labor_type=row.get("labor_type"),
                        fallback_min=tiempo_other,
                    )
                    _mark_warning(
                        assigned_df, idx,
                        warning_code="estimated_time_fallback",
                        detail=(
                            f"estimated_time missing for labor_type={row.get('labor_type')!r}; "
                            f"used duraciones fallback ({duration_min:.1f} min)."
                        ),
                    )

                aend = astart + timedelta(minutes=duration_min)
                assigned_df.loc[idx, "actual_start"] = astart
                assigned_df.loc[idx, "actual_end"] = aend
                assigned_df.loc[idx, "preassigned_overtime"] = aend > workday_end_dt
                if pd.isna(assigned_df.loc[idx, "dist_km"]):
                    assigned_df.loc[idx, "dist_km"] = 0.0
                _warn_if_intermediate_delay(
                    assigned_df=assigned_df,
                    idx=idx,
                    labor_sequence=row.get("labor_sequence"),
                    scheduled_ts=sched,
                    actual_start_ts=astart,
                )
                service_end_times[service_id] = aend

        assigned_df["dist_km"] = pd.to_numeric(assigned_df["dist_km"], errors="coerce").fillna(0.0)
        moves = build_driver_movements(
            labors_df=assigned_df.loc[group_sorted.index],
            directory_df=directorio_df,
            day_str=day_str,
            dist_method=dist_method,
            dist_dict=city_dist,
            ALFRED_SPEED=alfred_speed,
            city_key=city_key,
        )
        if isinstance(moves, pd.DataFrame) and not moves.empty:
            moves_parts.append(moves)

    moves_df = pd.concat(moves_parts, axis=0, copy=False) if moves_parts else pd.DataFrame()

    failed_count = int(assigned_df["preassigned_failed"].sum())
    overtime_count = int(assigned_df["preassigned_overtime"].sum())
    infeasible_count = int(assigned_df["is_infeasible"].sum())
    warning_count = int(assigned_df["is_warning"].sum())
    fallback_service_count = int(
        assigned_df.loc[assigned_df["preassigned_fallback_applied"], "service_id"].nunique()
    )
    metrics = {
        "preassigned_total": int(len(assigned_df)),
        "preassigned_failed": failed_count,
        "preassigned_overtime": overtime_count,
        "preassigned_infeasible": infeasible_count,
        "preassigned_warnings": warning_count,
        "preassigned_fallback_services": fallback_service_count,
        "preassigned_failure_reasons": assigned_df["preassigned_failure_reason"]
        .dropna()
        .value_counts()
        .to_dict(),
        "preassigned_infeasibility_causes": assigned_df["infeasibility_cause_code"]
        .dropna()
        .value_counts()
        .to_dict(),
        "preassigned_warning_causes": assigned_df["warning_code"]
        .dropna()
        .value_counts()
        .to_dict(),
    }

    _log_preassigned_reconstruction_summary(metrics)
    return _finalize_preassigned_df(assigned_df), unassigned_df, moves_df, metrics


def _prepare_service_rows_for_reassignment(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rows = df.copy()
    if "original_assigned_driver" not in rows.columns:
        rows["original_assigned_driver"] = rows.get("assigned_driver")
    else:
        missing_original = rows["original_assigned_driver"].isna()
        rows.loc[missing_original, "original_assigned_driver"] = rows.loc[missing_original, "assigned_driver"]

    rows["preassignment_infeasible_detected"] = rows.get("is_infeasible", False).fillna(False)
    rows["preassignment_infeasibility_cause_code"] = rows.get("infeasibility_cause_code")
    rows["preassignment_infeasibility_cause_detail"] = rows.get("infeasibility_cause_detail")
    rows["preassignment_warning_detected"] = rows.get("is_warning", False).fillna(False)
    rows["preassignment_warning_code"] = rows.get("warning_code")
    rows["preassignment_warning_detail"] = rows.get("warning_detail")

    rows["reassignment_candidate"] = True
    rows["reassignment_priority"] = 1
    rows["assigned_driver"] = pd.NA
    rows["actual_start"] = pd.NaT
    rows["actual_end"] = pd.NaT
    rows["actual_status"] = pd.NA
    rows["is_infeasible"] = False
    rows["infeasibility_cause_code"] = None
    rows["infeasibility_cause_detail"] = None
    rows["is_warning"] = False
    rows["warning_code"] = None
    rows["warning_detail"] = None
    return rows


def _nontransport_duration(
    *,
    duraciones_df: pd.DataFrame,
    city: str,
    labor_type: Any,
    fallback_min: float,
) -> float:
    if duraciones_df is None or duraciones_df.empty:
        return float(fallback_min)

    if labor_type is None:
        return float(fallback_min)

    dur_row = duraciones_df[
        (duraciones_df["city"] == city) &
        (duraciones_df["labor_type"] == labor_type)
    ]
    if not dur_row.empty:
        return float(dur_row["p75_min"].iloc[0])

    labor_rows = duraciones_df[duraciones_df["labor_type"] == labor_type]
    if not labor_rows.empty:
        return float(labor_rows["p75_min"].mean())

    return float(fallback_min)
