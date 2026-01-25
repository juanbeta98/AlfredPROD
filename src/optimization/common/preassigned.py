import logging
from datetime import timedelta
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


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
    if "map_start_point" not in df.columns and "start_address_point" in df.columns:
        df["map_start_point"] = df["start_address_point"]
    if "map_end_point" not in df.columns and "end_address_point" in df.columns:
        df["map_end_point"] = df["end_address_point"]

    assigned_driver = df.get("assigned_driver")
    if assigned_driver is None:
        assigned_mask = pd.Series(False, index=df.index)
    else:
        assigned_mask = assigned_driver.notna() & assigned_driver.astype(str).str.strip().ne("")

    assigned_df = df.loc[assigned_mask].copy()
    unassigned_df = df.loc[~assigned_mask].copy()

    if assigned_df.empty:
        return assigned_df, unassigned_df, pd.DataFrame()

    if "actual_start" in assigned_df.columns and assigned_df["actual_start"].isna().any():
        logger.warning("preassigned_missing_actual_start rows=%s", int(assigned_df["actual_start"].isna().sum()))
    if "actual_end" in assigned_df.columns and assigned_df["actual_end"].isna().any():
        logger.warning("preassigned_missing_actual_end rows=%s", int(assigned_df["actual_end"].isna().sum()))

    if "schedule_date" not in assigned_df.columns:
        return assigned_df, unassigned_df, pd.DataFrame()

    from src.optimization.algorithms.offline.offline_algorithms import build_driver_movements

    moves_parts = []
    grouped = assigned_df.dropna(subset=["schedule_date"]).groupby(
        [assigned_df["city"], assigned_df["schedule_date"].dt.date],
        sort=False,
    )
    for (city, day), group in grouped:
        day_str = str(day)
        city_key = str(city)
        city_dist = dist_dict.get(city_key, dist_dict.get(city, {})) if isinstance(dist_dict, dict) else {}
        moves = build_driver_movements(
            labors_df=group,
            directory_df=directorio_df,
            day_str=day_str,
            dist_method=dist_method,
            dist_dict=city_dist,
            ALFRED_SPEED=alfred_speed,
            city_name=city_key,
        )
        if isinstance(moves, pd.DataFrame) and not moves.empty:
            moves_parts.append(moves)

    moves_df = pd.concat(moves_parts, axis=0, copy=False) if moves_parts else pd.DataFrame()
    return assigned_df, unassigned_df, moves_df


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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "preassigned_total": 0,
            "preassigned_failed": 0,
            "preassigned_overtime": 0,
        }

    df = labors_df.copy()
    if "schedule_date" in df.columns:
        df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce", utc=True)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    if "map_start_point" not in df.columns and "start_address_point" in df.columns:
        df["map_start_point"] = df["start_address_point"]
    if "map_end_point" not in df.columns and "end_address_point" in df.columns:
        df["map_end_point"] = df["end_address_point"]

    assigned_driver = df.get("assigned_driver")
    if assigned_driver is None:
        assigned_mask = pd.Series(False, index=df.index)
    else:
        assigned_mask = assigned_driver.notna() & assigned_driver.astype(str).str.strip().ne("")

    assigned_df = df.loc[assigned_mask].copy()
    unassigned_df = df.loc[~assigned_mask].copy()

    if assigned_df.empty:
        return assigned_df, unassigned_df, pd.DataFrame(), {
            "preassigned_total": 0,
            "preassigned_failed": 0,
            "preassigned_overtime": 0,
        }

    assigned_df["actual_start"] = pd.NaT
    assigned_df["actual_end"] = pd.NaT
    assigned_df["actual_status"] = "COMPLETED"
    assigned_df["preassigned_failed"] = False
    assigned_df["preassigned_failure_reason"] = None
    assigned_df["preassigned_overtime"] = False

    from src.optimization.algorithms.offline.offline_algorithms import assign_task_to_driver, init_drivers
    from src.optimization.algorithms.offline.offline_algorithms import build_driver_movements
    from src.optimization.common.distance_utils import distance
    from src.optimization.common.utils import compute_workday_end

    tiempo_previo = model_params.tiempo_previo_min
    tiempo_gracia = model_params.tiempo_gracia_min
    tiempo_alistar = model_params.tiempo_alistar_min
    tiempo_finalizacion = model_params.tiempo_finalizacion_min
    tiempo_other = model_params.tiempo_other_min
    alfred_speed = model_params.alfred_speed_kmh
    vehicle_transport_speed = model_params.vehicle_transport_speed_kmh
    workday_end_str = model_params.workday_end_str

    moves_parts = []

    grouped = assigned_df.dropna(subset=["schedule_date"]).groupby(
        [assigned_df["city"], assigned_df["schedule_date"].dt.date],
        sort=False,
    )

    for (city, day), group in grouped:
        if group.empty:
            continue
        day_str = str(day)
        city_key = str(city)
        city_dist = dist_dict.get(city_key, dist_dict.get(city, {})) if isinstance(dist_dict, dict) else {}

        drivers = init_drivers(
            labors_df=group,
            directorio_df=directorio_df,
            city=city_key,
        )

        service_end_times: Dict[Any, Any] = {}
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
            if service_end_times.get(service_id) is pd.NaT:
                assigned_df.loc[idx, "actual_status"] = "FAILED"
                assigned_df.loc[idx, "preassigned_failed"] = True
                assigned_df.loc[idx, "preassigned_failure_reason"] = "blocked_after_infeasible"
                continue

            prev_end = service_end_times.get(service_id)

            if row.get("labor_category") == "VEHICLE_TRANSPORTATION":
                drv = str(row.get("assigned_driver", "")).strip()
                if not drv or drv not in drivers:
                    assigned_df.loc[idx, "actual_status"] = "FAILED"
                    assigned_df.loc[idx, "preassigned_failed"] = True
                    assigned_df.loc[idx, "preassigned_failure_reason"] = "driver_not_found"
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

                dist_km, city_dist = distance(
                    driver_state["position"],
                    row["map_start_point"],
                    method=dist_method,
                    dist_dict=city_dist,
                )
                travel_min = 0 if pd.isna(dist_km) else (dist_km / alfred_speed * 60)
                arrival = av + timedelta(minutes=travel_min)

                sched = row["schedule_date"]
                early = prev_end or (sched - timedelta(minutes=tiempo_previo))
                late = (prev_end + timedelta(minutes=tiempo_gracia)) if prev_end else (sched + timedelta(minutes=tiempo_gracia))

                if arrival > late:
                    assigned_df.loc[idx, "actual_status"] = "FAILED"
                    assigned_df.loc[idx, "preassigned_failed"] = True
                    assigned_df.loc[idx, "preassigned_failure_reason"] = "missed_schedule"
                    service_end_times[service_id] = pd.NaT
                    continue

                is_last = group_sorted[
                    (group_sorted["service_id"] == service_id)
                    & (group_sorted["labor_sequence"] > row["labor_sequence"])
                ].empty

                astart, aend, _ = assign_task_to_driver(
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
                assigned_df.loc[idx, "preassigned_overtime"] = aend > workday_end_dt
                service_end_times[service_id] = aend
            else:
                sched = row["schedule_date"]
                astart = prev_end or sched

                duration_min = _nontransport_duration(
                    duraciones_df=duraciones_df,
                    city=city_key,
                    labor_type=row.get("labor_type"),
                    fallback_min=tiempo_other,
                )

                aend = astart + timedelta(minutes=duration_min)
                assigned_df.loc[idx, "actual_start"] = astart
                assigned_df.loc[idx, "actual_end"] = aend
                assigned_df.loc[idx, "preassigned_overtime"] = aend > workday_end_dt
                service_end_times[service_id] = aend

        moves = build_driver_movements(
            labors_df=assigned_df.loc[group_sorted.index],
            directory_df=directorio_df,
            day_str=day_str,
            dist_method=dist_method,
            dist_dict=city_dist,
            ALFRED_SPEED=alfred_speed,
            city_name=city_key,
        )
        if isinstance(moves, pd.DataFrame) and not moves.empty:
            moves_parts.append(moves)

    moves_df = pd.concat(moves_parts, axis=0, copy=False) if moves_parts else pd.DataFrame()

    failed_count = int(assigned_df["preassigned_failed"].sum())
    overtime_count = int(assigned_df["preassigned_overtime"].sum())
    metrics = {
        "preassigned_total": int(len(assigned_df)),
        "preassigned_failed": failed_count,
        "preassigned_overtime": overtime_count,
        "preassigned_failure_reasons": assigned_df["preassigned_failure_reason"]
        .dropna()
        .value_counts()
        .to_dict(),
    }

    return assigned_df, unassigned_df, moves_df, metrics


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
