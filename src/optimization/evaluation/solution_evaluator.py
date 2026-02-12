from __future__ import annotations

import logging
from datetime import time, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.data.id_normalization import normalize_id_value

logger = logging.getLogger(__name__)

VEHICLE_TRANSPORTATION = "VEHICLE_TRANSPORTATION"
DRIVER_MOVE = "DRIVER_MOVE"
FREE_TIME = "FREE_TIME"
KPI_DECIMALS = 2


def evaluate_solution(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    *,
    driver_directory_df: Optional[pd.DataFrame] = None,
    grace_minutes: int = 15,
    default_shift_end: str = "19:00:00",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich labor-level KPI columns and build a robust local evaluation report.

    Payload-facing KPIs are labor-level and remain in `labors_df`:
      - labor_distance_km
      - driver_move_distance_km
    """
    if not isinstance(labors_df, pd.DataFrame):
        raise TypeError("labors_df must be a pandas DataFrame")

    if labors_df.empty:
        report = _empty_report()
        report["skipped"] = True
        report["reason"] = "labors_empty"
        return labors_df.copy(), report

    enriched = _enrich_labor_kpis(labors_df=labors_df, moves_df=moves_df)
    report = _build_evaluation_report(
        enriched_df=enriched,
        moves_df=moves_df,
        driver_directory_df=driver_directory_df,
        grace_minutes=grace_minutes,
        default_shift_end=default_shift_end,
    )

    logger.info(
        "solution_evaluation_completed services=%s labors=%s drivers=%s vt_labors=%s labor_distance_km=%s driver_move_distance_km=%s util_wo_moves=%s util_with_moves=%s",
        report["summary"]["services_total"],
        report["summary"]["labors_total"],
        report["summary"]["drivers_used"],
        report["summary"]["vt_labors_total"],
        report["summary"]["total_labor_distance_km"],
        report["summary"]["total_driver_move_distance_km"],
        report["summary"]["utilization_without_moves_pct"],
        report["summary"]["utilization_with_moves_pct"],
    )

    return enriched, report


def _enrich_labor_kpis(*, labors_df: pd.DataFrame, moves_df: pd.DataFrame) -> pd.DataFrame:
    df = labors_df.copy()

    labor_distance = _first_numeric_series(
        df,
        columns=("labor_distance_km", "dist_km", "distance_km", "labor_distance"),
    )
    df["labor_distance_km"] = labor_distance.fillna(0.0).round(KPI_DECIMALS)

    driver_move_by_labor = _driver_move_distance_by_labor(moves_df)
    labor_key = _id_key_series(df.get("labor_id"), index=df.index)
    df["driver_move_distance_km"] = labor_key.map(driver_move_by_labor).fillna(0.0)
    df["driver_move_distance_km"] = df["driver_move_distance_km"].astype(float).round(KPI_DECIMALS)

    return df


def _build_evaluation_report(
    *,
    enriched_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    driver_directory_df: Optional[pd.DataFrame],
    grace_minutes: int,
    default_shift_end: str,
) -> Dict[str, Any]:
    assignment = _assignment_metrics(enriched_df)
    distance = _distance_metrics(enriched_df)
    time_allocation = _time_allocation_metrics(moves_df)
    punctuality = _punctuality_metrics(enriched_df, grace_minutes=grace_minutes)
    utilization = _utilization_metrics(
        moves_df=moves_df,
        driver_directory_df=driver_directory_df,
        default_shift_end=default_shift_end,
    )
    quality_checks = _quality_checks(
        enriched_df=enriched_df,
        moves_df=moves_df,
        driver_directory_df=driver_directory_df,
    )

    # Backward-compatible summary keys + extended utilization signals.
    summary = {
        "services_total": assignment["services_total"],
        "labors_total": assignment["labors_total"],
        "drivers_used": assignment["drivers_used"],
        "services_with_vt_labors": assignment["services_with_vt_labors"],
        "services_successfully_assigned": assignment["services_successfully_assigned"],
        "services_unassigned_or_failed": assignment["services_unassigned_or_failed"],
        "failed_services_total": assignment["failed_services_total"],
        "vt_labors_total": assignment["vt_labors_total"],
        "vt_labors_assigned": assignment["vt_labors_assigned"],
        "vt_labors_unassigned": assignment["vt_labors_unassigned"],
        "failed_labors_total": assignment["failed_labors_total"],
        "unassigned_labors_total": assignment["unassigned_labors_total"],
        "total_labor_distance_km": distance["total_labor_distance_km"],
        "total_driver_move_distance_km": distance["total_driver_move_distance_km"],
        "service_assignment_rate_pct": assignment["service_assignment_rate_pct"],
        "vt_assignment_rate_pct": assignment["vt_assignment_rate_pct"],
        "utilization_without_moves_pct": utilization["system"]["utilization_without_moves_pct"],
        "utilization_with_moves_pct": utilization["system"]["utilization_with_moves_pct"],
        "driver_move_utilization_pct": utilization["system"]["driver_move_utilization_pct"],
    }

    return {
        "summary": summary,
        "assignment": assignment,
        "distance": distance,
        "time_allocation": time_allocation,
        "punctuality": punctuality,
        "utilization": utilization,
        "quality_checks": quality_checks,
    }


def _empty_report() -> Dict[str, Any]:
    return {
        "summary": {
            "services_total": 0,
            "labors_total": 0,
            "drivers_used": 0,
            "services_with_vt_labors": 0,
            "services_successfully_assigned": 0,
            "services_unassigned_or_failed": 0,
            "failed_services_total": 0,
            "vt_labors_total": 0,
            "vt_labors_assigned": 0,
            "vt_labors_unassigned": 0,
            "failed_labors_total": 0,
            "unassigned_labors_total": 0,
            "total_labor_distance_km": 0.0,
            "total_driver_move_distance_km": 0.0,
            "service_assignment_rate_pct": 0.0,
            "vt_assignment_rate_pct": 0.0,
            "utilization_without_moves_pct": 0.0,
            "utilization_with_moves_pct": 0.0,
            "driver_move_utilization_pct": 0.0,
        },
        "assignment": {},
        "distance": {},
        "time_allocation": {},
        "punctuality": {},
        "utilization": {},
        "quality_checks": {},
    }


def _assignment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    service_key = _id_key_series(df.get("service_id"), index=df.index)
    driver_series = _clean_text_series(df.get("assigned_driver"), index=df.index)
    status_series = _clean_text_series(df.get("actual_status"), index=df.index).str.upper()

    labor_category = _clean_text_series(df.get("labor_category"), index=df.index).str.upper()
    is_vt = labor_category.eq(VEHICLE_TRANSPORTATION)
    is_assigned = driver_series.notna()

    vt_df = df.loc[is_vt].copy()
    vt_df["__service_key"] = _id_key_series(vt_df.get("service_id"), index=vt_df.index)
    vt_df["__is_assigned"] = _clean_text_series(vt_df.get("assigned_driver"), index=vt_df.index).notna()
    vt_df["__status"] = _clean_text_series(vt_df.get("actual_status"), index=vt_df.index).str.upper()

    services_with_vt = int(vt_df["__service_key"].nunique()) if not vt_df.empty else 0
    services_success = 0
    failed_services_total = 0
    if not vt_df.empty:
        by_service = vt_df.groupby("__service_key", dropna=True).agg(
            assigned_sum=("__is_assigned", "sum"),
            assigned_count=("__is_assigned", "count"),
            failed_rows=("__status", lambda s: int((s == "FAILED").sum())),
        )
        services_success = int((by_service["assigned_sum"] == by_service["assigned_count"]).sum())
        failed_services_total = int((by_service["failed_rows"] > 0).sum())

    vt_assigned = int((is_vt & is_assigned).sum())
    vt_total = int(is_vt.sum())

    services_total = int(service_key.nunique(dropna=True))
    services_unassigned_or_failed = max(services_with_vt - services_success, 0)
    failed_labors_total = int(status_series.eq("FAILED").sum())
    unassigned_labors_total = int((~is_assigned).sum())

    return {
        "services_total": services_total,
        "labors_total": int(len(df)),
        "drivers_used": int(driver_series.nunique(dropna=True)),
        "services_with_vt_labors": services_with_vt,
        "services_successfully_assigned": services_success,
        "services_unassigned_or_failed": services_unassigned_or_failed,
        "failed_services_total": failed_services_total,
        "vt_labors_total": vt_total,
        "vt_labors_assigned": vt_assigned,
        "vt_labors_unassigned": max(vt_total - vt_assigned, 0),
        "failed_labors_total": failed_labors_total,
        "unassigned_labors_total": unassigned_labors_total,
        "service_assignment_rate_pct": _pct(services_success, services_with_vt),
        "vt_assignment_rate_pct": _pct(vt_assigned, vt_total),
    }


def _distance_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    labor_category = _clean_text_series(df.get("labor_category"), index=df.index).str.upper()
    is_vt = labor_category.eq(VEHICLE_TRANSPORTATION)

    vt_labor_distance = pd.to_numeric(df.loc[is_vt, "labor_distance_km"], errors="coerce").fillna(0.0)
    driver_move_distance = pd.to_numeric(df["driver_move_distance_km"], errors="coerce").fillna(0.0)

    return {
        "total_labor_distance_km": _round_kpi(vt_labor_distance.sum()),
        "total_driver_move_distance_km": _round_kpi(driver_move_distance.sum()),
        "avg_vt_labor_distance_km": _round_kpi(vt_labor_distance.mean()) if len(vt_labor_distance) > 0 else 0.0,
        "avg_driver_move_distance_km_per_labor": _round_kpi(driver_move_distance.mean())
        if len(driver_move_distance) > 0
        else 0.0,
    }


def _time_allocation_metrics(moves_df: pd.DataFrame) -> Dict[str, Any]:
    moves = _prepare_moves_df(moves_df)
    if moves.empty:
        return {
            "timeline_total_min": 0.0,
            "free_time_min": 0.0,
            "driver_move_min": 0.0,
            "labor_work_min": 0.0,
            "free_time_pct": 0.0,
            "driver_move_pct": 0.0,
            "labor_work_pct": 0.0,
        }

    total = float(moves["duration_min"].sum())
    free = float(moves.loc[moves["labor_category_norm"].eq(FREE_TIME), "duration_min"].sum())
    driver_move = float(moves.loc[moves["labor_category_norm"].eq(DRIVER_MOVE), "duration_min"].sum())
    labor_work = max(total - free - driver_move, 0.0)

    return {
        "timeline_total_min": _round_kpi(total),
        "free_time_min": _round_kpi(free),
        "driver_move_min": _round_kpi(driver_move),
        "labor_work_min": _round_kpi(labor_work),
        "free_time_pct": _pct_float(free, total),
        "driver_move_pct": _pct_float(driver_move, total),
        "labor_work_pct": _pct_float(labor_work, total),
    }


def _punctuality_metrics(df: pd.DataFrame, *, grace_minutes: int) -> Dict[str, Any]:
    work = df.copy()
    if work.empty:
        return {
            "grace_minutes": int(grace_minutes),
            "services_considered": 0,
            "late_services_count": 0,
            "late_services_pct": 0.0,
            "total_lateness_min": 0.0,
            "avg_lateness_min_late_only": 0.0,
            "avg_lateness_min_all_considered": 0.0,
            "normalized_tardiness_pct": 0.0,
        }

    work["labor_category_norm"] = _clean_text_series(work.get("labor_category"), index=work.index).str.upper()
    vt = work.loc[work["labor_category_norm"].eq(VEHICLE_TRANSPORTATION)].copy()
    if vt.empty:
        return {
            "grace_minutes": int(grace_minutes),
            "services_considered": 0,
            "late_services_count": 0,
            "late_services_pct": 0.0,
            "total_lateness_min": 0.0,
            "avg_lateness_min_late_only": 0.0,
            "avg_lateness_min_all_considered": 0.0,
            "normalized_tardiness_pct": 0.0,
        }

    vt["service_key"] = _id_key_series(vt.get("service_id"), index=vt.index)
    vt["schedule_ts"] = pd.to_datetime(vt.get("schedule_date"), errors="coerce")
    vt["actual_start_ts"] = pd.to_datetime(vt.get("actual_start"), errors="coerce")
    if "labor_sequence" in vt.columns:
        vt["labor_sequence_num"] = pd.to_numeric(vt["labor_sequence"], errors="coerce")
    else:
        vt["labor_sequence_num"] = pd.NA

    vt = vt.sort_values(["service_key", "labor_sequence_num", "schedule_ts"], na_position="last")
    first_vt = vt.groupby("service_key", dropna=True, as_index=False).first()
    first_vt = first_vt.dropna(subset=["schedule_ts", "actual_start_ts"])
    services_considered = int(len(first_vt))
    if services_considered == 0:
        return {
            "grace_minutes": int(grace_minutes),
            "services_considered": 0,
            "late_services_count": 0,
            "late_services_pct": 0.0,
            "total_lateness_min": 0.0,
            "avg_lateness_min_late_only": 0.0,
            "avg_lateness_min_all_considered": 0.0,
            "normalized_tardiness_pct": 0.0,
        }

    grace_delta = pd.Timedelta(minutes=max(int(grace_minutes), 0))
    deadlines = first_vt["schedule_ts"] + grace_delta
    lateness_min = (
        (first_vt["actual_start_ts"] - deadlines).dt.total_seconds().div(60.0).clip(lower=0.0).fillna(0.0)
    )

    late_count = int((lateness_min > 0).sum())
    total_lateness = float(lateness_min.sum())
    avg_late_only = float(lateness_min[lateness_min > 0].mean()) if late_count > 0 else 0.0
    avg_all = float(lateness_min.mean()) if services_considered > 0 else 0.0

    denom = max(int(grace_minutes), 1) * services_considered
    norm_pct = (total_lateness / float(denom) * 100.0) if denom > 0 else 0.0

    return {
        "grace_minutes": int(grace_minutes),
        "services_considered": services_considered,
        "late_services_count": late_count,
        "late_services_pct": _pct(late_count, services_considered),
        "total_lateness_min": _round_kpi(total_lateness),
        "avg_lateness_min_late_only": _round_kpi(avg_late_only),
        "avg_lateness_min_all_considered": _round_kpi(avg_all),
        "normalized_tardiness_pct": _round_kpi(norm_pct),
    }


def _utilization_metrics(
    *,
    moves_df: pd.DataFrame,
    driver_directory_df: Optional[pd.DataFrame],
    default_shift_end: str,
) -> Dict[str, Any]:
    moves = _prepare_moves_df(moves_df)
    active = moves.loc[~moves["labor_category_norm"].eq(FREE_TIME)].copy()
    if active.empty:
        return {
            "system": {
                "drivers_considered": 0,
                "total_reference_window_min": 0.0,
                "total_labor_work_min": 0.0,
                "total_driver_move_min": 0.0,
                "utilization_without_moves_pct": 0.0,
                "utilization_with_moves_pct": 0.0,
                "driver_move_utilization_pct": 0.0,
                "driver_move_share_of_active_pct": 0.0,
            },
            "driver_distribution": {
                "utilization_without_moves_pct_avg": 0.0,
                "utilization_without_moves_pct_p50": 0.0,
                "utilization_without_moves_pct_p90": 0.0,
                "utilization_with_moves_pct_avg": 0.0,
                "utilization_with_moves_pct_p50": 0.0,
                "utilization_with_moves_pct_p90": 0.0,
            },
            "drivers": [],
        }

    shifts = _driver_shift_lookup(driver_directory_df, default_shift_end=default_shift_end)

    per_driver_day: list[Dict[str, Any]] = []
    active = active.sort_values(["assigned_driver_key", "actual_start_ts", "actual_end_ts"])
    for (driver_key, day), g in active.groupby(["assigned_driver_key", "work_day"], dropna=True):
        labor_min = float(g.loc[~g["labor_category_norm"].eq(DRIVER_MOVE), "duration_min"].sum())
        move_min = float(g.loc[g["labor_category_norm"].eq(DRIVER_MOVE), "duration_min"].sum())
        active_min = labor_min + move_min

        start_working = g["actual_start_ts"].min()
        end_working = g["actual_end_ts"].max()
        if pd.isna(start_working) or pd.isna(end_working):
            continue

        shift_start_ts, shift_end_ts, has_shift = _resolve_shift_window_for_day(
            driver_key=driver_key,
            day=day,
            reference_ts=start_working,
            shifts=shifts,
        )
        if shift_start_ts is None or shift_end_ts is None:
            shift_start_ts = start_working
            shift_end_ts = end_working
            has_shift = False

        window_start = min(start_working, shift_start_ts)
        window_end = max(end_working, shift_end_ts)
        denom_min = max((window_end - window_start).total_seconds() / 60.0, 0.0)
        if denom_min <= 0:
            continue

        util_without = (labor_min / denom_min) * 100.0
        util_with = (active_min / denom_min) * 100.0
        move_util = (move_min / denom_min) * 100.0
        move_share_active = (move_min / active_min * 100.0) if active_min > 0 else 0.0

        per_driver_day.append(
            {
                "driver_id": driver_key,
                "day": str(day),
                "shift_window_found": bool(has_shift),
                "reference_window_min": _round_kpi(denom_min),
                "labor_work_min": _round_kpi(labor_min),
                "driver_move_min": _round_kpi(move_min),
                "active_work_min": _round_kpi(active_min),
                "utilization_without_moves_pct": _round_kpi(util_without),
                "utilization_with_moves_pct": _round_kpi(util_with),
                "driver_move_utilization_pct": _round_kpi(move_util),
                "driver_move_share_of_active_pct": _round_kpi(move_share_active),
            }
        )

    if not per_driver_day:
        return {
            "system": {
                "drivers_considered": 0,
                "total_reference_window_min": 0.0,
                "total_labor_work_min": 0.0,
                "total_driver_move_min": 0.0,
                "utilization_without_moves_pct": 0.0,
                "utilization_with_moves_pct": 0.0,
                "driver_move_utilization_pct": 0.0,
                "driver_move_share_of_active_pct": 0.0,
            },
            "driver_distribution": {
                "utilization_without_moves_pct_avg": 0.0,
                "utilization_without_moves_pct_p50": 0.0,
                "utilization_without_moves_pct_p90": 0.0,
                "utilization_with_moves_pct_avg": 0.0,
                "utilization_with_moves_pct_p50": 0.0,
                "utilization_with_moves_pct_p90": 0.0,
            },
            "drivers": [],
        }

    util_df = pd.DataFrame(per_driver_day)
    ref_total = float(util_df["reference_window_min"].sum())
    labor_total = float(util_df["labor_work_min"].sum())
    move_total = float(util_df["driver_move_min"].sum())
    active_total = labor_total + move_total

    system = {
        "drivers_considered": int(util_df["driver_id"].nunique()),
        "driver_day_rows": int(len(util_df)),
        "total_reference_window_min": _round_kpi(ref_total),
        "total_labor_work_min": _round_kpi(labor_total),
        "total_driver_move_min": _round_kpi(move_total),
        "total_active_work_min": _round_kpi(active_total),
        "utilization_without_moves_pct": _pct_float(labor_total, ref_total),
        "utilization_with_moves_pct": _pct_float(active_total, ref_total),
        "driver_move_utilization_pct": _pct_float(move_total, ref_total),
        "driver_move_share_of_active_pct": _pct_float(move_total, active_total),
    }

    distribution = {
        "utilization_without_moves_pct_avg": _round_kpi(util_df["utilization_without_moves_pct"].mean()),
        "utilization_without_moves_pct_p50": _round_kpi(util_df["utilization_without_moves_pct"].quantile(0.50)),
        "utilization_without_moves_pct_p90": _round_kpi(util_df["utilization_without_moves_pct"].quantile(0.90)),
        "utilization_with_moves_pct_avg": _round_kpi(util_df["utilization_with_moves_pct"].mean()),
        "utilization_with_moves_pct_p50": _round_kpi(util_df["utilization_with_moves_pct"].quantile(0.50)),
        "utilization_with_moves_pct_p90": _round_kpi(util_df["utilization_with_moves_pct"].quantile(0.90)),
    }

    return {
        "system": system,
        "driver_distribution": distribution,
        "drivers": per_driver_day,
    }


def _quality_checks(
    *,
    enriched_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    driver_directory_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    moves = _prepare_moves_df(moves_df)
    labor_ids_with_move_row = set()
    if not moves.empty and "labor_id_key" in moves.columns:
        labor_ids_with_move_row = set(
            moves.loc[moves["labor_category_norm"].eq(DRIVER_MOVE), "labor_id_key"].dropna().tolist()
        )

    labor_id_key = _id_key_series(enriched_df.get("labor_id"), index=enriched_df.index)
    vt_mask = _clean_text_series(enriched_df.get("labor_category"), index=enriched_df.index).str.upper().eq(
        VEHICLE_TRANSPORTATION
    )
    vt_assigned_mask = vt_mask & _clean_text_series(enriched_df.get("assigned_driver"), index=enriched_df.index).notna()
    vt_assigned_labor_keys = set(labor_id_key.loc[vt_assigned_mask].dropna().tolist())

    move_rows = moves.loc[moves["labor_category_norm"].eq(DRIVER_MOVE)].copy()
    missing_driver_move_distance_rows = int(move_rows["distance_km_num"].isna().sum()) if not move_rows.empty else 0
    negative_duration_rows = int((moves["duration_min"] < 0).sum()) if not moves.empty else 0

    missing_move_rows_for_assigned_vt = len(vt_assigned_labor_keys - labor_ids_with_move_row)

    assigned_driver_keys = set(
        _id_key_series(
            _clean_text_series(enriched_df.get("assigned_driver"), index=enriched_df.index),
            index=enriched_df.index,
        )
        .dropna()
        .tolist()
    )
    directory_keys = set()
    if isinstance(driver_directory_df, pd.DataFrame) and not driver_directory_df.empty:
        directory_keys = set(
            _id_key_series(driver_directory_df.get("driver_id"), index=driver_directory_df.index).dropna().tolist()
        )
    missing_in_directory = len(assigned_driver_keys - directory_keys) if directory_keys else 0

    return {
        "moves_rows": int(len(moves)),
        "missing_driver_move_distance_rows": int(missing_driver_move_distance_rows),
        "negative_duration_rows": int(negative_duration_rows),
        "assigned_vt_labors_without_driver_move_row": int(missing_move_rows_for_assigned_vt),
        "assigned_drivers_missing_from_directory": int(missing_in_directory),
    }


def _driver_move_distance_by_labor(moves_df: pd.DataFrame) -> Dict[str, float]:
    moves = _prepare_moves_df(moves_df)
    if moves.empty:
        return {}

    driver_moves = moves.loc[moves["labor_category_norm"].eq(DRIVER_MOVE)].copy()
    if driver_moves.empty:
        return {}

    grouped = driver_moves.groupby("labor_id_key", dropna=True)["distance_km_num"].sum()
    return {
        labor_key: _round_kpi(value)
        for labor_key, value in grouped.items()
        if labor_key is not None
    }


def _prepare_moves_df(moves_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(moves_df, pd.DataFrame) or moves_df.empty:
        return pd.DataFrame()

    moves = moves_df.copy()
    moves["labor_category_norm"] = _normalize_move_category(moves)
    moves["distance_km_num"] = pd.to_numeric(moves.get("distance_km"), errors="coerce")
    moves["assigned_driver_key"] = _id_key_series(moves.get("assigned_driver"), index=moves.index)
    moves["labor_id_key"] = _id_key_series(moves.get("labor_id"), index=moves.index)

    moves["actual_start_ts"] = pd.to_datetime(moves.get("actual_start"), errors="coerce")
    moves["actual_end_ts"] = pd.to_datetime(moves.get("actual_end"), errors="coerce")

    if "duration_min" in moves.columns:
        duration = pd.to_numeric(moves["duration_min"], errors="coerce")
    else:
        duration = pd.Series(pd.NA, index=moves.index, dtype="float64")
    derived = (moves["actual_end_ts"] - moves["actual_start_ts"]).dt.total_seconds().div(60.0)
    moves["duration_min"] = duration.where(duration.notna(), derived)
    moves["duration_min"] = moves["duration_min"].fillna(0.0)

    valid_time = moves["actual_start_ts"].notna() & moves["actual_end_ts"].notna()
    moves = moves.loc[valid_time].copy()
    moves = moves.loc[moves["assigned_driver_key"].notna()].copy()

    moves["work_day"] = moves["actual_start_ts"].dt.date
    return moves


def _normalize_move_category(df: pd.DataFrame) -> pd.Series:
    if "labor_category" in df.columns:
        series = _clean_text_series(df["labor_category"], index=df.index)
    elif "labor_name" in df.columns:
        series = _clean_text_series(df["labor_name"], index=df.index)
    else:
        series = pd.Series(pd.NA, index=df.index, dtype="string")
    return series.str.upper()


def _driver_shift_lookup(driver_directory_df: Optional[pd.DataFrame], *, default_shift_end: str) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not isinstance(driver_directory_df, pd.DataFrame) or driver_directory_df.empty:
        return lookup

    default_end_t = _parse_hms(default_shift_end)
    for _, row in driver_directory_df.iterrows():
        driver_id = normalize_id_value(row.get("driver_id"))
        if driver_id is None:
            continue
        start_t = _parse_hms(row.get("start_time"))
        if start_t is None:
            continue
        end_t = _parse_hms(row.get("end_time")) or default_end_t or start_t
        lookup[driver_id] = {
            "start_time": start_t,
            "end_time": end_t,
        }

    return lookup


def _resolve_shift_window_for_day(
    *,
    driver_key: str,
    day: Any,
    reference_ts: pd.Timestamp,
    shifts: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
    info = shifts.get(driver_key)
    if not info:
        return None, None, False

    start_t = info.get("start_time")
    end_t = info.get("end_time")
    if start_t is None or end_t is None:
        return None, None, False

    shift_start = _combine_day_time(day=day, t=start_t, reference_ts=reference_ts)
    shift_end = _combine_day_time(day=day, t=end_t, reference_ts=reference_ts)
    if shift_start is None or shift_end is None:
        return None, None, False

    if shift_end < shift_start:
        shift_end = shift_end + pd.Timedelta(days=1)
    return shift_start, shift_end, True


def _combine_day_time(day: Any, t: time, reference_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
    try:
        day_ts = pd.Timestamp(day)
    except Exception:
        return None

    value = pd.Timestamp.combine(day_ts.date(), t)
    if reference_ts.tzinfo is not None:
        try:
            return value.tz_localize(reference_ts.tz)
        except TypeError:
            # already tz-aware in rare mixed inputs
            return value.tz_convert(reference_ts.tz)
    return value


def _parse_hms(value: Any) -> Optional[time]:
    if value is None:
        return None
    if isinstance(value, time):
        return value

    txt = str(value).strip()
    if not txt:
        return None

    parsed = pd.to_datetime(txt, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.time()


def _first_numeric_series(df: pd.DataFrame, *, columns: tuple[str, ...]) -> pd.Series:
    for col in columns:
        if col not in df.columns:
            continue
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(index=df.index, dtype="float64")


def _id_key_series(series: pd.Series | None, *, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    keys = series.map(normalize_id_value)
    return pd.Series(keys, index=series.index, dtype="string")


def _clean_text_series(series: pd.Series | None, *, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    txt = series.astype("string").str.strip()
    txt = txt.where(
        txt.notna()
        & txt.ne("")
        & ~txt.str.lower().isin({"none", "null", "nan"}),
        pd.NA,
    )
    return txt


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _round_kpi((float(numerator) / float(denominator)) * 100.0)


def _pct_float(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return _round_kpi((float(numerator) / float(denominator)) * 100.0)


def _round_kpi(value: Any) -> float:
    return round(float(value), KPI_DECIMALS)
