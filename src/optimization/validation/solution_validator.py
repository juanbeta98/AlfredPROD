from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.location import row_location_key
from src.optimization.common.distance_utils import distance
from src.optimization.settings.model_params import ModelParams
from src.optimization.settings.solver_settings import DEFAULT_DISTANCE_METHOD

logger = logging.getLogger(__name__)

DEFAULT_BLOCKING_CHECKS = frozenset(
    {
        "driver_overlap",
        "time_discontinuity",
    }
)


def _city_dist_slice(dist_dict: Optional[Dict[Any, Any]], city_key: Any) -> Dict[Any, Any]:
    if not isinstance(dist_dict, dict):
        return {}
    if city_key in dist_dict:
        value = dist_dict.get(city_key, {})
        return value if isinstance(value, dict) else {}
    city_key_txt = str(city_key)
    for key, value in dist_dict.items():
        if str(key) == city_key_txt:
            return value if isinstance(value, dict) else {}
    return {}


def validate_solution(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    *,
    model_params: ModelParams,
    dist_method: str = DEFAULT_DISTANCE_METHOD,
    dist_dict: Optional[Dict[Any, Any]] = None,
    strict_time_check: bool = False,
    blocking_checks: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Validate solution feasibility using labors + moves.

    Returns:
        report: dict summary (JSON-serializable)
        issues_df: DataFrame of individual issues
    """
    report: Dict[str, Any] = {}

    if labors_df is None or labors_df.empty:
        report.update(
            {
                "skipped": True,
                "reason": "labors_empty",
                "labors_rows": 0,
                "moves_rows": len(moves_df) if isinstance(moves_df, pd.DataFrame) else 0,
            }
        )
        return report, pd.DataFrame()

    if moves_df is None or moves_df.empty:
        report.update(
            {
                "skipped": True,
                "reason": "moves_empty",
                "labors_rows": len(labors_df),
                "moves_rows": 0,
            }
        )
        return report, pd.DataFrame()

    labors = _drop_all_city(labors_df)
    moves = _drop_all_city(moves_df)

    overlaps_df = check_driver_overlaps(labors)
    if not overlaps_df.empty:
        overlaps_df = overlaps_df.assign(check="driver_overlap")

    issues_moves_df, move_stats = validate_moves_df(
        moves,
        model_params=model_params,
        dist_method=dist_method,
        dist_dict=dist_dict,
        strict_time_check=strict_time_check,
    )

    issues_df = _combine_issues(overlaps_df, issues_moves_df)

    issues_by_check = (
        issues_df["check"].value_counts().to_dict() if "check" in issues_df.columns else {}
    )

    blocking_set = set(blocking_checks) if blocking_checks is not None else set(DEFAULT_BLOCKING_CHECKS)
    blocking_issues = {
        check: count for check, count in issues_by_check.items() if check in blocking_set and count > 0
    }

    report = {
        "summary": {
            "labors_rows": len(labors),
            "moves_rows": len(moves),
            "overlaps": int(len(overlaps_df)),
            "moves_inconsistencies": int(len(issues_moves_df)),
            "total_issues": int(len(issues_df)),
            **move_stats,
        },
        "issues_by_check": issues_by_check,
        "blocking_checks": sorted(blocking_set),
        "blocking_issues": blocking_issues,
        "blocking_failed": bool(blocking_issues),
        "strict_time_check": strict_time_check,
        "dist_method": dist_method,
    }

    return report, issues_df


def compare_labors(
    labors_base: pd.DataFrame,
    labors_other: pd.DataFrame,
    tol_min: float = 1.0,
) -> pd.DataFrame:
    """
    Compare labors between baseline and another algorithm.
    """
    for df in [labors_base, labors_other]:
        df["duration_min"] = (df["actual_end"] - df["actual_start"]).dt.total_seconds() / 60

    base = labors_base[["labor_id", "map_start_point", "map_end_point", "duration_min"]].rename(
        columns={
            "map_start_point": "base_start",
            "map_end_point": "base_end",
            "duration_min": "base_duration",
        }
    )
    other = labors_other[["labor_id", "map_start_point", "map_end_point", "duration_min"]].rename(
        columns={
            "map_start_point": "other_start",
            "map_end_point": "other_end",
            "duration_min": "other_duration",
        }
    )

    merged = base.merge(other, on="labor_id", how="outer", indicator=True)

    merged["match_found"] = merged["_merge"] == "both"
    merged["start_match"] = merged["base_start"] == merged["other_start"]
    merged["end_match"] = merged["base_end"] == merged["other_end"]
    merged["duration_diff_min"] = (merged["other_duration"] - merged["base_duration"]).abs().round(1)
    merged["duration_match"] = merged["duration_diff_min"] < tol_min

    return merged


def check_driver_overlaps(
    labors_df: pd.DataFrame,
    time_tolerance_min: int = 0,
) -> pd.DataFrame:
    """
    Check for overlapping labors assigned to the same driver on the same day.
    """
    df = labors_df.copy()
    for col in ["actual_start", "actual_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if "date" not in df.columns:
        df["date"] = df["actual_start"].dt.date

    df = df.dropna(subset=["assigned_driver", "actual_start", "actual_end"])

    overlaps: List[Dict[str, Any]] = []

    for (driver, date), g in df.groupby(["assigned_driver", "date"], group_keys=False):
        g = g.sort_values("actual_start")
        for i in range(len(g) - 1):
            curr, nxt = g.iloc[i], g.iloc[i + 1]
            gap = (nxt["actual_start"] - curr["actual_end"]).total_seconds() / 60
            if pd.isna(gap):
                continue
            if gap < time_tolerance_min:
                overlaps.append(
                    {
                        "assigned_driver": driver,
                        "date": date,
                        "labor_id_1": curr["labor_id"],
                        "start_1": curr["actual_start"],
                        "end_1": curr["actual_end"],
                        "labor_id_2": nxt["labor_id"],
                        "start_2": nxt["actual_start"],
                        "end_2": nxt["actual_end"],
                        "overlap_minutes": round(max(0.0, -gap), 1),
                    }
                )

    overlap_df = pd.DataFrame(overlaps)
    logger.info(
        "solution_validation_driver_overlap drivers=%s overlaps=%s",
        df["assigned_driver"].nunique(),
        len(overlap_df),
    )
    return overlap_df


def validate_moves_df(
    moves_df: pd.DataFrame,
    *,
    model_params: ModelParams,
    dist_method: str = DEFAULT_DISTANCE_METHOD,
    dist_dict: Optional[Dict[Any, Any]] = None,
    strict_time_check: bool = True,
    dist_tol_km: float = 1e-3,
    time_tol_min: float = 1e-2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates the consistency of move sequences for each driver per day.
    """
    df = moves_df.copy()
    for col in ["actual_start", "actual_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["date"] = df["actual_start"].dt.date
    df = df.sort_values(["assigned_driver", "date", "actual_start"]).reset_index(drop=True)

    inconsistencies: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = dict(
        drivers_checked=0,
        rows_checked=0,
        pairs_checked=0,
        distance_calls=0,
    )

    dist_dict = dist_dict or {}
    distance_cache: Dict[Tuple[str, str, str, str], float] = {}

    for (driver, date), g in df.groupby(["assigned_driver", "date"], sort=False):
        stats["drivers_checked"] += 1
        g = g.sort_values(["actual_start", "actual_end"]).reset_index(drop=True)

        for i in range(len(g)):
            row = g.loc[i]

            city_key = row_location_key(row.to_dict())
            city_dist_dict = _city_dist_slice(dist_dict, city_key)

            stats["rows_checked"] += 1
            labor_id = row.get("labor_id")
            cat = row.get("labor_category")
            start_pt, end_pt = row.get("start_point"), row.get("end_point")
            reported_dist, reported_dur = row.get("distance_km"), row.get("duration_min")
            actual_start, actual_end = row.get("actual_start"), row.get("actual_end")

            computed_dist = None
            if pd.notna(start_pt) and pd.notna(end_pt):
                cache_key = (str(city_key), str(start_pt), str(end_pt), dist_method)
                if cache_key in distance_cache:
                    computed_dist = distance_cache[cache_key]
                else:
                    try:
                        computed_dist, _ = distance(
                            start_pt,
                            end_pt,
                            method=dist_method,
                            dist_dict=city_dist_dict,
                        )
                        distance_cache[cache_key] = computed_dist
                        stats["distance_calls"] += 1
                    except Exception as exc:
                        inconsistencies.append(
                            {
                                "check": "distance_exception",
                                "assigned_driver": driver,
                                "labor_id": labor_id,
                                "date": date,
                                "message": str(exc),
                            }
                        )
                        continue

            if cat == "FREE_TIME" and pd.notna(reported_dist) and abs(reported_dist) > dist_tol_km:
                inconsistencies.append(
                    {
                        "check": "free_time_nonzero_distance",
                        "assigned_driver": driver,
                        "labor_id": labor_id,
                        "date": date,
                        "reported_dist": reported_dist,
                    }
                )

            if computed_dist is not None and pd.notna(reported_dur):
                if cat == "DRIVER_MOVE":
                    speed_kmh = model_params.alfred_speed_kmh
                    expected_dur_min = (computed_dist / speed_kmh) * 60
                elif cat == "VEHICLE_TRANSPORTATION":
                    speed_kmh = model_params.vehicle_transport_speed_kmh
                    expected_dur_min = (
                        (computed_dist / speed_kmh) * 60
                        + model_params.tiempo_alistar_min
                        + model_params.tiempo_finalizacion_min
                    )
                else:
                    speed_kmh = model_params.alfred_speed_kmh
                    expected_dur_min = (computed_dist / speed_kmh) * 60

                expected_dur_min = round(expected_dur_min, 1)
                diff_min = abs(reported_dur - expected_dur_min)

                if diff_min > time_tol_min and cat != "FREE_TIME":
                    inconsistencies.append(
                        {
                            "check": "duration_mismatch",
                            "assigned_driver": driver,
                            "labor_id": labor_id,
                            "date": date,
                            "diff_min": diff_min,
                            "reported": reported_dur,
                            "expected": expected_dur_min,
                        }
                    )

            if i > 0 and pd.notna(actual_start) and pd.notna(actual_end):
                prev = g.loc[i - 1]
                time_diff = (row["actual_start"] - prev["actual_end"]).total_seconds() / 60

                if pd.isna(time_diff):
                    continue

                if strict_time_check:
                    if abs(time_diff) > time_tol_min:
                        inconsistencies.append(
                            {
                                "check": "time_discontinuity",
                                "assigned_driver": driver,
                                "labor_id": labor_id,
                                "date": date,
                                "diff_min": time_diff,
                            }
                        )
                else:
                    if cat == "FREE_TIME" and time_diff < -time_tol_min:
                        inconsistencies.append(
                            {
                                "check": "free_time_overlap",
                                "assigned_driver": driver,
                                "labor_id": labor_id,
                                "date": date,
                                "diff_min": time_diff,
                            }
                        )

            stats["pairs_checked"] += 1

    inconsistencies_df = pd.DataFrame(inconsistencies)
    stats["inconsistencies"] = len(inconsistencies_df)

    logger.info(
        "solution_validation_moves drivers=%s rows=%s inconsistencies=%s",
        stats["drivers_checked"],
        stats["rows_checked"],
        stats["inconsistencies"],
    )

    return inconsistencies_df, stats


def _drop_all_city(df: pd.DataFrame) -> pd.DataFrame:
    if "city_code" in df.columns:
        return df[df["city_code"] != "ALL"]
    return df


def _combine_issues(*dfs: pd.DataFrame) -> pd.DataFrame:
    parts = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)
