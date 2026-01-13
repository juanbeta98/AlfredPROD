import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from src.data.distance_utils import distance
from src.config.config import (
    ALFRED_SPEED,
    VEHICLE_TRANSPORT_SPEED,
    TIEMPO_ALISTAR,
    TIEMPO_FINALIZACION,
)

def validate_all_algorithms(
    algorithms: List[Dict[str, Any]], 
    dist_method: str = "haversine",
    dist_dict: dict = {}):
    """
    Run full validation for all loaded solutions.
    """
    results = []
    all_moves_issues = []

    for algo in algorithms:
        name = algo["name"]
        labors_df = algo["labors_df"]
        moves_df = algo["moves_df"]

        labors_df = labors_df[labors_df['city'] != 'ALL']
        moves_df = moves_df[moves_df['city'] != 'ALL']

        print(f"\n🚀 Validating {name} ...")
        overlaps = check_driver_overlaps(labors_df)
        moves_issues, summary = validate_moves_df(
            moves_df, 
            dist_method=dist_method,
            dist_dict=dist_dict, 
            strict_time_check=False)

        results.append({
            "algorithm": name,
            "n_labors": len(labors_df),
            "n_moves": len(moves_df),
            "overlaps": len(overlaps),
            "move_inconsistencies": len(moves_issues),
            **summary,
        })

        all_moves_issues.append(moves_issues)

    results_df = pd.DataFrame(results)
    print("\n📊 Validation Summary:")
    # display(results_df)
    return results_df, all_moves_issues

# =============================
# 1. LABOR COMPARISON
# =============================

def compare_labors(
    labors_base: pd.DataFrame, 
    labors_other: pd.DataFrame, 
    tol_min: float = 1.0
) -> pd.DataFrame:
    """
    Compare labors between baseline and another algorithm.
    """

    for df in [labors_base, labors_other]:
        df["duration_min"] = (df["actual_end"] - df["actual_start"]).dt.total_seconds() / 60

    base = labors_base[["labor_id", "map_start_point", "map_end_point", "duration_min"]].rename(columns={
        "map_start_point": "base_start",
        "map_end_point": "base_end",
        "duration_min": "base_duration"
    })
    other = labors_other[["labor_id", "map_start_point", "map_end_point", "duration_min"]].rename(columns={
        "map_start_point": "other_start",
        "map_end_point": "other_end",
        "duration_min": "other_duration"
    })

    merged = base.merge(other, on="labor_id", how="outer", indicator=True)

    merged["match_found"] = merged["_merge"] == "both"
    merged["start_match"] = merged["base_start"] == merged["other_start"]
    merged["end_match"] = merged["base_end"] == merged["other_end"]
    merged["duration_diff_min"] = (merged["other_duration"] - merged["base_duration"]).abs().round(1)
    merged["duration_match"] = merged["duration_diff_min"] < tol_min

    print(f"🧩 Matched labors: {merged['match_found'].sum()} / {len(merged)}")
    print(f"💯 Perfect matches: {(merged['start_match'] & merged['end_match'] & merged['duration_match']).sum()}")

    return merged


# =============================
# 2. DRIVER OVERLAP CHECK
# =============================

def check_driver_overlaps(
    labors_df: pd.DataFrame, 
    time_tolerance_min: int = 0
) -> pd.DataFrame:
    """
    Check for overlapping labors assigned to the same driver on the same day.
    """

    df = labors_df.copy()
    for col in ['actual_start', 'actual_end']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'date' not in df.columns:
        df['date'] = df['actual_start'].dt.date

    overlaps = []

    for (driver, date), g in df.groupby(['assigned_driver', 'date'], group_keys=False):
        g = g.sort_values('actual_start')
        for i in range(len(g) - 1):
            curr, nxt = g.iloc[i], g.iloc[i + 1]
            gap = (nxt['actual_start'] - curr['actual_end']).total_seconds() / 60
            if gap < time_tolerance_min:
                overlaps.append({
                    "assigned_driver": driver, "date": date,
                    "labor_id_1": curr['labor_id'], "start_1": curr['actual_start'], "end_1": curr['actual_end'],
                    "labor_id_2": nxt['labor_id'], "start_2": nxt['actual_start'], "end_2": nxt['actual_end'],
                    "overlap_minutes": round(max(0, -gap), 1)
                })

    overlap_df = pd.DataFrame(overlaps)
    print(f"🧑‍🔧 Checked {df['assigned_driver'].nunique()} drivers, found {len(overlap_df)} overlaps")

    return overlap_df


# =============================
# 3. MOVES VALIDATION
# =============================

def validate_moves_df(
    moves_df: pd.DataFrame,
    dist_method: str = "haversine",
    dist_dict: dict = {},
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
    # if "date" not in df.columns:
    df["date"] = df["actual_start"].dt.date
    df = df.sort_values(["assigned_driver", "date", "actual_start"]).reset_index(drop=True)

    inconsistencies: List[Dict[str, Any]] = []
    stats = dict(drivers_checked=0, rows_checked=0, pairs_checked=0, distance_calls=0)

    for (driver, date), g in df.groupby(["assigned_driver", "date"], sort=False):
        stats["drivers_checked"] += 1
        g = g.sort_values(["actual_start", 'actual_end']).reset_index()

        for i in range(len(g)):
            row = g.loc[i]

            city = row['city']
            city_dist_dict = dist_dict.get(city, {})

            stats["rows_checked"] += 1
            labor_id = row.get('labor_id')
            cat = row.get("labor_category")
            start_pt, end_pt = row.get("start_point"), row.get("end_point")
            reported_dist, reported_dur = row.get("distance_km"), row.get("duration_min")
            actual_start, actual_end = row.get("actual_start"), row.get("actual_end")

            # Recompute distance
            computed_dist = None
            if pd.notna(start_pt) and pd.notna(end_pt):
                try:
                    computed_dist, _ = distance(
                        start_pt, 
                        end_pt, 
                        method=dist_method,
                        dist_dict=city_dist_dict)
                    stats["distance_calls"] += 1
                except Exception as e:
                    inconsistencies.append({
                        "check": "distance_exception",
                        "assigned_driver": driver, 'labor_id': labor_id,
                        "date": date, "message": str(e)
                    })
                    continue

            # FREE_TIME must have zero distance
            if cat == "FREE_TIME" and reported_dist and abs(reported_dist) > dist_tol_km:
                inconsistencies.append({
                    "check": "free_time_nonzero_distance", "assigned_driver": driver,
                    'labor_id': labor_id, "date": date, "reported_dist": reported_dist
                })

            # Duration check
            if computed_dist is not None and pd.notna(reported_dur):
                if cat == "DRIVER_MOVE":
                    speed_kmh = ALFRED_SPEED
                    expected_dur_min = (computed_dist / speed_kmh) * 60
                elif cat == "VEHICLE_TRANSPORTATION":
                    speed_kmh = VEHICLE_TRANSPORT_SPEED
                    expected_dur_min = (computed_dist / speed_kmh) * 60 + TIEMPO_ALISTAR + TIEMPO_FINALIZACION
                else:
                    speed_kmh = ALFRED_SPEED
                    expected_dur_min = (computed_dist / speed_kmh) * 60

                expected_dur_min = round(expected_dur_min, 1)
                diff_min = abs(reported_dur - expected_dur_min)

                if diff_min > time_tol_min and cat != "FREE_TIME":
                    inconsistencies.append({
                        "check": "duration_mismatch",
                        "assigned_driver": driver, 'labor_id': labor_id, "date": date,
                        "diff_min": diff_min, "reported": reported_dur, "expected": expected_dur_min
                    })

            # Continuity
            if i > 0:
                prev = g.loc[i - 1]
                time_diff = (row["actual_start"] - prev["actual_end"]).total_seconds() / 60

                if strict_time_check:
                    if abs(time_diff) > time_tol_min:
                        inconsistencies.append({
                            "check": "time_discontinuity",
                            "assigned_driver": driver, 'labor_id': labor_id, 
                            "date": date, "diff_min": time_diff
                        })
                else:
                    if cat == "FREE_TIME" and time_diff < -time_tol_min:
                        inconsistencies.append({
                            "check": "free_time_overlap",
                            "assigned_driver": driver, 'labor_id': labor_id,
                            "date": date, "diff_min": time_diff
                        })

            stats["pairs_checked"] += 1

    inconsistencies_df = pd.DataFrame(inconsistencies)
    stats["inconsistencies"] = len(inconsistencies_df)

    # print(f"✅ Validated {stats['drivers_checked']} drivers, {stats['rows_checked']} rows, {stats['inconsistencies']} inconsistencies")
    print(f"✅ Validated {stats['drivers_checked']} drivers, {stats['rows_checked']} rows, {stats['inconsistencies']} inconsistencies")

    return inconsistencies_df, stats
