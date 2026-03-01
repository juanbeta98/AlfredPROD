"""
insert_algorithms.py — Core logic for the INSERT algorithm.

Inserts new (unassigned) labors into an already-built preassigned schedule by
evaluating four cases per driver:
  1. Empty schedule   → direct insertion
  2. Before first     → insertion before the driver's first labor
  3. Between labors   → insertion between two consecutive labors
  4. After last labor → append at end

For >1 new service, the caller runs multiple iterations with randomised service
ordering (BUFFER_FIXED style) and selects the best result.

Bug fixes vs. the original INSERT_algorithm.py reference:
  - Case 3:  timedelta(TIEMPO_GRACIA) → timedelta(minutes=TIEMPO_GRACIA)
  - Case 2: end_pos=new_labor['map_start_point'] → map_end_point
  - _simulate_downstream_shift: variable shadowing 'start_pos' → 'labor_start_idx'
  - _simulate_downstream_shift: travel_time=next_dist_to_reach → next_time_to_reach
  - get_best_insertion: plan.get("alfred") → plan.get("driver_id")
  - Case 4: inconsistent dict keys standardised → dist_to_new / dist_to_next
  - commit_new_labor_insertion: end_address_point → moves-derived end_point
  - commit_new_labor_insertion: duplicate labors_df filter removed
  - get_drivers / filter_dfs_for_insertion: flexible_filter replaced with
    explicit department_code column filtering
  - Directory lookup: 'alfred' column replaced by driver_id / latitud / longitud
"""
import logging
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...common.distance_utils import distance
from ...common.movements import (
    _extract_driver_key,
    _filter_drivers_by_city,
    _location_fields,
)

logger = logging.getLogger(__name__)

DistDict = Dict[str, Any] | None


# ===========================================================================
# DRIVER DIRECTORY HELPERS
# ===========================================================================

def _get_driver_home_pos(directorio_df: pd.DataFrame, driver: str) -> Optional[str]:
    """Return the driver's home start position as a WKT POINT string, or None."""
    driver_str = str(driver).strip()
    for col in ("driver_id", "ALFRED'S"):
        if col not in directorio_df.columns:
            continue
        mask = directorio_df[col].astype(str).str.strip() == driver_str
        rows = directorio_df.loc[mask]
        if rows.empty:
            continue
        r = rows.iloc[0]
        lat, lon = r.get("latitud"), r.get("longitud")
        if pd.notna(lat) and pd.notna(lon):
            return f"POINT ({lon} {lat})"
    return None


# ===========================================================================
# TIMING / DISTANCE PRIMITIVES
# ===========================================================================

def _compute_arrival_to_next_labor(
    current_end_time,
    current_end_pos: str,
    target_pos: str,
    speed: float,
    distance_method: str,
    dist_dict: DistDict,
    **kwargs,
) -> Tuple:
    """Return (arrival_time, dist_km, travel_min)."""
    dist_km, _ = distance(
        current_end_pos, target_pos,
        method=distance_method, dist_dict=dist_dict, **kwargs,
    )
    if dist_km is None or (isinstance(dist_km, float) and math.isnan(dist_km)):
        dist_km = 0.0
    travel_min = dist_km / speed * 60
    return current_end_time + timedelta(minutes=travel_min), dist_km, travel_min


def _adjust_for_early_arrival(
    would_arrive_at,
    travel_time: float,
    schedule_date,
    early_buffer: float = 30,
) -> Tuple:
    """
    Ensure the driver does not arrive more than `early_buffer` minutes early.
    Returns (real_arrival_time, move_start_time).
    """
    earliest_allowed = schedule_date - timedelta(minutes=early_buffer)
    real_arrival = max(would_arrive_at, earliest_allowed)
    move_start = real_arrival - timedelta(minutes=travel_time)
    return real_arrival, move_start


def _compute_service_end_time(
    arrival_time,
    start_pos: str,
    end_pos: str,
    distance_method: str,
    dist_dict: DistDict,
    vehicle_speed: float,
    prep_time: float,
    finish_time: float,
    **kwargs,
) -> Tuple:
    """Return (finish_time, end_pos, total_duration_min, labor_distance_km)."""
    labor_dist, _ = distance(
        start_pos, end_pos,
        method=distance_method, dist_dict=dist_dict, **kwargs,
    )
    if labor_dist is None or (isinstance(labor_dist, float) and math.isnan(labor_dist)):
        labor_dist = 0.0
    travel_min = labor_dist / vehicle_speed * 60
    total_duration = prep_time + travel_min + finish_time
    return arrival_time + timedelta(minutes=total_duration), end_pos, total_duration, labor_dist


def _can_reach_next_labor(
    new_finish_time,
    new_finish_pos: str,
    next_start_time,
    next_start_pos: str,
    distance_method: str,
    dist_dict: DistDict,
    driver_speed: float,
    grace_time: float,
    early_buffer: float,
    **kwargs,
) -> Tuple:
    """
    Check if driver can arrive at next_start_pos on time after finishing new service.
    Returns (feasible, next_real_arrival, next_move_start, dist_km, travel_min).
    """
    dist_km, _ = distance(
        new_finish_pos, next_start_pos,
        method=distance_method, dist_dict=dist_dict, **kwargs,
    )
    if dist_km is None or (isinstance(dist_km, float) and math.isnan(dist_km)):
        dist_km = 0.0
    travel_min = dist_km / driver_speed * 60
    next_arrival = new_finish_time + timedelta(minutes=travel_min)
    next_real, next_move_start = _adjust_for_early_arrival(
        would_arrive_at=next_arrival,
        travel_time=travel_min,
        schedule_date=next_start_time,
        early_buffer=early_buffer,
    )
    feasible = next_real <= next_start_time + timedelta(minutes=grace_time)
    return feasible, next_real, next_move_start, dist_km, travel_min


def _get_driver_context(moves_driver_df: pd.DataFrame, idx: int) -> Tuple:
    """Return (curr_end_time, curr_end_pos, next_start_time, next_start_pos)."""
    curr_end_time = moves_driver_df.loc[idx, "actual_end"]
    curr_end_pos = moves_driver_df.loc[idx, "end_point"]
    next_start_time = moves_driver_df.loc[idx + 3, "schedule_date"]
    next_start_pos = moves_driver_df.loc[idx + 3, "start_point"]
    return curr_end_time, curr_end_pos, next_start_time, next_start_pos


# ===========================================================================
# MOVES BLOCK BUILDER
# ===========================================================================

def _build_moves_block(
    labor: pd.Series,
    driver: str,
    start_free_time,
    start_move_time,
    move_duration: float,
    move_distance: float,
    arrival_time,
    labor_distance: float,
    labor_duration: float,
    finish_time,
    start_pos: Optional[str],
    start_address: str,
    end_address: str,
    block_label: str = "new",
) -> pd.DataFrame:
    """
    Build the standardized triplet (FREE_TIME, DRIVER_MOVE, LABOR) for any
    labor insertion or rescheduling.
    """
    service_id = labor.get("service_id")
    labor_id = labor["labor_id"]
    schedule_date = labor["schedule_date"]
    date = labor.get("date", schedule_date.date() if hasattr(schedule_date, "date") else None)
    location = _location_fields(labor)

    # Ensure free_time does not start after move starts
    if start_free_time is None or start_free_time > start_move_time:
        start_free_time = start_move_time
    end_free_time = start_move_time

    free_start_pt = start_pos if start_pos is not None else start_address

    rows = [
        # 1) FREE_TIME
        {
            "service_id": service_id,
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_free",
            "labor_name": "FREE_TIME",
            "labor_category": "FREE_TIME",
            "assigned_driver": driver,
            "schedule_date": schedule_date,
            "actual_start": start_free_time,
            "actual_end": end_free_time,
            "start_point": free_start_pt,
            "end_point": free_start_pt,
            "distance_km": 0.0,
            "duration_min": round(
                (end_free_time - start_free_time).total_seconds() / 60.0, 1
            ),
            "date": date,
            **location,
        },
        # 2) DRIVER_MOVE
        {
            "service_id": service_id,
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_move",
            "labor_name": "DRIVER_MOVE",
            "labor_category": "DRIVER_MOVE",
            "assigned_driver": driver,
            "schedule_date": schedule_date,
            "actual_start": start_move_time,
            "actual_end": arrival_time,
            "start_point": free_start_pt,
            "end_point": start_address,
            "distance_km": move_distance,
            "duration_min": round(move_duration, 1),
            "date": date,
            **location,
        },
        # 3) LABOR
        {
            "service_id": service_id,
            "labor_id": labor_id,
            "labor_context_id": f"{labor_id}_labor",
            "labor_name": labor["labor_name"],
            "labor_category": labor["labor_category"],
            "assigned_driver": driver,
            "schedule_date": schedule_date,
            "actual_start": arrival_time,
            "actual_end": finish_time,
            "start_point": start_address,
            "end_point": end_address,
            "distance_km": labor_distance,
            "duration_min": round(labor_duration, 1),
            "date": date,
            **location,
        },
    ]

    df = pd.DataFrame(rows)
    df.attrs["block_label"] = block_label
    return df


# ===========================================================================
# DOWNSTREAM SHIFT SIMULATOR
# ===========================================================================

def _simulate_downstream_shift(
    moves_driver_df: pd.DataFrame,
    driver: str,
    start_idx: int,
    start_time,
    start_pos: str,
    previous_start_time,
    distance_method: str,
    dist_dict: DistDict,
    ALFRED_SPEED: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    TIEMPO_GRACIA: float,
    EARLY_BUFFER: float,
    **kwargs,
) -> Tuple[bool, List[pd.DataFrame]]:
    """
    Simulate how downstream labors shift after inserting a new labor.
    Returns (feasible, downstream_shift_blocks).
    """
    shift = (start_time - previous_start_time).total_seconds() / 60
    if abs(shift) < 1:
        return True, []

    curr_end_time = start_time
    curr_end_pos = start_pos

    # List of _labor row indices in chronological order
    labor_rows = (
        moves_driver_df[
            moves_driver_df["labor_context_id"].astype(str).str.endswith("_labor")
        ]
        .index.tolist()
    )

    # Locate our starting point within the list (rename to avoid shadowing start_pos)
    try:
        labor_start_idx = labor_rows.index(start_idx)
    except ValueError:
        labor_start_idx = 0

    downstream_shifts: List[pd.DataFrame] = []

    for i in labor_rows[labor_start_idx + 1:]:
        next_labor = moves_driver_df.loc[i]
        next_start_time = next_labor["schedule_date"]
        next_start_pos = next_labor["start_point"]
        next_end_pos = next_labor["end_point"]

        next_arrival, next_dist, next_travel = _compute_arrival_to_next_labor(
            current_end_time=curr_end_time,
            current_end_pos=curr_end_pos,
            target_pos=next_start_pos,
            speed=ALFRED_SPEED,
            distance_method=distance_method,
            dist_dict=dist_dict,
            **kwargs,
        )

        if next_arrival > next_start_time + timedelta(minutes=TIEMPO_GRACIA):
            return False, []

        next_real_arrival, next_move_start = _adjust_for_early_arrival(
            would_arrive_at=next_arrival,
            travel_time=next_travel,  # Fixed: was next_dist (km) in the reference
            schedule_date=next_start_time,
            early_buffer=EARLY_BUFFER,
        )

        next_finish, next_end_pos, next_duration, next_labor_dist = _compute_service_end_time(
            arrival_time=next_real_arrival,
            start_pos=next_start_pos,
            end_pos=next_end_pos,
            distance_method=distance_method,
            dist_dict=dist_dict,
            vehicle_speed=VEHICLE_TRANSPORT_SPEED,
            prep_time=TIEMPO_ALISTAR,
            finish_time=TIEMPO_FINALIZACION,
            **kwargs,
        )

        next_moves = _build_moves_block(
            labor=next_labor,
            driver=driver,
            start_free_time=curr_end_time,
            start_move_time=next_move_start,
            move_duration=next_travel,
            move_distance=next_dist,
            arrival_time=next_real_arrival,
            labor_distance=next_labor_dist,
            labor_duration=next_duration,
            finish_time=next_finish,
            start_pos=curr_end_pos,
            start_address=next_start_pos,
            end_address=next_end_pos,
            block_label="downstream",
        )

        curr_end_time = next_finish
        curr_end_pos = next_end_pos
        downstream_shifts.append(next_moves)

        # Early exit: if this labor did not shift, downstream labors won't either
        shift = (next_finish - next_labor["actual_end"]).total_seconds() / 60
        if abs(shift) < 1:
            break

    return True, downstream_shifts


# ===========================================================================
# CASE 1: EMPTY DRIVER SCHEDULE
# ===========================================================================

def _direct_insertion_empty_driver(
    new_labor: pd.Series,
    driver: str,
    directorio_df: pd.DataFrame,
    distance_method: str,
    dist_dict: DistDict,
    alfred_speed: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    EARLY_BUFFER: float,
    **kwargs,
) -> Tuple[bool, str, Optional[dict]]:
    home_pos = _get_driver_home_pos(directorio_df, driver)
    if home_pos is None:
        return False, "Driver not found in directory or missing location.", None

    start_time = new_labor["schedule_date"]
    arrival_time = start_time - timedelta(minutes=EARLY_BUFFER)

    _, dist_to_new, travel_time_to_new = _compute_arrival_to_next_labor(
        current_end_time=arrival_time,
        current_end_pos=home_pos,
        target_pos=new_labor["map_start_point"],
        speed=alfred_speed,
        distance_method=distance_method,
        dist_dict=dist_dict,
        **kwargs,
    )

    start_move_time = arrival_time - timedelta(minutes=travel_time_to_new)

    finish_time, finish_pos, labor_duration, labor_distance = _compute_service_end_time(
        arrival_time=arrival_time,
        start_pos=new_labor["map_start_point"],
        end_pos=new_labor["map_end_point"],
        distance_method=distance_method,
        dist_dict=dist_dict,
        vehicle_speed=VEHICLE_TRANSPORT_SPEED,
        prep_time=TIEMPO_ALISTAR,
        finish_time=TIEMPO_FINALIZACION,
        **kwargs,
    )

    new_moves = _build_moves_block(
        labor=new_labor,
        driver=driver,
        start_free_time=None,
        start_move_time=start_move_time,
        move_duration=travel_time_to_new,
        move_distance=dist_to_new,
        arrival_time=arrival_time,
        labor_distance=labor_distance,
        labor_duration=labor_duration,
        finish_time=finish_time,
        start_pos=home_pos,
        start_address=new_labor["map_start_point"],
        end_address=new_labor["map_end_point"],
        block_label="new",
    )

    insertion_plan = {
        "driver_id": driver,
        "new_labor_id": new_labor["labor_id"],
        "prev_labor_id": None,
        "next_labor_id": None,
        "dist_to_new": dist_to_new,
        "dist_to_next": 0.0,
        "new_moves": new_moves,
        "next_moves": None,
        "downstream_shifts": [],
    }
    return True, "", insertion_plan


# ===========================================================================
# CASE 2: BEFORE FIRST LABOR
# ===========================================================================

def _is_creation_before_first_labor(
    moves_driver_df: pd.DataFrame,
    directorio_df: pd.DataFrame,
    driver: str,
) -> bool:
    """Return True if the driver is still at their home base (first move starts there)."""
    home_pos = _get_driver_home_pos(directorio_df, driver)
    if home_pos is None:
        return False
    first_start = str(moves_driver_df.iloc[0]["start_point"]).strip()
    return first_start == home_pos.strip()


def _evaluate_and_execute_insertion_before_first_labor(
    new_labor: pd.Series,
    moves_driver_df: pd.DataFrame,
    driver: str,
    directorio_df: pd.DataFrame,
    distance_method: str,
    dist_dict: DistDict,
    vehicle_transport_speed: float,
    alfred_speed: float,
    tiempo_alistar: float,
    tiempo_finalizacion: float,
    tiempo_gracia: float,
    early_buffer: float,
    **kwargs,
) -> Tuple[bool, str, Optional[dict]]:
    next_labor = moves_driver_df.loc[2]  # First _labor row in the reset-index triplet
    home_pos = _get_driver_home_pos(directorio_df, driver)
    if home_pos is None:
        return False, "Driver not found in directory.", None

    arrival_time = new_labor["schedule_date"] - timedelta(minutes=early_buffer)

    _, new_dist_to_reach, new_time_to_reach = _compute_arrival_to_next_labor(
        current_end_time=arrival_time,
        current_end_pos=home_pos,
        target_pos=new_labor["map_start_point"],
        speed=alfred_speed,
        distance_method=distance_method,
        dist_dict=dist_dict,
        **kwargs,
    )

    new_move_start_time = arrival_time - timedelta(minutes=new_time_to_reach)
    next_labor_id_candidate = moves_driver_df.loc[0, "labor_id"]

    new_labor_finish_time, new_labor_finish_pos, new_duration, new_distance = (
        _compute_service_end_time(
            arrival_time=arrival_time,
            start_pos=new_labor["map_start_point"],
            end_pos=new_labor["map_end_point"],  # Fixed: reference had map_start_point
            distance_method=distance_method,
            dist_dict=dist_dict,
            vehicle_speed=vehicle_transport_speed,
            prep_time=tiempo_alistar,
            finish_time=tiempo_finalizacion,
            **kwargs,
        )
    )

    feasible_next, next_real_arrival, next_move_start, next_dist_to_reach, next_time_to_reach = (
        _can_reach_next_labor(
            new_finish_time=new_labor_finish_time,
            new_finish_pos=new_labor_finish_pos,
            next_start_time=next_labor["schedule_date"],
            next_start_pos=next_labor["start_point"],
            distance_method=distance_method,
            dist_dict=dist_dict,
            driver_speed=alfred_speed,
            grace_time=tiempo_gracia,
            early_buffer=early_buffer,
            **kwargs,
        )
    )

    if not feasible_next:
        return False, "Driver would not make it to the next labor after early insertion.", None

    next_finish, next_end_pos, next_duration, next_distance = _compute_service_end_time(
        arrival_time=next_real_arrival,
        start_pos=next_labor["start_point"],
        end_pos=next_labor["end_point"],
        distance_method=distance_method,
        dist_dict=dist_dict,
        vehicle_speed=vehicle_transport_speed,
        prep_time=tiempo_alistar,
        finish_time=tiempo_finalizacion,
        **kwargs,
    )

    downstream_ok, downstream_shifts = _simulate_downstream_shift(
        moves_driver_df=moves_driver_df,
        driver=driver,
        start_idx=2,
        start_time=next_real_arrival,
        start_pos=next_labor["start_point"],
        previous_start_time=next_labor["actual_end"],
        distance_method=distance_method,
        dist_dict=dist_dict,
        ALFRED_SPEED=alfred_speed,
        VEHICLE_TRANSPORT_SPEED=vehicle_transport_speed,
        TIEMPO_ALISTAR=tiempo_alistar,
        TIEMPO_FINALIZACION=tiempo_finalizacion,
        TIEMPO_GRACIA=tiempo_gracia,
        EARLY_BUFFER=early_buffer,
        **kwargs,
    )

    if not downstream_ok:
        return False, "Downstream labors become infeasible after early insertion.", None

    new_moves = _build_moves_block(
        labor=new_labor,
        driver=driver,
        start_free_time=None,
        start_move_time=new_move_start_time,
        move_duration=new_time_to_reach,
        move_distance=new_dist_to_reach,
        arrival_time=arrival_time,
        labor_distance=new_distance,
        labor_duration=new_duration,
        finish_time=new_labor_finish_time,
        start_pos=home_pos,
        start_address=new_labor["map_start_point"],
        end_address=new_labor["map_end_point"],
        block_label="new",
    )

    next_moves = _build_moves_block(
        labor=next_labor,
        driver=driver,
        start_free_time=new_labor_finish_time,
        start_move_time=next_move_start,
        move_duration=next_time_to_reach,
        move_distance=next_dist_to_reach,
        arrival_time=next_real_arrival,
        labor_distance=next_distance,
        labor_duration=next_duration,
        finish_time=next_finish,
        start_pos=new_labor_finish_pos,
        start_address=next_labor["start_point"],
        end_address=next_labor["end_point"],
        block_label="next",
    )

    insertion_plan = {
        "driver_id": driver,
        "new_labor_id": new_labor["labor_id"],
        "prev_labor_id": None,
        "next_labor_id": next_labor_id_candidate,
        "dist_to_new": new_dist_to_reach,
        "dist_to_next": next_dist_to_reach,
        "new_moves": new_moves,
        "next_moves": next_moves,
        "downstream_shifts": downstream_shifts,
    }

    return True, "", insertion_plan


# ===========================================================================
# MAIN FEASIBILITY EVALUATOR
# ===========================================================================

def evaluate_driver_feasibility(
    new_labor: pd.Series,
    driver: str,
    moves_driver_df: pd.DataFrame,
    directorio_df: pd.DataFrame,
    distance_method: str,
    dist_dict: DistDict,
    ALFRED_SPEED: float,
    VEHICLE_TRANSPORT_SPEED: float,
    TIEMPO_ALISTAR: float,
    TIEMPO_FINALIZACION: float,
    TIEMPO_GRACIA: float,
    EARLY_BUFFER: float,
    forced_start_time=None,
    **kwargs,
) -> Tuple[bool, str, Optional[dict]]:
    """
    Determine whether driver can insert new_labor and build the full insertion plan.
    Returns (feasible, reason, insertion_plan).
    """
    feasible = False
    infeasible_log = ""
    insertion_plan = None

    new_start_time = forced_start_time if forced_start_time else new_labor["schedule_date"]

    # ------------------------------------------------------------------
    # Case 1: Driver has no assigned labors
    # ------------------------------------------------------------------
    if moves_driver_df.empty:
        return _direct_insertion_empty_driver(
            new_labor=new_labor,
            driver=driver,
            directorio_df=directorio_df,
            distance_method=distance_method,
            dist_dict=dist_dict,
            alfred_speed=ALFRED_SPEED,
            VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
            TIEMPO_ALISTAR=TIEMPO_ALISTAR,
            TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
            EARLY_BUFFER=EARLY_BUFFER,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Case 2: Insertion before the driver's first labor
    # ------------------------------------------------------------------
    if _is_creation_before_first_labor(moves_driver_df, directorio_df, driver):
        if new_labor["schedule_date"] <= moves_driver_df.loc[2, "schedule_date"]:
            return _evaluate_and_execute_insertion_before_first_labor(
                new_labor=new_labor,
                moves_driver_df=moves_driver_df,
                driver=driver,
                directorio_df=directorio_df,
                distance_method=distance_method,
                dist_dict=dist_dict,
                vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                alfred_speed=ALFRED_SPEED,
                tiempo_alistar=TIEMPO_ALISTAR,
                tiempo_finalizacion=TIEMPO_FINALIZACION,
                tiempo_gracia=TIEMPO_GRACIA,
                early_buffer=EARLY_BUFFER,
                **kwargs,
            )

    labor_iter = 2
    n_rows = len(moves_driver_df)

    # ------------------------------------------------------------------
    # Case 3: Insertion between existing labors
    # ------------------------------------------------------------------
    while labor_iter < n_rows:
        if labor_iter + 3 > n_rows:
            break

        curr_end_time, curr_end_pos, next_start_time, next_start_pos = _get_driver_context(
            moves_driver_df, labor_iter
        )

        if next_start_time < new_start_time:
            labor_iter += 3
            continue

        curr_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]
        next_labor = moves_driver_df.loc[labor_iter + 3]
        next_labor_id_candidate = next_labor["labor_id"]

        new_arrival, new_dist_to_reach, new_time_to_reach = _compute_arrival_to_next_labor(
            current_end_time=curr_end_time,
            current_end_pos=curr_end_pos,
            target_pos=new_labor["map_start_point"],
            speed=ALFRED_SPEED,
            distance_method=distance_method,
            dist_dict=dist_dict,
            **kwargs,
        )

        # Fixed: was timedelta(TIEMPO_GRACIA) which is days, not minutes
        if new_arrival > new_start_time + timedelta(minutes=TIEMPO_GRACIA):
            infeasible_log = "Driver would not arrive on time to the new labor."
            break

        new_real_arrival, new_move_start = _adjust_for_early_arrival(
            would_arrive_at=new_arrival,
            travel_time=new_time_to_reach,
            schedule_date=new_start_time,
            early_buffer=EARLY_BUFFER,
        )

        new_finish, new_finish_pos, new_duration, new_distance = _compute_service_end_time(
            arrival_time=new_real_arrival,
            start_pos=new_labor["map_start_point"],
            end_pos=new_labor["map_end_point"],
            distance_method=distance_method,
            dist_dict=dist_dict,
            vehicle_speed=VEHICLE_TRANSPORT_SPEED,
            prep_time=TIEMPO_ALISTAR,
            finish_time=TIEMPO_FINALIZACION,
            **kwargs,
        )

        feasible_next, next_real_arrival, next_move_start, next_dist_to_reach, next_time_to_reach = (
            _can_reach_next_labor(
                new_finish_time=new_finish,
                new_finish_pos=new_finish_pos,
                next_start_time=next_start_time,
                next_start_pos=next_start_pos,
                distance_method=distance_method,
                dist_dict=dist_dict,
                driver_speed=ALFRED_SPEED,
                grace_time=TIEMPO_GRACIA,
                early_buffer=EARLY_BUFFER,
                **kwargs,
            )
        )

        if not feasible_next:
            infeasible_log = "Driver would not reach the next scheduled labor in time."
            break

        next_finish, next_end_pos, next_duration, next_distance = _compute_service_end_time(
            arrival_time=next_real_arrival,
            start_pos=next_labor["start_point"],
            end_pos=next_labor["end_point"],
            distance_method=distance_method,
            dist_dict=dist_dict,
            vehicle_speed=VEHICLE_TRANSPORT_SPEED,
            prep_time=TIEMPO_ALISTAR,
            finish_time=TIEMPO_FINALIZACION,
            **kwargs,
        )

        downstream_ok, downstream_shifts = _simulate_downstream_shift(
            moves_driver_df=moves_driver_df,
            driver=driver,
            start_idx=labor_iter + 3,
            start_time=next_finish,
            start_pos=next_end_pos,
            previous_start_time=moves_driver_df.loc[labor_iter + 3, "actual_end"],
            distance_method=distance_method,
            dist_dict=dist_dict,
            ALFRED_SPEED=ALFRED_SPEED,
            VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
            TIEMPO_ALISTAR=TIEMPO_ALISTAR,
            TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
            TIEMPO_GRACIA=TIEMPO_GRACIA,
            EARLY_BUFFER=EARLY_BUFFER,
            **kwargs,
        )

        if not downstream_ok:
            infeasible_log = "Downstream labors become infeasible after insertion."
            break

        feasible = True

        new_moves = _build_moves_block(
            labor=new_labor,
            driver=driver,
            start_free_time=curr_end_time,
            start_move_time=new_move_start,
            move_duration=new_time_to_reach,
            move_distance=new_dist_to_reach,
            arrival_time=new_real_arrival,
            labor_distance=new_distance,
            labor_duration=new_duration,
            finish_time=new_finish,
            start_pos=curr_end_pos,
            start_address=new_labor["map_start_point"],
            end_address=new_labor["map_end_point"],
            block_label="new",
        )

        next_moves_block = _build_moves_block(
            labor=next_labor,
            driver=driver,
            start_free_time=new_finish,
            start_move_time=next_move_start,
            move_duration=next_time_to_reach,
            move_distance=next_dist_to_reach,
            arrival_time=next_real_arrival,
            labor_distance=next_distance,
            labor_duration=next_duration,
            finish_time=next_finish,
            start_pos=new_finish_pos,
            start_address=next_labor["start_point"],
            end_address=next_labor["end_point"],
            block_label="next",
        )

        insertion_plan = {
            "driver_id": driver,
            "new_labor_id": new_labor["labor_id"],
            "prev_labor_id": curr_labor_id,
            "next_labor_id": next_labor_id_candidate,
            "dist_to_new": new_dist_to_reach,
            "dist_to_next": next_dist_to_reach,
            "new_moves": new_moves,
            "next_moves": next_moves_block,
            "downstream_shifts": downstream_shifts,
        }
        break

    # ------------------------------------------------------------------
    # Case 4: Append at end of driver's schedule
    # ------------------------------------------------------------------
    if not feasible and infeasible_log == "" and labor_iter >= n_rows - 3:
        prev_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]
        curr_end_time = moves_driver_df.loc[labor_iter, "actual_end"]
        curr_end_pos = moves_driver_df.loc[labor_iter, "end_point"]

        if not isinstance(curr_end_time, pd.Timestamp):
            logger.debug(
                "INSERT Case 4: unexpected end_time type %s for driver %s",
                type(curr_end_time), driver,
            )

        would_arrive, dist_to_new, travel_time_to_new = _compute_arrival_to_next_labor(
            current_end_time=curr_end_time,
            current_end_pos=curr_end_pos,
            target_pos=new_labor["map_start_point"],
            speed=ALFRED_SPEED,
            distance_method=distance_method,
            dist_dict=dist_dict,
            **kwargs,
        )

        latest_arrival = new_labor["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)
        if would_arrive > latest_arrival:
            infeasible_log = "Driver would arrive too late to the new labor."
            return feasible, infeasible_log, insertion_plan

        feasible = True
        real_arrival, move_start_time = _adjust_for_early_arrival(
            would_arrive_at=would_arrive,
            travel_time=travel_time_to_new,
            schedule_date=new_labor["schedule_date"],
            early_buffer=EARLY_BUFFER,
        )

        finish_time, finish_pos, labor_duration, labor_distance = _compute_service_end_time(
            arrival_time=real_arrival,
            start_pos=new_labor["map_start_point"],
            end_pos=new_labor["map_end_point"],
            distance_method=distance_method,
            dist_dict=dist_dict,
            vehicle_speed=VEHICLE_TRANSPORT_SPEED,
            prep_time=TIEMPO_ALISTAR,
            finish_time=TIEMPO_FINALIZACION,
            **kwargs,
        )

        new_moves = _build_moves_block(
            labor=new_labor,
            driver=driver,
            start_free_time=curr_end_time,
            start_move_time=move_start_time,
            move_duration=travel_time_to_new,
            move_distance=dist_to_new,
            arrival_time=real_arrival,
            labor_distance=labor_distance,
            labor_duration=labor_duration,
            finish_time=finish_time,
            start_pos=curr_end_pos,
            start_address=new_labor["map_start_point"],
            end_address=new_labor["map_end_point"],
            block_label="new",
        )

        # Fixed: standardised keys (was dist_to_new_service / dist_to_next_labor)
        insertion_plan = {
            "driver_id": driver,
            "new_labor_id": new_labor["labor_id"],
            "prev_labor_id": prev_labor_id,
            "next_labor_id": None,
            "dist_to_new": dist_to_new,
            "dist_to_next": 0.0,
            "new_moves": new_moves,
            "next_moves": None,
            "downstream_shifts": [],
        }

    return feasible, infeasible_log, insertion_plan


# ===========================================================================
# COMMIT HELPERS
# ===========================================================================

def _get_affected_labors_from_moves(
    new_moves: pd.DataFrame,
    next_moves: Optional[pd.DataFrame] = None,
    downstream_shifts: Optional[List[pd.DataFrame]] = None,
) -> List[str]:
    """Return unique labor_ids from all provided move blocks."""
    dfs = []
    if isinstance(new_moves, pd.DataFrame) and not new_moves.empty:
        dfs.append(new_moves)
    if isinstance(next_moves, pd.DataFrame) and not next_moves.empty:
        dfs.append(next_moves)
    if downstream_shifts:
        dfs.extend(
            [d for d in downstream_shifts if isinstance(d, pd.DataFrame) and not d.empty]
        )
    if not dfs:
        return []
    combined = pd.concat(dfs, ignore_index=True)
    return combined["labor_id"].dropna().unique().tolist()


def commit_new_labor_insertion(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    insertion_plan: dict,
    new_labor: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, str]:
    """
    Apply a validated insertion plan to labors_df and moves_df.

    Returns (updated_labors_df, updated_moves_df, new_labor_end_time, new_labor_end_pos).
    """
    assert isinstance(insertion_plan["new_moves"], pd.DataFrame), \
        "insertion_plan['new_moves'] must be a DataFrame"

    driver_id = insertion_plan["driver_id"]
    new_labor_id = new_labor["labor_id"]
    new_moves = insertion_plan["new_moves"].copy()
    next_moves = insertion_plan.get("next_moves")
    downstream_shifts = insertion_plan.get("downstream_shifts", [])

    affected_labors = _get_affected_labors_from_moves(new_moves, next_moves, downstream_shifts)

    # --- Collect all updated move rows ---
    moves_to_add: List[pd.DataFrame] = [new_moves]
    if isinstance(next_moves, pd.DataFrame) and not next_moves.empty:
        moves_to_add.append(next_moves)
    moves_to_add.extend(
        [m for m in downstream_shifts if isinstance(m, pd.DataFrame) and not m.empty]
    )
    new_moves_combined = pd.concat(moves_to_add, ignore_index=True) if moves_to_add else pd.DataFrame()

    # --- Update moves_df (remove stale rows, add new ones) ---
    if moves_df.empty:
        moves_df_updated = new_moves_combined.copy()
    else:
        moves_df_updated = pd.concat(
            [moves_df[~moves_df["labor_id"].isin(affected_labors)], new_moves_combined],
            ignore_index=True,
        )

    # --- Update labors_df ---
    labors_df_updated = labors_df.copy()

    # 1. Sync actual_start / actual_end for existing labors that shifted
    labor_rows_from_moves = moves_df_updated[
        moves_df_updated["labor_context_id"].astype(str).str.endswith("_labor")
    ]
    existing_ids = set(labors_df["labor_id"].values)
    for labor_id in affected_labors:
        if labor_id == new_labor_id or labor_id not in existing_ids:
            continue
        move_row = labor_rows_from_moves[labor_rows_from_moves["labor_id"] == labor_id]
        if not move_row.empty:
            r = move_row.iloc[0]
            labors_df_updated.loc[
                labors_df_updated["labor_id"] == labor_id,
                ["actual_start", "actual_end"],
            ] = [r["actual_start"], r["actual_end"]]

    # 2. Build and add the new labor row
    new_labor_move_row = new_moves[
        new_moves["labor_context_id"].astype(str).str.endswith("_labor")
    ].iloc[0]

    new_labor_row = new_labor.copy()
    new_labor_row["assigned_driver"] = driver_id
    new_labor_row["actual_start"] = new_labor_move_row["actual_start"]
    new_labor_row["actual_end"] = new_labor_move_row["actual_end"]
    new_labor_row["dist_km"] = new_labor_move_row["distance_km"]
    new_labor_row["map_start_point"] = new_labor_move_row["start_point"]
    new_labor_row["map_end_point"] = new_labor_move_row["end_point"]
    new_labor_row["date"] = new_labor_move_row["date"]
    new_labor_row["actual_status"] = "COMPLETED"

    # Remove any pre-existing row then append (Fixed: no double-filter bug)
    labors_df_updated = labors_df_updated[labors_df_updated["labor_id"] != new_labor_id]
    labors_df_updated = pd.concat(
        [labors_df_updated, pd.DataFrame([new_labor_row])], ignore_index=True
    )

    end_time = new_labor_move_row["actual_end"]
    end_pos = new_labor_move_row["end_point"]
    return labors_df_updated, moves_df_updated, end_time, end_pos


# ===========================================================================
# BEST INSERTION SELECTION
# ===========================================================================

def get_best_insertion(
    candidate_insertions: List[Tuple[str, dict]],
    selection_mode: str = "min_total_distance",
    random_state: Optional[int] = None,
) -> Tuple[Optional[str], Optional[dict], pd.DataFrame]:
    """
    Select the best insertion plan from feasible candidates.
    Returns (selected_driver, best_plan, selection_df).
    """
    if not candidate_insertions:
        return None, None, pd.DataFrame()

    records = []
    for driver, plan in candidate_insertions:
        records.append({
            "driver_id": plan.get("driver_id"),  # Fixed: was plan.get("alfred")
            "prev_labor_id": plan.get("prev_labor_id"),
            "next_labor_id": plan.get("next_labor_id"),
            "dist_to_new": plan.get("dist_to_new", np.nan),   # Fixed: standardised key
            "dist_to_next": plan.get("dist_to_next", np.nan),  # Fixed: standardised key
        })

    selection_df = pd.DataFrame(records)
    selection_df["dist_to_new"] = pd.to_numeric(
        selection_df["dist_to_new"], errors="coerce"
    ).fillna(0)
    selection_df["dist_to_next"] = pd.to_numeric(
        selection_df["dist_to_next"], errors="coerce"
    ).fillna(0)
    selection_df["total_distance"] = selection_df["dist_to_new"] + selection_df["dist_to_next"]

    if selection_mode == "random":
        if selection_df.empty:
            return None, None, selection_df
        chosen_idx = selection_df.sample(1, random_state=random_state).index[0]
    elif selection_mode == "min_dist_to_new_labor":
        chosen_idx = selection_df["dist_to_new"].idxmin()
    elif selection_mode == "min_total_distance":
        chosen_idx = selection_df["total_distance"].idxmin()
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode!r}")

    driver, best_plan = candidate_insertions[chosen_idx]
    return driver, best_plan, selection_df


# ===========================================================================
# HIGH-LEVEL ORCHESTRATION (one labor, all drivers)
# ===========================================================================

def commit_labor_insertion(
    labors_algo_df: pd.DataFrame,
    moves_algo_df: pd.DataFrame,
    new_labor: pd.Series,
    directorio_df: pd.DataFrame,
    drivers: List[str],
    city: str,
    fecha: str,
    dist_dict: DistDict,
    distance_method: str,
    alfred_speed: float,
    vehicle_transport_speed: float,
    tiempo_alistar: float,
    tiempo_finalizacion: float,
    tiempo_gracia: float,
    early_buffer: float,
    forced_start_time=None,
    selection_mode: str = "min_total_distance",
    **kwargs,
) -> Tuple[bool, pd.DataFrame, pd.DataFrame, Optional[Any], Optional[str]]:
    """
    Evaluate all drivers for inserting new_labor and commit the best feasible plan.
    Returns (success, labors_df, moves_df, end_time, end_pos).
    """
    candidate_insertions = []

    for driver in drivers:
        _, moves_driver_df = filter_dfs_for_insertion(
            labors_algo_df=labors_algo_df,
            moves_algo_df=moves_algo_df,
            city=city,
            fecha=fecha,
            driver=driver,
        )

        feasible, _, insertion_plan = evaluate_driver_feasibility(
            new_labor=new_labor,
            driver=driver,
            moves_driver_df=moves_driver_df,
            directorio_df=directorio_df,
            distance_method=distance_method,
            dist_dict=dist_dict,
            ALFRED_SPEED=alfred_speed,
            VEHICLE_TRANSPORT_SPEED=vehicle_transport_speed,
            TIEMPO_ALISTAR=tiempo_alistar,
            TIEMPO_FINALIZACION=tiempo_finalizacion,
            TIEMPO_GRACIA=tiempo_gracia,
            EARLY_BUFFER=early_buffer,
            forced_start_time=forced_start_time,
            **kwargs,
        )

        if feasible:
            candidate_insertions.append((driver, insertion_plan))

    if not candidate_insertions:
        return False, labors_algo_df, moves_algo_df, None, None

    _, best_plan, _ = get_best_insertion(
        candidate_insertions, selection_mode=selection_mode
    )

    labors_updated, moves_updated, end_time, end_pos = commit_new_labor_insertion(
        labors_df=labors_algo_df,
        moves_df=moves_algo_df,
        insertion_plan=best_plan,
        new_labor=new_labor,
    )

    return True, labors_updated, moves_updated, end_time, end_pos


# ===========================================================================
# FILTERING HELPERS
# ===========================================================================

def get_drivers(
    labors_algo_df: pd.DataFrame,
    directorio_df: pd.DataFrame,
    city: str,
    fecha=None,
    get_all: bool = True,
) -> List[str]:
    """Return driver IDs for a city. Uses directory if get_all=True, otherwise labors."""
    if get_all:
        dir_city = _filter_drivers_by_city(directorio_df, city)
        drivers: List[str] = []
        for _, row in dir_city.iterrows():
            key = _extract_driver_key(row)
            if key:
                drivers.append(key)
        return list(dict.fromkeys(drivers))  # deduplicate, preserve order
    else:
        mask = labors_algo_df["department_code"] == city
        if fecha is not None:
            fecha_date = pd.to_datetime(fecha).date()
            mask = mask & (labors_algo_df["schedule_date"].dt.date == fecha_date)
        drivers_series = labors_algo_df[mask]["assigned_driver"].dropna().astype(str)
        return [d for d in drivers_series.unique() if d.strip() not in ("", "nan", "None")]


def filter_dfs_for_insertion(
    labors_algo_df: pd.DataFrame,
    moves_algo_df: pd.DataFrame,
    city: str,
    fecha,
    driver: str,
    created_at=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter labors and moves to the relevant subset for a given driver / day.

    When created_at is provided (online mode), only labors ending after that
    reference time are kept. When None (batch mode), all are returned.
    """
    fecha_date = pd.to_datetime(fecha).date()
    driver_str = str(driver).strip()

    def _driver_mask(df: pd.DataFrame, col: str = "assigned_driver") -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return df[col].astype(str).str.strip() == driver_str

    # --- Labors ---
    lab_mask = (
        (labors_algo_df["department_code"] == city)
        & (labors_algo_df["schedule_date"].dt.date == fecha_date)
        & _driver_mask(labors_algo_df)
    )
    labors_pref = labors_algo_df[lab_mask].sort_values(
        ["schedule_date", "actual_start", "actual_end"], ignore_index=True
    )

    # --- Moves ---
    if moves_algo_df.empty:
        moves_pref = pd.DataFrame()
    else:
        mov_mask = (
            (moves_algo_df["schedule_date"].dt.date == fecha_date)
            & _driver_mask(moves_algo_df)
        )
        if "department_code" in moves_algo_df.columns:
            mov_mask = mov_mask & (moves_algo_df["department_code"] == city)
        moves_pref = moves_algo_df[mov_mask].sort_values(
            ["schedule_date", "actual_start", "actual_end"], ignore_index=True
        )

    if labors_pref.empty:
        empty_moves = pd.DataFrame(columns=moves_pref.columns) if not moves_pref.empty else pd.DataFrame()
        return labors_pref, empty_moves

    # --- created_at filter ---
    if created_at is not None:
        future = labors_pref[labors_pref["actual_end"] > created_at]
        if not future.empty:
            active_ids = future["labor_id"].tolist()
            first_pos = labors_pref.index[labors_pref["labor_id"] == active_ids[0]][0]
            if first_pos > 0:
                prev_id = labors_pref.iloc[first_pos - 1]["labor_id"]
                active_ids = [prev_id] + active_ids
        else:
            active_ids = [labors_pref["labor_id"].iloc[-1]]

        labors_pref = labors_pref[labors_pref["labor_id"].isin(active_ids)].copy()
        if not moves_pref.empty:
            moves_pref = moves_pref[moves_pref["labor_id"].isin(active_ids)].copy()

    return labors_pref, moves_pref


# ===========================================================================
# NON-TRANSPORT HELPERS
# ===========================================================================

def get_nontransport_labor_duration(
    duraciones_df: pd.DataFrame,
    city: str,
    labor_type: str,
) -> float:
    """
    Return the representative duration (p75_min) for a non-transport labor in a city.
    Falls back to the 75th-percentile across all cities if no city-specific entry exists.
    """
    required = {"city", "labor_type", "p75_min"}
    missing = required - set(duraciones_df.columns)
    if missing:
        raise ValueError(f"duraciones_df missing required columns: {missing}")

    city_match = duraciones_df[
        (duraciones_df["city"] == city)
        & (duraciones_df["labor_type"] == labor_type)
        & duraciones_df["p75_min"].notna()
    ]
    if not city_match.empty:
        return float(city_match.iloc[0]["p75_min"])

    all_cities = duraciones_df[
        (duraciones_df["labor_type"] == labor_type) & duraciones_df["p75_min"].notna()
    ]
    if all_cities.empty:
        raise FileNotFoundError(f"Labor type '{labor_type}' not found in any city.")

    per_city_vals = all_cities.groupby("city")["p75_min"].mean().values
    return float(np.percentile(per_city_vals, 75))


def commit_nontransport_labor_insertion(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    new_labor: pd.Series,
    start_pos: str,
    start_time,
    duration: float,
    fecha: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Commit a non-transport labor (no driver movement needed)."""
    new_end_time = start_time + timedelta(minutes=duration)
    new_labor_id = new_labor["labor_id"]

    new_labor_row = new_labor.copy()
    new_labor_row["assigned_driver"] = ""
    new_labor_row["actual_start"] = start_time
    new_labor_row["actual_end"] = new_end_time
    new_labor_row["dist_km"] = 0.0
    new_labor_row["map_start_point"] = start_pos
    new_labor_row["map_end_point"] = start_pos
    new_labor_row["date"] = fecha

    labors_updated = labors_df[labors_df["labor_id"] != new_labor_id].copy()
    labors_updated = pd.concat(
        [labors_updated, pd.DataFrame([new_labor_row])], ignore_index=True
    )
    return labors_updated, moves_df.copy(), new_end_time


# ===========================================================================
# ITERATION RUNNER
# ===========================================================================

def run_insertion_worker(
    base_labors_df: pd.DataFrame,
    base_moves_df: pd.DataFrame,
    new_labors_df: pd.DataFrame,
    seed: int,
    city: str,
    fecha: str,
    directorio_df: pd.DataFrame,
    drivers: List[str],
    dist_dict: DistDict,
    distance_method: str,
    alfred_speed: float,
    vehicle_transport_speed: float,
    tiempo_alistar: float,
    tiempo_finalizacion: float,
    tiempo_gracia: float,
    early_buffer: float,
    workday_end_dt,
    duraciones_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Attempt to insert all new labors into the base schedule using a randomised
    service ordering determined by `seed`.

    Returns a result dict with keys: valid, seed, num_inserted, dist, results, moves.
    """
    working_labors = base_labors_df.copy()
    working_moves = base_moves_df.copy()

    # Shuffle service order by seed
    service_ids = (
        new_labors_df[["service_id"]]
        .drop_duplicates()
        .sample(frac=1, random_state=seed)["service_id"]
        .tolist()
    )

    total_inserted = 0

    for svc in service_ids:
        svc_labors = (
            new_labors_df[new_labors_df["service_id"] == svc]
            .sort_values("labor_sequence")
        )
        tmp_labors = working_labors.copy()
        tmp_moves = working_moves.copy()
        curr_end_time = None
        curr_end_pos = None
        svc_ok = True

        for _, labor in svc_labors.iterrows():
            is_transport = labor.get("labor_category") == "VEHICLE_TRANSPORTATION"

            if is_transport:
                if curr_end_time is not None and curr_end_time > workday_end_dt:
                    # Workday exceeded — stop this service's remaining transport labors
                    break

                success, tmp_labors, tmp_moves, curr_end_time, curr_end_pos = (
                    commit_labor_insertion(
                        labors_algo_df=tmp_labors,
                        moves_algo_df=tmp_moves,
                        new_labor=labor,
                        directorio_df=directorio_df,
                        drivers=drivers,
                        city=city,
                        fecha=fecha,
                        dist_dict=dist_dict,
                        distance_method=distance_method,
                        alfred_speed=alfred_speed,
                        vehicle_transport_speed=vehicle_transport_speed,
                        tiempo_alistar=tiempo_alistar,
                        tiempo_finalizacion=tiempo_finalizacion,
                        tiempo_gracia=tiempo_gracia,
                        early_buffer=early_buffer,
                        forced_start_time=curr_end_time,
                        selection_mode="random",
                    )
                )

                if not success:
                    svc_ok = False
                    break

            else:
                # Non-transport: schedule relative to current position
                est_time = labor.get("estimated_time")
                if est_time is not None and pd.notna(est_time):
                    duration = float(est_time)
                else:
                    try:
                        duration = (
                            get_nontransport_labor_duration(
                                duraciones_df, city, labor.get("labor_type", "")
                            )
                            if duraciones_df is not None and not duraciones_df.empty
                            else 30.0
                        )
                    except (ValueError, FileNotFoundError):
                        duration = 30.0

                start_pos = labor.get("map_start_point") or ""
                start_time = curr_end_time if curr_end_time is not None else labor["schedule_date"]

                tmp_labors, tmp_moves, curr_end_time = commit_nontransport_labor_insertion(
                    labors_df=tmp_labors,
                    moves_df=tmp_moves,
                    new_labor=labor,
                    start_pos=start_pos,
                    start_time=start_time,
                    duration=duration,
                    fecha=fecha,
                )
                curr_end_pos = start_pos

        if svc_ok:
            working_labors = tmp_labors
            working_moves = tmp_moves
            total_inserted += len(svc_labors)

    total_dist = (
        float(working_moves["distance_km"].sum())
        if "distance_km" in working_moves.columns
        else 0.0
    )

    return {
        "valid": True,
        "seed": seed,
        "num_inserted": total_inserted,
        "dist": total_dist,
        "results": working_labors,
        "moves": working_moves,
    }


def select_best_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select the best iteration: maximise inserted services, break ties by min distance."""
    valid = [r for r in results if r and r.get("valid")]
    if not valid:
        return None
    return sorted(
        valid,
        key=lambda r: (
            -r["num_inserted"],
            r["dist"] if r["dist"] is not None else float("inf"),
        ),
    )[0]
