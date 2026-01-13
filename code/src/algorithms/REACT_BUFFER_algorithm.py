import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time

import random

from typing import Tuple, Optional, Dict, Any, List, Union, Callable

from src.algorithms.online_algorithms import filter_dynamic_df
from src.utils.filtering import flexible_filter
from src.data.distance_utils import distance


''' --------- REACT algorithm ---------'''
def filter_dfs_for_REACT(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    directorio_df: pd.DataFrame,
    freeze_cutoff: pd.Timestamp,
    driver_col: str = "assigned_driver",
    start_time_col: str = "actual_start",
    end_time_col: str = "actual_end",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labors / moves into frozen vs reassignable sets and derive driver states.

    Returns (in this exact order):
      labors_reassign_df, labors_frozen_df, moves_reassign_df, moves_frozen_df, driver_states_df

    driver_states_df columns:
      ['alfred', 'address_point', 'available_time']

    Rules implemented:
    - A move that is ongoing at cutoff (move.actual_start <= cutoff < move.actual_end)
      causes the target labor (and all previous labors) to be frozen; later labors are reassignable.
    - A labor that is ongoing at cutoff (labor.actual_start <= cutoff < labor.actual_end)
      results in that labor and all previous labors frozen; later labors reassignable.
    - Past labors (actual_end <= cutoff) are frozen; labors with actual_start > cutoff are reassignable.
    - If driver has no labors before cutoff, all their labors/moves are reassignable and driver's state
      is taken from directorio_df (address_point) and available_time = freeze_cutoff.
    - Uses labor_context_id suffixes (_free/_move/_labor) when present; otherwise detects move rows
      via labor_category == 'DRIVER_MOVE' or labor_name.
    """
    # --- Helpers --------------------------------------------------------------
    def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
        """Return empty DataFrame preserving dtypes of df."""
        return pd.DataFrame({col: pd.Series(dtype=df[col].dtype) for col in df.columns})

    def _ensure_datetime_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    def _get_base_labor_id_from_context(lcid: str) -> str:
        # labors may be like "12345_free", "12345_move", "12345_labor" or plain "12345"
        s = str(lcid)
        if "_" in s:
            return s.split("_")[0]
        return s

    # keep consistent datetimes
    datetime_cols = [start_time_col, end_time_col, "schedule_date", "created_at"]
    labors_df = _ensure_datetime_cols(labors_df.copy(), datetime_cols)
    moves_df = _ensure_datetime_cols(moves_df.copy(), datetime_cols)

    # normalize inputs (avoid missing columns)
    if driver_col not in labors_df.columns:
        labors_df[driver_col] = np.nan
    if driver_col not in moves_df.columns:
        moves_df[driver_col] = np.nan

    # Determine candidate drivers from all three sources (labors, moves, directorio)
    drivers_from_labors = labors_df[driver_col].dropna().astype(str).unique().tolist()
    drivers_from_moves = moves_df[driver_col].dropna().astype(str).unique().tolist()
    drivers_from_dir = directorio_df['alfred'].dropna().astype(str).unique().tolist() if 'alfred' in directorio_df.columns else []
    all_drivers = list(dict.fromkeys(drivers_from_labors + drivers_from_moves + drivers_from_dir))

    frozen_labors_parts = []
    reassign_labors_parts = []
    frozen_moves_parts = []
    reassign_moves_parts = []
    driver_states: List[Dict[str, Any]] = []

    for drv in all_drivers:
        # per-driver slices (keep original indices to identify previous/next)
        labors_d = labors_df[labors_df[driver_col].astype(str) == str(drv)].copy()
        moves_d = moves_df[moves_df[driver_col].astype(str) == str(drv)].copy()

        # sort chronologically defensively
        labors_d = labors_d.sort_values([start_time_col, end_time_col]).reset_index(drop=True)
        moves_d = moves_d.sort_values([start_time_col, end_time_col]).reset_index(drop=True)

        # default: everything reassignable (if no labors)
        if labors_d.empty and moves_d.empty:
            # driver state from directorio if available
            drow = directorio_df[directorio_df['alfred'].astype(str) == str(drv)]
            if not drow.empty:
                d0 = drow.iloc[0]
                addr = d0.get("address_point")
                driver_states.append({"alfred": drv, "address_point": addr, "available_time": freeze_cutoff})
            else:
                driver_states.append({"alfred": drv, "address_point": None, "available_time": freeze_cutoff})

            # no labors/moves to add
            continue

        # ----- 1) Check for an ongoing MOVE at cutoff (preferred detection) -----
        ongoing_move_idx = None
        ongoing_move_row = None
        if not moves_d.empty:
            # find rows that correspond to 'move' rows

            move_rows = moves_d[moves_d["labor_context_id"].astype(str).str.endswith("_move")]

            # find any move that is active at cutoff
            mask_move_ongoing = (move_rows[start_time_col] <= freeze_cutoff) & (move_rows[end_time_col] > freeze_cutoff)
            if mask_move_ongoing.any():
                ongoing_move_row = move_rows[mask_move_ongoing].iloc[0]

        if ongoing_move_row is not None:
            # Move is ongoing. Identify base labor id
            base_lid = ongoing_move_row.get("labor_id")

            # freeze labors up to and including that base_lid
            # find position in labors_d of that labor
            lab_pos = labors_d[labors_d["labor_id"].astype(str) == base_lid].index
            if not lab_pos.empty:
                last_frozen_idx = lab_pos[0]
                frozen_labors = labors_d.loc[:last_frozen_idx].copy()
                reassign_labors = labors_d.loc[last_frozen_idx+1:].copy()
            else:
                # can't find linked labor in labors_d: be conservative and freeze all moves related up to the move
                frozen_labors = labors_d.copy()
                reassign_labors = labors_d.iloc[0:0].copy()

            # moves: freeze all moves whose labor_context_id base is in frozen_labors' ids
            frozen_prefixes = set(map(str, frozen_labors["labor_id"].astype(str).tolist()))

            frozen_moves_mask = moves_d["labor_context_id"].astype(str).apply(
                lambda x: any(str(x).startswith(pref) for pref in frozen_prefixes)
            )

            frozen_moves = moves_d[frozen_moves_mask].copy()
            reassign_moves = moves_d[~frozen_moves_mask].copy()

            # driver state: finish position/time of the labor corresponding to the ongoing move's target labor
            # If labors contained that labor, use its end; otherwise fall back to move's end point and time
            if not lab_pos.empty:
                last_row = labors_d.loc[last_frozen_idx]
                addr_pt = last_row.get("end_address_point") if "end_address_point" in last_row else last_row.get("end_point")
                avail_time = last_row.get(end_time_col)
            else:
                addr_pt = ongoing_move_row.get("end_point")
                avail_time = ongoing_move_row.get(end_time_col)

            # collect
            frozen_labors_parts.append(frozen_labors)
            reassign_labors_parts.append(reassign_labors)
            frozen_moves_parts.append(frozen_moves)
            reassign_moves_parts.append(reassign_moves)
            driver_states.append({"alfred": drv, "address_point": addr_pt, "available_time": avail_time if pd.notna(avail_time) else freeze_cutoff})
            continue  # done with this driver

        # ----- 2) Check for an ongoing LABOR at cutoff -----
        ongoing_lab_mask = (labors_d[start_time_col] <= freeze_cutoff) & (labors_d[end_time_col] > freeze_cutoff)
        if ongoing_lab_mask.any():
            # freeze up to and including that labor; future labors reassignable
            ongoing_idx = labors_d.index[ongoing_lab_mask][0]
            frozen_labors = labors_d.loc[:ongoing_idx].copy()
            reassign_labors = labors_d.loc[ongoing_idx+1:].copy()

            # freeze moves linked to frozen labors
            frozen_prefixes = set(map(str, frozen_labors["labor_id"].astype(str).tolist()))
            frozen_moves_mask = moves_d["labor_context_id"].astype(str).apply(
                    lambda x: any(str(x).startswith(pref) for pref in frozen_prefixes)
                )

            frozen_moves = moves_d[frozen_moves_mask].copy()
            reassign_moves = moves_d[~frozen_moves_mask].copy()

            # driver state = end point/time of ongoing labor
            ongoing_row = labors_d.loc[ongoing_idx]
            addr_pt = ongoing_row.get("end_address_point") if "end_address_point" in ongoing_row else ongoing_row.get("end_point")
            avail_time = ongoing_row.get(end_time_col)

            frozen_labors_parts.append(frozen_labors)
            reassign_labors_parts.append(reassign_labors)
            frozen_moves_parts.append(frozen_moves)
            reassign_moves_parts.append(reassign_moves)
            driver_states.append({"alfred": drv, "address_point": addr_pt, "available_time": avail_time if pd.notna(avail_time) else freeze_cutoff})
            continue

        # ----- 3) No ongoing: any past (ended) labors? -----
        past_mask = labors_d[end_time_col] <= freeze_cutoff
        if past_mask.any():
            past_labors = labors_d.loc[past_mask].copy()
            frozen_labors_parts.append(past_labors)

            # moves ended before or at cutoff -> frozen
            if not moves_d.empty:
                frozen_moves_mask = moves_d[end_time_col] <= freeze_cutoff
                frozen_moves = moves_d.loc[frozen_moves_mask].copy()
                reassign_moves = moves_d.loc[~frozen_moves_mask].copy()
            else:
                frozen_moves = moves_d.iloc[0:0].copy()
                reassign_moves = moves_d.iloc[0:0].copy()

            reassign_labors_mask = labors_d[start_time_col] > freeze_cutoff
            reassign_labors = labors_d.loc[reassign_labors_mask].copy()

            if not reassign_labors.empty:
                reassign_labors_parts.append(reassign_labors)
            if not reassign_moves.empty:
                reassign_moves_parts.append(reassign_moves)
            if not frozen_moves.empty:
                frozen_moves_parts.append(frozen_moves)

            # driver state = last past labor end
            last_past = past_labors.sort_values(end_time_col).iloc[-1]
            addr_pt = last_past.get("end_address_point") if "end_address_point" in last_past else last_past.get("end_point")
            avail_time = last_past.get(end_time_col)
            driver_states.append({"alfred": drv, "address_point": addr_pt, "available_time": freeze_cutoff})
            continue

        # ----- 4) No labors before cutoff: idle from start-of-day -----
        # everything reassignable
        if not labors_d.empty:
            reassign_labors_parts.append(labors_d.copy())
        if not moves_d.empty:
            reassign_moves_parts.append(moves_d.copy())

        # driver initial position from directorio if present, else None
        drow = directorio_df[directorio_df['alfred'].astype(str) == str(drv)]
        if not drow.empty:
            d0 = drow.iloc[0]
            addr = d0.get("address_point")
        else:
            addr = None
        driver_states.append({"alfred": drv, "address_point": addr, "available_time": freeze_cutoff})

    # --- Concatenate preserving dtypes (or return empty like original) ----------------
    labors_frozen_df = pd.concat(frozen_labors_parts, ignore_index=True) if frozen_labors_parts else _empty_like(labors_df)
    labors_reassign_df = pd.concat(reassign_labors_parts, ignore_index=True) if reassign_labors_parts else _empty_like(labors_df)
    moves_frozen_df = pd.concat(frozen_moves_parts, ignore_index=True) if frozen_moves_parts else _empty_like(moves_df)
    moves_reassign_df = pd.concat(reassign_moves_parts, ignore_index=True) if reassign_moves_parts else _empty_like(moves_df)

    # ensure datetime columns are datetimes
    labors_frozen_df = _ensure_datetime_cols(labors_frozen_df, datetime_cols)
    labors_reassign_df = _ensure_datetime_cols(labors_reassign_df, datetime_cols)
    moves_frozen_df = _ensure_datetime_cols(moves_frozen_df, datetime_cols)
    moves_reassign_df = _ensure_datetime_cols(moves_reassign_df, datetime_cols)

    driver_states_df = pd.DataFrame(driver_states)[["alfred", "address_point", "available_time"]]

    # final return order (explicit)
    return labors_reassign_df, labors_frozen_df,  moves_frozen_df, driver_states_df


def compute_disruption_factor(
    new_labors: pd.DataFrame,
    return_value: str = 'number',
) -> float:
    """
    Computes how many labors kept the same assigned driver between old and new plans.

    Vectorized version (no Python loops).
    """
    # Keep only relevant columns & drop NAs
    # old = old_labors[['labor_id', 'old_assigned_driver']]
    filtered_labors = new_labors.dropna(subset=['old_assigned_driver'])

    # # Merge on labor_id to align old vs new
    # merged = old.merge(new, on='labor_id', how='inner', suffixes=('_old', '_new'))

    # Compare driver assignments
    same_driver_mask = filtered_labors['assigned_driver'] == filtered_labors['old_assigned_driver']
    factor = same_driver_mask.sum()

    # Return count or ratio
    if return_value == 'number':
        return int(factor)
    else:
        total = len(filtered_labors)
        return float(factor / total) if total > 0 else 0.0


def rename_old_solution_columns(labors_df):
    rename_columns = ['assigned_driver', 'actual_start', 'actual_end']
    for col in rename_columns:
        labors_df[f'old_{col}'] = labors_df[col]
    
    return labors_df


def prep_dfs_for_REACT(
    labors_dynamic_df,
    labors_algo_df,
    moves_algo_df,
    city,
    fecha
):
    # dynamic labors for this city/date
    labors_dynamic_filtered_df = filter_dynamic_df(
        labors_dynamic_df=labors_dynamic_df,
        city=city,
        fecha=fecha
    )

    # working copies for dynamic solution for this city/date
    labors_algo_dynamic_filt_df = flexible_filter(
        labors_algo_df,
        city=city,
        schedule_date=fecha
    ).copy().reset_index(drop=True)

    moves_algo_dynamic_filt_df = flexible_filter(
        moves_algo_df,
        city=city,
        schedule_date=fecha
    ).copy().reset_index(drop=True)

    return labors_dynamic_filtered_df, labors_algo_dynamic_filt_df, moves_algo_dynamic_filt_df