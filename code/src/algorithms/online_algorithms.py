import pandas as pd
import numpy as np

from datetime import datetime, timedelta, time

import random

from typing import Tuple, Optional, Dict, Any, List, Union, Callable

from src.utils.filtering import flexible_filter
from src.data.distance_utils import distance


''' Batching functions '''
def get_batches_for_date(
    labors_df_date: pd.DataFrame,
    batch_interval_minutes: int,
    start_of_day_str: str = "07:00",
    end_of_day: pd.Timestamp = None
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Build list of batches for a single date DataFrame (labors for that city/date).
    - If batch_interval_minutes == 0 => return per-service batches ordered by created_at (REACT).
    - First batch is [00:00, start_of_day) (so orders created < start_of_day belong to first batch).
    Returns list of (batch_start, batch_end, batch_df).
    """
    if labors_df_date is None or labors_df_date.empty:
        return []

    # Ensure created_at is tz-aware or naive consistently
    df = labors_df_date.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    the_date = df["created_at"].dt.normalize().iloc[0]  # midnight of that day
    start_of_day = datetime.combine(the_date, datetime.strptime(start_of_day_str, "%H:%M").time())

    # Decide end_of_day: either provided or max created_at + small buffer
    if end_of_day is None:
        max_created = df["created_at"].max()
        # if max_created is same day, end_of_day set to max_created + 1s (so last batch includes it)
        end_of_day_ts = max_created + timedelta(seconds=1)
    else:
        end_of_day_ts = pd.to_datetime(end_of_day)

    batches = []

    # CASE: batch_interval == 0 => REACT behavior: each service separately by created_at order
    if batch_interval_minutes == 0:
        # group by service_id preserving order of creation
        for _, group in df.sort_values("created_at").groupby("service_id"):
            created_min = group["created_at"].min()
            # definition: decision_time = created_at (process immediately) -> emulate old REACT's created_at + freeze logic elsewhere
            batches.append((created_min, created_min, group.copy().reset_index(drop=True)))
        return batches

    # Build first batch: [00:00, start_of_day)
    tz = "America/Bogota"

    batch0_start = pd.Timestamp(the_date)
    batch0_end = pd.Timestamp(start_of_day)

    if batch0_start.tzinfo is None:
        batch0_start = batch0_start.tz_localize(tz)
    else:
        batch0_start = batch0_start.tz_convert(tz)

    if batch0_end.tzinfo is None:
        batch0_end = batch0_end.tz_localize(tz)
    else:
        batch0_end = batch0_end.tz_convert(tz)


    batch0_df = df[(df["created_at"] >= batch0_start) & (df["created_at"] < batch0_end)].copy()
    batches.append((batch0_start, batch0_end, batch0_df.reset_index(drop=True)))

    # Build subsequent rolling windows starting at start_of_day with size batch_interval_minutes
    cursor = batch0_end
    interval = timedelta(minutes=batch_interval_minutes)
    while cursor < end_of_day_ts:
        batch_start = cursor
        batch_end = cursor + interval
        # include orders with created_at in (batch_start, batch_end] - as agreed earlier
        batch_df = df[(df["created_at"] > batch_start) & (df["created_at"] <= batch_end)].copy()
        batches.append((batch_start, batch_end, batch_df.reset_index(drop=True)))
        cursor = batch_end

    # Might include some empty batches; calling code can skip those with zero services or we keep them for metrics
    return batches


def attach_service_batch_to_reassign(labors_reassign_df: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the whole batch (possibly multiple services) to the reassign df.
    Ensures map_start_point / map_end_point are present.
    Returns a new expanded DataFrame (does not modify inputs).
    """
    base = labors_reassign_df.copy() if labors_reassign_df is not None else pd.DataFrame()
    batch_copy = batch_df.copy()

    # create mapping columns if missing (these are used in run_assignment_algorithm / pipeline)
    if "map_start_point" not in batch_copy.columns and "start_address_point" in batch_copy.columns:
        batch_copy["map_start_point"] = batch_copy["start_address_point"]
    if "map_end_point" not in batch_copy.columns and "end_address_point" in batch_copy.columns:
        batch_copy["map_end_point"] = batch_copy["end_address_point"]

    # concat and reset index
    expanded = pd.concat([base, batch_copy], ignore_index=True).reset_index(drop=True)
    return expanded


''' Filtering functions '''
def filter_dynamic_df(labors_dynamic_df, city, fecha):
    labors_dynamic_filtered_df = flexible_filter(
        labors_dynamic_df,
        city=city,
        schedule_date=fecha
        ).sort_values(['created_at', 'schedule_date', 'labor_start_date']).reset_index(drop=True)

    return labors_dynamic_filtered_df


# ''' --------- INSERT algorithm ---------'''
# def commit_labor_insertion(
#     labors_algo_df: pd.DataFrame,
#     moves_algo_df: pd.DataFrame,
#     new_labor: pd.Series,
#     directorio_hist_df: pd.DataFrame,
#     unassigned_services: list,
#     drivers,
#     city: str,
#     fecha: str,
#     distance_method: str,
#     dist_dict: dict,
#     vehicle_transport_speed: float,
#     alfred_speed: float,
#     tiempo_alistar: float,
#     tiempo_finalizacion: float,
#     tiempo_gracia: float,
#     early_buffer: float,
#     selection_mode: str = 'min_total_distance',
#     forced_start_time = None,
#     **kwargs
# ):
#     candidate_insertions = []

#     for driver in drivers:
#         labors_driver_df, moves_driver_df = filter_dfs_for_insertion(
#             labors_algo_df=labors_algo_df,
#             moves_algo_df=moves_algo_df,
#             city=city,
#             fecha=fecha,
#             driver=driver,
#             created_at=new_labor["created_at"],
#         )

#         # kwargs['city_code'] = city

#         feasible, infeasible_log, insertion_plan = evaluate_driver_feasibility(
#             new_labor=new_labor,
#             driver=driver,
#             moves_driver_df=moves_driver_df,
#             directory_df=directorio_hist_df,
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             ALFRED_SPEED=alfred_speed,
#             VEHICLE_TRANSPORT_SPEED=vehicle_transport_speed,
#             TIEMPO_ALISTAR=tiempo_alistar,
#             TIEMPO_FINALIZACION=tiempo_finalizacion,
#             TIEMPO_GRACIA=tiempo_gracia,
#             EARLY_BUFFER=early_buffer,
#             forced_start_time=forced_start_time,
#             city=city,
#             **kwargs
#         )

#         if feasible:
#             candidate_insertions.append((driver, insertion_plan))

#     if len(candidate_insertions) == 0:
#         unassigned_services.append(new_labor)
#         return False, labors_algo_df, moves_algo_df, None, None, unassigned_services

#     selected_driver, insertion_point, selection_df = get_best_insertion(
#         candidate_insertions, selection_mode=selection_mode, random_state=None
#     )

#     labors_algo_dynamic_df, moves_algo_dynamic_df, curr_end_time, curr_end_pos = commit_new_labor_insertion(
#             labors_df=labors_algo_df,
#             moves_df=moves_algo_df,
#             insertion_plan=insertion_point,
#             new_labor=new_labor,
#         )

#     return True, labors_algo_dynamic_df, moves_algo_dynamic_df, curr_end_time, curr_end_pos, unassigned_services

    

# ''' INSERT algorithm auxiliary functions '''
# def evaluate_driver_feasibility(
#     new_labor: pd.Series,
#     driver:float,
#     moves_driver_df: pd.DataFrame,
#     directory_df: pd.DataFrame,
#     distance_method: str,
#     dist_dict: dict,
#     ALFRED_SPEED: float,
#     VEHICLE_TRANSPORT_SPEED: float,
#     TIEMPO_ALISTAR: float,
#     TIEMPO_FINALIZACION: float,
#     TIEMPO_GRACIA: float,
#     EARLY_BUFFER: int,
#     forced_start_time = None,
#     **kwargs
# ) -> Tuple[bool, str, Optional[dict]]:
#     """
#     Determine whether a driver can insert a new labor into their route *and* build a full insertion plan.

#     Returns:
#     --------
#     feasible : bool
#         Whether the insertion is possible.
#     reason : str
#         If infeasible, description of the reason.
#     insertion_plan : dict or None
#         A structured plan describing how the driver’s schedule would look after insertion.
#     """

#     feasible = False
#     infeasible_log = ''
#     insertion_plan = None

#     if forced_start_time:
#         new_start_time = forced_start_time
#     else:
#         new_start_time = new_labor['schedule_date']

#     # ---------------------------------------------------------------------------------------
#     # Case 1: Driver has no other labors assigned
#     # ---------------------------------------------------------------------------------------
#     if moves_driver_df.empty:
#         feasible, infeasible_log, insertion_plan = _direct_insertion_empty_driver(
#             new_labor=new_labor,
#             driver=driver,
#             directory_df=directory_df,
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             alfred_speed=ALFRED_SPEED,
#             VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
#             TIEMPO_ALISTAR=TIEMPO_ALISTAR,
#             TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
#             EARLY_BUFFER=EARLY_BUFFER,
#             **kwargs
#         )
#         return feasible, infeasible_log, insertion_plan


#     # ---------------------------------------------------------------------------------------
#     # Case 2: insertion before first labor
#     # ---------------------------------------------------------------------------------------
#     is_before_first_labor = _is_creation_before_first_labor(
#         moves_driver_df,
#         directory_df,
#         driver
#     )
#     if is_before_first_labor:
#         # --- Skip if next labor starts before new labor’s starts moving
#         if new_labor['schedule_date'] <= moves_driver_df.loc[2,'schedule_date']:
#             feasible, infeasible_log, insertion_plan = _evaluate_and_execute_insertion_before_first_labor(
#                 new_labor=new_labor,
#                 moves_driver_df=moves_driver_df,
#                 driver=driver,
#                 directory_df=directory_df,
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
#                 alfred_speed=ALFRED_SPEED,
#                 tiempo_alistar=TIEMPO_ALISTAR,
#                 tiempo_finalizacion=TIEMPO_FINALIZACION,
#                 tiempo_gracia=TIEMPO_GRACIA,
#                 early_buffer=EARLY_BUFFER,
#                 **kwargs
#             )

#             return feasible, infeasible_log, insertion_plan

#     labor_iter = 2
           
#     # ---------------------------------------------------------------------------------------
#     # Case 3: Insertion between existing labors
#     # ---------------------------------------------------------------------------------------
#     n_rows = len(moves_driver_df)

#     while labor_iter < len(moves_driver_df):
#         # --- Break if the current labor is the next labor
#         if labor_iter + 3 > len(moves_driver_df):
#                 break
        
#         # --- Context: previous labor & next labor
#         curr_end_time, curr_end_pos, next_start_time, next_start_pos = \
#             _get_driver_context(moves_driver_df, labor_iter)
        
#         # --- Skip if next labor starts before new labor’s schedule
#         if next_start_time < new_start_time:
#             labor_iter += 3
#             continue

#         curr_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]

#         next_labor = moves_driver_df.loc[labor_iter + 3,:]
#         next_labor_id_candidate = next_labor["labor_id"]

#         # --- Compute arrival to new service
#         new_arrival_time, new_dist_to_reach, new_time_to_reach = _compute_arrival_to_next_labor(
#             current_end_time=curr_end_time,
#             current_end_pos=curr_end_pos,
#             target_pos=new_labor['start_address_point'],
#             speed=ALFRED_SPEED,
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             **kwargs
#         )

#         # --- Break if driver wouldn't arrive on time to new labor
#         if new_arrival_time > new_start_time + timedelta(TIEMPO_GRACIA):
#             infeasible_log = "Driver would not arrive on time to the new labor."
#             break

#         # --- Adjust if arriving too early (driver waits)
#         new_real_arrival_time, new_move_start_time = _adjust_for_early_arrival(
#             would_arrive_at=new_arrival_time,
#             travel_time=new_time_to_reach,
#             schedule_date=new_start_time, 
#             early_buffer=EARLY_BUFFER
#         )

#         # --- Compute new labor end time
#         new_finish_time, new_finish_pos, new_duration, new_distance = \
#             _compute_service_end_time(
#                 arrival_time=new_real_arrival_time,
#                 start_pos=new_labor['start_address_point'],
#                 end_pos=new_labor['end_address_point'],
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 vehicle_speed=VEHICLE_TRANSPORT_SPEED,
#                 prep_time=TIEMPO_ALISTAR,
#                 finish_time=TIEMPO_FINALIZACION,
#                 **kwargs
#             )

#         # --- Check feasibility with next scheduled labor
#         feasible_next, next_real_arrival_time, next_move_start_time, next_dist_to_reach, next_time_to_reach = \
#             _can_reach_next_labor(
#                 new_finish_time=new_finish_time, 
#                 new_finish_pos=new_finish_pos,
#                 next_start_time=next_start_time, 
#                 next_start_pos=next_start_pos,
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 driver_speed=ALFRED_SPEED, 
#                 grace_time=TIEMPO_GRACIA,
#                 early_buffer=EARLY_BUFFER,
#                 **kwargs
#             )

#         if not feasible_next:
#             infeasible_log = (
#                 "Driver would not make it to the next scheduled labor in time "
#                 "if new labor is inserted."
#             )
#             break
        
#         next_finish_time, next_end_pos, next_duration, next_distance = \
#             _compute_service_end_time(
#                 arrival_time=next_real_arrival_time,
#                 start_pos=next_labor['start_point'],
#                 end_pos=next_labor['end_point'],
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 vehicle_speed=VEHICLE_TRANSPORT_SPEED,
#                 prep_time=TIEMPO_ALISTAR,
#                 finish_time=TIEMPO_FINALIZACION,
#                 **kwargs
#             ) 

#         # --- DOWNSTREAM FEASIBILITY CHECK
#         downstream_ok, downstream_shifts = _simulate_downstream_shift(
#             moves_driver_df=moves_driver_df,
#             driver=driver,
#             start_idx=labor_iter+3,
#             start_time=next_finish_time,
#             start_pos=next_end_pos,
#             previous_start_time=moves_driver_df.loc[labor_iter+3, 'actual_end'],
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             ALFRED_SPEED=ALFRED_SPEED,
#             VEHICLE_TRANSPORT_SPEED=VEHICLE_TRANSPORT_SPEED,
#             TIEMPO_ALISTAR=TIEMPO_ALISTAR,
#             TIEMPO_FINALIZACION=TIEMPO_FINALIZACION,
#             TIEMPO_GRACIA=TIEMPO_GRACIA,
#             EARLY_BUFFER=EARLY_BUFFER,
#             **kwargs
#         )

#         if not downstream_ok:
#             infeasible_log = "Downstream labors become infeasible after insertion."
#             break

#         # --- FEASIBLE INSERTION FOUND
#         feasible = True

#         # Precompute updated moves for the new labor and next labor
#         new_moves = _build_moves_block(
#             labor=new_labor,
#             driver=driver,
#             start_free_time=curr_end_time,
#             start_move_time=new_move_start_time,
#             move_duration=new_time_to_reach,
#             move_distance=new_dist_to_reach,
#             arrival_time=new_real_arrival_time,
#             labor_distance=new_distance,
#             labor_duration=new_duration,
#             finish_time=new_finish_time,
#             start_pos=curr_end_pos,
#             start_address=new_labor["start_address_point"],
#             end_address=new_labor["end_address_point"],
#             block_label='new'
#         )

#         next_moves = _build_moves_block(
#             labor=next_labor,
#             driver=driver,
#             start_free_time=new_finish_time,
#             start_move_time=next_move_start_time,
#             move_duration=next_time_to_reach,
#             move_distance=next_dist_to_reach,
#             arrival_time=next_real_arrival_time,
#             labor_distance=next_distance,
#             labor_duration=next_duration,
#             finish_time=next_finish_time,
#             start_pos=new_finish_pos,
#             start_address=next_labor['start_point'],
#             end_address=next_labor['end_point'],
#             block_label='next'
#         )


#         # Build insertion plan dict (the full blueprint)
#         insertion_plan = {
#             "driver_id": driver,
#             "new_labor_id": new_labor["labor_id"],
#             "prev_labor_id": curr_labor_id,
#             "next_labor_id": next_labor_id_candidate,
#             'dist_to_new': new_dist_to_reach,
#             "dist_to_next": next_dist_to_reach,
#             "new_moves": new_moves,
#             'next_moves': next_moves,
#             "downstream_shifts": downstream_shifts,
#         }

#         break  # stop after finding first feasible insertion
    

#     # ---------------------------------------------------------------------------------------
#     # Case 4: Insert labor at the end of the shift
#     # ---------------------------------------------------------------------------------------
#     if not feasible and infeasible_log == '' and labor_iter >= n_rows - 3:
#         # feasible = True
#         prev_labor_id = moves_driver_df.loc[labor_iter, "labor_id"]

#         # Compute basic timing info
#         curr_end_time = moves_driver_df.loc[labor_iter, "actual_end"]
#         curr_end_pos = moves_driver_df.loc[labor_iter, "end_point"]
#         would_arrive_at, dist_to_new, travel_time_to_new = _compute_arrival_to_next_labor(
#             curr_end_time,
#             curr_end_pos,
#             new_labor["start_address_point"],
#             ALFRED_SPEED,
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             **kwargs
#         )

#         # Check if driver would arrive on time to the new labor
#         if would_arrive_at > new_labor["latest_arrival_time"]:
#             infeasible_log = "Driver would arrive too late to the new labor."
#             return feasible, infeasible_log, insertion_plan


#         # Feasible insertion
#         feasible = True

#         real_arrival_time, move_start_time = _adjust_for_early_arrival(
#             would_arrive_at=would_arrive_at,
#             travel_time = travel_time_to_new,
#             schedule_date=new_labor["schedule_date"], 
#             early_buffer=EARLY_BUFFER
#         )

#         finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
#             _compute_service_end_time(
#                 arrival_time=real_arrival_time,
#                 start_pos=new_labor["start_address_point"],
#                 end_pos=new_labor["end_address_point"],
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 vehicle_speed=VEHICLE_TRANSPORT_SPEED,
#                 prep_time=TIEMPO_ALISTAR,
#                 finish_time=TIEMPO_FINALIZACION,
#                 **kwargs
#             )

#         new_moves = _build_moves_block(
#             labor=new_labor,
#             driver=driver,
#             start_free_time=curr_end_time,
#             start_move_time=move_start_time,
#             move_duration=travel_time_to_new,
#             move_distance=dist_to_new,
#             arrival_time=real_arrival_time,
#             labor_distance=new_labor_distance,
#             labor_duration=new_labor_duration,
#             finish_time=finish_new_labor_time,
#             start_pos=curr_end_pos,
#             start_address=new_labor["start_address_point"],
#             end_address=new_labor["end_address_point"],
#             block_label='new'
#         )

#         insertion_plan = {
#             "driver_id": driver,
#             "new_labor_id": new_labor["labor_id"],
#             "prev_labor_id": prev_labor_id,
#             "next_labor_id": None,
#             "dist_to_new_service": dist_to_new,
#             "dist_to_next_labor": 0,
#             "new_moves":  new_moves,
#             'next_moves': None,
#             "downstream_shifts": [],
#         }


#     # ============================================================
#     # FINAL RETURN
#     # ============================================================

#     return feasible, infeasible_log, insertion_plan


# def evaluate_driver_feasibility_subsequent_labor():
#     pass


# def _direct_insertion_empty_driver(
#     new_labor: pd.Series,
#     driver:float,
#     directory_df:pd.DataFrame,
#     distance_method: str,
#     dist_dict: dict,
#     alfred_speed: float,
#     VEHICLE_TRANSPORT_SPEED: float,
#     TIEMPO_ALISTAR: float,
#     TIEMPO_FINALIZACION: float,
#     EARLY_BUFFER: float,
#     **kwargs
# ) -> Tuple:
#     home_pos = directory_df.loc[directory_df['alfred'] == driver, 'address_point'].iloc[0]

#     # 1. Compute start and end times for the new labor (driver starts “from base” or assumed start)
#     start_time = new_labor["schedule_date"]
#     arrival_time = start_time - timedelta(minutes=EARLY_BUFFER)  # no travel, assume immediate availability

#     _, dist_to_new, travel_time_to_new = _compute_arrival_to_next_labor(
#         current_end_time=arrival_time, 
#         current_end_pos=home_pos,
#         target_pos=new_labor['start_address_point'],
#         speed=alfred_speed,
#         distance_method=distance_method,
#         dist_dict=dist_dict,
#         **kwargs
#     )
    
#     start_move_time = arrival_time - timedelta(minutes=travel_time_to_new)
    
#     finish_new_labor_time, finish_new_labor_pos, new_labor_duration, new_labor_distance = \
#         _compute_service_end_time(
#         arrival_time=arrival_time,
#         start_pos=new_labor["start_address_point"],
#         end_pos=new_labor["end_address_point"],
#         distance_method=distance_method,
#         dist_dict=dist_dict,
#         vehicle_speed=VEHICLE_TRANSPORT_SPEED,
#         prep_time=TIEMPO_ALISTAR,
#         finish_time=TIEMPO_FINALIZACION,
#         **kwargs
#     )

#     # Precompute updated moves for the new labor and next labor
#     new_moves = _build_moves_block(
#         labor=new_labor,
#         driver=driver,
#         start_free_time=None,
#         start_move_time=start_move_time,
#         move_duration=travel_time_to_new,
#         move_distance=dist_to_new,
#         arrival_time=arrival_time,
#         labor_distance=new_labor_distance,
#         labor_duration=new_labor_duration,
#         finish_time=finish_new_labor_time,
#         start_pos=home_pos,
#         start_address=new_labor["start_address_point"],
#         end_address=new_labor["end_address_point"],
#         block_label='new'
#     )

#     # 3. Prepare minimal insertion plan
#     insertion_plan = {
#         "driver_id": driver,
#         "new_labor_id": new_labor["labor_id"],
#         "prev_labor_id": None,
#         "next_labor_id": None,
#         "dist_to_new": 0,
#         "dist_to_next": 0,
#         "new_moves": new_moves,
#         "downstream_shifts": [],
#     }

#     return True, "", insertion_plan


# def _is_creation_before_first_labor(
#     moves_driver_df: pd.DataFrame,
#     directory_df: pd.DataFrame,
#     driver: str,
# ) -> bool:
#     """
#     Determine if a new labor can be inserted *before* the driver's first scheduled labor.

#     Logic:
#     -------
#     If the first movement (the '_free' segment) starts from the same location
#     as the driver's registered starting address in the directory, we can consider
#     the driver as 'at base' before any assigned labor → insertion before first labor is valid.

#     Returns
#     -------
#     bool : True if insertion before first labor is possible.
#     """
#     # --- Extract driver's base location from directory ---
#     try:
#         driver_base = (
#             directory_df.loc[directory_df["alfred"] == driver, "address_point"]
#             .dropna()
#             .iloc[0]
#         )
#     except IndexError:
#         # Driver not found in directory → cannot determine base position
#         return False

#     first_start = str(moves_driver_df.iloc[0]["start_point"]).strip()
#     driver_base = str(driver_base).strip()

#     # --- Compare normalized positions ---
#     return first_start == driver_base


# def _evaluate_and_execute_insertion_before_first_labor(
#     new_labor, 
#     moves_driver_df,
#     driver: str,
#     directory_df: pd.DataFrame,
#     distance_method: str,
#     dist_dict: dict,
#     vehicle_transport_speed: float,
#     alfred_speed: float,
#     tiempo_alistar: float,
#     tiempo_finalizacion: float,
#     tiempo_gracia: float,
#     early_buffer: float,
#     **kwargs
# ) -> Tuple:
    
#     next_labor = moves_driver_df.loc[2,:]
#     home_pos = directory_df.loc[directory_df['alfred'] == driver, 'address_point'].iloc[0]

#     arrival_time = new_labor["schedule_date"] - timedelta(minutes=early_buffer)
    
#     _, new_dist_to_reach, new_time_to_reach = _compute_arrival_to_next_labor(
#         current_end_time=arrival_time, 
#         current_end_pos=home_pos,
#         target_pos=new_labor['start_address_point'],
#         speed=alfred_speed,
#         distance_method=distance_method,
#         dist_dict=dist_dict,
#         **kwargs)
    
#     new_move_start_time = arrival_time - timedelta(minutes=new_time_to_reach)
#     next_labor_id_candidate = moves_driver_df.loc[0, 'labor_id']

#     # Compute new labor end time
#     new_labor_finish_time, new_labor_finish_pos, new_duration, new_distance = \
#         _compute_service_end_time(
#             arrival_time=arrival_time,
#             start_pos=new_labor['start_address_point'],
#             end_pos=new_labor['end_address_point'],
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             vehicle_speed=vehicle_transport_speed,
#             prep_time=tiempo_alistar,
#             finish_time=tiempo_finalizacion,
#             **kwargs
#         )
    
#     # --- Check feasibility with next scheduled labor
#     feasible_next, next_real_arrival_time, next_move_start_time, next_dist_to_reach, next_time_to_reach = \
#         _can_reach_next_labor(
#             new_finish_time=new_labor_finish_time, 
#             new_finish_pos=new_labor_finish_pos,
#             next_start_time=next_labor['schedule_date'],
#             next_start_pos=next_labor['start_point'],
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             driver_speed=alfred_speed, 
#             grace_time=tiempo_gracia,
#             early_buffer=early_buffer,
#             **kwargs
#         )

#     if not feasible_next:
#         infeasible_log = (
#             "Driver would not make it to the next scheduled labor in time "
#             "if new labor is inserted."
#         )
#         return False, infeasible_log, None

#     next_finish_time, next_end_pos, next_duration, next_distance = \
#     _compute_service_end_time(
#         arrival_time=next_real_arrival_time,
#         start_pos=next_labor['start_point'],
#         end_pos=next_labor['end_point'],
#         distance_method=distance_method,
#         dist_dict=dist_dict,
#         vehicle_speed=vehicle_transport_speed,
#         prep_time=tiempo_alistar,
#         finish_time=tiempo_finalizacion,
#         **kwargs
#     ) 
        
#     # Now simulate downstream shift from the FIRST labor
#     downstream_ok, downstream_shifts = _simulate_downstream_shift(
#         moves_driver_df=moves_driver_df,
#         driver=driver,
#         start_idx=2,
#         start_time=next_real_arrival_time,
#         start_pos=next_labor['start_point'],
#         previous_start_time=next_labor['actual_end'],
#         distance_method=distance_method,
#         dist_dict=dist_dict,
#         ALFRED_SPEED=alfred_speed,
#         VEHICLE_TRANSPORT_SPEED=vehicle_transport_speed,
#         TIEMPO_ALISTAR=tiempo_alistar,
#         TIEMPO_FINALIZACION=tiempo_finalizacion,
#         TIEMPO_GRACIA=tiempo_gracia,
#         EARLY_BUFFER=early_buffer,
#         **kwargs
#     )

#     if not downstream_ok:
#         return False, "Downstream labors become infeasible after early insertion.", None
    
#     new_moves = _build_moves_block(
#         labor=new_labor,
#         driver=driver,
#         start_free_time=None,
#         start_move_time=new_move_start_time,
#         move_duration=new_time_to_reach,
#         move_distance=new_dist_to_reach,
#         arrival_time=arrival_time,
#         labor_distance=new_distance,
#         labor_duration=new_duration,
#         finish_time=new_labor_finish_time,
#         start_pos=home_pos,
#         start_address=new_labor["start_address_point"],
#         end_address=new_labor["end_address_point"],
#         block_label='new'
#     )

#     next_moves = _build_moves_block(
#         labor=next_labor,
#         driver=driver,
#         start_free_time=new_labor_finish_time,
#         start_move_time=next_move_start_time,
#         move_duration=next_time_to_reach,
#         move_distance=next_dist_to_reach,
#         arrival_time=next_real_arrival_time,
#         labor_distance=next_distance,
#         labor_duration=next_duration,
#         finish_time=next_finish_time,
#         start_pos=new_labor_finish_pos,
#         start_address=next_labor['start_point'],
#         end_address=next_labor['end_point'],
#         block_label='next'
#     )

#     # Build insertion plan same as normal
#     insertion_plan = {
#         "driver_id": driver,
#         "new_labor_id": new_labor["labor_id"],
#         "prev_labor_id": None,
#         "next_labor_id": next_labor_id_candidate,
#         "dist_to_new": new_dist_to_reach,
#         "dist_to_next": next_dist_to_reach,
#         "new_moves": new_moves,
#         'next_moves': next_moves,
#         "downstream_shifts": downstream_shifts,
#     }

#     return True, "", insertion_plan


# def _build_moves_block(
#     labor: pd.Series,
#     driver: str,
#     start_free_time: datetime,
#     start_move_time: datetime,
#     move_duration: float,
#     move_distance: float,
#     arrival_time: datetime,
#     labor_distance: float,
#     labor_duration: float,
#     finish_time: datetime,
#     start_pos: Optional[str],
#     start_address: str,
#     end_address: str,
#     block_label: str = "new"
# ) -> pd.DataFrame:
#     """
#     Generic builder for the standardized triplet of moves rows (FREE_TIME, DRIVER_MOVE, LABOR)
#     corresponding to any labor insertion or rescheduling.

#     This replaces _build_new_moves_block and _build_next_moves_block.

#     Parameters
#     ----------
#     labor : pd.Series
#         Labor row (either the new labor or the next labor being shifted).
#     driver : str
#         Driver identifier.
#     start_free_time, start_move_time, arrival_time, finish_time : datetime
#         Timestamps for start/end of free, move, and labor segments.
#     move_duration_min, move_distance : float
#         Travel metrics for the driver move segment.
#     labor_distance, labor_duration : float
#         Work metrics for the labor itself.
#     start_pos, start_address, end_address : str
#         Spatial start/end references.
#     block_label : str
#         Optional tag (e.g. 'new', 'next', 'shifted') for readability/logging.

#     Returns
#     -------
#     pd.DataFrame
#         3-row standardized DataFrame ready for concatenation into moves_df.
#     """

#     # ---- Extract base info ----
#     service_id = labor.get("service_id", None)
#     labor_id = labor["labor_id"]
#     schedule_date = labor["schedule_date"]
#     city = labor["city"]
#     date = labor.get("date", schedule_date.date())

#     # ---- Compute timing sanity ----
#     move_end_time = arrival_time
#     if start_free_time is None or start_free_time > start_move_time:
#         start_free_time = end_free_time = start_move_time
#     else:
#         end_free_time = start_move_time

#     rows = []

#     # ============================================================
#     # 1️⃣ FREE TIME ROW
#     # ============================================================
#     rows.append({
#         "service_id": service_id,
#         "labor_id": labor_id,
#         "labor_context_id": f"{labor_id}_free",
#         "labor_name": "FREE_TIME",
#         "labor_category": "FREE_TIME",
#         "assigned_driver": driver,
#         "schedule_date": schedule_date,
#         "actual_start": start_free_time,
#         "actual_end": end_free_time,
#         "start_point": start_pos if start_pos is not None else start_address,
#         "end_point": start_pos if start_pos is not None else start_address,
#         "distance_km": 0.0,
#         "duration_min": round((end_free_time - start_free_time).total_seconds() / 60.0,1),
#         "city": city,
#         "date": date,
#     })

#     # ============================================================
#     # 2️⃣ DRIVER MOVE ROW
#     # ============================================================
#     rows.append({
#         "service_id": service_id,
#         "labor_id": labor_id,
#         "labor_context_id": f"{labor_id}_move",
#         "labor_name": "DRIVER_MOVE",
#         "labor_category": "DRIVER_MOVE",
#         "assigned_driver": driver,
#         "schedule_date": schedule_date,
#         "actual_start": start_move_time,
#         "actual_end": move_end_time,
#         "start_point": start_pos if start_pos is not None else start_address,
#         "end_point": start_address,
#         "distance_km": move_distance,
#         "duration_min": round(move_duration,1),
#         "city": city,
#         "date": date,
#     })

#     # ============================================================
#     # 3️⃣ LABOR ROW
#     # ============================================================
#     rows.append({
#         "service_id": service_id,
#         "labor_id": labor_id,
#         "labor_context_id": f"{labor_id}_labor",
#         "labor_name": labor["labor_name"],
#         "labor_category": labor["labor_category"],
#         "assigned_driver": driver,
#         "schedule_date": schedule_date,
#         "actual_start": arrival_time,
#         "actual_end": finish_time,
#         "start_point": start_address,
#         "end_point": end_address,
#         "distance_km": labor_distance,
#         "duration_min": round(labor_duration,1),
#         "city": city,
#         "date": date,
#     })

#     # ---- Final alignment ----
#     cols = [
#         "service_id", "labor_id", "labor_context_id", "labor_name",
#         "labor_category", "assigned_driver", "schedule_date", "actual_start",
#         "actual_end", "start_point", "end_point", "distance_km",
#         "duration_min", "city", "date"
#     ]
#     df = pd.DataFrame(rows)[cols]
#     df.attrs["block_label"] = block_label
#     return df


# def _simulate_downstream_shift(
#     moves_driver_df: pd.DataFrame,
#     driver: str,
#     start_idx: int,
#     start_time: datetime,
#     start_pos: str,
#     previous_start_time: datetime,
#     distance_method:str,
#     dist_dict:dict,
#     ALFRED_SPEED: float,
#     VEHICLE_TRANSPORT_SPEED: float,
#     TIEMPO_ALISTAR: float,
#     TIEMPO_FINALIZACION: float,
#     TIEMPO_GRACIA: float,
#     EARLY_BUFFER: int,
#     **kwargs
#     ):
#     """
#     Simulates how downstream labors are affected after inserting a new labor
#     (either before the first labor or between two existing labors).
#     """
#     # --- Initialization ---
#     donwstream_shifts = []     # Store info of shifts per labor
#     feasible = True            # Default to feasible until proven otherwise

#     # --- Early exit condition: THe labor after the insertion didn't shift (it's end time)
#     shift = (start_time - previous_start_time).total_seconds() / 60
#     if abs(shift) < 1:
#         # No temporal shift → no downstream propagation
#         return True, []

#     # # --- Prepare iteration variables ---
#     curr_end_time = start_time
#     curr_end_pos = start_pos

#     labor_rows = moves_driver_df[
#     moves_driver_df["labor_context_id"].astype(str).str.endswith("_labor")
#     ].index.tolist()

#     # Find current labor position among them
#     try:
#         start_pos = labor_rows.index(start_idx)
#     except ValueError:
#         # Fallback in case start_idx was a move/free index
#         start_pos = 0

#     # --- Iterate through all subsequent labors ---
#     for i in labor_rows[start_pos + 1:]:
#     # for i in range(start_idx + 3, len(moves_driver_df), 3):  # Move by triplets
#         # iterate naturally over downstream labor indices
#         next_labor = moves_driver_df.loc[i,:]

#         next_start_time = next_labor['schedule_date']
#         next_start_pos = next_labor['start_point']
#         next_end_pos = next_labor['end_point']

#         # --- Compute arrival to new service
#         next_arrival_time, next_dist_to_reach, next_time_to_reach = _compute_arrival_to_next_labor(
#             current_end_time=curr_end_time,
#             current_end_pos=curr_end_pos, 
#             target_pos=next_start_pos,
#             speed=ALFRED_SPEED,
#             distance_method=distance_method,
#             dist_dict=dist_dict,
#             **kwargs
#         )

#         # --- Break if driver wouldn't arrive on time to new labor
#         next_max_arrival_time = next_start_time + timedelta(minutes=TIEMPO_GRACIA)
#         if next_arrival_time > next_max_arrival_time:
#             return False, []

#         # --- Adjust if arriving too early (driver waits)
#         next_real_arrival_time, next_move_start_time = _adjust_for_early_arrival(
#             would_arrive_at=next_arrival_time, 
#             travel_time= next_dist_to_reach,
#             schedule_date=next_start_time,
#             early_buffer=EARLY_BUFFER
#         )
        
#         # --- Compute new labor end time
#         next_finish_time, next_finish_pos, next_duration, next_distance = \
#             _compute_service_end_time(
#                 arrival_time=next_real_arrival_time,
#                 start_pos=next_start_pos,
#                 end_pos=next_end_pos,
#                 distance_method=distance_method,
#                 dist_dict=dist_dict,
#                 vehicle_speed=VEHICLE_TRANSPORT_SPEED,
#                 prep_time=TIEMPO_ALISTAR,
#                 finish_time=TIEMPO_FINALIZACION,
#                 **kwargs
#             )
        
#         next_moves = _build_moves_block(
#             labor=next_labor,
#             driver=driver,
#             start_free_time=curr_end_time,
#             start_move_time=next_move_start_time,
#             move_duration=next_time_to_reach,
#             move_distance=next_dist_to_reach,
#             arrival_time=next_real_arrival_time,
#             labor_distance=next_distance,
#             labor_duration=next_duration,
#             finish_time=next_finish_time,
#             start_pos=curr_end_pos,
#             start_address=next_labor['start_point'],
#             end_address=next_labor['end_point'],
#             block_label='downstream'
#         )
        
#         # --- Update tracking variables ---
#         curr_end_time = next_finish_time
#         curr_end_pos = next_finish_pos

#         # --- Log downstream changes ---
#         donwstream_shifts.append(next_moves)

#         shift = (next_finish_time - next_labor['actual_end']).total_seconds() / 60
#         if abs(shift) < 1:
#             break

#     # --- Return final result ---
#     return feasible, donwstream_shifts


# def _get_driver_context(
#     moves_driver_df: pd.DataFrame,
#     idx: int,
# ) -> Tuple:
#     """Return the current and next labor context for driver."""
#     # The most recent labor is still in place
#     curr_end_time = moves_driver_df.loc[idx, 'actual_end']
#     curr_end_pos = moves_driver_df.loc[idx, 'end_point']
#     next_start_time = moves_driver_df.loc[idx + 3, 'schedule_date']
#     next_start_pos = moves_driver_df.loc[idx + 3, 'start_point']
    
#     return curr_end_time, curr_end_pos, next_start_time, next_start_pos


# def _compute_arrival_to_next_labor(
#     current_end_time, 
#     current_end_pos, 
#     target_pos: str,
#     speed: float,
#     distance_method: str,
#     dist_dict: dict,
#     **kwargs
# ) -> Tuple:
#     """Compute when driver would arrive at the target position."""
#     dist, _ = distance(current_end_pos, target_pos, method=distance_method, dist_dict=dist_dict, **kwargs)
#     travel_time = dist / speed * 60
#     return current_end_time + timedelta(minutes=travel_time), dist, travel_time


# def _adjust_for_early_arrival(
#     would_arrive_at,
#     travel_time,
#     schedule_date, 
#     early_buffer: float = 30):
#     """
#     Adjusts arrival time if driver would arrive too early.
#     Ensures driver waits to arrive no earlier than (schedule_date - early_buffer).
#     """
#     earliest_allowed = schedule_date - timedelta(minutes=early_buffer)

#     real_arrival_time = max(would_arrive_at, earliest_allowed)
#     move_start_time =  real_arrival_time - timedelta(minutes=travel_time)

#     return real_arrival_time, move_start_time


# def _compute_service_end_time(
#     arrival_time, 
#     start_pos: str, 
#     end_pos: str,
#     distance_method: str,
#     dist_dict: dict, 
#     vehicle_speed: float, 
#     prep_time: float, 
#     finish_time,
#     **kwargs
# ) -> Tuple:
#     """Compute finish time and position of performing the new service."""
#     labor_distance, _ = distance(start_pos, end_pos, method=distance_method, dist_dict=dist_dict, **kwargs)
#     travel_time = labor_distance / vehicle_speed * 60
#     labor_total_duration = prep_time + travel_time + finish_time
#     finish_time = arrival_time + timedelta(minutes=labor_total_duration)
#     return finish_time, end_pos, labor_total_duration, labor_distance


# def _can_reach_next_labor(
#     new_finish_time, 
#     new_finish_pos: str,
#     next_start_time: datetime, 
#     next_start_pos: str,
#     distance_method: str,
#     dist_dict: dict, 
#     driver_speed: float, 
#     grace_time: float,
#     early_buffer: float,
#     **kwargs
# ) -> Tuple:
#     """Check if driver can arrive to next labor in time after finishing new service."""
#     dist, _ = distance(new_finish_pos, next_start_pos, method=distance_method, dist_dict=dist_dict, **kwargs)
#     travel_time = dist / driver_speed * 60

#     next_arrival = new_finish_time + timedelta(minutes=travel_time)
    
#     next_real_arrival, next_move_start_time = _adjust_for_early_arrival(
#         would_arrive_at=next_arrival,
#         travel_time=travel_time,
#         schedule_date=next_start_time,
#         early_buffer=early_buffer
#     )

#     feasible = next_real_arrival <= next_start_time + timedelta(minutes=grace_time)

#     return feasible, next_real_arrival, next_move_start_time, dist, travel_time


# ''' Non transport labor functions'''
# def commit_nontransport_labor_insertion(
#     labors_df: pd.DataFrame,
#     moves_df: pd.DataFrame,
#     new_labor: pd.Series,
#     start_pos: str,
#     start_time: datetime,
#     duration: float,
#     fecha: str
# ):
#     new_end_time = start_time + timedelta(minutes=duration)

#     labors_updated_df = labors_df.copy()
#     moves_updated_df = moves_df.copy()

#     # Add the new labor entry (complete row)
#     new_labor_row = new_labor.copy()
#     new_labor_id = new_labor_row["labor_id"]

#     new_labor_row["assigned_driver"] = ''
#     new_labor_row["actual_start"] = start_time
#     new_labor_row["actual_end"] = new_end_time
#     new_labor_row["dist_km"] = 0
#     new_labor_row["start_address_point"] = start_pos
#     new_labor_row["end_address_point"] = start_pos
#     new_labor_row["date"] = fecha
#     new_labor_row["n_drivers"] = np.nan  # Can be filled later

#     labors_updated_df = labors_updated_df[labors_updated_df["labor_id"] != new_labor_id]
#     labors_updated_df = pd.concat([labors_updated_df, pd.DataFrame([new_labor_row])], ignore_index=True)

#     return labors_updated_df, moves_updated_df, new_end_time


# def get_nontransport_labor_duration(
#     duraciones_df: pd.DataFrame,
#     city: str,
#     labor_type: str,
#     verbose: bool = False
# ) -> Union[float, int]:
#     """
#     Get the representative duration (p75_min) for a non-transport labor type in a city.

#     Behavior:
#       - If the (city, labor_type) exists in `duraciones_df`, returns that city's p75_min (first match).
#       - If not, computes the 75th percentile across *cities that do have this labor_type*.
#         For each city with at least one row for the labor_type we take that city's representative
#         p75_min (mean if multiple rows), then compute the 75th percentile across those city values.
#       - Raises FileNotFoundError if the labor_type is not present in any city.

#     Parameters
#     ----------
#     duraciones_df : pd.DataFrame
#         DataFrame containing at least columns ['city', 'labor_type', 'p75_min'].
#     city : str
#         City code to look up first.
#     labor_type : str
#         Labor type to search for.
#     verbose : bool
#         If True, print diagnostics.

#     Returns
#     -------
#     float or int
#         Duration value (same units as 'p75_min').
#     """
#     # Defensive checks
#     required_cols = {"city", "labor_type", "p75_min"}
#     missing = required_cols - set(duraciones_df.columns)
#     if missing:
#         raise ValueError(f"duraciones_df is missing required columns: {missing}")

#     # Try to find the direct match for city + labor_type
#     duration_filt_df = duraciones_df[
#         (duraciones_df["city"] == city) &
#         (duraciones_df["labor_type"] == labor_type) &
#         duraciones_df["p75_min"].notna()
#     ]

#     if not duration_filt_df.empty:
#         # If there are multiple rows, take the first (or you could take mean/median)
#         val = float(duration_filt_df.iloc[0]["p75_min"])
#         if verbose:
#             print(f"Found direct duration for city={city}, labor_type={labor_type}: {val}")
#         return val

#     # No direct match — compute 75th percentile across cities that have this labor_type
#     other = duraciones_df[
#         (duraciones_df["labor_type"] == labor_type) &
#         duraciones_df["p75_min"].notna()
#     ]

#     if other.empty:
#         raise FileNotFoundError(f"Labor type '{labor_type}' not found in any city.")

#     # Representative value per city (if multiple rows per city, take the mean of p75_min)
#     per_city = other.groupby("city", as_index=False)["p75_min"].mean()["p75_min"].values

#     # Compute 75th percentile across city representatives
#     p75_across_cities = float(np.percentile(per_city, 75))

#     if verbose:
#         print(
#             f"No direct duration for city={city}. "
#             f"Using 75th percentile across {len(per_city)} cities: {p75_across_cities}"
#         )

#     return p75_across_cities


# def get_drivers(labors_algo_df, directorio_hist_df, city, fecha, get_all=True):
#     if get_all:
#         directorio_hist_filtered_df = flexible_filter(
#                 directorio_hist_df, city=city, date=fecha
#             )
        
#         drivers = (
#             directorio_hist_filtered_df['alfred']
#             .dropna()
#             .astype(str)
#         )
#     else:
#         labors_algo_filtered_df = flexible_filter(
#             labors_algo_df,
#             city=city,
#             schedule_date=fecha
#         )

#         drivers = (
#             labors_algo_filtered_df['assigned_driver']
#             .dropna()                                 # Remove NaN values
#             .astype(str)                              # Ensure all are strings
#         )

#     # Remove empty strings and pure whitespace
#     drivers = [d for d in drivers.unique() if d.strip() != '']

#     return drivers


# def filter_dfs_for_insertion(
#     labors_algo_df: pd.DataFrame,
#     moves_algo_df: pd.DataFrame,
#     city: str,
#     fecha,
#     driver,
#     created_at
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Prepare filtered labors and moves for evaluating insertions for a given driver / day.

#     Behaviour summary
#     -----------------
#     1. Filter labors and moves by city, date and driver.
#     2. Keep only labors whose actual_end > created_at (future or ongoing labors).
#     3. For moves, keep rows whose labor_id starts with any of the active labor ids
#        (so free/move/labor triplets are preserved).
#     4. If the first remaining moves row corresponds to a triplet whose previous labor
#        was filtered out (i.e. the driver is in the _free/_move/_labor of an already-started
#        block), and the created_at occurs *before* the end of that first row, then
#        prepend the whole previous triplet (previous labor + its moves) so we retain
#        the context where the driver currently is.

#     Returns
#     -------
#     (labors_algo_filtered_df, moves_algo_filtered_df)
#     """
#     # --- Step 1: Prefilter by city, date and driver ---
#     labors_pref = (
#         flexible_filter(labors_algo_df, city=city, schedule_date=fecha, assigned_driver=driver)
#         .sort_values(["schedule_date", "actual_start", "actual_end"], ignore_index=True)
#     )

#     moves_pref = (
#         flexible_filter(moves_algo_df, city=city, schedule_date=fecha, assigned_driver=driver)
#         .sort_values(["schedule_date", "actual_start", "actual_end"], ignore_index=True)
#     )

#     # 🧩 Driver has no assigned labors
#     if labors_pref.empty:
#         return labors_pref, pd.DataFrame(columns=moves_pref.columns)

#     # --- Step 2: Filter labors that are ongoing or future relative to created_at ---
#     labors_filtered = labors_pref.query("actual_end > @created_at").copy()

#     if not labors_filtered.empty:
#         active_ids = labors_filtered["labor_id"].tolist()

#         # Corresponding moves for those labors
#         moves_filtered = (
#             moves_pref[moves_pref["labor_id"].isin(active_ids)]
#             .reset_index(drop=True)
#             .copy()
#         )

#         # Handle the case where driver is idle before first active labor
#         if (
#             not moves_filtered.empty and
#             created_at < moves_filtered.loc[1,"actual_start"]
#         ):
#             # Get index (position) of first active labor
#             idx_current = labors_pref.index[labors_pref["labor_id"] == active_ids[0]][0]

#             # If there’s a previous labor, include it
#             if idx_current > 0:
#                 previous_labor_id = labors_pref.loc[idx_current - 1, "labor_id"]
#                 active_ids = [previous_labor_id] + active_ids

#     else:
#         # No ongoing/future labors — fallback to last past labor
#         last_active_labor = labors_pref["labor_id"].iloc[-1]
#         active_ids = [last_active_labor]

#     # --- Step 3: Final filtering and sorting ---
#     labors_filtered = labors_pref[labors_pref["labor_id"].isin(active_ids)].copy()
#     moves_filtered = (
#         moves_pref[moves_pref["labor_id"].isin(active_ids)]
#         .sort_values(["schedule_date", "actual_start", "actual_end"], ignore_index=True)
#     )

#     return labors_filtered, moves_filtered


# def get_best_insertion(
#     candidate_insertions: List[Dict[str, Any]],
#     selection_mode: str = "min_total_distance",
#     random_state: Optional[int] = None
# ) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
#     """
#     Selects the best insertion plan among feasible options.

#     Parameters
#     ----------
#     candidate_insertions : list of dict
#         Each element is an insertion_plan produced by evaluate_driver_feasibility().
#         Must include at least:
#           - 'driver_id'
#           - 'dist_to_new_service'
#           - 'dist_to_next_labor'
#           - 'feasible' (optional but recommended)
#           - (others like downstream_shifts, new_moves, etc.)
#     selection_mode : str, optional
#         Criterion for selection:
#           - "min_total_distance" (default): minimize (dist_to_new_service + dist_to_next_labor)
#           - "min_dist_to_new_labor": minimize dist_to_new_service
#           - "random": choose randomly among feasible insertions
#     random_state : int, optional
#         Random seed for reproducibility.

#     Returns
#     -------
#     best_plan : dict or None
#         The chosen insertion_plan ready for commit_new_labor_insertion.
#     selection_df : pd.DataFrame or None
#         DataFrame summary of all feasible candidates (for diagnostics or logging).
#     """

#     if not candidate_insertions:
#         return None, pd.DataFrame()

#     # --- Convert to DataFrame for easy filtering / comparison ---
#     selection_records = []
#     for driver, plan in candidate_insertions:
#         selection_records.append({
#             "driver_id": plan.get("alfred"),
#             "prev_labor_id": plan.get("prev_labor_id"),
#             "next_labor_id": plan.get("next_labor_id"),
#             "dist_to_new": plan.get("dist_to_new", np.nan),
#             "dist_to_next": plan.get("dist_to_next", np.nan),
#             "feasible": plan.get("feasible", True)
#         })

#     selection_df = pd.DataFrame(selection_records)

#     # --- Keep only feasible ones ---
#     if "feasible" in selection_df.columns:
#         selection_df = selection_df[selection_df["feasible"] == True]
#     if selection_df.empty:
#         return None, selection_df

#     selection_df["dist_to_new"] = pd.to_numeric(selection_df["dist_to_new"], errors="coerce").fillna(0)
#     selection_df["dist_to_next"] = pd.to_numeric(selection_df["dist_to_next"], errors="coerce").fillna(0)
#     selection_df["total_distance"] = selection_df["dist_to_new"] + selection_df["dist_to_next"]

#     # --- Select best plan according to selection_mode ---
#     if selection_mode == "random":
#         chosen_idx = (
#             selection_df.sample(1, random_state=random_state).index[0]
#             if not selection_df.empty else None
#         )

#     elif selection_mode == "min_dist_to_new_labor":
#         chosen_idx = selection_df["dist_to_new_service"].idxmin()

#     elif selection_mode == "min_total_distance":
#         chosen_idx = selection_df["total_distance"].idxmin()

#     else:
#         raise ValueError(f"Unknown selection_mode: {selection_mode}")

#     if chosen_idx is None:
#         return None, selection_df

#     # --- Retrieve the corresponding plan directly ---
#     driver, best_plan = candidate_insertions[chosen_idx]

#     return driver, best_plan, selection_df


# def _get_affected_labors_from_moves(
#     new_moves: pd.DataFrame,
#     next_moves: Optional[pd.DataFrame] = None,
#     downstream_shifts: Optional[list[pd.DataFrame]] = None
# ) -> list[str]:
#     """
#     Derive the list of affected labor_ids from the move blocks.

#     This function scans the provided move blocks (new, next, downstream)
#     and returns the unique set of all labor_ids that appear across them.

#     Parameters
#     ----------
#     new_moves : pd.DataFrame
#         The moves block associated with the newly inserted labor.
#     next_moves : pd.DataFrame, optional
#         The updated moves block of the labor immediately following the insertion.
#     downstream_shifts : list[pd.DataFrame], optional
#         List of DataFrames corresponding to subsequent downstream labor adjustments.

#     Returns
#     -------
#     affected_labors : list[str]
#         Unique labor_ids affected by the insertion (including the new one).
#     """
#     dfs = []

#     if isinstance(new_moves, pd.DataFrame) and not new_moves.empty:
#         dfs.append(new_moves)

#     if isinstance(next_moves, pd.DataFrame) and not next_moves.empty:
#         dfs.append(next_moves)

#     if downstream_shifts:
#         dfs.extend([d for d in downstream_shifts if isinstance(d, pd.DataFrame) and not d.empty])

#     if not dfs:
#         return []

#     # Concatenate all moves and extract unique labor_ids
#     combined = pd.concat(dfs, ignore_index=True)
#     affected_labors = combined["labor_id"].dropna().unique().tolist()

#     return affected_labors


# ''' Updating dataframe functions'''
# def commit_new_labor_insertion(
#     labors_df: pd.DataFrame,
#     moves_df: pd.DataFrame,
#     insertion_plan: dict,
#     new_labor: pd.Series,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Commit the effects of a validated insertion plan into the global dataframes.

#     This function takes the `insertion_plan` (built by `evaluate_driver_feasibility`)
#     and applies its changes transactionally to both the `moves_df` and `labors_df`.

#     -------------------------------
#     Responsibilities:
#     -------------------------------
#     1. Update `moves_df`
#        - Remove all rows whose `labor_id` appears in `insertion_plan["affected_labors"]`.
#        - Append new rows from `new_moves`, `next_moves`, and all `downstream_shifts`.
#        - Ensure final structure integrity (no duplicates, aligned columns).

#     2. Update `labors_df`
#        - For already-existing labors:
#          update only their `actual_start` and `actual_end`.
#          (optionally, synchronize start/end points if available)
#        - For the newly inserted labor:
#          build a complete new row using the base `new_labor` data,
#          filling `assigned_driver`, `actual_start`, `actual_end`,
#          `dist_km`, and `date` fields.

#     -------------------------------
#     Parameters:
#     -------------------------------
#     labors_df : pd.DataFrame
#         Main dataframe containing all labors (historical + scheduled).
#     moves_df : pd.DataFrame
#         Main dataframe containing all driver movements.
#     insertion_plan : dict
#         Plan produced by `evaluate_driver_feasibility`. Must include:
#         {
#             "driver_id": str,
#             "new_labor_id": str,
#             "affected_labors": list[str],
#             "new_moves": pd.DataFrame,
#             "next_moves": Optional[pd.DataFrame],
#             "downstream_shifts": list[pd.DataFrame]
#         }
#     new_labor : pd.Series
#         The original row from `labors_dynamic_filtered_df` for the new labor
#         (missing only assigned_driver, actual_start, actual_end, dist_km, date).

#     -------------------------------
#     Returns:
#     -------------------------------
#     tuple[pd.DataFrame, pd.DataFrame]
#         Updated (labors_df, moves_df) pair, ready to persist.

#     -------------------------------
#     Notes:
#     -------------------------------
#     - This function assumes all dataframes use consistent column naming.
#     - The `moves_df` triplet structure (FREE_TIME, DRIVER_MOVE, LABOR)
#       must already be respected by the blocks being appended.
#     - The function does NOT reorder rows by time — this can be done externally.
#     """

#     # === Safety checks ===
#     assert isinstance(insertion_plan["new_moves"], pd.DataFrame), \
#         "insertion_plan['new_moves'] must be a DataFrame"

#     driver_id = insertion_plan["driver_id"]
#     new_moves = insertion_plan["new_moves"].copy()
#     next_moves = insertion_plan.get("next_moves", None)
#     downstream_shifts = insertion_plan.get("downstream_shifts", [])

#     affected_labors = _get_affected_labors_from_moves(new_moves, next_moves, downstream_shifts)

#     # --- Collect all new move rows to insert ---
#     moves_to_add = [new_moves]
#     if next_moves is not None and not next_moves.empty:
#         moves_to_add.append(next_moves)
#     if downstream_shifts:
#         moves_to_add.extend([m for m in downstream_shifts if m is not None and not m.empty])
#     moves_to_add = pd.concat(moves_to_add, ignore_index=True) if moves_to_add else pd.DataFrame()

#     # === MOVES_DF UPDATE ===
#     # Remove old rows for affected labors to avoid duplicates
#     moves_df_updated = moves_df[~moves_df["labor_id"].isin(affected_labors)].copy()
#     moves_df_updated = pd.concat([moves_df_updated, moves_to_add], ignore_index=True)

#     # === LABORS_DF UPDATE ===
#     labors_df_updated = labors_df.copy()

#     # 1️⃣ Update existing affected labors (if they already exist in labors_df)
#     affected_existing = [lid for lid in affected_labors if lid in labors_df["labor_id"].values]
#     labor_rows_from_moves = moves_df_updated[moves_df_updated["labor_context_id"].str.endswith("_labor")]

#     for labor_id in affected_existing:
#         move_row = labor_rows_from_moves.query("labor_id == @labor_id")
#         if not move_row.empty:
#             start, end = move_row.iloc[0]["actual_start"], move_row.iloc[0]["actual_end"]
#             labors_df_updated.loc[
#                 labors_df_updated["labor_id"] == labor_id, ["actual_start", "actual_end"]
#             ] = [start, end]

#             # Optional: keep start/end points synchronized
#             if "start_point" in move_row and "start_address_point" in labors_df_updated.columns:
#                 labors_df_updated.loc[
#                     labors_df_updated["labor_id"] == labor_id, "start_address_point"
#                 ] = move_row.iloc[0]["start_point"]
#             if "end_point" in move_row and "end_address_point" in labors_df_updated.columns:
#                 labors_df_updated.loc[
#                     labors_df_updated["labor_id"] == labor_id, "end_address_point"
#                 ] = move_row.iloc[0]["end_point"]

#     # 2️⃣ Add the new labor entry (complete row)
#     new_labor_row = new_labor.copy()
#     new_labor_id = new_labor_row["labor_id"]
#     new_labor_move_row = new_moves.query("labor_context_id.str.endswith('_labor')", engine="python").iloc[0]

#     new_labor_row["assigned_driver"] = driver_id
#     new_labor_row["actual_start"] = new_labor_move_row["actual_start"]
#     new_labor_row["actual_end"] = new_labor_move_row["actual_end"]
#     new_labor_row["dist_km"] = new_labor_move_row["distance_km"]
#     new_labor_row["start_address_point"] = new_labor_move_row["start_point"]
#     new_labor_row["end_address_point"] = new_labor_move_row["end_point"]
#     new_labor_row["date"] = new_labor_move_row["date"]
#     new_labor_row["n_drivers"] = np.nan  # Can be filled later

#     # Remove existing row if any (to prevent duplicates)
#     labors_df_updated = labors_df_updated[labors_df_updated["labor_id"] != new_labor_id]
#     labors_df_updated = pd.concat([labors_df_updated, pd.DataFrame([new_labor_row])], ignore_index=True)

#     # === Final consistency checks ===
#     assert moves_df_updated["labor_id"].nunique() == len(
#         moves_df_updated["labor_id"].unique()
#     ), "Duplicate labor_id detected in moves_df after insertion!"

#     return labors_df_updated, moves_df_updated, new_labor_row['actual_end'], new_labor_row["end_address_point"]


