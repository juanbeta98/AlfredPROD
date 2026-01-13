import os
import pickle
import pandas as pd
from datetime import timedelta

import json

from time import perf_counter
from tqdm import tqdm
from itertools import product

import pandas as pd
from datetime import datetime, timedelta
import pytz

from datetime import datetime, timedelta, time as dt_time
from typing import Tuple, Dict, Any, List


from src.algorithms.ALFRED_algorithm import (
    generate_alfred_parameters, 
    load_alfred_parameters,
    alfred_algorithm_assignment
)
from src.algorithms.OFFLINE_algorithm import OFFLINE_algorithm_wrapper
from src.algorithms.BUFFER_FIXED_algorithm import BUFFER_FIXED_wrapper
from src.algorithms.BUFFER_REACT_algorithm import BUFFER_REACT_wrapper


from src.utils.utils import prep_algorithm_inputs, compute_workday_end
from src.config.experimentation_config import *
from src.config.config import *
from src.config.alfred_algorithm_config import (
    OFFLINE_ALGORITHM_CUTOFF,
    FIRST_RUN_TIME,
    BATCH_RUN_INTERVAL_MINS,
    BUFFER_REACT_RUN_TIME,
    DEFAULT_ALGORITHM,
    LAST_RUN_TIME
)
from src.utils.filtering import flexible_filter

ZONE = "America/Bogota"

def generate_time_intervals():
    """
    Return list of (previous_run, current_time) as 'HH:MM' strings.
    First interval covers [00:00, start_time).
    Then rolling windows from start_time to end_time in `increment` minutes.
    """
    fmt = "%H:%M"
    intervals = [(None, OFFLINE_ALGORITHM_CUTOFF), (OFFLINE_ALGORITHM_CUTOFF, FIRST_RUN_TIME)]
    current = datetime.strptime(FIRST_RUN_TIME, fmt)
    end = datetime.strptime(LAST_RUN_TIME, fmt)

    while current + timedelta(minutes=BATCH_RUN_INTERVAL_MINS) <= end:
        lower = current.strftime(fmt)
        upper = (current + timedelta(minutes=BATCH_RUN_INTERVAL_MINS)).strftime(fmt)
        intervals.append((lower, upper))
        current = current + timedelta(minutes=BATCH_RUN_INTERVAL_MINS)

    return intervals


def default_algorithm_dispatch():
    """
    Map algorithm names to wrapper callables.
    Each wrapper MUST follow the signature:
        wrapper(assignment_state: Tuple[pd.DataFrame, pd.DataFrame], params: Dict) -> (assignment_state_out, orchestration_log_entry)
    The orchestration_log_entry is a dict describing what happened for this run (chosen iter, duration, metrics, etc).
    """
    return {
        "OFFLINE": OFFLINE_algorithm_wrapper,
        "BUFFER_REACT": BUFFER_REACT_wrapper,
        "BUFFER_FIXED": BUFFER_FIXED_wrapper,
        # add any other algorithms here
    }


def select_algorithm_to_run(current_time):
    '''
    Determines the algorithm to run depending on the time and the predefined parameters
    '''
    if current_time == OFFLINE_ALGORITHM_CUTOFF:
        algorithm_to_run = "OFFLINE"
    elif current_time == BUFFER_REACT_RUN_TIME:  # 11:00
        algorithm_to_run = "BUFFER_REACT"  # BUFFER_REACT
    else:
        algorithm_to_run = DEFAULT_ALGORITHM  # BUFFER_FIXED (insertion style)
    
    return algorithm_to_run


def save_iteration_state(base_output_dir: str, city: str, fecha: str, iteration_idx: int,
                         assignment_state: Tuple[pd.DataFrame, pd.DataFrame], orchestration_log: Dict[str, Any]):
    """
    Save assignment_state and a small orchestration log for traceability.
    - labors_state saved as CSV (or pickle if you prefer).
    - moves_state saved as CSV.
    - orchestration_log saved as JSON.
    """
    # os.makedirs(base_output_dir, exist_ok=True)
    log_directory = os.path.join(base_output_dir, 'alfred_log')
    os.makedirs(log_directory, exist_ok=True)
    
    prefix = f"{fecha}_city{city}_iter{iteration_idx}"

    labors_path = os.path.join(log_directory, f"{prefix}_labors.csv")
    moves_path = os.path.join(log_directory, f"{prefix}_moves.csv")
    log_path = os.path.join(log_directory, f"{prefix}_log.json")

    labors_state, moves_state = assignment_state

    labors_state.to_csv(os.path.join(base_output_dir, 'labors_state.csv'), index=False)
    moves_state.to_csv(os.path.join(base_output_dir, 'moves_state.csv'), index=False)

    labors_state.to_csv(labors_path, index=False)
    moves_state.to_csv(moves_path, index=False)
    with open(log_path, "w") as f:
        json.dump(orchestration_log, f, default=str, indent=2)


def run_ALFRED(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    time_previous_freeze: int,
    save_results: bool = True,
    multiprocessing: bool = True,
    n_processes: int = None,
    experiment_type: str = 'online_operation',
    base_seed: int = 12345,
):
    """
    High-level orchestrator skeleton.

    - Loads inputs via prep_online_algorithm_inputs
    - Generates decision windows (list of (prev, curr))
    - For each date / city / decision moment:
        - builds params
        - chooses algorithm to run
        - calls the wrapper with current assignment_state and params
        - saves / logs results (one snapshot per decision moment)
    """

    start = perf_counter()

    # --- prepare environment & inputs ------------------------------------
    (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df, 
        directorio_hist_df,
        global_dist_dict,
        fechas,
        alpha
    ) = prep_algorithm_inputs(instance, distance_method, optimization_obj)

    # algorithm dispatch (wrapper functions)
    dispatch = default_algorithm_dispatch()

    base_output_dir = os.path.join(data_path, "resultados", experiment_type, instance, distance_method)
    temp_output_dir = os.path.join(base_output_dir, 'alfred_run')
    os.makedirs(temp_output_dir, exist_ok=True)

    # --- produce decision windows ------------------------------------------------
    decision_windows = generate_time_intervals()

    # Extra / static orchestrator params
    hyper_params = {
        'instance': instance,
        'distance_method': distance_method,
        'duraciones_df': duraciones_df,

        'alpha': 0.3,
        'optimization_obj': 'driver_distance',

        "time_previous_freeze": time_previous_freeze,
        "base_seed": base_seed,
        "multiprocessing": multiprocessing,
        'n_processes': n_processes,
    }

    # ---------------------------------------------------------
    # Main loops: date -> city -> decision window
    # ---------------------------------------------------------
    labors_algo_df = pd.DataFrame()
    moves_algo_df = pd.DataFrame()

    for fecha in fechas:
        print(f"\n{'-'*120}\n▶ DATE: {fecha} (remaining {len(fechas)-fechas.index(fecha)-1})")

        for city in valid_cities:
            print(f'\tCity: {city}')

            directorio_hist_filtered_df = flexible_filter(
                directorio_hist_df, city=city, date=fecha
            )
            city_dist_dict = global_dist_dict.get(city, {})
            n_iterations = max_iterations[city]

            # For every decision window
            # for i, (prev_run, curr_time) in tqdm(enumerate(decision_windows), desc=f"City: {city}", leave=False, unit="window"):
            for i, (prev_run, curr_time) in enumerate(decision_windows):
                # Algorithm selection (parametrized policy)
                algorithm_to_run = select_algorithm_to_run(curr_time)

                # if algorithm_to_run != 'OFFLINE': continue
                status = f'\t\tprev_run: {prev_run} - curr_time: {curr_time} - Algorithm: {algorithm_to_run}'

                # Retrieve current states
                def get_assignment_state(
                    algorithm_to_run,
                    temp_output_dir,
                ):
                    if algorithm_to_run == 'OFFLINE':
                        labors_state = pd.DataFrame(columns=labors_real_df.columns)
                        moves_state = pd.DataFrame(columns=['service_id', 'labor_id', 'labor_context_id', 'labor_name',
                            'labor_category', 'assigned_driver', 'schedule_date', 'actual_start',
                            'actual_end', 'start_point', 'end_point', 'distance_km', 'city',
                            'duration_min', 'date'])
                    else:
                        labors_state = pd.read_csv(f'{temp_output_dir}/labors_state.csv')
                        moves_state = pd.read_csv(f'{temp_output_dir}/moves_state.csv')

                        # Reconstruir fechas en modo ISO8601
                        date_cols = [
                            "labor_created_at", 
                            "labor_start_date", 
                            "labor_end_date", 
                            "created_at", 
                            "schedule_date", 
                            'actual_start', 
                            'actual_end'
                        ]

                        ### DEBUGGING
                        problem_col = "actual_start"

                        def _strip_microseconds(x):
                            if isinstance(x, str):
                                # split on the first '.' but keep timezone suffix
                                parts = x.split('.', 1)
                                if len(parts) == 2:
                                    # Find the timezone part
                                    rest = parts[1]
                                    if '-' in rest or '+' in rest:
                                        # microseconds + timezone
                                        tz_index = rest.find('-')
                                        if tz_index == -1:
                                            tz_index = rest.find('+')
                                        return f"{parts[0]}{rest[tz_index:]}"
                                    else:
                                        # no timezone (unlikely)
                                        return parts[0]
                            return x
                        
                        for col in date_cols:
                            if col in labors_state.columns:
                                cleaned = labors_state[col].apply(_strip_microseconds)
                                parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
                                labors_state[col] = parsed.dt.tz_convert('America/Bogota')

                            if col in moves_state.columns:
                                cleaned = moves_state[col].apply(_strip_microseconds)
                                parsed = pd.to_datetime(cleaned, errors="coerce", utc=True)
                                moves_state[col] = parsed.dt.tz_convert('America/Bogota')

                        for col in ['city', 'alfred', 'service_id', 'labor_id', 'assigned_driver']:
                            if col in labors_state.columns:
                                labors_state[col] = (
                                    labors_state[col]
                                    .apply(lambda x: '' if pd.isna(x) else str(int(float(x))))
                                )
                            if col in moves_state.columns:
                                moves_state[col] = (
                                    moves_state[col]
                                    .apply(lambda x: '' if pd.isna(x) else str(int(float(x))))
                                )

                    assignment_state = (labors_state, moves_state)
                    
                    return assignment_state
                
                    
                assignment_state = get_assignment_state(
                    algorithm_to_run,
                    temp_output_dir
                )


                # Retrieve pending labors
                def get_pending_labors(
                    labors_df,
                    city,
                    fecha,
                    prev_run,
                    curr_time,
                ):
                    # Filtrar labores para esa ciudad y esa fecha
                    labors_filtered_df = flexible_filter(
                        labors_df,
                        city=city,
                        schedule_date=fecha
                    )

                    tz = pytz.timezone("America/Bogota")

                    # ---- Helper to build datetime from fecha + HH:mm (same date) ----
                    def build_datetime(date_str, hhmm):
                        hour, minute = map(int, hhmm.split(":"))
                        dt_naive = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute)
                        return tz.localize(dt_naive)

                    # ---- Case 1: prev_run is None ----
                    if prev_run is None:
                        if curr_time.startswith("-"):
                            # Negative time → day before fecha
                            time_part = curr_time[1:]  # strip "-"
                            base_date = (datetime.strptime(fecha, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
                            reference_dt = build_datetime(base_date, time_part)
                        else:
                            # Positive time → same fecha
                            reference_dt = build_datetime(fecha, curr_time)

                        # Keep rows created BEFORE OR EQUAL to the reference datetime
                        mask = labors_filtered_df["created_at"] <= reference_dt

                        return labors_filtered_df.loc[mask]

                    # ---- Case 2: prev_run is NOT None ----
                    # prev_run and curr_time are guaranteed to be positive
                    prev_dt = build_datetime(fecha, prev_run)
                    curr_dt = build_datetime(fecha, curr_time)

                    # Keep rows in the window: prev_dt < created_at <= curr_dt
                    mask = (labors_filtered_df["created_at"] > prev_dt) & (labors_filtered_df["created_at"] <= curr_dt)
                    return labors_filtered_df.loc[mask]

                pending_labors = get_pending_labors(
                    labors_real_df,
                    city,
                    fecha,
                    prev_run,
                    curr_time
                )

                if pending_labors.empty:
                    print(f'{status} - Pending labors: 0')

                    assignment_columns = ['actual_start', 'actual_end', 'assigned_driver']
                    for col in assignment_columns:
                        if col not in assignment_state[0].columns:
                            assignment_state[0][col] = None

                    save_iteration_state(temp_output_dir, city, fecha, i, assignment_state, {})
                    continue
                
                print(f'{status} - Pending labors: {len(pending_labors)}')
                continue 

                # Compose params to feed wrapper
                def build_alfred_params(
                    algorithm_to_run,
                    hyper_params,
                    iter_params,
                    base_params):
                    params = {**iter_params}
                    params.update({
                        'multiprocessing': hyper_params['multiprocessing'],
                        'n_processes': hyper_params['n_processes']
                    })

                    if algorithm_to_run in 'OFFLINE':
                        base_params['instance'] = hyper_params['instance']
                        base_params['distance_method'] = hyper_params['distance_method']
                        base_params['duraciones_df'] = hyper_params['duraciones_df']
                        base_params['alpha'] = hyper_params['alpha']
                        base_params['driver_init_mode'] = driver_init_mode

                        params['base_args'] = base_params
                    
                    elif algorithm_to_run == 'BUFFER_FIXED':
                        params.update({**base_params})
                        params.update({**hyper_params})
                        params['driver_init_mode'] = driver_init_mode
                        params['assignment_type'] = 'algorithm'
                        
                    
                    return params
                
                iter_params = {
                    'n_iterations': n_iterations,
                }
                base_params = {
                    "city": city,
                    "fecha": fecha,
                    'dist_dict': city_dist_dict,
                    'directorio_hist_filtered_df': directorio_hist_filtered_df,
                }
                params = build_alfred_params(
                    algorithm_to_run,
                    hyper_params,
                    iter_params,
                    base_params
                )

                # params = {}
                # params["n_iterations"] = params.get("n_iterations", 20)
                params["time_previous_freeze"] = time_previous_freeze

                # --- Algorithm execution ------------------------------------------------
                wrapper = dispatch.get(algorithm_to_run)

                try:
                    # wrapper must return (assignment_state_out, orchestration_log)
                    assignment_state, orch_log = wrapper(
                        assignment_state=assignment_state,
                        pending_labors=pending_labors,
                        params=params
                    )
                except Exception as e:
                    # Robust logging and continue: save exception details into orchestration log and continue with prior state
                    orch_log = {
                        "success": False,
                        "error": repr(e),
                        "algorithm": algorithm_to_run,
                        "city": city,
                        "fecha": fecha,
                        "decision_window": (prev_run, curr_time)
                    }
                    print(f"  [ERROR] wrapper raised: {e}. Continuing with previous state.")
                    # keep previous assignment state unchanged

                # Save iteration snapshot and orchestration log           
                if (assignment_state[1]['labor_id'].value_counts().reset_index()['count'].max() > 3):
                    print('ERROR')
                    
                save_iteration_state(temp_output_dir, city, fecha, i, assignment_state, orch_log)

            #-------- 
            labors_city_date = assignment_state[0].copy()
            moves_city_date = assignment_state[1].copy()

            # Here attach the
            if not labors_city_date.empty:
                labors_algo_df = pd.concat([labors_algo_df, labors_city_date])
                moves_algo_df = pd.concat([moves_algo_df, moves_city_date])

    print()
    print(f" ✅ Completed ALFRED algorithm in {round(perf_counter() - start, 1)}s total. \n")

    # final save of the final assignment_state
    if save_results:

        with open(os.path.join(base_output_dir, f'res_algo_ALFRED.pkl'), "wb") as f:
            pickle.dump([labors_algo_df, moves_algo_df, None], f)

    return assignment_state
