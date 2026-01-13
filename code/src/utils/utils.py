import pandas as pd
import pickle

import sys
import os
import traceback

from datetime import datetime
import argparse

from src.data.data_load import load_inputs, load_distances, load_instance, upload_ONLINE_static_solution
from src.config.experimentation_config import instance_map, fechas_map, hyperparameter_selection, codificacion_ciudades
from src.config.config import REPO_PATH


def print_iteration_header(details):
    print(f'\n------ Algorithm solution search for {details}')
    print('fecha \t\tcity \titer \tfound \trn_t \tn_ser \textr_t \tdriv_dist')


def get_city_name(city_code: str, cities_df: pd.DataFrame) -> str:
    """
    Retorna el nombre de la ciudad dado su código.

    Parámetros
    ----------
    city_code : str
        Código de la ciudad (columna 'cod_ciudad').
    cities_df : pd.DataFrame
        DataFrame con las columnas 'cod_ciudad' y 'ciudad'.

    Retorna
    -------
    str
        Nombre de la ciudad correspondiente al código. 
        Si el código no se encuentra, devuelve 'DESCONOCIDO'.
    """
    match = cities_df.loc[cities_df['cod_ciudad'].astype(str) == str(city_code), 'ciudad']
    if not match.empty:
        return match.iloc[0]
    return "DESCONOCIDO"


def get_city_name_from_code(city_code):
    if city_code == 'ALL':
        return 'Global'
    return codificacion_ciudades.get(city_code, 'DESCONOCIDO')


def process_instance(instance_input): 
    instance = f'inst{instance_input}'
    assert instance in instance_map.keys(), f'Non-existing instance "{instance}"!'
    instance_type = instance_map[instance]

    return instance, instance_type


def process_dynamic_instance(instance_input):
    if instance_input=='r':
        instance = f'instRD1'
    elif instance_input=='a':
        instance = f'instAD1'
    else:
        raise ValueError('Select a valid instance')
    instance_type = instance_map[instance]

    return instance, instance_type


def get_max_drivers(instance, city, max_drivers, start_date, initial_day):
    base_day = pd.to_datetime(start_date).date()
    max_drivers_dict = max_drivers.get(instance, {}).get(city, None)
    if max_drivers_dict:
        max_drivers_num = max_drivers_dict[base_day.day - initial_day]
    else:
        max_drivers_num = None
    
    return max_drivers_num


def build_algos_header(
    run_historic_baseline: bool,
    run_algo_baseline: bool,
    run_online_static_algo: bool,
    run_INSERT_algo: bool,
    run_INSERT_BUFFER_algo: bool,
    run_REACT_algo: bool,
    run_REACT_BUFFER_algo: bool,
    run_ALFRED_algo: bool,
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool,
    multiprocessing: bool,
    experiment_type: str = 'Artificial Instances'
):
    active_algos = []
    if run_historic_baseline:
        active_algos.append("Hist. Baseline")
    if run_algo_baseline:
        active_algos.append("Offline Baseline")
    if run_online_static_algo:
        active_algos.append("Online Static")
    if run_INSERT_algo:
        active_algos.append("INSERT")
    if run_INSERT_BUFFER_algo:
        active_algos.append("INSERT BUFFER")
    if run_REACT_algo:
        active_algos.append("REACT")
    if run_REACT_BUFFER_algo:
        active_algos.append("REACT BUFFER")
    if run_ALFRED_algo:
        active_algos.append("ALFRED")

    active_summary = ", ".join(active_algos) if active_algos else "⚠️ None (check flags!)"

    # Pretty header
    print("\n" + "=" * 120)
    print(f"🚀 Running Orchestration - {experiment_type}")
    print("=" * 120)
    print(f"📏 Instance               : {instance}")
    print(f"⌚️ Start time             : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    print(f"🎯 Optimization objective : {optimization_obj}")
    print(f"📏 Distance method        : {distance_method}")
    print(f"💾 Save results           : {save_results}")
    print(f"🤖 Algorithms to run      : {active_summary}")
    print(f"🔥 Parallelization        : {multiprocessing}")
    print("=" * 120 + "\n")


def process_instance_selection(
    DEBUG_MODE: bool,
    DEFAULT_INSTANCE: str,
    DEFAULT_DISTANCE_METHOD: str,
    DEFAULT_SAVE: bool,
) -> tuple:
    parser = argparse.ArgumentParser(description="Run online dynamic insertion simulation")
    parser.add_argument("--inst", help="Instance name (e.g. instAD2b)")
    parser.add_argument("--dist_method", help="Distance method (haversine or precalced)")
    parser.add_argument("--save", help="Save the results of the run")
    args = parser.parse_args()

    # ====== INSTANCE SELECTION ======
    if DEBUG_MODE:
        instance = DEFAULT_INSTANCE
        distance_method = DEFAULT_DISTANCE_METHOD
        save_results = DEFAULT_SAVE
        print(f"[DEBUG MODE] Using default instance: {instance}")
    else:
        if not args.inst:
            parser.error("You must provide --instance when not in debug mode.")
        # instance = args.instance
        instance = f'inst{args.inst}'
        distance_method = args.dist_method
        if args.save == 'True':
            save_results = True
        elif args.save=='False':
            save_results = False
        else:
            parser.error("Select a boolean option for --save.")
    
    return instance, distance_method, save_results


def clear_last_n_lines(n: int):
    for _ in range(n):
        # Move cursor up one line
        sys.stdout.write("\033[F")
        # Clear line
        sys.stdout.write("\033[K")
    sys.stdout.flush()


def prep_algorithm_inputs(instance, distance_method, optimization_obj, **kwargs):
    data_path = f'{REPO_PATH}/data'
    distance_type = 'osrm'
    assignment_type = 'algorithm'
    driver_init_mode = 'historic_directory'

    fechas = fechas_map(instance)
    duraciones_df, valid_cities, labors_real_df, directorio_hist_df = load_inputs(data_path, instance, **kwargs)

    dist_dict = load_distances(data_path, distance_type, instance, distance_method)

    alpha = hyperparameter_selection[optimization_obj]

    return_items = (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df, 
        directorio_hist_df,
        dist_dict,
        fechas,
        alpha
    )
    return return_items


def prep_online_algorithm_inputs(
    instance, 
    distance_method, 
    optimization_obj,
    experiment_type='online_operation',
):
    
    (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df, 
        directorio_hist_df,
        dist_dict,
        fechas,
        alpha
    ) = prep_algorithm_inputs(instance, distance_method, optimization_obj)

    labors_dynamic_df = load_instance(
        data_path, 
        instance, 
        labors_real_df,
        online_type='_dynamic')
    
    # # --- Consolidar resultados previos (estáticos) ---
    labors_algo_static_df, moves_algo_static_df, postponed_labors = upload_ONLINE_static_solution(
        data_path, 
        instance, 
        distance_method,
        experiment_type
    )

    labors_algo_dynamic_df = labors_algo_static_df.copy()
    moves_algo_dynamic_df = moves_algo_static_df.copy()
    
    return_items = (
        data_path,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        valid_cities, 
        labors_real_df, 
        directorio_hist_df,
        dist_dict,
        fechas,
        alpha, 
        labors_dynamic_df,
        labors_algo_dynamic_df,
        moves_algo_dynamic_df,
        postponed_labors
    )

    return return_items


def safe_run(name, func, *args, **kwargs):
    print(f"\n ▶ Running {name}...")
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f" ❌ {name} failed with error: {type(e).__name__}: {e}")
        tb = traceback.format_exc()
        print(" --- Traceback ---")
        print(tb)



def compute_workday_end(day_str: str, workday_end_str: str, tzinfo=None):
    """
    Builds the Timestamp corresponding to the workday end on the given day.
    
    Inputs:
        day_str: 'YYYY-MM-DD'
        workday_end_str: 'HH:MM'
        tzinfo: timezone to apply (optional)
    
    Output:
        pd.Timestamp with correct date and time
    """
    # Combine into a full timestamp string
    full_str = f"{day_str} {workday_end_str}"

    # Parse
    ts = pd.to_datetime(full_str)

    # Apply timezone if provided
    if tzinfo is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tzinfo)
        else:
            ts = ts.tz_convert(tzinfo)

    return ts


