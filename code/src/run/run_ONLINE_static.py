import pandas as pd
import numpy as np
import pickle

from time import perf_counter
import os

from tqdm import tqdm

import multiprocessing as mp

from src.data.data_load import load_inputs, load_distances
from src.utils.filtering import filter_labors_by_date, flexible_filter
from src.data.metrics import concat_run_results, compute_metrics_with_moves, compute_iteration_metrics
from src.utils.utils import print_iteration_header, clear_last_n_lines, prep_algorithm_inputs
from src.config.config import *
from src.config.experimentation_config import *
from src.algorithms.pipeline import run_single_iteration, select_best_iteration, run_city_pipeline
from src.algorithms.solution_search_utils import should_update_incumbent, update_incumbent_state


def run_ONLINE_static(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool,
    multiprocessing: bool,
    n_processes = None,
    experiment_type: str = 'online_operation'
):
    if multiprocessing: 
        run_ONLINE_static_parallel(
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results,
            n_processes=n_processes,
            experiment_type=experiment_type
        )
    else:
        run_ONLINE_static_sequential(
            instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            save_results=save_results
        )

def run_ONLINE_static_sequential(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: str
):
    data_path = f'{REPO_PATH}/data'
    
    ''' START RUN PARAMETERS'''
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']
    assignment_type = 'algorithm'
    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']
    ''' END RUN PARAMETERS'''

    fechas = fechas_map(instance)
    initial_day = int(fechas[0].rsplit('-')[2])

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance, online_type='_static')

    ### Optimization config
    print_iteration_header(
        details='online static scheduling'
    )

    alpha = hyperparameter_selection[optimization_obj]
    # 🔄 Progress bar replaces manual iteration printing

    run_results = []
    for fecha in fechas:
        print('----------------------------------------------------------------------------')

        dist_dict = load_distances(data_path, distance_type, instance, distance_method)
        df_day = filter_labors_by_date(
            labors_real_df, fecha=fecha, end_date='one day lag'
        )
            
        for city in valid_cities:
            directorio_hist_filtered_df = flexible_filter(
                directorio_hist_df, city=city, date=fecha
            )
            dist_dict_city = dist_dict.get(city, {})

            start = perf_counter()
            inc_vt_labors = 0
            inc_extra_time = 1e9
            inc_dist = 1e9
            inc_state = None

            # Iterative search 
            for iter in tqdm(range(1, max_iterations[city] + 1), desc=f"{city}/{fecha}", unit="iter", leave=False):
                ''' START RUN ITERATION FOR SET ALPHA '''
                
                results_df, moves_df = run_city_pipeline(
                    city, 
                    fecha, 
                    df_day, 
                    directorio_hist_filtered_df, 
                    duraciones_df,
                    assignment_type, 
                    alpha=alpha,
                    DIST_DICT=dist_dict_city,
                    dist_method=distance_method,
                    instance=instance,
                    driver_init_mode=driver_init_mode
                )

                if results_df.empty:
                    continue

                metrics = compute_metrics_with_moves(
                    results_df, 
                    moves_df,
                    fechas=[fecha],
                    dist_dict=dist_dict,
                    workday_hours=8,
                    city=city,
                    skip_weekends=False,
                    assignment_type=assignment_type,
                    dist_method=distance_method
                )

                iter_vt_labors, iter_extra_time, iter_dist = compute_iteration_metrics(metrics)

                ''' UPDATE INCUMBENT '''
                if iter_vt_labors >= inc_vt_labors:
                    update = False
                    
                    update, new_score, inc_score = should_update_incumbent(
                    optimization_obj,
                    iter_dist=iter_dist,
                    iter_extra_time=iter_extra_time,
                    inc_dist=inc_dist,
                    inc_extra_time=inc_extra_time
                    )

                    results_df['date'] = fecha
                    moves_df['city'] = city
                    moves_df['date'] = fecha

                    if update:
                        inc_state = update_incumbent_state(
                            iter_idx=iter,
                            iter_vt_labors=iter_vt_labors,
                            iter_extra_time=iter_extra_time,
                            iter_dist=iter_dist,
                            results_df=results_df,
                            moves_df=moves_df,
                            metrics=metrics,
                            start_time=start
                        )

                    # unpack into your variables (if you still want separate vars)
                    inc_vt_labors = inc_state["vt_labors"]
                    inc_extra_time = inc_state["extra_time"]
                    inc_dist = inc_state["dist"]
                    inc_values = inc_state["values"]

                    ''' CHECKPOINT SAVE '''
                    if iter in iterations_nums[city]:
                        duration = round(perf_counter() - start, 1)
                        tqdm.write(  # ✅ ensures checkpoint is printed on its own line
                            f"{fecha} \t{city} \t{iter} \t{inc_values[0]} \t"
                            f"{round(duration)}s \t\t{inc_vt_labors} \t\t"
                            f"{round(inc_extra_time/60,1)}h \t\t{inc_dist} km"
                        )

            if inc_state:
                run_results.append([inc_state['results'], inc_state['moves']])
            ''' END RUN ITERATION '''

    # Build consolidated dfs for this run
    results_df, moves_df = concat_run_results(run_results)

    # ------ Ensure output directory exists ------
    if save_results:
        output_dir = os.path.join(data_path, 
                                    "resultados", 
                                    "online_operation", 
                                    instance,
                                    distance_method
                                    )
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing
        
        with open(os.path.join(output_dir, f'res_algo_ONLINE_static.pkl'), 'wb') as f:
            duration = round(perf_counter() - start, 1)
            pickle.dump([results_df, moves_df], f)

    print(f' ✅ Successfully ran ONLINE static algorithm \n')
    return True


def run_ONLINE_static_parallel(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool,
    n_processes: int = None,  # defaults to all available CPUs
    experiment_type: str = 'online_operation',
):
    
    start = perf_counter()
    
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
    ) = prep_algorithm_inputs(
        instance, 
        distance_method, 
        optimization_obj,
        online_type='_static'
    )

    run_results = []
    postponed_labors = []

    for fecha in fechas:
        print(f"{'-'*120}\n▶ Processing date: {fecha} / {fechas[-1]}")
        df_day = filter_labors_by_date(labors_real_df, start_date=fecha, end_date='one day lag')

        for city in valid_cities:
            if df_day[df_day['city'] == city].empty:
                continue

            directorio_hist_filtered_df = flexible_filter(
                directorio_hist_df, city=city, date=fecha
            )
            dist_dict_city = dist_dict.get(city, {})

            max_iter = max_iterations[city]
            iter_args = [
                {
                    "city": city,
                    "fecha": fecha,
                    "iter_idx": i,
                    "df_day": df_day,
                    "dist_dict": dist_dict_city,
                    "directorio_hist_filtered_df": directorio_hist_filtered_df,
                    "duraciones_df": duraciones_df,
                    "distance_method": distance_method,
                    "assignment_type": assignment_type,
                    "alpha": alpha,
                    "instance": instance,
                    "driver_init_mode": driver_init_mode,
                }
                for i in range(1, max_iter + 1)
            ]

            # --- Run in parallel ---
            with mp.Pool(processes=n_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(run_single_iteration, iter_args), 
                        total=len(iter_args),
                        leave=False,
                        desc=f"City: {city}"
                        )
                    )

            # --- Select incumbent ---
            df_results = pd.DataFrame(results)
            best_idx = select_best_iteration(df_results, optimization_obj)
            inc_state = df_results.iloc[best_idx]

            postponed_labors += inc_state["postponed_labors"]

            run_results.append([inc_state["results"], inc_state["moves"]])
        
        clear_last_n_lines(2)

    # --- Combine all results ---
    results_df, moves_df = concat_run_results(run_results)

    if save_results:
        output_dir = os.path.join(data_path, "resultados", experiment_type, instance, distance_method)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'res_algo_ONLINE_static.pkl'), "wb") as f:
            pickle.dump([results_df, moves_df, postponed_labors], f)

    print(f"\n ✅ Completed ONLINE static in {round(perf_counter() - start, 1)}s total. \n")
    return True



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    run_ONLINE_static(args.instance)
