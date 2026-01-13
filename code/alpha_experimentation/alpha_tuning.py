import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import perf_counter
from multiprocessing import Pool, cpu_count

from src.data_load import load_distances
from src.filtering import filter_labors_by_date, flexible_filter
from src.metrics import compute_metrics_with_moves, compute_iteration_metrics
from src.pipeline import run_city_pipeline
from src.alpha_tuning_utils import should_update_incumbent, update_incumbent_state, dist_norm_factor, extra_time_norm_factor

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from time import perf_counter
import os

from src.data_load import load_inputs, load_distances
from src.filtering import filter_labors_by_date, flexible_filter
from src.metrics import compute_metrics_with_moves, compute_iteration_metrics
from src.utils import process_instance
from src.config import *
from src.experimentation_config import fechas_map
from src.pipeline import run_city_pipeline
from src.alpha_tuning_utils import should_update_incumbent, update_incumbent_state, \
    metrics, alphas, dist_norm_factor, extra_time_norm_factor

from tqdm import tqdm


def run_single_iteration_task(task):
    """
    Task runner for a single (city, date, iteration) run.
    task = (instance, distance_method, optimization_variable, alpha, city, start_date, iter, context)
    context: dictionary of read-only shared data
    """
    (instance, distance_method, optimization_variable, alpha, city, start_date, iter, context) = task
    df_day = context["df_days"][(city, start_date)]
    directorio_hist_filtered_df = context["directorio_days"][(city, start_date)]
    duraciones_df = context["duraciones_df"]
    assignment_type = context["assignment_type"]
    DIST_DICT_city = context["dist_dict"].get(city, {})
    distance_method = context["distance_method"]
    driver_init_mode = context["driver_init_mode"]
    instance = context["instance"]

    results_df, moves_df = run_city_pipeline(
        city,
        start_date,
        df_day,
        directorio_hist_filtered_df,
        duraciones_df,
        assignment_type,
        alpha=alpha,
        DIST_DICT=DIST_DICT_city,
        dist_method=distance_method,
        instance=instance,
        driver_init_mode=driver_init_mode
    )

    if results_df.empty:
        return None

    metrics_dict = compute_metrics_with_moves(
        results_df,
        moves_df,
        fechas=[start_date],
        dist_dict=context["dist_dict"],
        workday_hours=8,
        city=city,
        skip_weekends=False,
        assignment_type=assignment_type,
        dist_method=distance_method,
    )

    iter_vt_labors, iter_extra_time, iter_dist = compute_iteration_metrics(metrics_dict)

    return {
        "instance": instance,
        "distance_method": distance_method,
        "optimization_variable": optimization_variable,
        "alpha": alpha,
        "city": city,
        "date": start_date,
        "iteration": iter,
        "vt_labors": iter_vt_labors,
        "extra_time": iter_extra_time,
        "dist": iter_dist,
        "metrics": metrics_dict,
        "results_df": results_df,
        "moves_df": moves_df
    }


def run_alpha_calibration_parallel(
    instance,
    data_path,
    distance_type,
    distance_method,
    assignment_type,
    driver_init_mode,
    duraciones_df,
    directorio_hist_df,
    labors_real_df,
    valid_cities,
    fechas,
    alphas,
    max_iterations,
    iterations_nums,
    optimization_objs
):
    """Run alpha calibration per (city, date) with multiprocessed iterations."""

    save_dir = f"{data_path}/resultados/alpha_calibration/{instance}/{distance_method}"
    os.makedirs(save_dir, exist_ok=True)

    # Precompute context to reduce pickling overhead
    dist_dict = load_distances(data_path, distance_type, instance, distance_method)
    df_days = {}
    directorio_days = {}
    for city in valid_cities:
        for start_date in fechas:
            df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date="one day lag")
            df_days[(city, start_date)] = df_day[df_day["city"] == city]
            directorio_days[(city, start_date)] = flexible_filter(directorio_hist_df, city=city, date=start_date)

    context = {
        "dist_dict": dist_dict,
        "df_days": df_days,
        "directorio_days": directorio_days,
        "duraciones_df": duraciones_df,
        "assignment_type": assignment_type,
        "distance_method": distance_method,
        "driver_init_mode": driver_init_mode,
        "instance": instance
    }

    all_traces = []
    best_results = {}

    # We'll run each alpha sequentially
    for optimization_variable in optimization_objs:
        print(f"\n🧭 Starting calibration for {optimization_variable.upper()}")

        for alpha in alphas:
            print(f"\n🚀 Running alpha = {alpha:.2f}")

            # Build all tasks for this alpha
            tasks = []
            for start_date in fechas:
                for city in valid_cities:
                    max_iter = max_iterations[city]
                    for iter in range(1, max_iter + 1):
                        tasks.append((instance, distance_method, optimization_variable, alpha, city, start_date, iter, context))

            # Multiprocessing pool
            with Pool(processes=max(cpu_count() - 2, 1)) as pool:
                for result in tqdm(pool.imap(run_single_iteration_task, tasks), total=len(tasks), desc=f"Alpha {alpha}", leave=True):
                    if result is None:
                        continue

                    city = result["city"]
                    start_date = result["date"]
                    iter = result["iteration"]
                    vt_labors = result["vt_labors"]
                    extra_time = result["extra_time"]
                    dist = result["dist"]
                    metrics_dict = result["metrics"]
                    results_df = result["results_df"]
                    moves_df = result["moves_df"]

                    # initialize incumbent for this (city,date) if missing
                    if (city, start_date) not in best_results:
                        best_results[(city, start_date)] = {
                            "alpha": alpha,
                            "vt_labors": vt_labors,
                            "extra_time": extra_time,
                            "dist": dist,
                            "results": results_df,
                            "moves": moves_df,
                            "metrics": metrics_dict,
                        }
                        inc_vt_labors = vt_labors
                        inc_extra_time = extra_time
                        inc_dist = dist
                    else:
                        inc_state = best_results[(city, start_date)]
                        inc_vt_labors = inc_state["vt_labors"]
                        inc_extra_time = inc_state["extra_time"]
                        inc_dist = inc_state["dist"]

                        # update incumbent if better
                        update = False
                        update, new_score, inc_score = should_update_incumbent(
                            optimization_variable,
                            iter_dist=dist,
                            iter_extra_time=extra_time,
                            inc_dist=inc_dist,
                            inc_extra_time=inc_extra_time,
                            dist_norm_factor=dist_norm_factor,
                            extra_time_norm_factor=extra_time_norm_factor,
                        )

                        if update:
                            best_results[(city, start_date)] = {
                                "alpha": alpha,
                                "vt_labors": vt_labors,
                                "extra_time": extra_time,
                                "dist": dist,
                                "results": results_df,
                                "moves": moves_df,
                                "metrics": metrics_dict,
                            }
                            inc_vt_labors = vt_labors
                            inc_extra_time = extra_time
                            inc_dist = dist

                    # Save checkpoint traces
                    if iter in iterations_nums[city]:
                        all_traces.append({
                            "optimization_variable": optimization_variable,
                            "city": city,
                            "date": start_date,
                            "alpha": alpha,
                            "iteration": iter,
                            "vt_labors": vt_labors,
                            "extra_time": extra_time,
                            "dist": dist,
                        })

    # --- Save aggregated results ---
    trace_df = pd.DataFrame(all_traces)
    trace_path = f"{save_dir}/result_traces.csv"
    trace_df.to_csv(trace_path, index=False)

    # best_path = f"{save_dir}/best.pkl"
    # with open(best_path, "wb") as f:
    #     pickle.dump(best_results, f)

    print(f"\n✅ Saved aggregated trace to: {trace_path}")
    print(f"-------- Finished {optimization_variable} --------\n")



if __name__ == "__main__":
    data_path = f'{REPO_PATH}/data'
    # instance_input = input('Input the instance -inst{}- to run?: ')
    # instance, instance_type = process_instance(instance_input)
    instance = 'instAD2'
    
    ''' START RUN PARAMETERS'''
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']
    distance_method = 'haversine'       # Options: ['precalced', 'haversine']

    assignment_type = 'algorithm'

    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']
    ''' END RUN PARAMETERS'''

    fechas = fechas_map(instance)
    initial_day = int(fechas[0].rsplit('-')[2])

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance)
    
    max_iterations = {
        '149': 1300,
        '1': 1000,
        '1004': 1000,
        '126': 150,
        '150': 150,
        '844': 150,
        '830': 150 ,
    }

    iterations_nums = {city:[int(p * max_iterations[city]) for p in np.linspace(0, 1.0, 20)] for city in max_iterations}

    run_alpha_calibration_parallel(
        instance,
        data_path,
        distance_type,
        distance_method,
        assignment_type,
        driver_init_mode,
        duraciones_df,
        directorio_hist_df,
        labors_real_df,
        valid_cities,
        fechas,
        alphas,
        max_iterations,
        iterations_nums,
        optimization_objs=['hybrid', 'driver_distance']
    )
