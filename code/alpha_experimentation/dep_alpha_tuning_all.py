import pandas as pd
import pickle
from datetime import timedelta
from multiprocessing import Pool

from time import perf_counter

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_inputs, load_distances
from src.filtering import filter_labors_by_date, flexible_filter
from src.metrics import collect_results_from_dicts, compute_metrics_with_moves, compute_iteration_metrics
from src.utils import process_instance, get_max_drivers, consolidate_run_results
from src.config import *
from src.experimentation_config import instance_map, fechas_dict, max_drivers
from src.pipeline import run_city_pipeline
from src.alpha_tuning_utils import print_header, print_iter_header, should_update_incumbent, update_incumbent_state, \
    metrics, alphas, iterations_nums, dist_norm_factor, extra_time_norm_factor

from tqdm import tqdm

# ——————————————————————————
# Configuración previa
# ——————————————————————————

if __name__ == "__main__":
    data_path = f'{REPO_PATH}/data'
    instance_input = input('Input the instance -inst{}- to run?: ')
    instance, instance_type = process_instance(instance_input)
    
    ''' START RUN PARAMETERS'''
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']
    distance_method = 'precalced'       # Options: ['precalced', 'haversine']

    assignment_type = 'algorithm'

    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']
    ''' END RUN PARAMETERS'''

    max_iterations = max(iterations_nums)
    fechas = fechas_dict[instance]
    initial_day = int(fechas[0].rsplit('-')[2])

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance)

    ### Optimization config
    # for optimization_variable in [metrics[2]]:
    for optimization_variable in metrics:
        print_header(instance, distance_method, assignment_type, optimization_variable)    

        for alpha in alphas:
            print_iter_header(alpha) 

            start = perf_counter()
            inc_vt_labors = 0
            inc_extra_time = 1e9
            inc_dist = 1e9

            # 🔄 Progress bar replaces manual iteration printing
            for iter in tqdm(range(1, max_iterations + 1), desc=f"Alpha {alpha}", unit="iter"):
                ''' START RUN ITERATION FOR SET ALPHA '''
                run_results = []
                for start_date in fechas:
                    dist_dict = load_distances(data_path, distance_type, instance, distance_method=distance_method)
                    df_day = filter_labors_by_date(
                        labors_real_df, start_date=start_date, end_date='one day lag'
                    )
                    results = []
                    
                    for city in valid_cities:
                        directorio_hist_filtered_df = flexible_filter(
                            directorio_hist_df, city=city, date=start_date
                        )

                        res = run_city_pipeline(
                            city, 
                            start_date, 
                            df_day, 
                            directorio_hist_filtered_df, 
                            duraciones_df,
                            assignment_type, 
                            alpha=alpha,
                            DIST_DICT=dist_dict.get(city, {}),
                            dist_method=distance_method,
                            instance=instance,
                            driver_init_mode=driver_init_mode
                        )
                        results.append(res)

                    results_by_city = consolidate_run_results(results)
                    run_results.append(results_by_city)
                ''' END RUN ITERATION '''

                # Build consolidated dfs for this run
                results_df, moves_df = collect_results_from_dicts(
                    run_results, fechas, assignment_type
                )

                metrics = {}
                for city in valid_cities:
                    metrics[city] = compute_metrics_with_moves(
                        results_df, 
                        moves_df,
                        fechas=fechas,
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
                        optimization_variable,
                        iter_dist=iter_dist,
                        iter_extra_time=iter_extra_time,
                        inc_dist=inc_dist,
                        inc_extra_time=inc_extra_time,
                        dist_norm_factor=dist_norm_factor,
                        extra_time_norm_factor=extra_time_norm_factor
                    )

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
                    inc_results = inc_state["results"]
                    inc_moves = inc_state["moves"]
                    inc_metrics = inc_state["metrics"]


                ''' CHECKPOINT SAVE '''
                if iter in iterations_nums:
                    duration = round(perf_counter() - start, 1)
                    tqdm.write(  # ✅ ensures checkpoint is printed on its own line
                        f"{iter} \t{inc_values[0]} \t\t"
                        f"{round(duration)}s \t\t{inc_vt_labors} \t\t"
                        f"{round(inc_extra_time,1)} \t{inc_dist} km"
                    )

                    save_full_path = (
                        f"{data_path}/resultados/alpha_calibration/{instance}/"
                        f"{distance_method}/{optimization_variable}/res_{alpha:.1f}_{iter}.pkl"
                    )
                    with open(save_full_path, 'wb') as f:
                        pickle.dump([inc_values, duration, inc_results, inc_moves, inc_metrics], f)


        print(f'-------- Successfully ran -{instance}- for objective {optimization_variable} --------\n')


