import os
import pickle
import pandas as pd
from datetime import timedelta

from time import perf_counter
from tqdm import tqdm
from itertools import product

from src.algorithms.offline_algorithms import run_assignment_algorithm
from src.algorithms.online_algorithms import filter_dynamic_df
from src.algorithms.REACT_BUFFER_algorithm import (
    filter_dfs_for_REACT,
    compute_disruption_factor,
    rename_old_solution_columns
)
from src.data.metrics import compute_metrics_with_moves, compute_iteration_metrics, concat_run_results
from src.config.experimentation_config import *
from src.config.config import *
from src.utils.filtering import flexible_filter
from src.utils.utils import get_city_name_from_code, clear_last_n_lines, prep_online_algorithm_inputs
from src.algorithms.solution_search_utils import should_update_incumbent, update_incumbent_state

def run_REACT(instance: str,
    optimization_obj: str,
    distance_method: str,
    time_previous_freeze: int,
    save_results: bool,
    multiprocessing: bool,
    n_processes = None
):
    if multiprocessing:
        pass
    else:
        run_REACT_sequential(
            instance=instance,
            optimization_obj=optimization_obj,
            distance_method=distance_method,
            time_previous_freeze=time_previous_freeze,
            save_results=save_results
        )


def run_REACT_sequential(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    time_previous_freeze: int,
    save_results: bool):
    """
    Ejecuta la simulación online dinámica para una instancia dada.
    Puede correrse como módulo independiente o ser llamado desde otro script.
    """
    
    global_start = perf_counter()

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
        alpha,
        labors_dynamic_df,
        labors_algo_static_df,
        moves_algo_static_df,
        labors_algo_dynamic_df,
        moves_algo_dynamic_df
    ) = prep_online_algorithm_inputs(instance, distance_method, optimization_obj)

    labors_algo_dynamic_df = rename_old_solution_columns(labors_algo_dynamic_df)

    unassigned_services = []
    run_results = []

    # --- Ejecución por ciudad y fecha ---
    for fecha in fechas:
        print(f"{'-'*100}\n▶ Processing date: {fecha} / {fechas[-1]}")

        for city in valid_cities:

            directorio_hist_filtered_df = flexible_filter(
                    directorio_hist_df, city=city, date=fecha
                )
            
            labors_dynamic_filtered_df = filter_dynamic_df(
                labors_dynamic_df=labors_dynamic_df, 
                city=city, 
                fecha=fecha
            )

            labors_algo_dynamic_filt_df = flexible_filter(
                labors_algo_dynamic_df,
                city=city,
                schedule_date=fecha
            ).copy().reset_index(drop=True)

            moves_algo_dynamic_filt_df = flexible_filter(
                moves_algo_dynamic_df,
                city=city,
                schedule_date=fecha
            ).copy().reset_index(drop=True)

            for service_id, service_df in labors_dynamic_filtered_df.groupby('service_id'):
                inc_vt_labors = 0
                inc_extra_time = 1e9
                inc_dist = 1e9
                inc_state = None

                freeze_cutoff = service_df.iloc[0]['created_at'] + timedelta(minutes=time_previous_freeze)

                labors_reassign_df, labors_frozen_df, moves_frozen_df, directorio_online_df = \
                        filter_dfs_for_REACT(
                            labors_df=labors_algo_dynamic_filt_df,
                            moves_df=moves_algo_dynamic_filt_df,
                            directorio_df=directorio_hist_filtered_df,
                            freeze_cutoff=freeze_cutoff,
                        )

                for iter in tqdm(range(1, max_iterations[city] + 1), desc=f"City: {city}", unit="iter", leave=False):
                    ''' START RUN ITERATION FOR A BREAKUP '''
                    def attach_new_labor_to_reassign(labors_df: pd.DataFrame, new_service_df: pd.DataFrame) -> pd.DataFrame:
                        new_service_df = new_service_df.copy()
                        new_service_df['map_start_point'] = new_service_df['start_address_point']
                        new_service_df['map_end_point'] = new_service_df['end_address_point']

                        expanded_labors_df = pd.concat(
                            [labors_df, new_service_df],
                            ignore_index=True
                        )

                        return expanded_labors_df    

                    labors_reassign_iter_df = attach_new_labor_to_reassign(
                        labors_df=labors_reassign_df,
                        new_service_df=service_df,
                    )

                    results_df, moves_df = run_assignment_algorithm( 
                        df_cleaned_template=labors_reassign_iter_df,
                        directorio_df=directorio_online_df,
                        duraciones_df=duraciones_df,
                        day_str=fecha, 
                        ciudad=get_city_name_from_code(city),
                        dist_m1ethod=distance_method,
                        dist_dict=dist_dict,
                        assignment_type='algorithm',
                        alpha=alpha,
                        instance=instance,
                        driver_init_mode=driver_init_mode,
                        update_dist_dict=True
                    )

                    if moves_df.empty:
                        continue

                    metrics = compute_metrics_with_moves(
                        results_df, 
                        moves_df,
                        fechas=[fecha],
                        dist_dict=dist_dict,
                        workday_hours=8,
                        city=city,
                        skip_weekends=False,
                        assignment_type='algorithm',
                        dist_method=distance_method
                    )

                    iter_vt_labors, iter_extra_time, iter_dist = compute_iteration_metrics(metrics)
                    disruption_factor = compute_disruption_factor(
                        new_labors=results_df)

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
                                metrics=metrics
                            )

                        # unpack into your variables (if you still want separate vars)
                        inc_vt_labors = inc_state["vt_labors"]
                        inc_extra_time = inc_state["extra_time"]
                        inc_dist = inc_state["dist"]
                        inc_values = inc_state["values"]

                if inc_state:
                    labors_algo_dynamic_filt_df = pd.concat([labors_frozen_df, inc_state['results']]).reset_index(drop=True)
                    moves_algo_dynamic_filt_df = pd.concat([moves_frozen_df, inc_state['moves']]).reset_index(drop=True)
                else:
                    labors_algo_dynamic_filt_df = labors_frozen_df
                    moves_algo_dynamic_filt_df = moves_frozen_df

            run_results.append([labors_algo_dynamic_filt_df, moves_algo_dynamic_filt_df])
            ''' END RUN ITERATION '''
        
        clear_last_n_lines(2)

    # Build consolidated dfs for this run
    results_df, moves_df = concat_run_results(run_results)

    if save_results:
        # ------ Ensure output directory exists ------
        output_dir = os.path.join(data_path, 
                                    "resultados", 
                                    "online_operation", 
                                    instance,
                                    distance_method
                                    )
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing
        
        with open(os.path.join(output_dir, f'res_algo_REACT.pkl'), 'wb') as f:
            pickle.dump([results_df, moves_df], f)

    print(f'\n ✅ Completed REACT algorithm in {round(perf_counter() - global_start, 1)}s total.\n')
    return True


# --- Allow running standalone ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()

    run_REACT(args.instance)

