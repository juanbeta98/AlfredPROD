import os
import pickle
import pandas as pd
from datetime import timedelta

from time import perf_counter
from tqdm import tqdm
from itertools import product

from src.algorithms.INSERT_algorithm import insert_single_labor, get_drivers, filter_dynamic_df
from src.utils.utils import prep_online_algorithm_inputs, compute_workday_end
from src.config.experimentation_config import *
from src.config.config import *

from src.utils.filtering import flexible_filter


def run_INSERT(
    instance: str,
    optimization_obj: str,
    distance_method: str,
    save_results: bool,
    experiment_type: str = 'online_operation'
):
    """
    Ejecuta la simulación online dinámica para una instancia dada.
    Puede correrse como módulo independiente o ser llamado desde otro script.
    """

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
        alpha,
        labors_dynamic_df,
        labors_algo_dynamic_df,
        moves_algo_dynamic_df,
        postponed_labors
    ) = prep_online_algorithm_inputs(
        instance, 
        distance_method, 
        optimization_obj,
        experiment_type=experiment_type)

    labors_dynamic_df["latest_arrival_time"] = labors_dynamic_df["schedule_date"] + timedelta(minutes=TIEMPO_GRACIA)
    unassigned_services = []

    # --- Ejecución por ciudad y fecha ---
    for city, fecha in tqdm(product(valid_cities, fechas),
                        total=len(valid_cities) * len(fechas),
                        desc="Processing city/date pairs",
                        unit="iteration",
                        leave=False):
        
        # if city != '149' or fecha != '2026-01-06': continue

        labors_dynamic_filtered_df = filter_dynamic_df(
            labors_dynamic_df=labors_dynamic_df, 
            city=city, 
            fecha=fecha
        )

        dist_dict_city = dist_dict.get(city, {})

        drivers = get_drivers(
            labors_algo_df=labors_algo_dynamic_df,
            directorio_hist_df=directorio_hist_df,
            city=city,
            fecha=fecha,
            get_all=True,
        )

        new_postponed_labors = []

        workday_end_dt = compute_workday_end(
            day_str=fecha,
            workday_end_str=WORKDAY_END,
            tzinfo="America/Bogota"
        )

        # for _, new_labor in labors_dynamic_filtered_df.iterrows():
        for service_id, service_df in labors_dynamic_filtered_df.groupby("service_id"):

            # Work on a temporary copy until verified valid
            tmp_labors = labors_algo_dynamic_df.copy()
            tmp_moves = moves_algo_dynamic_df.copy()
            tmp_unassigned = list(unassigned_services)

            service_block_after_first_afterhours_nontransport = False

            curr_end_time = None
            curr_end_pos = None
            # success_flag = True

            # Iterate through the labors of the service IN ORDER
            for i, labor in service_df.sort_values("labor_sequence").iterrows():
                
                success, postponed, tmp_labors, tmp_moves, curr_end_time, curr_end_pos = insert_single_labor(
                    labor=labor,
                    labors_df=tmp_labors,
                    moves_df=tmp_moves,
                    curr_end_time=curr_end_time,
                    curr_end_pos=curr_end_pos,
                    directorio_hist_df=directorio_hist_df,
                    unassigned_services=unassigned_services,
                    drivers=drivers,
                    city=city,
                    fecha=fecha,
                    distance_method=distance_method,
                    dist_dict=dist_dict_city,
                    duraciones_df=duraciones_df,
                    vehicle_transport_speed=VEHICLE_TRANSPORT_SPEED,
                    alfred_speed=ALFRED_SPEED,
                    tiempo_alistar=TIEMPO_ALISTAR,
                    tiempo_finalizacion=TIEMPO_FINALIZACION,
                    tiempo_gracia=TIEMPO_GRACIA,
                    early_buffer=TIEMPO_PREVIO,
                    workday_end_dt=workday_end_dt,
                    service_block_after_first_afterhours_nontransport=service_block_after_first_afterhours_nontransport,
                    instance=instance,
                    selection_mode='min_total_distance',
                    update_dist_dict=True
                )

                if postponed:
                    new_postponed_labors.append({
                        "labor_row": labor,
                        "service_id": service_id,
                        "reason": 'After workday hours',
                        "prev_end": curr_end_time,
                        "workday_end": workday_end_dt
                    })

                if not success:
                    break

            # ---------- FINAL COMMIT OR DISCARD ----------
            if success:
                labors_algo_dynamic_df = tmp_labors
                moves_algo_dynamic_df = tmp_moves
            else:
                unassigned_services.append(service_df)

                # Cleanup logic, include all the services in the
                service_df['actual_status'] = 'FAILED'
                labors_algo_dynamic_df = pd.concat([labors_algo_dynamic_df, service_df])
    
    postponed_labors += new_postponed_labors

    if save_results:
        # ------ Ensure output directory exists ------
        output_dir = os.path.join(data_path, 
                                    "resultados", 
                                    experiment_type, 
                                    instance,
                                    distance_method
                                    )
        os.makedirs(output_dir, exist_ok=True)  # Creates folder if missing

        extra_output_dir = os.path.join(output_dir, 'extra_info')
        os.makedirs(extra_output_dir, exist_ok=True)  # Creates folder if missing
        
        with open(os.path.join(output_dir, f'res_algo_INSERT.pkl'), 'wb') as f:
            pickle.dump([labors_algo_dynamic_df, moves_algo_dynamic_df, postponed_labors], f)

        with open(os.path.join(extra_output_dir, f'unassigned_INSERT.pkl'), 'wb') as f:
            pickle.dump(unassigned_services, f)

    print(f'\n ✅ Completed INSERT algorithm in {round(perf_counter() - start, 1)}s total.\n')
    return {"unassigned_labors": len(unassigned_services)}


# --- Allow running standalone ---
if __name__ == "__main__":
    instance='instAD2b'
    optimization_obj='hybrid'
    distance_method='haversine'

    import argparse

    parser = argparse.ArgumentParser(description="Run online dynamic insertion simulation")
    parser.add_argument("--instance", required=True, help="Instance name (e.g. instAD2b)")
    args = parser.parse_args()

    run_INSERT(
        # args.instance
        instance=instance,
        optimization_obj=optimization_obj,
        distance_method=distance_method)
