import pandas as pd
import pickle
from datetime import timedelta

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_inputs, load_distances
from src.filtering import filter_labors_by_date, flexible_filter # type: ignore
from src.utils import process_instance, get_max_drivers, consolidate_run_results

from src.config import * # type: ignore
from src.experimentation_config import *
from src.pipeline import run_city_pipeline


# ——————————————————————————
# Configuración previa
# ——————————————————————————
if __name__ == "__main__":
    data_path = f'{REPO_PATH}/data'
    instance_input = input('Input the instance -inst{}- to run?: ')
    instance, instance_type = process_instance(instance_input)
    
    
    distance_type = 'osrm'              # Options: ['osrm', 'manhattan']
    distance_method = 'haversine'      # Options: ['precalced', 'haversine']
    
    assignment_types = ['algorithm', 'historic']  # 👈 choose one or both

    driver_init_mode = 'historic_directory' # Options: ['historic_directory', 'driver_directory']

    save_path = f'{data_path}/resultados'

    print(f'------------------ Running -{instance}- ------------------')
    print(f'------- Run mode: sequential')
    print(f'------- Distances: {distance_method}')
    print(f'------- Driver init mode: {driver_init_mode}\n')

    # Load shared inputs only once
    (directorio_df, labor_raw_df, cities_df, duraciones_df,
        valid_cities, labors_real_df, directorio_hist_df) = load_inputs(data_path, instance)

    # Parámetros de fecha
    fechas = fechas_dict[instance]
    initial_day = int(fechas[0].rsplit('-')[2])

    for assignment_type in assignment_types:
        print(f"\n=== Running assignment_type = {assignment_type} ===")

        for start_date in fechas:

            dist_dict = load_distances(data_path, distance_type, instance, distance_method)
            df_day = filter_labors_by_date(
                labors_real_df, start_date=start_date, end_date='one day lag'
            )
            results = []

            for city in valid_cities:

                max_drivers_num = get_max_drivers(instance, city, max_drivers, start_date, initial_day)
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
                    alpha=0,
                    DIST_DICT=dist_dict.get(city, {}),
                    dist_method=distance_method,
                    max_drivers_num=max_drivers_num,
                    instance=instance,
                    driver_init_mode=driver_init_mode
                    )
                results.append(res)
                print('       ✅ ' + city)

            # Resultados como diccionario
            results_by_city = consolidate_run_results(results)

            # Guardar en pickle
            if assignment_type == 'algorithm':
                save_full_path = f'{save_path}/offline_operation/{instance}/{distance_method}/res_{start_date}.pkl'
            elif assignment_type == 'historic':
                save_full_path = f'{save_path}/alfred_baseline/{instance}/{distance_method}/res_hist_{start_date}.pkl'

            with open(save_full_path, 'wb') as f:
                pickle.dump(results_by_city, f)

            print('  ✅ ' + start_date)

    print("\n✅ Finished all runs")