import pandas as pd
import pickle
from datetime import timedelta
from multiprocessing import Pool

from time import perf_counter

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_tables, load_artificial_instance, load_distances
from src.filtering import filter_labors_by_date
from src.metrics import collect_results_from_dicts, compute_metrics_with_moves
from src.utils import codificacion_ciudades
from src.config import *
from src.pipeline import run_city_pipeline
from src.alpha_tuning_utils import print_header, print_iter_header, fechas, alphas, iterations_nums, \
    dist_norm_factor, extra_time_norm_factor


# ——————————————————————————
# Configuración previa
# ——————————————————————————
import pickle
import pandas as pd
from time import perf_counter

if __name__ == "__main__":
    data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
    # instance = 'inst4a'
    instance_input = input('Input the instance -inst4{}- to run?: ')
    instance = f'inst4{instance_input}'

    distance_type = 'osrm'
    distance_method = 'precalced'
    assignment_type = 'algorithm'
    # save_path = f'{data_path}/resultados/artif_inst/{instance}'

    max_iterations = max(iterations_nums)

    ### Optimization config
    optimization_variable = "hybrid"    # "driver_distance", "extra_time", "hybrid"

    print_header(instance, distance_type, assignment_type, optimization_variable)    

    directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
    labors_real_df = load_artificial_instance(data_path, instance, labor_raw_df)

    for alpha in alphas:
        print_iter_header(alpha)

        start = perf_counter()
        inc_vt_labors = 0
        inc_extra_time = 1e9
        inc_dist = 1e9

        for i in range(1, max_iterations + 1):  # run full loop only once

            ''' START RUN ITERATION FOR SET ALPHA '''
            run_results = []
            for start_date in fechas:
                dist_dict = load_distances(data_path, distance_type, instance)

                df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')
                results = []
                for city in valid_cities:
                    res = run_city_pipeline(city, start_date, df_day, directorio_df, duraciones_df, 
                                            assignment_type, alpha=0, DIST_DICT=dist_dict[city], 
                                            dist_method=distance_method, instance=instance)
                    results.append(res)

                results_by_city = {
                    city: (df_cleaned, df_moves, n_drivers)
                    for city, df_cleaned, df_moves, n_drivers in results
                }
                run_results.append(results_by_city)
            ''' END RUN ITERATION '''

            # Build consolidated dfs for this run
            results_df, moves_df = collect_results_from_dicts(run_results, fechas, assignment_type)

            metrics = {}
            for city in valid_cities:                    
                metrics[city] = compute_metrics_with_moves(
                    results_df, moves_df,
                    fechas=fechas,
                    dist_dict=dist_dict,
                    workday_hours=8,
                    city=city,
                    skip_weekends=False,
                    assignment_type=assignment_type,
                    dist_method=distance_method
                )

            iter_vt_labors = sum(sum(metrics['vt_count']) for metrics in metrics.values())
            iter_extra_time = round(sum(sum(metrics['driver_extra_time']) for metrics in metrics.values()), 2)
            iter_dist = round(sum(sum(metrics['driver_move_distance']) for metrics in metrics.values()), 2)

            ''' UPDATE INCUMBENT '''
            if iter_vt_labors >= inc_vt_labors:
                update = False

                if optimization_variable == "driver_distance":
                    update = iter_dist < inc_dist

                elif optimization_variable == "driver_extra_time":
                    update = iter_extra_time < inc_extra_time or inc_extra_time == 0

                elif optimization_variable == "hybrid":
                    # Puntaje nuevo
                    norm_dist_new = iter_dist / dist_norm_factor
                    norm_extra_new = iter_extra_time / extra_time_norm_factor
                    new_score = norm_dist_new + norm_extra_new

                    # Puntaje incumbente
                    norm_dist_inc = inc_dist / dist_norm_factor
                    norm_extra_inc = inc_extra_time / extra_time_norm_factor
                    inc_score = norm_dist_inc + norm_extra_inc

                    update = new_score < inc_score

                if update:
                    duration = round(perf_counter() - start, 1)
                    inc_vt_labors = iter_vt_labors
                    inc_extra_time = iter_extra_time
                    inc_dist = iter_dist
                    inc_values = (i, inc_extra_time, inc_dist, duration)

                    inc_results = results_df
                    inc_moves = moves_df
                    inc_metrics = metrics

            ''' CHECKPOINT SAVE '''
            if i in iterations_nums:
                duration = round(perf_counter() - start, 1)
                print(f'{i} \t \t{inc_values[0]} \t\t{duration} \t\t{inc_vt_labors} \t\t{"na"} \t\t{round(inc_extra_time,1)} \t{inc_dist}')

                save_full_path = f'{data_path}/resultados/alpha_calibration/{instance}/{optimization_variable}/res_{alpha:.1f}_{i}.pkl'

                with open(save_full_path, 'wb') as f:
                    pickle.dump([inc_values, duration, inc_results, inc_moves, inc_metrics], f)

    print(f'-------- Successfully ran -{instance}- --------')



# if __name__ == "__main__":
#     data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
#     instance = 'inst3'
#     distance_type = 'real'
#     assignment_type = 'algorithm'
#     save_path = f'{data_path}/resultados/artif_col_inst/{instance}'

#     ### Iteration parameters
#     alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     iterations_nums = [10, 25, 50, 100, 250, 500]  # checkpoints
#     max_iterations = max(iterations_nums)

#     print(f'------------------ Running alpha tuning for -{instance}- ------------------')
#     print(f'------- Run mode: sequential')
#     print(f'------- Distances: {distance_type}')
#     print(f'------- Assignment: {assignment_type}')

#     directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
#     labors_real_df = load_artificial_instance(data_path, instance, labor_raw_df)
#     dist_dict = load_distances(data_path, 'real', instance)

#     # Parámetros de fecha    
#     fechas = pd.date_range("2025-09-08", "2025-09-14").strftime("%Y-%m-%d").tolist()

#     for alpha in alphas[::-1]:
#         print(f'\n- alpha: {alpha}')
#         print('num-iter \tfound_on \trun_time \tnum_ser \tserv/con \textra_time \tdriv_dist')

#         start = perf_counter()
#         found_on = 0
#         inc_vt_labors = 0
#         inc_extra_time = 0
#         inc_dist = 1e9

#         for i in range(1, max_iterations + 1):  # run full loop only once

#             ''' START RUN ITERATION FOR SET ALPHA '''
#             run_results = []
#             for start_date in fechas:
#                 df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')
#                 results = []
#                 for city in valid_cities:
#                     res = run_city_pipeline(city, start_date, df_day, directorio_df,
#                                             duraciones_df, assignment_type, alpha, dist_dict)
#                     results.append(res)

#                 results_by_city = {
#                     city: (df_cleaned, df_moves, n_drivers)
#                     for city, df_cleaned, df_moves, n_drivers in results
#                 }
#                 run_results.append(results_by_city)
#             ''' END RUN ITERATION '''

#             # Build consolidated dfs for this run
#             results_df, moves_df = collect_results_from_dicts(run_results, fechas, assignment_type)

#             metrics = {}
#             for city in valid_cities:                    
#                 metrics[city] = compute_metrics_with_moves(
#                                                                 results_df, moves_df,
#                                                                 fechas=fechas,
#                                                                 dist_dict=dist_dict,
#                                                                 workday_hours=8,
#                                                                 city=city,
#                                                                 skip_weekends=False,
#                                                                 assignment_type=assignment_type,
#                                                                 distance_method='haversine'
#                                                             )

#             iter_vt_labors = sum(sum(metrics['vt_count']) for city, metrics in metrics.items())
#             iter_extra_time = round(sum(sum(metrics['tiempo_extra']) for city, metrics in metrics.items()), 2)
#             iter_dist = round(sum(sum(metrics['driver_move_distance']) for city, metrics in metrics.items()), 2)

#             ''' UPDATE INCUMBENT '''
#             if iter_vt_labors >= inc_vt_labors and iter_dist < inc_dist:
#                 found_on = i
#                 inc_vt_labors = iter_vt_labors
#                 inc_extra_time = iter_extra_time
#                 inc_dist = iter_dist

#                 inc_results = results_df
#                 inc_moves = moves_df
#                 inc_metrics = metrics

#             ''' CHECKPOINT SAVE '''
#             if i in iterations_nums:
#                 duration = round(perf_counter() - start, 1)
#                 print(f'{i} \t \t{found_on} \t\t{duration} \t\t{inc_vt_labors} \t\t{"na"} \t\t{round(inc_extra_time,1)} \t{inc_dist}')
#                 save_full_path = f'{data_path}/resultados/alpha_calibration/alpha_{alpha:.1f}/res_{i}.pkl'
#                 with open(save_full_path, 'wb') as f:
#                     pickle.dump([found_on, duration, inc_results, inc_moves, inc_metrics], f)

#     print(f'-------- Successfully ran -{instance}- --------')




# if __name__ == "__main__":
#     data_path = '/Users/juanbeta/Library/CloudStorage/GoogleDrive-juan.beta98@gmail.com/My Drive/Work/Alfred/Alfred/data'
#     instance='inst3'
#     distance_type = 'real'
#     assignment_type = 'algorithm'
#     # assignment_type = 'historic'
#     save_path = f'{data_path}/resultados/artif_col_inst/{instance}'

#     ### Iteration parameters
#     alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#     iterations_nums = [10, 25, 50, 100, 250, 500]
#     # iterations_nums = [int(i/10) for i in iterations_nums]

#     print(f'------------------ Running alpha tunning for -{instance}- ------------------')
#     print(f'------- Run mode: sequential')
#     print(f'------- Distances: {distance_type}')
#     print(f'------- Assignment: {assignment_type}')

#     # data_path = '../data'
#     directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
#     labors_real_df = load_artificial_instance(data_path, instance, labor_raw_df)
#     dist_dict = load_distances(data_path, 'real', instance)

#     # Parámetros de fecha    
#     fechas = pd.date_range("2025-09-08", "2025-09-14").strftime("%Y-%m-%d").tolist()

#     results_dicts = []

#     for alpha in alphas[::]:
#         print(f'\n- alpha: {alpha}')
#         print('num-iter \tfound_on \trun_time \tnum_ser \tserv/con \textra_time \tdriv_dist')
        
#         for num_iterations in iterations_nums:
#             start = perf_counter()
#             found_on = 0
#             inc_vt_labors = 0
#             inc_extra_time = 0
#             inc_dist = 1e9

#             for i in range(num_iterations):
#                 ''' START RUN ITERATION FOR SET ALPHA AND NUM INTERATIONS'''

#                 run_results = []

#                 for start_date in fechas:
#                     df_day = filter_labors_by_date(labors_real_df, start_date=start_date, end_date='one day lag')
#                     results = []
#                     for city in valid_cities:
#                         res = run_city_pipeline(city, start_date, df_day, directorio_df, 
#                                                 duraciones_df, assignment_type, alpha, dist_dict)
#                         results.append(res)

#                     results_by_city = {city: (df_cleaned, df_moves, n_drivers)
#                                     for city, df_cleaned, df_moves, n_drivers in results}
#                     run_results.append(results_by_city)
                
#                 ''' END RUN ITERATION FOR SET ALPHA AND NUM INTERATIONS'''


#                 ''' START EVALUATION OF RUN '''
#                 # Build consolidated dfs for this run
#                 results_df, moves_df = collect_results_from_dicts(run_results, fechas, assignment_type)

#                 metrics = {}
#                 for city in valid_cities:                    
#                     metrics[city] = compute_metrics_with_moves( results_df, 
#                                                                 moves_df,
#                                                                 fechas=fechas,
#                                                                 dist_dict=dist_dict,
#                                                                 workday_hours=8,
#                                                                 city=city,
#                                                                 skip_weekends=False,
#                                                                 assignment_type=assignment_type,
#                                                                 distance_method='haversine')
                
#                 iter_vt_labors = sum(sum(metrics['vt_count']) for city,metrics in metrics.items())
#                 iter_extra_time = round(sum(sum(metrics['tiempo_extra']) for city,metrics in metrics.items()),2)
#                 iter_dist = round(sum(sum(metrics['driver_move_distance']) for city,metrics in metrics.items()),2)
#                 ''' END EVALUATION OF RUN '''

#                 ''' UPDATE INCUMBENT'''
#                 if iter_vt_labors >= inc_vt_labors and iter_dist < inc_dist :
#                     found_on = i
#                     inc_vt_labors = iter_vt_labors
#                     inc_extra_time = round(iter_extra_time,2)
#                     inc_dist = round(iter_dist,2)

#                     inc_results = results_df
#                     inc_moves = moves_df
#                     inc_metrics = metrics

#             end = perf_counter()
#             duration = round(end - start, 1)
            
#             print(f'{num_iterations} \t \t{found_on} \t\t{duration} \t\t{inc_vt_labors} \t\t{"na"} \t\t{inc_extra_time} \t{inc_dist}')
#             '''
#             Saving results
#             '''
#             save_path = f'{data_path}/resultados/alpha_calibration/alpha_{alpha:.1f}/res_{num_iterations}.pkl'
#             with open(save_path, 'wb') as f:
#                 pickle.dump([inc_results, inc_moves, inc_metrics], f)

#     print(f'-------- Successfully ran -{instance}- --------')