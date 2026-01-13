import pandas as pd

import os

import ipywidgets as widgets
from ipywidgets import interact

from src.data.data_load import load_tables, load_online_instance, load_distances
from src.data.solution_load import load_solution_dfs
from src.utils.filtering import flexible_filter
from src.utils.plotting import plot_metrics_comparison_dynamic
from src.data.metrics import collect_results_to_df, compute_metrics_with_moves, get_day_plotting_df
from src.config.experimentation_config import *
from src.config.config import *

data_path = '../data'

distance_type = 'osrm'              # Options: ['osrm', 'manhattan']

optimization_obj = 'driver_distance'

directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)

metricas = ['service_count', 'vt_count', 'num_drivers', 'driver_extra_time', 'driver_move_distance']

city = 'ALL'

algo_selection = {
    'hist': True,
    'offline': True,
    'INSERT': True,
    'INSERT_BUFFER': True,
    'REACT': True,
    'REACT_BUFFER': True
}

metrics_with_gap = [
    "total_distance",
    "driver_move_distance",
    "labor_extra_time",
    "driver_extra_time"
]

def generate_online_results_summary(instance, dist_method):

    labors_real_df, labors_static_df, labors_dynamic_df = load_online_instance(data_path, instance, labors_raw_df)
    dist_dict = load_distances(data_path, distance_type, instance, dist_method)

    fechas = fechas_map(instance)

    save_dir = f'{data_path}/resultados/online_operation/summary/{instance}/{dist_method}/'

    os.makedirs(save_dir, exist_ok=True)

    algorithms = []

    if algo_selection['hist']:
        labors_hist_df, moves_hist_df, postponed_labors_hist = load_solution_dfs(
            data_path, 
            instance, 
            dist_method,
            algorithm='hist',
            include_all_city=True
        )
        algorithms.append({
            "name": "Histórico",
            "labors_df": labors_hist_df,
            "moves_df": moves_hist_df,
            "type": "historic",
            "color": "#C0392B",  # 🔴 Deep crimson red – “you were here”
            "visible": True,
        })

    if algo_selection['offline']:    
        labors_algo_baseline_df, moves_algo_baseline_df, postponed_labors_OFFLINE = load_solution_dfs(
            data_path, 
            instance, 
            dist_method,
            algorithm='OFFLINE',
            include_all_city=True
        )
        algorithms.append({
            "name": "Offline",
            "labors_df": labors_algo_baseline_df,
            "moves_df": moves_algo_baseline_df,
            "type": "algorithm",
            "color": "#2980B9",  # 🔵 Royal blue – “best possible solution”
            "visible": True,
        })

    labors_algo_static_df, moves_algo_static_df, postponed_labors_ONLINE_static = load_solution_dfs(
        data_path,
        instance,
        dist_method,
        algorithm='ONLINE_static',
        include_all_city=True
    )

    if algo_selection['INSERT']:
        labors_algo_INSERT_df, moves_algo_INSERT_df, postponed_labors_INSERT = load_solution_dfs(
            data_path,
            instance,
            dist_method,
            algorithm='INSERT',
            include_all_city=True
        )
        algorithms.append({
            "name": "INSERT",
            "labors_df": labors_algo_INSERT_df,
            "moves_df": moves_algo_INSERT_df,
            "type": "algorithm",
            "color": "#27AE60",  # 🟢 Vibrant green – “first smart improvement”
            "visible": True,
        })

    if algo_selection['INSERT_BUFFER']:
        labors_algo_INSERT_BUFFER_df, moves_algo_INSERT_BUFFER_df, postponed_labors_INSERT_BUFFER = load_solution_dfs(
            data_path,
            instance,
            dist_method,
            algorithm='INSERT_BUFFER',
            include_all_city=True
        )
        algorithms.append({
            "name": "INSERT_BUFFER",
            "labors_df": labors_algo_INSERT_BUFFER_df,
            "moves_df": moves_algo_INSERT_BUFFER_df,
            "type": "algorithm",
            "color": "#16A085",
            "visible": True,
        })

    if algo_selection['REACT']:
        labors_algo_REACT_df, moves_algo_REACT_df, postponed_labors_REACT_BUFFER_0 = load_solution_dfs(
            data_path, 
            instance, 
            dist_method,
            algorithm='REACT_BUFFER_0',
            include_all_city=True
        )
        algorithms.append({
            "name": "REACT",
            "labors_df": labors_algo_REACT_df,
            "moves_df": moves_algo_REACT_df,
            "type": "algorithm",
            "color": "#E67E22",  # 🟠 Bold orange – “fast adaptation”
            "visible": True,
        })

    if algo_selection['REACT_BUFFER']:
        labors_algo_REACT_BUFFER_df, moves_algo_REACT_BUFFER_df, postponed_labors_REACT_BUFFER = load_solution_dfs(
            data_path, 
            instance, 
            dist_method,
            algorithm='REACT_BUFFER',
            include_all_city=True
        )
        algorithms.append({
            "name": "REACT_BUFFER",
            "labors_df": labors_algo_REACT_BUFFER_df,
            "moves_df": moves_algo_REACT_BUFFER_df,
            "type": "algorithm",
            "color": "#9B59B6",  # 🟣 Electric purple – “intelligent batching”
            "visible": True,
        })

    # algo_copy = [{**algo, "visible": algo_selection[algo["name"]]} for algo in algorithms]
    plot_metrics_comparison_dynamic(
        algorithms,
        city='ALL',
        metricas=metricas,
        dist_dict=dist_dict,
        fechas=fechas,
        metrics_with_gap=metrics_with_gap,
        save_dir=save_dir
    )


for instance in ['AD3', 'AD4', 'RD3', 'RD4']:
    for dist_method in ['haversine', 'precalced']:
        try:
            generate_online_results_summary(f'inst{instance}', dist_method)
        except:
            print(f'Not possible to generate solution summary for {instance} - {dist_method}')

        