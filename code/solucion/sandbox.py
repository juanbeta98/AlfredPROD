import pandas as pd

import sys
import os

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

from src.data_load import load_tables, load_artificial_instance, load_distances
from src.filtering import filter_labors_by_city, filter_labors_by_date
from src.plotting import plot_indicator_evolution, plot_metrics_comparison, plot_gantt_labors_by_driver, plot_results
from src.metrics import collect_vt_metrics_range, show_day_report_dayonly, compute_indicators, collect_results_to_df
from src.config import *
from src.utils import get_city_name_from_code, get_city_name
from code.src.offline_algorithms import build_driver_movements

data_path = '../data'
instance = 'inst3'
directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path, generate_labors=False)
labors_real_df = load_artificial_instance(data_path, instance, labors_raw_df)
dist_dict = load_distances(data_path, 'real', instance)

# Parámetros de fecha    
fechas = pd.date_range("2025-09-08", "2025-09-13").strftime("%Y-%m-%d").tolist()
# labors_algo_df, moves_algo_df = collect_results_to_df(data_path, instance, fechas, assignment_type='algorithm')
labors_hist_df, moves_hist_df = collect_results_to_df(data_path, instance, fechas, assignment_type='historic')