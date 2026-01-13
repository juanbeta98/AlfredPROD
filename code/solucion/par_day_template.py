import pandas as pd
import pickle
from datetime import timedelta
from multiprocessing import Pool

import sys
import os

# sys.path.append(os.path.abspath(os.path.join('../src')))  # Adjust as needed

# from src.distance_utils import distance # type: ignore
from src.data_load import load_tables # type: ignore
from src.filtering import filter_labors_by_date
from src.utils import codificacion_ciudades
from src.config import * # type: ignore
from src.pipeline import run_city_pipeline


# ——————————————————————————
# Configuración previa
# ——————————————————————————
if __name__ == "__main__":
    # Pre-cargar globals para todos
    # with open('distances.pkl', 'rb') as f:
    #     DIST_DICT = pickle.load(f)

    data_path = '../data'
    directorio_df, df_dist, cities_df, valid_cities = load_tables(data_path)

    # Parámetros de fecha
    start_date = "2025-07-07"

    # Filtrar por rango de fechas (start_date -> start_date+1)
    end_date = pd.to_datetime(start_date) + timedelta(days=4)
    df_day = filter_labors_by_date(df_dist, start_date=start_date, end_date=end_date)

    # Paralelizar
    with Pool(processes=None) as pool:  # ajusta procesos según tu CPU
        results = pool.starmap(run_city_pipeline, [(c, start_date, df_day, 
                                                    directorio_df) for c in valid_cities])

    # Resultados como diccionario
    results_by_city = {city: (df_cleaned, df_moves, n_drivers) for city, df_cleaned, df_moves, n_drivers in results}

    # Guardar en pickle
    with open(f'{data_path}/resultados/parallel/res_{start_date}.pkl', 'wb') as f:
        pickle.dump(results_by_city, f)
