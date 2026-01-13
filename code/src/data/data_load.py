import pandas as pd
import numpy as np
import ast

import pickle
import os

from pathlib import Path
from typing import Tuple, Union, Optional
import logging

from src.utils.filtering import filter_valid_services
from src.data.preprocessing import generate_labors_raw_df
from src.config.config import *
from src.config.experimentation_config import instance_map, codificacion_ciudades

logger = logging.getLogger(__name__)

def load_tables(
    data_path: Union[str, Path],
    generate_labors: bool = False,
    tz: str = "America/Bogota",
    strict_datetime_format: Optional[str] = "%Y-%m-%d %H:%M:%S.%f %z",
    read_directorio: bool = True,
    read_cities: bool = True,
    read_durations: bool = True,
    read_distances: bool = True,
    distance_type: str = None
    ):
    """
    Carga las tablas preprocesadas de ALFRED con la opción de generar o leer 
    directamente el dataset pesado `labors_raw_df`.

    Comportamiento
    --------------
    - Si `generate_labors=True` o si no existe `labors_raw_df.csv`, 
      se reconstruye `labors_raw_df` mediante `generate_labors_raw_df` 
      y se guarda en `data_clean/labors_raw_df.csv`.
    - En caso contrario, se carga el CSV guardado (mucho más rápido).
    - Otras tablas auxiliares (`directorio_df`, `cities_df`, `duraciones_df`) 
      se leen directamente de sus archivos procesados si están disponibles.

    Parámetros
    ----------
    data_path : str | Path
        Ruta base del proyecto con subcarpetas `data_raw/`, `data_pre/`, y `data_clean/`.
    generate_labors : bool, opcional
        Si es True, fuerza la regeneración de `labors_raw_df` desde los CSV crudos.
        Por defecto = False.
    tz : str, opcional
        Zona horaria de destino para conversión de fechas (solo usado si se genera).
    strict_datetime_format : str | None, opcional
        Formato estricto para parseo de fechas, usado por `generate_labors_raw_df`.
    read_directorio : bool, opcional
        Si True, carga `data_pre/directorio_df.csv`. Por defecto = True.
    read_cities : bool, opcional
        Si True, carga y filtra `cities.csv`. Por defecto = True.
    read_durations : bool, opcional
        Si True, carga `data_clean/duraciones.csv`. Por defecto = True.

    Retorna
    -------
    tuple
        (directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities)

        - directorio_df : pd.DataFrame  
          Información de directorio preprocesada (puede estar vacío).  
        - labors_raw_df : pd.DataFrame  
          Dataset de labores unificado (cargado desde CSV o generado).  
        - cities_df : pd.DataFrame  
          Tabla de ciudades de referencia (puede estar vacía).  
        - duraciones_df : pd.DataFrame  
          Tabla de duraciones de servicios (puede estar vacía).  
        - valid_cities : list[str]  
          Lista de códigos de ciudades válidas definidas en `codificacion_ciudades`.

    Notas
    -----
    - Usar `generate_labors=True` únicamente cuando los CSV crudos hayan cambiado.
    - Para mejorar el rendimiento, preferir cargar el CSV ya guardado en `data_clean/`.
    """
    data_path = Path(data_path)
    pre_dir = data_path / "data_pre"
    clean_dir = data_path / "data_clean"

    # Rutas
    directorio_fp = pre_dir / "directorio_df.csv"
    cities_fp = data_path / "data_raw" / "cities.csv"
    duraciones_fp = clean_dir / "duraciones.csv"
    labors_fp = clean_dir / "labors_raw_df.csv"

    # Labors_raw_df
    if generate_labors or not labors_fp.exists():
        labors_raw_df = generate_labors_raw_df(data_path, tz, strict_datetime_format)
        labors_raw_df.to_csv(labors_fp, index=False, date_format="%Y-%m-%d %H:%M:%S.%f%z")

    else:
        labors_raw_df = pd.read_csv(labors_fp, low_memory=False)

        # Reconstruir fechas en modo ISO8601
        for col in ["labor_created_at", "labor_start_date", "labor_end_date", "created_at", "schedule_date"]:
            if col in labors_raw_df.columns:
                labors_raw_df[col] = (
                pd.to_datetime(labors_raw_df[col], errors="coerce", utc=True)
                .dt.tz_convert(tz)   # <- conversión a la zona horaria de destino
            )

        for col in ['city', 'alfred', 'service_id', 'labor_id']:
            if col in labors_raw_df.columns:
                labors_raw_df[col] = (
                    labors_raw_df[col]
                    .apply(lambda x: '' if pd.isna(x) else str(int(float(x))))
                )

    # labors_raw_df = _order_labor_df(labors_raw_df, assignment_type='historic')
    
    # Directorio
    if read_directorio and directorio_fp.exists():
        directorio_df = pd.read_csv(directorio_fp)
        directorio_df = directorio_df[pd.notna(directorio_df['latitud']) & pd.notna(directorio_df['longitud'])]
    else:
        pd.DataFrame()

    # Cities
    if read_cities and cities_fp.exists():
        cities_df = upload_cities_table(data_path, list(labors_raw_df["city"].unique()))
    else:
        cities_df = pd.DataFrame()

    # Duraciones
    duraciones_df = pd.read_csv(duraciones_fp) if read_durations and duraciones_fp.exists() else pd.DataFrame()

    valid_cities = list(codificacion_ciudades.keys())

    return directorio_df, labors_raw_df, cities_df, duraciones_df, valid_cities


def upload_cities_table(data_path: str, city_codes: list = []) -> pd.DataFrame:
    """
    Carga la tabla 'cities.csv', extrae la información detallada contenida en la columna 'data',
    renombra las columnas y filtra por una lista de códigos de ciudad proporcionada.

    Parámetros:
        data_path (str): Ruta base que contiene la carpeta 'data_raw' con el archivo 'cities.csv'.
        city_codes (list): Lista de códigos de ciudad (como cadenas) que se desean incluir.

    Retorna:
        pd.DataFrame: DataFrame transformado con las columnas ['cod_ciudad', 'ciudad', 'cod_depto', 'depto'].
    """

    # Cargar el archivo cities.csv
    cities_df = pd.read_csv(f"{data_path}/data_raw/cities.csv")
    cities_df['city'] = cities_df['city'].astype(str)

    # Convertir la columna 'data' de cadena a diccionario (si es necesario)
    if isinstance(cities_df["data"].iloc[0], str):
        cities_df["data"] = cities_df["data"].apply(ast.literal_eval)

    # Extraer y convertir las claves del diccionario en nuevas columnas (como cadenas)
    cities_df["cod_ciudad"] = cities_df['city']
    cities_df["ciudad"] = cities_df["data"].apply(lambda x: x.get("city_name"))
    cities_df["cod_depto"] = cities_df["data"].apply(lambda x: str(x.get("department_code")))
    cities_df["depto"] = cities_df["data"].apply(lambda x: x.get("department_name"))

    # Filtrar las ciudades según la lista proporcionada
    if city_codes:
        filtered_df = cities_df[cities_df["cod_ciudad"].isin(city_codes)]
    else:
        filtered_df = cities_df

    # Retornar solo las columnas requeridas en el orden especificado
    return filtered_df[["cod_ciudad", "ciudad", "cod_depto", "depto"]].reset_index(drop=True)


def load_instance(
    data_path: str, 
    instance: str, 
    labors_raw_df: pd.DataFrame, 
    tz: str = "America/Bogota", 
    online_type:str = ''):
    """
    Load the artificial instance from CSV and align its dtypes with a reference dataframe.

    Parameters
    ----------
    data_path : str
        Path to the folder containing the CSV (expects `data_clean/labors_art_df.csv`).
    instance : str
        Name of the artificial instance folder.
    labors_raw_df : pd.DataFrame
        Reference dataframe with correct dtypes.
    tz : str, optional
        Target timezone for datetime columns. Default = "America/Bogota".

    Returns
    -------
    labors_art_df : pd.DataFrame
        Artificial instance dataframe with dtypes aligned to `labors_raw_df`.
    """
    # Load CSV
    instance_type = instance_map(instance)
    if instance_type != 'simu':
        labors_art_df = pd.read_csv(f"{data_path}/instances/{instance_type}_inst/{instance}/labors_{instance}{online_type}_df.csv")
    else:
        labors_art_df = pd.read_csv(f"{data_path}/instances/{instance_type}_inst/{instance}/labors_sim{online_type}_df.csv")

    # Align dtypes with reference dataframe
    for col in labors_raw_df.columns:
        if col in labors_art_df.columns:

            if col in ['city', 'alfred', 'service_id', 'labor_id']:
                labors_art_df[col] = (
                    labors_art_df[col]
                    .apply(lambda x: '' if pd.isna(x) else str(int(float(x))))
                )

            else:
                try:
                    labors_art_df[col] = labors_art_df[col].astype(labors_raw_df[col].dtype)

                except Exception:
                    print(col)
                    # If conversion fails, try datetime
                    if np.issubdtype(labors_raw_df[col].dtype, np.datetime64):
                        labors_art_df[col] = pd.to_datetime(labors_art_df[col], errors="coerce", utc=True)
                    # otherwise leave as is

    # Force timezone normalization for known datetime columns
    datetime_cols = [
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
        "created_at",
        "schedule_date",
        "actual_start",
        "actual_end",
        "historic_start", 
        'historic_end'
    ]

    for col in datetime_cols:
        if col in labors_art_df.columns:
            labors_art_df[col] = (
                pd.to_datetime(labors_art_df[col], errors="coerce", utc=True)
                  .dt.tz_convert(tz)
            )

    return labors_art_df


def load_online_instance(data_path, instance, labors_raw_df):
    labors_real_df = load_instance(data_path, instance, labors_raw_df)
    labors_static_df = load_instance(
        data_path, 
        instance, 
        labors_raw_df, 
        online_type='_static'
    )
    labors_dynamic_df = load_instance(
        data_path, 
        instance, 
        labors_raw_df, 
        online_type='_dynamic'
        ).sort_values(['created_at', 'schedule_date']).reset_index(drop=True)

    return labors_real_df, labors_static_df, labors_dynamic_df



def load_distances(data_path, distance_type, instance, distance_method='precalced'):
    # Distancias
    if distance_method=='precalced':
        with open(f'{data_path}/instances/{instance_map(instance)}_inst/dist/{distance_type}_dist_dict.pkl', "rb") as f:
            dist_dict = pickle.load(f)
        return dist_dict
    else:
        return {}


def load_directorio_hist_df(data_path, instance):
    directorio_hist_df = pd.read_csv(f'{data_path}/instances/{instance_map(instance)}_inst/{instance}/directorio_hist_df.csv')
    for col in ['city', 'alfred']:
        directorio_hist_df[col] = directorio_hist_df[col].astype(int).astype(str)

    return directorio_hist_df


def load_inputs(data_path, instance, **kwargs):
    directorio_df, labor_raw_df, cities_df, duraciones_df, valid_cities = load_tables(data_path)
    # labors_raw_df, duraciones_df, valid_cities = load_algo_tables(data_path)
    labors_real_df = load_instance(data_path, instance, labor_raw_df, **kwargs)
    directorio_hist_df = load_directorio_hist_df(data_path, instance)

    return (duraciones_df, valid_cities, labors_real_df, directorio_hist_df)


def load_duraciones_df(data_path):
    duraciones_df = pd.read_csv(f'{data_path}/data_clean/duraciones.csv')
    duraciones_df['city'] = duraciones_df['city'].astype(str)
    return duraciones_df


def upload_ONLINE_static_solution(
    data_path: str,
    instance: str,
    dist_method: str,
    instance_type: str = 'online_operation'
):
    """
    Carga resultados previos de optimización estática (alpha tuning) 
    y los consolida en un solo DataFrame de labores y movimientos.
    """
    inst_path = f"{data_path}/resultados/{instance_type}/{instance}/{dist_method}"
    labors_algo_df = pd.DataFrame()
    moves_algo_df = pd.DataFrame()

    upload_path = f"{inst_path}/res_algo_ONLINE_static.pkl"

    if not os.path.exists(upload_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with open(upload_path, "rb") as f:
        res = pickle.load(f)
        results_df, moves_df, postponed_labors = res

    if not results_df.empty:
        results_df = results_df.sort_values(["city", "date", "service_id", "labor_id"])
    if not moves_df.empty:
        moves_df = moves_df.sort_values(["city", "date", "service_id", "labor_id"])

    datetime_cols = [
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
        "created_at",
        "schedule_date",
        "actual_start",
        "actual_end",
    ]
    for df in (results_df, moves_df):
        for col in datetime_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce", utc=True)
                    .dt.tz_convert("America/Bogota")
                )

        for col in ["city", "alfred", "service_id", "assigned_driver"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: "" if (pd.isna(x) or x == "") else str(int(float(x)))
                )

    results_df["labor_id"] = results_df["labor_id"].apply(
        lambda x: "" if (pd.isna(x) or x == "") else str(int(float(x)))
    )

    labors_algo_df = pd.concat([labors_algo_df, results_df])
    moves_algo_df = pd.concat([moves_algo_df, moves_df])

    return labors_algo_df, moves_algo_df, postponed_labors


def _order_labor_df(labors_df:pd.DataFrame, 
                   assignment_type:str = 'historic'):
    base_cols = [   'service_id', 'labor_id', 'labor_type', 'labor_name', 'labor_category', 
        'schedule_date', 'created_at', 'shop']
    historic_cols = ['alfred','labor_start_date', 'labor_end_date']
    reconstructed_cols = ['historic_driver', 'historic_start', 'historic_end', 'dist_km']
    solution_cols = ['assigned_driver', 'actual_start', 'actual_end', 'dist_km']
    address_cols = ['address_id', 'address_point', 'address_name', 
                    'start_address_id', 'start_address_point',
                    'end_address_id', 'end_address_point',
                    'city', 'state_service']
    extra_solution_cols = ['map_start_point', 'map_end_point', 'date', 'n_drivers']

    cols = base_cols
    if assignment_type == 'algorithm':
        cols += solution_cols
    elif assignment_type == 'historic':
        for col in reconstructed_cols:
            if col in labors_df.columns:
                cols += [col]
    cols += historic_cols
    cols += address_cols
    for col in extra_solution_cols:
        if col in labors_df.columns:
            cols += [col]

    return labors_df[cols]