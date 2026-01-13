import pandas as pd
import pickle

import os


def load_solution_dfs(
    data_path: str, 
    instance: str, 
    dist_method: str,
    algorithm: str,
    include_all_city: bool,
    tz: str = "America/Bogota"
) -> tuple:
    """
    Carga resultados desde pickles y devuelve dos DataFrames consolidados:
        - results_df: contiene df_cleaned (labores finales)
        - moves_df: contiene df_moves (movimientos simulados)

    Params
    ------
        data_path : str
            Path base de los datos
        instance : str
            Nombre de la instancia artificial
        fecha_list : list[str]
            Lista de fechas en formato YYYY-MM-DD
        tz : str, opcional
            Zona horaria de destino para columnas datetime. Default="America/Bogota"

    Returns
    -------
        results_df (pd.DataFrame), moves_df (pd.DataFrame)
    """
    labors_df = pd.DataFrame()
    moves_df = pd.DataFrame()

    if algorithm != 'Histórico':
        algorithm = 'algo_' + algorithm
    else:
        algorithm = 'hist'

    upload_path = (f'{data_path}/resultados/online_operation/{instance}/'
                    f'{dist_method}/res_{algorithm}.pkl')

    if not os.path.exists(upload_path):
        raise FileNotFoundError(f"Expected results file not found: {f'res_{algorithm}.pkl'}")
    
    with open(upload_path, "rb") as f:
        res = pickle.load(f)
        labors_df, moves_df, postponed_labors = res

    if not labors_df.empty:
        labors_df = labors_df.sort_values(["city", "date", "service_id", "labor_id", 'schedule_date'])
    if not moves_df.empty:
        moves_df = moves_df.sort_values(["city", "date", "service_id", "labor_id", 'schedule_date'])

    # Normalize datetime columns to Bogotá tz
    datetime_cols = [
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
        "created_at",
        "schedule_date",
        "actual_start", 
        "actual_end",
        'historic_start',
        'historic_end',
        ]


    for df in (labors_df, moves_df):
        for col in datetime_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce", utc=True)
                    .dt.tz_convert(tz)
                )
    
    if include_all_city:
        labors_df = _include_all_city(labors_df)
        moves_df = _include_all_city(moves_df)
    
    return labors_df, moves_df, postponed_labors


def _include_all_city(df):
    temp_df = df.copy()
    temp_df['city'] = 'ALL'
    df = pd.concat([df, temp_df])

    return df