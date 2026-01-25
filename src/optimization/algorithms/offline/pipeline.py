import pandas as pd

from typing import Dict, Any

from src.utils.filtering import filter_labors_by_city, filter_labores
from src.data.preprocessing import remap_to_base_date, build_services_map_df, process_group
from src.config.config import *
from src.algorithms.offline_algorithms import run_assignment_algorithm
from src.utils.utils import get_city_name_from_code
from src.data.metrics import compute_metrics_with_moves, compute_iteration_metrics

# ——————————————————————————
# Función maestra por ciudad
# ——————————————————————————
def run_city_pipeline(  
    city_code: str, 
    start_date: str, 
    df_dist,
    directorio_df, 
    duraciones_df,
    alpha=1,
    dist_method='haversine',
    DIST_DICT=None,
    **kwargs
) -> tuple:
    """
    Ejecuta TODO el flujo para una ciudad y fecha dada.
    start_date: string o Timestamp, mismo para todas las ciudades.
    Devuelve: (city_code, df_cleaned, df_moves)
    """
    # 1. Filtrar por ciudad
    df_city = filter_labors_by_city(df_dist, str(city_code))
    
    # 3. Quitar cancelados y ordenar
    df_city_filtered = (
        df_city.query("state_service != 'CANCELED'")
        .sort_values(['service_id', 'labor_start_date'])
        .reset_index(drop=True)
    )

    # 4. Remapear fechas al día base
    base_day = pd.to_datetime(start_date).date()
    df_city_remaped = remap_to_base_date(
        df_city_filtered, 
        ['schedule_date', 'labor_start_date', 'labor_end_date'], 
        base_day
    )

    # 5. Construir mapa de servicios
    services_map_df = build_services_map_df(df_city_remaped)

    # 6. Procesar grupos
    cleaned = []
    for _, grp in df_city_remaped.groupby('service_id', sort=False):
        kwargs['city_code'] = city_code
        cleaned.append(process_group(grp, dist_method=dist_method, 
                                     dist_dict=DIST_DICT, **kwargs))

    if len(cleaned)==0:
        return pd.DataFrame(), pd.DataFrame()
    
    df_cleaned_template = pd.concat([c for c in cleaned if not c.empty], ignore_index=True)
    df_cleaned_template = df_cleaned_template.merge(
        services_map_df, 
        on=['service_id', 'labor_id'], 
        how='left'
    )

    df_cleaned_template = filter_labores(df_cleaned_template, hour_threshold=0)
    # avg_times_map = compute_avg_times(df_dist)

    # 7. Ejecutar el algorithmo de assignación
    df_result, df_moves = run_assignment_algorithm(  
        df_cleaned_template=df_cleaned_template,
        directorio_df=directorio_df,
        duraciones_df=duraciones_df,
        day_str=start_date, 
        ciudad=get_city_name_from_code(city_code),
        dist_dict=DIST_DICT,
        dist_method=dist_method,
        alpha=alpha,
        **kwargs
    )

    # 8. Devolver resultados
    return df_result, df_moves


''' Algorithm '''
def run_single_iteration(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single iteration of the assignment algorithm.
    Args is a dict (since Pool.map only passes one argument).
    Returns all iteration results & metrics.
    """
    city = args["city"]
    fecha = args["fecha"]
    iter_idx = args["iter_idx"]
    df_day = args["df_day"]
    dist_dict = args["dist_dict"]
    directorio_df = args["directorio_hist_filtered_df"]
    duraciones_df = args["duraciones_df"]
    distance_method = args["distance_method"]
    alpha = args["alpha"]
    instance = args["instance"]
    driver_init_mode = args["driver_init_mode"]

    # --- Algorithm execution ---
    df_cleaned_template = _prepare_algo_baseline_pipeline(
        city_code=city,
        start_date=fecha,
        df_dist=df_day,
        dist_method=distance_method,
        DIST_DICT=dist_dict
    )

    results_df, moves_df, postponed_labors = run_assignment_algorithm(
        df_cleaned_template=df_cleaned_template,
        directorio_df=directorio_df,
        duraciones_df=duraciones_df,
        day_str=fecha,
        ciudad=get_city_name_from_code(city),
        dist_method=distance_method,
        dist_dict=dist_dict,
        alpha=alpha,
        instance=instance,
        driver_init_mode=driver_init_mode,
        city=city,
        update_dist_dict=False
    )

    results_df['city'] = city
    results_df['date'] = fecha
    moves_df['city'] = city
    moves_df['date'] = fecha

    metrics = compute_metrics_with_moves(
        results_df,
        moves_df,
        fechas=[fecha],
        dist_dict=dist_dict,
        workday_hours=8,
        city=city,
        skip_weekends=False,
        dist_method=distance_method
    )

    vt_labors, extra_time, dist = compute_iteration_metrics(metrics)

    return {
        "city": city,
        "date": fecha,
        "iter": iter_idx,
        "vt_labors": vt_labors,
        "extra_time": extra_time,
        "dist": dist,
        "results": results_df,
        "moves": moves_df,
        'postponed_labors': postponed_labors,
        "metrics": metrics,
        'success': True
    }


def select_best_iteration(df_results: pd.DataFrame, optimization_obj: str) -> int:
    """
    Return index of best (incumbent) iteration.
    """
    if optimization_obj == "driver_distance":
        return df_results["dist"].idxmin()
    elif optimization_obj == "driver_extra_time":
        return df_results["extra_time"].idxmin()
    else:
        raise ValueError(f"Unknown optimization objective: {optimization_obj}")


def _prepare_algo_baseline_pipeline(
    city_code: str, 
    start_date: str, 
    df_dist,
    dist_method='haversine',
    DIST_DICT=None,
    **kwargs
):
    # 1. Filtrar por ciudad
    df_city = filter_labors_by_city(df_dist, str(city_code))
    
    # 3. Quitar cancelados y ordenar
    df_city_filtered = (
        df_city.query("state_service != 'CANCELED'")
        .sort_values(['service_id', 'labor_start_date'])
        .reset_index(drop=True)
    )

    # 4. Remapear fechas al día base
    base_day = pd.to_datetime(start_date).date()
    df_city_remaped = remap_to_base_date(
        df_city_filtered, 
        ['schedule_date', 'labor_start_date', 'labor_end_date'], 
        base_day
    )

    # 5. Construir mapa de servicios
    # services_map_df = build_services_map_df(df_city_remaped)

    # 6. Procesar grupos
    cleaned = []
    for _, grp in df_city_remaped.groupby('service_id', sort=False):
        kwargs['city_code'] = city_code
        cleaned.append(process_group(grp, dist_method=dist_method, 
                                     dist_dict=DIST_DICT, **kwargs))

    if len(cleaned)==0:
        return pd.DataFrame()
    
    df_cleaned_template = pd.concat([c for c in cleaned if not c.empty], ignore_index=True)
    # df_cleaned_template = df_cleaned_template.merge(
    #     services_map_df, 
    #     on=['service_id', 'labor_id'], 
    #     how='left'
    # )

    df_cleaned_template = filter_labores(df_cleaned_template, hour_threshold=0)
    df_cleaned_template = df_cleaned_template.reset_index(drop=True)


    return df_cleaned_template
