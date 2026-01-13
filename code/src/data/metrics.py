import pandas as pd
import pickle

import plotly.graph_objects as go
from datetime import timedelta

from IPython.display import display

from typing import Tuple, Dict

from src.data.data_load import _order_labor_df
from src.algorithms.offline_algorithms import init_drivers
from src.data.distance_utils import distance
from src.config.experimentation_config import instance_map


def collect_vt_metrics_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    dist_dict: dict,
    workday_hours: float = 8.0,
    city_code: str | int | None = "149",
    skip_weekends: bool = True,
    assignment_type: str = 'algorithm'
):
    """
    Calcula métricas diarias de transporte (VT) y genera visualizaciones para un rango de fechas.

    La función:
    1. Filtra opcionalmente por ciudad (asegurando tipos consistentes para códigos).
    2. Itera sobre cada día del rango [start_date, end_date], omitiendo fines de semana si se solicita.
    3. Llama a `vt_metrics` para calcular métricas por día.
    4. Crea un DataFrame con las métricas diarias.
    5. Genera dos gráficos interactivos (Plotly):
        - Labores por conductor vs Utilización promedio (%)
        - Número de conductores vs Número de labores VT

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con información de labores. Puede contener más columnas, pero al menos:
        - `labor_start_date` y `labor_end_date`
        - (opcional) `city_code` para filtrado por ciudad.
    start_date : str
        Fecha de inicio del análisis (ej. "2025-08-01").
    end_date : str
        Fecha de fin del análisis (ej. "2025-08-12").
    workday_hours : float, opcional
        Duración de la jornada laboral en horas. Default = 8.0.
    city_code : str, int o None, opcional
        Código de ciudad para filtrar. Si es None, no se filtra por ciudad. Default = "149".
    skip_weekends : bool, opcional
        Si es True, excluye sábados y domingos del análisis. Default = True.

    Retorna
    -------
    metrics_df : pd.DataFrame
        Tabla con métricas diarias:
        - `day` (str)
        - `vt_count` (int)
        - `num_drivers` (int)
        - `labores_por_conductor` (float)
        - `utilizacion_promedio_%` (float)
    fig_metrics : go.Figure o None
        Figura con métricas "Labores por conductor" y "Utilización promedio (%)".
    fig_counts : go.Figure o None
        Figura con conteo de conductores y labores VT.

    Notas
    -----
    - Depende de la función `vt_metrics`.
    - Puede devolver figuras como None si no hay datos.
    - Los gráficos devueltos son objetos Plotly y deben mostrarse con `.show()`.
    """

    # --- Copiar DataFrame para no modificar el original ---
    df_in = df.copy()

    # --- Filtrar por ciudad si se especifica ---
    # Convertir ambos a string para evitar errores de comparación por tipo.
    if city_code is not None and 'city' in df_in.columns:
        df_in['city'] = df_in['city'].astype(str)
        city_code = str(city_code)
        df_in = df_in[df_in['city'] == city_code].copy()

    # --- Construir lista de días del rango ---
    days = pd.date_range(start=start_date, end=end_date, freq='D')
    rows = []

    # --- Recorrer cada día y calcular métricas ---
    for d in days:
        if skip_weekends and d.weekday() >= 5:  # 5 = sábado, 6 = domingo
            continue
        day_str = d.strftime('%Y-%m-%d')

        # Llamar a vt_metrics (se asume que devuelve un dict con la estructura esperada)
        m = vt_metrics(
            df_in, 
            day_str=day_str, 
            workday_hours=workday_hours, 
            dist_dict=dist_dict, 
            assignment_type=assignment_type
        )

        # --- Calcular utilización promedio como float ---
        if not m["trabajo_por_conductor_df"].empty:
            util_prom = float(m["trabajo_por_conductor_df"]["utilizacion_%"].mean())
        else:
            util_prom = 0.0

        # --- Agregar fila con métricas ---
        rows.append({
            "day": day_str,
            'service_count': m['service_count'],
            "vt_count": m["vt_count"],
            "num_drivers": m["num_drivers"],
            "labores_por_conductor": round(m["labores_por_conductor"], 3),
            "utilizacion_promedio_%": round(util_prom, 1),
            'labor_extra_time': round(m['labor_extra_time'],1),
            'driver_extra_time': round(m['driver_extra_time'],1),
            'total_distance':m['total_distance_km']
        })

    # --- Crear DataFrame de métricas ---
    metrics_df = pd.DataFrame(rows)

    if metrics_df.empty:
        print("No hubo días hábiles en el rango o no hay datos.")
        return metrics_df, None, None

    # --- Conversión de columna de fechas ---
    x = pd.to_datetime(metrics_df["day"])

    return metrics_df


def vt_metrics(
    df: pd.DataFrame,
    day_str: str,
    workday_hours: float = 8.0,
    dist_dict: dict | None = None,
    assignment_type: str = "algorithm"
):
    """
    Calcula métricas de utilización de conductores, distancia total recorrida
    y tiempo extra (labor vs. conductor) en labores VEHICLE_TRANSPORTATION
    para un día específico.

    Retorna
    -------
    dict
        Diccionario con:
            - vt_count : int
            - alfred_col : str|None
            - alfred_ids : list
            - num_drivers : int
            - labores_por_conductor : float
            - total_distance_km : float
            - resumen_df : pd.DataFrame
            - trabajo_por_conductor_df : pd.DataFrame
            - filtered_df : pd.DataFrame
    """
    df = df.copy()

    # Selección dinámica de columnas de tiempo
    if assignment_type == "historic":
        start_col, end_col, alfred_col = "historic_start", "historic_end", "historic_driver"
    elif assignment_type == "algorithm":
        start_col, end_col, alfred_col = "actual_start", "actual_end", "assigned_driver"
    else:
        raise ValueError("assignment_type debe ser 'historic' o 'algorithm'")

    # Validar columnas mínimas
    _validate_columns(df, [start_col, end_col, "labor_category"])

    # Parsear fechas con preservación de zona horaria
    for col in [start_col, end_col]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert("America/Bogota")
        elif df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("America/Bogota")

    # Renombrar a columnas estándar internas
    df = df.rename(columns={start_col: "start_time", end_col: "end_time"})

    # Filtrar por día
    df_day = _filter_day(df, day_str, start_col="start_time", end_col="end_time")

    # Conteo de servicios
    service_count = int(len(df_day['service_id'].unique()))

    # Conteo de labores VT
    vt_mask = df_day["labor_category"].eq("VEHICLE_TRANSPORTATION")
    df_vt = df_day.loc[vt_mask].copy()
    vt_count = int(len(df_vt))

    if alfred_col:
        df_day[alfred_col] = df_day[alfred_col].astype(str)
        alfred_ids = sorted(x for x in df_day[alfred_col].dropna().unique() if x != '')
        num_drivers = len(alfred_ids)
    else:
        alfred_ids, num_drivers = [], 0

    # Métrica: labores promedio por conductor
    labores_por_conductor = (vt_count / num_drivers) if num_drivers > 0 else float("nan")

    # Tabla de carga laboral por conductor
    trabajo_por_conductor_df = pd.DataFrame()
    if alfred_col and vt_count > 0:
        trabajo_por_conductor_df = _compute_driver_workload(df_vt, alfred_col, workday_hours)

    # Métrica: utilización promedio
    util_prom = float(trabajo_por_conductor_df["utilizacion_%"].mean()) if not trabajo_por_conductor_df.empty else 0.0

    # Métrica: distancia total
    total_distance_km = _compute_total_distance(df_vt, dist_dict) if (dist_dict and vt_count > 0) else 0.0

    # --- Métricas de tiempo extra ---
    if not trabajo_por_conductor_df.empty:
        total_labor_ot   = float(trabajo_por_conductor_df["labor_extra_time_min"].sum())
        total_driver_ot  = float(trabajo_por_conductor_df["driver_extra_time_min"].sum())
        # total_early_ot   = float(trabajo_por_conductor_df["early_overtime_min"].sum())
        # total_late_ot    = float(trabajo_por_conductor_df["late_overtime_min"].sum())
    else:
        total_labor_ot = total_driver_ot = total_early_ot = total_late_ot = 0.0

    # Tabla resumen
    resumen_df = pd.DataFrame({
        "Métrica": [
            'Número de sevicios',
            "Labores VEHICLE_TRANSPORTATION (día filtrado)",
            "Conductores únicos (alfred)",
            "Labores por conductor",
            "Utilización promedio conductores",
            "Distancia total recorrida (km)",
            "Tiempo extra total por labor (min)",
            "Tiempo extra total por labor (horas)",
            "Tiempo extra total por conductor (min)",
            "Tiempo extra total por conductor (horas)",
            # "Tiempo extra temprano (min)",
            # "Tiempo extra tarde (min)"
        ],
        "Valor": [
            service_count,
            vt_count,
            num_drivers,
            round(labores_por_conductor, 3),
            f"{util_prom:.1f}%",
            round(total_distance_km, 2),
            round(total_labor_ot, 1),
            round(total_labor_ot / 60.0, 2),
            round(total_driver_ot, 1),
            round(total_driver_ot / 60.0, 2),
            # round(total_early_ot, 1),
            # round(total_late_ot, 1)
        ]
    })

    return {
        'service_count': service_count,
        "vt_count": vt_count,
        "alfred_col": alfred_col,
        "alfred_ids": alfred_ids,
        "num_drivers": num_drivers,
        "labores_por_conductor": labores_por_conductor,
        "total_distance_km": total_distance_km,
        "labor_extra_time": total_labor_ot,
        "driver_extra_time": total_driver_ot,
        "resumen_df": resumen_df,
        "trabajo_por_conductor_df": trabajo_por_conductor_df,
        "filtered_df": df_day
    }


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Valida que las columnas requeridas existan en el DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")


def _filter_day(
    df: pd.DataFrame, 
    day_str: str,
    start_col: str = "start_time",
    end_col: str = "end_time"
) -> pd.DataFrame:
    """
    Filtra filas cuya ventana temporal cae dentro de [día, día+1).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de tiempos de inicio y fin.
    day_str : str
        Día de referencia en formato 'YYYY-MM-DD'.
    start_col : str, opcional
        Nombre de la columna con tiempos de inicio. Default = "start_time".
    end_col : str, opcional
        Nombre de la columna con tiempos de fin. Default = "end_time".

    Retorna
    -------
    pd.DataFrame
        Subconjunto de filas del DataFrame que caen en la ventana [día, día+1).
    """
    start = pd.to_datetime(day_str).normalize()
    tz = df[start_col].dt.tz
    if tz is not None and start.tzinfo is None:
        start = start.tz_localize(tz)
    end = start + pd.Timedelta(days=1)

    mask = (
        df[start_col].between(start, end, inclusive="left") &
        df[end_col].between(start, end, inclusive="left")
    )
    return df.loc[mask].copy()


def _compute_driver_workload(
    df_vt: pd.DataFrame,
    alfred_col: str,
    workday_hours: float,
    work_start="09:00",
    work_end="17:00"
) -> pd.DataFrame:
    """Calcula carga de trabajo, utilización y horas extra por conductor.

    Incluye:
      - Tiempo extra por labor (suma de tiempos fuera de jornada en cada labor).
      - Tiempo extra por conductor (ventana desde primer hasta último servicio).
    """
    df_vt = df_vt.copy()
    df_vt["duration_min"] = (df_vt["end_time"] - df_vt["start_time"]).dt.total_seconds() / 60
    df_vt = df_vt.dropna(subset=["duration_min"])
    df_vt = df_vt[df_vt["duration_min"] >= 0]

    # --- 1) Overtime detallado por labor ---
    overtime = df_vt.apply(_compute_overtime_per_labor, axis=1, result_type="expand")
    overtime.columns = ["early_min", "late_min"]
    df_vt = pd.concat([df_vt, overtime], axis=1)

    agg = (
        df_vt.dropna(subset=[alfred_col])
             .groupby(alfred_col)
             .agg(
                 total_min=("duration_min", "sum"),
                 early_overtime_min=("early_min", "sum"),
                 late_overtime_min=("late_min", "sum"),
                 first_start=("start_time", "min"),
                 last_end=("end_time", "max")
             )
             .reset_index()
    )

    if agg.empty:
        return pd.DataFrame(columns=[
            alfred_col, "total_min", "total_horas", "utilizacion_%",
            "labor_extra_time_min", "labor_extra_time_horas",
            "early_overtime_min", "late_overtime_min",
            "driver_extra_time_min", "driver_extra_time_horas"
        ])

    # --- 2) Overtime por ventana completa de cada conductor ---
    driver_extra = []
    for _, row in agg.iterrows():
        start, end = row["first_start"], row["last_end"]

        if pd.isna(start) or pd.isna(end) or end <= start:
            driver_extra.append((0.0, 0.0))
            continue

        # Definir jornada laboral del mismo día
        day = start.normalize()
        tz  = start.tz or "America/Bogota"
        work_start_dt = pd.to_datetime(f"{day.date()} {work_start}").tz_localize(tz)
        work_end_dt   = pd.to_datetime(f"{day.date()} {work_end}").tz_localize(tz)

        early = max(0, (work_start_dt - start).total_seconds() / 60)
        late  = max(0, (end - work_end_dt).total_seconds() / 60)
        driver_extra.append((early + late, (early + late) / 60.0))

    driver_extra = pd.DataFrame(driver_extra, columns=["driver_extra_time_min", "driver_extra_time_horas"])
    agg = pd.concat([agg, driver_extra], axis=1)

    # --- 3) Métricas derivadas ---
    jornada_min = workday_hours * 60.0
    agg["total_horas"]            = agg["total_min"] / 60.0
    agg["utilizacion_%"]          = agg["total_min"] / jornada_min * 100.0
    agg["labor_extra_time_min"]   = agg["early_overtime_min"] + agg["late_overtime_min"]
    agg["labor_extra_time_horas"] = agg["labor_extra_time_min"] / 60.0

    # --- 4) Redondeo ---
    agg = agg.round({
        "total_min": 1,
        "total_horas": 2,
        "utilizacion_%": 1,
        "labor_extra_time_min": 1,
        "labor_extra_time_horas": 2,
        "early_overtime_min": 1,
        "late_overtime_min": 1,
        "driver_extra_time_min": 1,
        "driver_extra_time_horas": 2
    })

    # Limpiar columnas intermedias
    agg = agg.drop(columns=["first_start", "last_end"])

    return agg.sort_values("total_min", ascending=False).reset_index(drop=True)


def _compute_overtime_per_labor(row, work_start="09:00", work_end="17:00") -> tuple[float, float]:
    """
    Calcula minutos de tiempo extra antes y después de la jornada para una labor.
    """
    start = row["start_time"]
    end   = row["end_time"]

    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.0, 0.0

    # Ensure tz-aware (default Bogotá)
    if start.tz is None:
        start = start.tz_localize("America/Bogota")
    if end.tz is None:
        end = end.tz_localize(start.tz)

    # Define work window on the same day and tz
    day = start.normalize()
    work_start_dt = pd.to_datetime(f"{day.date()} {work_start}").tz_localize(start.tz)
    work_end_dt   = pd.to_datetime(f"{day.date()} {work_end}").tz_localize(start.tz)

    early_min = max(0, (min(end, work_start_dt) - start).total_seconds() / 60)
    late_min  = max(0, (end - max(start, work_end_dt)).total_seconds() / 60)

    return early_min, late_min


def _compute_total_distance(df_vt: pd.DataFrame, dist_dict: dict) -> float:
    """
    Calcula la distancia total recorrida en labores VEHICLE_TRANSPORTATION
    a partir de start_address_id y end_address_id.

    Parámetros
    ----------
    df_vt : pd.DataFrame
        DataFrame filtrado solo con labores VEHICLE_TRANSPORTATION.
        Debe contener columnas 'start_address_id' y 'end_address_id'.
    dist_dict : dict
        Diccionario de distancias {(origen, destino): distancia_km}.

    Retorna
    -------
    float
        Distancia total recorrida (en km).
    """
    if "start_address_id" not in df_vt.columns or "end_address_id" not in df_vt.columns:
        return 0.0

    total_distance = 0.0
    for _, row in df_vt.iterrows():
        key = (row["start_address_id"], row["end_address_id"])
        total_distance += dist_dict.get(key, 0.0)

    return float(total_distance)


def show_day_report_dayonly(
    df_full: pd.DataFrame,
    *,
    idx: int | None = None,          # Índice en metrics_df (día hábil)
    day_str: str | None = None,      # Fecha explícita en formato 'YYYY-MM-DD'
    workday_hours: float = 8.0,
    city_code: str | int | None = "149",
    only_vt: bool = False,            # True → Mostrar solo VEHICLE_TRANSPORTATION en el Gantt
    dist_dict: dict = None,
    return_plotting_df: bool = False,
    assignment_type: str = 'real'
):
    """
    Genera un reporte detallado para un día específico, mostrando métricas y un diagrama de Gantt
    únicamente con los conductores que trabajaron ese día.

    La función:
    1. Permite seleccionar el día por índice en metrics_df o por fecha directa.
    2. Filtra opcionalmente por código de ciudad (asegurando consistencia de tipos).
    3. Llama a `vt_metrics` para obtener datos y métricas del día.
    4. Muestra en pantalla:
        - Resumen general del día
        - Métricas por conductor (solo los que trabajaron ese día)
    5. Genera un diagrama de Gantt para las labores de esos conductores, filtrando por categoría si se indica.

    Parámetros
    ----------
    df_full : pd.DataFrame
        DataFrame con todas las labores, incluyendo posibles columnas como:
        - `city_code` (opcional, para filtrado)
        - `labor_category` (opcional, para filtrar solo VEHICLE_TRANSPORTATION)
    metrics_df : pd.DataFrame
        DataFrame con métricas diarias (como las generadas por collect_vt_metrics_range).
    idx : int o None, opcional
        Índice de la fila en metrics_df correspondiente al día deseado.
        Se debe pasar `idx` o `day_str`, pero no ambos. Default = None.
    day_str : str o None, opcional
        Fecha en formato 'YYYY-MM-DD'. Si se usa, ignora `idx`. Default = None.
    workday_hours : float, opcional
        Horas de jornada laboral a considerar. Default = 8.0.
    city_code : str, int o None, opcional
        Código de ciudad para filtrar. Si es None, no se filtra por ciudad. Default = "149".
    only_vt : bool, opcional
        Si es True, el Gantt solo mostrará labores de tipo "VEHICLE_TRANSPORTATION". Default = False.

    Retorna
    -------
    dict
        Contiene:
        - `day_str` (str): Día analizado.
        - `drivers_present` (list[str]): Conductores que trabajaron ese día.
        - `metrics` (dict): Salida completa de `vt_metrics`.
        - `trabajo_por_conductor_df` (pd.DataFrame): Métricas filtradas solo a los conductores presentes.

    Notas
    -----
    - Requiere las funciones `vt_metrics` y `plot_gantt_labors_by_driver`.
    - Muestra DataFrames directamente usando `display()`.
    - Si no hay conductores o datos para el día, puede devolver listas vacías y gráficas sin contenido.
    """
    # --- Validación de parámetros de entrada ---
    if (idx is None) and (day_str is None):
        raise ValueError("Debes pasar 'idx' o 'day_str'.")
    
    # Resolver day_str desde metrics_df si se proporciona idx
    day_str = pd.to_datetime(day_str).strftime("%Y-%m-%d")

    # --- Copiar DataFrame y filtrar por ciudad si aplica ---
    dfx = df_full.copy()
    if city_code is not None and "city" in dfx.columns:
        dfx["city"] = dfx["city"].astype(str)
        city_code = str(city_code)
        dfx = dfx[dfx["city"] == city_code].copy()

    # --- Obtener métricas del día ---
    m_day = vt_metrics(dfx, day_str=day_str, workday_hours=workday_hours, 
                       dist_dict=dist_dict, assignment_type=assignment_type)

    # --- Filtrar DataFrame del día ---
    df_day = m_day["filtered_df"].copy()
    driver_col = m_day["alfred_col"]
    if driver_col is None:
        raise KeyError(
            "No se encontró columna de conductor (alfred / ALFRED'S / assigned_driver / driver / conductor / alfred_id)."
        )

    # --- Lista de conductores presentes ---
    drivers_present = (
        df_day[driver_col].dropna().unique().tolist()
        if not df_day.empty else []
    )

    # --- Filtrar métricas solo a conductores presentes ---
    trabajo_df = m_day["trabajo_por_conductor_df"]
    if not trabajo_df.empty:
        trabajo_df = trabajo_df[trabajo_df[driver_col].isin(drivers_present)].reset_index(drop=True)

    # --- Mostrar métricas del día ---
    print(f"Reporte del día: {day_str}")
    display(m_day["resumen_df"])
    #display(trabajo_df)

    # --- Preparar DataFrame para el Gantt ---
    df_plot = df_day.copy()
    if only_vt and "labor_category" in df_plot.columns:
        df_plot = df_plot[df_plot["labor_category"] == "VEHICLE_TRANSPORTATION"]
    df_plot = df_plot[df_plot[driver_col].isin(drivers_present)]

    # --- Generar Gantt ---
    # plot_gantt_labors_by_driver(df_plot, day_str=day_str, driver_col=driver_col)
    
    summary = {
            "day_str": day_str,
            "drivers_present": drivers_present,
            "metrics": m_day,
            "trabajo_por_conductor_df": trabajo_df
        }
    
    if not return_plotting_df:
        return summary, ()
    else:
        return summary, (df_plot, driver_col)


def get_day_plotting_df(
    df_full: pd.DataFrame,
    *,
    day_str: str,
    workday_hours: float = 8.0,
    city_code: str | int | None = "149",
    only_vt: bool = False,
    dist_dict: dict | None = None,
    assignment_type: str = "real"
) -> tuple[pd.DataFrame, str]:
    """
    Prepara un DataFrame listo para graficar (Gantt u otros) 
    con las labores de un día específico y los conductores presentes.

    Parameters
    ----------
    df_full : pd.DataFrame
        DataFrame con todas las labores (de varias ciudades/días).
    day_str : str
        Fecha en formato 'YYYY-MM-DD'.
    workday_hours : float, default=8.0
        Horas de jornada laboral.
    city_code : str | int | None, default="149"
        Código de ciudad para filtrar. Si None, no se filtra por ciudad.
    only_vt : bool, default=False
        Si True, solo devuelve labores de tipo "VEHICLE_TRANSPORTATION".
    dist_dict : dict | None, optional
        Diccionario de distancias para cálculos de métricas.
    assignment_type : str, default="real"
        Tipo de asignación: "real" o "algorithm".

    Returns
    -------
    df_plot : pd.DataFrame
        DataFrame filtrado listo para graficar.
    driver_col : str
        Nombre de la columna que identifica a los conductores.
    """
    # --- Copiar DataFrame y filtrar por ciudad si aplica ---
    dfx = df_full.copy()
    if city_code is not None and "city" in dfx.columns:
        dfx["city"] = dfx["city"].astype(str)
        city_code = str(city_code)
        dfx = dfx[dfx["city"] == city_code].copy()

    # --- Obtener métricas del día ---
    m_day = vt_metrics(
        dfx,
        day_str=day_str,
        workday_hours=workday_hours,
        dist_dict=dist_dict,
        assignment_type=assignment_type
    )

    df_day = m_day["filtered_df"].copy()
    driver_col = m_day["alfred_col"]
    if driver_col is None:
        raise KeyError("No se encontró columna de conductor.")

    # --- Conductores presentes ---
    drivers_present = (
        df_day[driver_col].dropna().unique().tolist()
        if not df_day.empty else []
    )

    # --- Preparar DataFrame para graficar ---
    df_plot = df_day.copy()
    if only_vt and "labor_category" in df_plot.columns:
        df_plot = df_plot[df_plot["labor_category"] == "VEHICLE_TRANSPORTATION"]
    df_plot = df_plot[df_plot[driver_col].isin(drivers_present)]

    return df_plot, driver_col


def compute_indicators(
    df_cleaned: pd.DataFrame,
    df_moves: pd.DataFrame,
    num_drivers: int,
    tiempo_gracia: int
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calcula indicadores operativos para un día específico.

    Parámetros
    ----------
    df_cleaned : pd.DataFrame
        DataFrame con las labores completadas.
    df_moves : pd.DataFrame
        DataFrame con los movimientos/tareas programadas.
    tiempo_gracia : int
        Período de gracia en minutos para el cálculo de llegadas tarde.

    Retornos
    -------
    indicators : dict
        Indicadores numéricos crudos.
    ind_df : pd.DataFrame
        Indicadores formateados, listos para mostrar o exportar.
    """
    # 1) Precondición: si no hay movimientos, retorno vacíos
    if df_moves.empty:
        return {}, pd.DataFrame()

    # Cálculo de cada indicador
    num_labors   = len(df_cleaned)  # Número total de labors

    # Porcentaje de tiempo libre
    free_min     = df_moves.loc[df_moves['labor_category']=='FREE_TIME','duration_min'].sum()
    total_min    = df_moves['duration_min'].sum()
    pct_free     = 100 * free_min / total_min if total_min > 0 else 0

    # Porcentaje de servicios fallidos
    svc_fail     = (
        df_cleaned[df_cleaned['labor_category']=='VEHICLE_TRANSPORTATION']
          .groupby('service_id')['assigned_driver']
          .apply(lambda x: x.isna().all())
    )
    failed       = svc_fail.sum()
    total_s      = df_cleaned['service_id'].nunique()
    pct_failed   = 100 * failed / total_s if total_s > 0 else 0

    # Llegadas tarde
    n_late = 0
    total_tardy_norm = 0.0
    vt_df = df_cleaned[(df_cleaned['labor_category']=='VEHICLE_TRANSPORTATION') & df_cleaned['actual_start'].notna()]
    if not vt_df.empty:
        firsts_df      = vt_df.sort_values('schedule_date').groupby('service_id').first()
        late_deadlines = firsts_df['schedule_date'] + timedelta(minutes=tiempo_gracia)
        raw_late       = firsts_df['actual_start'] - late_deadlines
        n_late         = (raw_late > timedelta(0)).sum()
        capped_late    = raw_late.clip(lower=timedelta(0))
        total_tardy_sec= capped_late.dt.total_seconds().sum()
        assigned_s     = total_s - failed
        if assigned_s > 0:
            total_tardy_norm = total_tardy_sec / (tiempo_gracia * 60 * assigned_s)

    # Promedio de servicios por conductor
    assigned_drivers_df = df_cleaned[df_cleaned['assigned_driver'].notna()]
    avg_per_driver = (
        assigned_drivers_df.groupby('assigned_driver')['service_id']
        .nunique().mean()
        if not assigned_drivers_df.empty else 0
    )

    # Indicador ponderado (pesos iniciales)
    obj_weighted = pct_free*0 + pct_failed*0 + total_tardy_norm*0 + avg_per_driver*0.5

    # Diccionario con valores crudos
    indicators = {
        'num_labors':         num_labors,
        'num_drivers':        num_drivers,
        'pct_free':           pct_free,
        'pct_failed':         pct_failed,
        'n_late':             int(n_late),
        'total_tardy_norm':   total_tardy_norm,
        'avg_per_driver':     avg_per_driver,
        'obj_weighted':       obj_weighted
    }

    # DataFrame para mostrar
    ind_df = pd.DataFrame({
        'Indicador': [
            'N° Labores', 'N° Conductores Disponibles', '% Tiempo Libre',
            '% Fallidas', 'N° Llegadas Tarde', '% Tardanza (norm.)',
            'Prom. Servicios x Conductor', 'Obj. Ponderado'
        ],
        'Valor': [
            indicators['num_labors'],
            indicators['num_drivers'],
            f"{indicators['pct_free']:.1f}%",
            f"{indicators['pct_failed']:.1f}%",
            indicators['n_late'],
            f"{indicators['total_tardy_norm']*100:.1f}%",
            f"{indicators['avg_per_driver']:.2f}",
            f"{indicators['obj_weighted']:.2f}"
        ]
    })

    return indicators, ind_df



def collect_results_to_df(
    data_path: str,
    instance: str,
    fecha_list: list,
    assignment_type: str = 'algorithm',
    distance_method: str = 'haversine',
    tz: str = "America/Bogota",
) -> Tuple:
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
    all_results = []
    all_moves   = []

    for fecha in fecha_list:
        if assignment_type == 'algorithm':
            upload_path = f"{data_path}/resultados/offline_operation/{instance}/{distance_method}/res_{fecha}.pkl"
        elif assignment_type == 'historic':
            upload_path = f"{data_path}/resultados/alfred_baseline/{instance}/{distance_method}/res_hist_{fecha}.pkl"
        
        with open(upload_path, "rb") as f:
            res = pickle.load(f)  # dict: {city: (df_cleaned, df_moves, n_drivers)}

        for city, (df_cleaned, df_moves, n_drivers) in res.items():
            # --- df_cleaned records ---
            if df_cleaned is not None and not df_cleaned.empty:
                tmp = df_cleaned.copy()
                tmp["city"] = city
                tmp["date"] = fecha
                tmp["n_drivers"] = n_drivers
                all_results.append(tmp)

            # --- df_moves records ---
            if df_moves is not None and not df_moves.empty:
                tmpm = df_moves.copy()
                tmpm["city"] = city
                tmpm["date"] = fecha
                all_moves.append(tmpm)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    moves_df   = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()

    if not results_df.empty:
        results_df = results_df.sort_values(["city", "date", "service_id", "labor_id"])
    if not moves_df.empty:
        moves_df = moves_df.sort_values(["city", "date", "service_id", "labor_id"])

    # Normalize datetime columns to Bogotá tz
    datetime_cols = [
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
        "created_at",
        "schedule_date",
        
    ]
    if assignment_type == 'algorithm':
        datetime_cols += ["actual_start", "actual_end"]
    elif assignment_type == 'historic':
        datetime_cols += ['historic_start', 'historic_end']

    for df in (results_df, moves_df):
        for col in datetime_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce", utc=True)
                      .dt.tz_convert(tz)
                )
    
    # results_df = _order_labor_df(results_df, assignment_type=assignment_type)

    return results_df, moves_df


def collect_results_from_dicts(
    results_by_city_list: list,
    fecha_list: list,
    assignment_type: str = "algorithm",
    tz: str = "America/Bogota"
):
    """
    Consolida resultados desde múltiples diccionarios (uno por fecha) en dos DataFrames:
        - results_df: contiene df_cleaned (labores finales)
        - moves_df: contiene df_moves (movimientos simulados)

    Params
    ------
        results_by_city_list : list[dict]
            Lista de diccionarios en el formato {city: (df_cleaned, df_moves, n_drivers)}.
            El orden debe corresponder al orden de `fecha_list`.
        fecha_list : list[str]
            Lista de fechas (YYYY-MM-DD).
        assignment_type : str, default="algorithm"
            Puede ser "algorithm" o "historic".
        tz : str, default="America/Bogota"
            Zona horaria para normalizar datetimes.

    Returns
    -------
        results_df (pd.DataFrame), moves_df (pd.DataFrame)
    """
    all_results = []
    all_moves   = []

    for fecha, res in zip(fecha_list, results_by_city_list):
        for city, (df_cleaned, df_moves, n_drivers) in res.items():
            # --- df_cleaned records ---
            if df_cleaned is not None and not df_cleaned.empty:
                tmp = df_cleaned.copy()
                tmp["city"] = city
                tmp["date"] = fecha
                tmp["n_drivers"] = n_drivers
                all_results.append(tmp)

            # --- df_moves records ---
            if df_moves is not None and not df_moves.empty:
                tmpm = df_moves.copy()
                tmpm["city"] = city
                tmpm["date"] = fecha
                all_moves.append(tmpm)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    moves_df   = pd.concat(all_moves, ignore_index=True) if all_moves else pd.DataFrame()

    # Normalize datetime columns to tz
    datetime_cols = [
        "labor_created_at",
        "labor_start_date",
        "labor_end_date",
        "created_at",
        "schedule_date",
    ]
    if assignment_type == 'algorithm':
        datetime_cols += ["actual_start", "actual_end"]
    elif assignment_type == 'historic':
        datetime_cols += ['historic_start', 'historic_end']

    for df in (results_df, moves_df):
        for col in datetime_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_datetime(df[col], errors="coerce", utc=True)
                      .dt.tz_convert(tz)
                )

    return results_df, moves_df


def concat_run_results(run_results: list):
    all_labors = pd.DataFrame()
    all_moves = pd.DataFrame()

    for res in run_results: 
        all_labors = pd.concat([all_labors, res[0]])
        all_moves = pd.concat([all_moves, res[1]])

    # Ordenamiento
    if not all_labors.empty:
        all_labors = all_labors.sort_values(["city", "date", "service_id", "labor_id"])
    if not all_moves.empty:
        all_moves = all_moves.sort_values(["city", "date", "service_id", "labor_id"])

    return all_labors, all_moves


def compute_metrics_with_moves(
    labors_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    fechas: list[str],
    dist_dict: dict,
    workday_hours: int,
    city: str,
    assignment_type: str,
    skip_weekends: bool = False,
    dist_method: str = "haversine"
) -> pd.DataFrame:
    """
    Compute VT metrics + driver movement distance for a given city and assignment type.

    Parameters
    ----------
    labors_df : pd.DataFrame
        Labor assignments (cleaned).
    moves_df : pd.DataFrame
        Driver moves (including DRIVER_MOVE).
    fechas : list[str]
        List of dates (YYYY-MM-DD). First and last define the range.
    dist_dict : dict
        Precomputed distances.
    workday_hours : int
        Workday length in hours.
    city : str
        City code.
    assignment_type : str
        'historic' or 'algorithm'.
    skip_weekends : bool, default False
        Whether to skip weekends.
    distance_method : str, default "haversine"
        Method to compute missing distances.

    Returns
    -------
    pd.DataFrame
        Metrics DataFrame enriched with driver_move_distance.
    """

    # 1. Compute standard VT metrics
    metrics_df = collect_vt_metrics_range(
        labors_df,
        start_date=fechas[0],
        end_date=fechas[-1],
        dist_dict=dist_dict,
        workday_hours=workday_hours,
        city_code=city,
        skip_weekends=skip_weekends,
        assignment_type=assignment_type,
    )

    # 2. Aggregate driver movement distance
    # --- Filtrar por ciudad si se especifica ---
    # Convertir ambos a string para evitar errores de comparación por tipo.
    if city is not None and 'city' in moves_df.columns:
        moves_df['city'] = moves_df['city'].astype(str)
        city = str(city)
        moves_df = moves_df[moves_df['city'] == city]

    df_driver_moves = moves_df[moves_df["labor_category"] == "DRIVER_MOVE"].copy()
    if "distance_km" not in df_driver_moves.columns:
        df_driver_moves["distance_km"] = df_driver_moves.apply(
            lambda r: distance(r["start_point"], r["end_point"], method=dist_method, dist_dict=dist_dict)[0],
            axis=1
        )

    df_driver_moves["day"] = pd.to_datetime(df_driver_moves["schedule_date"]).dt.strftime("%Y-%m-%d")

    moves_exp_df = (
        df_driver_moves.groupby("day", as_index=False)["distance_km"]
        .sum()
        .rename(columns={"distance_km": "driver_move_distance"})
    )

    # 3. Merge metrics + move distances
    metrics_df = metrics_df.merge(moves_exp_df, on="day", how="left").fillna(0)

    return metrics_df


def compute_iteration_metrics(metrics):
    iter_vt_labors = sum(metrics['vt_count'])
    iter_extra_time = round(sum(metrics['driver_extra_time']), 2)
    iter_dist = round(sum(metrics['driver_move_distance']), 2)

    return iter_vt_labors, iter_extra_time, iter_dist