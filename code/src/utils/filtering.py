import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

from datetime import timedelta
import re


def filter_valid_services(df_dist, city_list=None):
    """
    Filtra los servicios válidos del dataframe de distribución.

    Condiciones de filtrado:
    - Excluye servicios con estado 'CANCELED'.
    - Incluye únicamente servicios que tengan al menos un labor con categoría 'VEHICLE_TRANSPORTATION'.
    - Excluye servicios que no tengan fecha de inicio (`labor_start_date` nula).
    - (Opcional) Filtra solo las ciudades incluidas en `city_list`.

    Parámetros
    ----------
    df_dist : pd.DataFrame
        DataFrame con información de los servicios.
    city_list : list, default=None
        Lista de ciudades a incluir. Si None, no se filtra por ciudad.

    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame filtrado con los servicios válidos.
    """
    df_filtered = df_dist[
        (df_dist['state_service'] != 'CANCELED')
        & df_dist.groupby("service_id")["labor_category"]
                 .transform(lambda x: (x == "VEHICLE_TRANSPORTATION").any())
        & df_dist['labor_start_date'].notna()
    ]

    if city_list is not None:
        df_filtered = df_filtered[df_filtered['city'].isin(city_list)]

    return df_filtered


def filter_labors_by_city(df, city_ids):
    """
    Filtra el DataFrame por uno o varios ID de ciudad.

    Parámetros:
    - df: DataFrame que contenga la columna 'city'.
    - city_ids: int o lista de ints con los ID de ciudad a filtrar.

    Retorna:
    - Subset del df con filas cuyo 'city' esté en city_ids.
    """
    # Aseguramos una lista de IDs
    if isinstance(city_ids, (str)):
        city_list = [city_ids]
    else:
        city_list = list(city_ids)

    filtered = df[df['city'].isin(city_list)]
    
    return filtered


def filter_labors_by_date(df, start_date=None, end_date=None):
    """
    Filtra y ordena el DataFrame por rangos de schedule_date.

    Parámetros:
    - df: DataFrame con columna 'schedule_date'.
    - start_date: fecha inicial (inclusive) en formato 'YYYY-MM-DD'.
    - end_date: fecha final (inclusive) en formato 'YYYY-MM-DD'.

    Retorna:
    - Subset del df con filas cuyo schedule_date está entre start_date y end_date,
      ordenado ascendentemente por schedule_date.
    """
    df_filtered = df.copy()

    # Asegurar datetime
    if not is_datetime64_any_dtype(df_filtered['schedule_date']):
        df_filtered['schedule_date'] = pd.to_datetime(df_filtered['schedule_date'])

    # Filtrar por start_date
    if start_date is not None:
        start = pd.to_datetime(start_date)
        if is_datetime64tz_dtype(df_filtered['schedule_date'].dtype):
            tz = df_filtered['schedule_date'].dt.tz
            start = start.tz_localize(tz)
        df_filtered = df_filtered[df_filtered['schedule_date'] >= start]

    # Filtrar por end_date
    if end_date == 'one day lag':
        end_date = pd.to_datetime(start_date) + timedelta(days=1)
        
    if end_date is not None:
        end = pd.to_datetime(end_date)
        if is_datetime64tz_dtype(df_filtered['schedule_date'].dtype):
            tz = df_filtered['schedule_date'].dt.tz
            end = end.tz_localize(tz)
        df_filtered = df_filtered[df_filtered['schedule_date'] <= end]

    # Ordenar antes de devolver
    df_filtered = df_filtered.sort_values('schedule_date')
    return df_filtered


def end_date_for_day(day_str: str) -> str:
    ts = pd.to_datetime(day_str).normalize()
    return (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def get_df_dia_single_day(df_dist: pd.DataFrame, day_str: str, city_code: str = '149') -> pd.DataFrame:
    start_date = day_str
    end_date   = end_date_for_day(day_str)
    # Filtrado por ciudad y por rango de fechas
    df_city = filter_labors_by_city(df_dist, city_code)
    df_dia  = filter_labors_by_date(df_city, start_date=start_date, end_date=end_date)
    df_dia  = (
        df_dia.query("state_service != 'CANCELED'")
              .sort_values(['service_id', 'labor_start_date'])
              .reset_index(drop=True)
    )
    return df_dia


def filter_labores(df: pd.DataFrame, hour_threshold: int = 7) -> pd.DataFrame:
    """
    Filtra un DataFrame de labores eliminando:
      1. Las programadas antes de una hora específica.
      2. Las con ubicaciones inválidas 'POINT (0 0)'.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'schedule_date', 'map_start_point', 'map_end_point'.
    hour_threshold : int
        Hora mínima (exclusiva) para filtrar labores. Default = 7 (7 AM).

    Retorna
    -------
    pd.DataFrame
        Copia filtrada del DataFrame.
    """
    df_filtered = df[df['schedule_date'].dt.hour > hour_threshold].copy()
    df_filtered = df_filtered[
        (df_filtered['map_start_point'] != 'POINT (0 0)') &
        (df_filtered['map_end_point'] != 'POINT (0 0)')
    ].copy()
    return df_filtered


def flexible_filter(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """
    Dynamically filters a DataFrame using keyword arguments.

    Rules:
      - callable:        mask = val(series)
      - list/tuple/set:  .isin()
      - "notna":         keep non-null
      - "na":            keep null
      - date-string on datetime column ("YYYY-MM-DD"): match on date
      - otherwise:       equality
    """
    # Strict YYYY-MM-DD pattern
    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    mask = pd.Series(True, index=df.index)

    for col, val in filters.items():
        series = df[col]

        # 1. Callable condition
        if callable(val):
            mask &= val(series)
            continue

        # 2. Collection → use .isin
        if isinstance(val, (list, tuple, set)):
            mask &= series.isin(val)
            continue

        # 3. "notna" / "na"
        if isinstance(val, str):
            if val.lower() == "notna":
                mask &= series.notna()
                continue
            if val.lower() == "na":
                mask &= series.isna()
                continue

        # 4. Strict date filter for datetime-like columns
        if (
            isinstance(val, str)
            and series.dtype.kind == "M"
            and DATE_RE.match(val)     # safe date pattern
        ):
            # print(col, val)
            target_date = pd.to_datetime(val).date()
            mask &= (series.dt.date == target_date)
            continue

        # 5. Fallback: exact equality
        mask &= (series.astype(object) == val)

    return df[mask]

